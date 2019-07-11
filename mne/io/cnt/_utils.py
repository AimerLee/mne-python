# Author: Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)

from struct import Struct
from collections import namedtuple
from math import modf
from datetime import datetime
import numpy as np
from os import SEEK_END

from ...utils import warn


def _read_teeg(f, teeg_offset):
    """
    Read TEEG structure from an open CNT file.

    # from TEEG structure in http://paulbourke.net/dataformats/eeg/
    typedef struct {
       char Teeg;       /* Either 1 or 2                    */
       long Size;       /* Total length of all the events   */
       long Offset;     /* Hopefully always 0               */
    } TEEG;
    """
    # we use a more descriptive names based on TEEG doc comments
    Teeg = namedtuple('Teeg', 'event_type total_length offset')
    teeg_parser = Struct('<Bll')

    f.seek(teeg_offset)
    return Teeg(*teeg_parser.unpack(f.read(teeg_parser.size)))


CNTEventType1 = namedtuple('CNTEventType1',
                           ('StimType KeyBoard KeyPad_Accept Offset'))
# typedef struct {
#    unsigned short StimType;     /* range 0-65535                           */
#    unsigned char  KeyBoard;     /* range 0-11 corresponding to fcn keys +1 */
#    char KeyPad_Accept;          /* 0->3 range 0-15 bit coded response pad  */
#                                 /* 4->7 values 0xd=Accept 0xc=Reject       */
#    long Offset;                 /* file offset of event                    */
# } EVENT1;


CNTEventType2 = namedtuple('CNTEventType2',
                           ('StimType KeyBoard KeyPad_Accept Offset Type '
                            'Code Latency EpochEvent Accept2 Accuracy'))
# unsigned short StimType; /* range 0-65535                           */
# unsigned char  KeyBoard; /* range 0-11 corresponding to fcn keys +1 */
# char KeyPad_Accept;      /* 0->3 range 0-15 bit coded response pad  */
#                          /* 4->7 values 0xd=Accept 0xc=Reject       */
# long Offset;             /* file offset of event                    */
# short Type;
# short Code;
# float Latency;
# char EpochEvent;
# char Accept2;
# char Accuracy;


# needed for backward compat: EVENT type 3 has the same structure as type 2
CNTEventType3 = namedtuple('CNTEventType3',
                           ('StimType KeyBoard KeyPad_Accept Offset Type '
                            'Code Latency EpochEvent Accept2 Accuracy'))


def _get_event_parser(event_type):
    if event_type == 1:
        event_maker = CNTEventType1
        struct_pattern = '<HBcl'
    elif event_type == 2:
        event_maker = CNTEventType2
        struct_pattern = '<HBclhhfccc'
    elif event_type == 3:
        event_maker = CNTEventType3
        struct_pattern = '<HBclhhfccc'  # Same as event type 2
    else:
        raise ValueError('unknown CNT even type %s' % event_type)

    def parser(buffer):
        struct = Struct(struct_pattern)
        for chunk in struct.iter_unpack(buffer):
            yield event_maker(*chunk)

    return parser


def _session_date_2_meas_date(session_date, date_format):
    try:
        frac_part, int_part = modf(datetime
                                   .strptime(session_date, date_format)
                                   .timestamp())
    except ValueError:
        warn('  Could not parse meas date from the header. Setting to None.')
        return None
    else:
        return (int_part, frac_part)


def _compute_robust_event_table_position(fid, data_format):
    """Compute `event_table_position`.

    When recording event_table_position is computed (as accomulation). If the
    file recording is large then this value overflows and ends up pointing
    somewhere else. (SEE #gh-6535)

    If the file is smaller than 2G the value in the SETUP is returned.
    Otherwise, the address of the table position is computed from:
    n_samples, n_channels, and the bytes size.

    Returns
    -------
    x_xxxxxxxx : xxxxxxxx xx Xxxxxxxxxxx
        Xxx xxxxxxxxxxx.
    """
    SETUP_NCHANNELS_OFFSET = 370
    SETUP_NSAMPLES_OFFSET = 864
    SETUP_EVENTTABLEPOS_OFFSET = 886

    def get_most_possible_sol(fid, possible_n_bytes, n_samples, n_channels):
        """Find the most possible solution

        Since both event table position and n_bytes has many possible values,
        and n_samples might be not so accurate, distance between the possible
        event table position and calculated event table position is used to
        find the most possible combination of the event table position and
        n_bytes.

        When the distance of the solution find is not equals to 0, there is
        a mismatch between the n_samples and the event_table_pos.
        """
        sol_table = []
        for event_table_pos in possible_event_table_pos(fid):
            for n_bytes in possible_n_bytes:
                calc_event_table_pos = 900 + 75 * n_channels + \
                                       n_bytes * n_channels * n_samples
                distance = abs(calc_event_table_pos - event_table_pos)

                if distance == 0:
                    return event_table_pos, n_bytes, distance
                sol_table.append((event_table_pos, n_bytes, distance))
        return sorted(sol_table, key=lambda x: x[2])[0]

    def possible_event_table_pos(fid):
        """Yield all the possible event table position"""
        fid.seek(SETUP_EVENTTABLEPOS_OFFSET)
        event_table_pos = int(np.frombuffer(fid.read(4), dtype='<u4')[0])
        file_size = fid.seek(0, SEEK_END)

        while event_table_pos <= file_size:
            yield event_table_pos
            event_table_pos = event_table_pos + np.iinfo(np.uint32).max + 1

    fid_origin = fid.tell()  # save the state

    fid.seek(SETUP_NSAMPLES_OFFSET)
    n_samples = int(np.frombuffer(fid.read(4), dtype='<i4')[0])

    fid.seek(SETUP_NCHANNELS_OFFSET)
    n_channels = int(np.frombuffer(fid.read(2), dtype='<u2')[0])

    if data_format == 'auto':
        possible_n_bytes = [2, 4]
    elif data_format == 'int16':
        possible_n_bytes = [2]
    elif data_format == 'int32':
        possible_n_bytes = [4]
    else:
        raise Exception("Correct data format required: 'auto','int16' or 'int32'.")

    event_table_pos, n_bytes, distance = get_most_possible_sol(fid, possible_n_bytes,
                                                               n_samples, n_channels)
    if distance != 0:
        warn("Metadata doesn't match so well, the samples might not loaded completely.")
    fid.seek(fid_origin)  # restore the state
    return n_channels, n_samples, event_table_pos, n_bytes
