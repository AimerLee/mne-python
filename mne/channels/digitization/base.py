from mne.transforms import _coord_frame_name
from mne.io.constants import FIFF

def _format_dig_points(dig):
    """Format the dig points nicely."""
    return [DigPoint(d) for d in dig] if dig is not None else dig


_dig_kind_dict = {
    'cardinal': FIFF.FIFFV_POINT_CARDINAL,
    'hpi': FIFF.FIFFV_POINT_HPI,
    'eeg': FIFF.FIFFV_POINT_EEG,
    'extra': FIFF.FIFFV_POINT_EXTRA,
}
_dig_kind_ints = tuple(sorted(_dig_kind_dict.values()))
_dig_kind_proper = {'cardinal': 'Cardinal',
                    'hpi': 'HPI',
                    'eeg': 'EEG',
                    'extra': 'Extra',
                    'unknown': 'Unknown'}
_dig_kind_rev = {val: key for key, val in _dig_kind_dict.items()}
_cardinal_kind_rev = {1: 'LPA', 2: 'Nasion', 3: 'RPA', 4: 'Inion'}


class DigPoint(dict):
    """Container for a digitization point.

    This is a simple subclass of the standard dict type designed to provide
    a readable string representation.

    Parameters
    ----------
    kind : int
        Digitization kind, e.g. ``FIFFV_POINT_EXTRA``.
    ident : int
        Identifier.
    r : ndarray, shape (3,)
        Position.
    coord_frame : int
        Coordinate frame, e.g. ``FIFFV_COORD_HEAD``.
    """

    def __repr__(self):  # noqa: D105
        if self['kind'] == FIFF.FIFFV_POINT_CARDINAL:
            id_ = _cardinal_kind_rev.get(
                self.get('ident', -1), 'Unknown cardinal')
        else:
            id_ = _dig_kind_proper[
                _dig_kind_rev.get(self.get('kind', -1), 'unknown')]
            id_ = ('%s #%s' % (id_, self.get('ident', -1)))
        id_ = id_.rjust(10)
        cf = _coord_frame_name(self['coord_frame'])
        pos = ('(%0.1f, %0.1f, %0.1f) mm' % tuple(1000 * self['r'])).ljust(25)
        return ('<DigPoint | %s : %s : %s frame>' % (id_, pos, cf))
