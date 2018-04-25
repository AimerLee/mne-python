# Authors: Alexandre Gramfort <alexandre.gramfort@telecom-paristech.fr>
#          Matti Hamalainen <msh@nmr.mgh.harvard.edu>
#          Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import copy as cp

import numpy as np

from .. import Epochs, compute_proj_evoked, compute_proj_epochs
from ..utils import logger, verbose, warn
from .. import pick_types
from ..io import make_eeg_average_ref_proj
from .ecg import find_ecg_events
from .eog import find_eog_events


def _safe_del_key(dict_, key):
    """Aux function.

    Use this function when preparing rejection parameters
    instead of directly deleting keys.
    """
    if key in dict_:
        del dict_[key]


def _deprecate_average(average):
    # XXX: When deprecation cycle is done, default average option should be
    #      changed in:
    #      - mne/commands/mne_compute_proj_ecg.py
    #      - mne/commands/mne_compute_proj_eog.py
    if average is None:
        warn('The default parameter `average=False` is deprecated'
             ' and will be replaced by `average=True` in 0.17.',
             DeprecationWarning)
    return False if average is None else average


@verbose
def _compute_exg_proj(mode, raw, raw_event, tmin, tmax,
                      n_grad, n_mag, n_eeg, l_freq, h_freq,
                      average, filter_length, n_jobs, ch_name,
                      reject, flat, bads, avg_ref, no_proj, event_id,
                      exg_l_freq, exg_h_freq, tstart, qrs_threshold,
                      filter_method, iir_params, return_drop_log, copy,
                      verbose):
    """Compute SSP/PCA projections for ECG or EOG artifacts."""
    average = _deprecate_average(average)
    raw = raw.copy() if copy else raw
    del copy
    raw.load_data()  # we will filter it later

    if no_proj:
        projs = []
    else:
        projs = cp.deepcopy(raw.info['projs'])
        logger.info('Including %d SSP projectors from raw file'
                    % len(projs))

    if avg_ref:
        eeg_proj = make_eeg_average_ref_proj(raw.info)
        projs.append(eeg_proj)

    if raw_event is None:
        raw_event = raw

    assert mode in ('ECG', 'EOG')  # internal function
    logger.info('Running %s SSP computation' % mode)
    if mode == 'ECG':
        events, _, _ = find_ecg_events(raw_event, ch_name=ch_name,
                                       event_id=event_id, l_freq=exg_l_freq,
                                       h_freq=exg_h_freq, tstart=tstart,
                                       qrs_threshold=qrs_threshold,
                                       filter_length=filter_length)
    else:  # mode == 'EOG':
        events = find_eog_events(raw_event, event_id=event_id,
                                 l_freq=exg_l_freq, h_freq=exg_h_freq,
                                 filter_length=filter_length, ch_name=ch_name,
                                 tstart=tstart)

    # Check to make sure we actually got at least one useable event
    if events.shape[0] < 1:
        warn('No %s events found, returning None for projs' % mode)
        return (None, events) + (([],) if return_drop_log else ())

    logger.info('Computing projector')
    my_info = cp.deepcopy(raw.info)
    my_info['bads'] += bads

    # Handler rejection parameters
    if reject is not None:  # make sure they didn't pass None
        if len(pick_types(my_info, meg='grad', eeg=False, eog=False,
                          ref_meg=False, exclude='bads')) == 0:
            _safe_del_key(reject, 'grad')
        if len(pick_types(my_info, meg='mag', eeg=False, eog=False,
                          ref_meg=False, exclude='bads')) == 0:
            _safe_del_key(reject, 'mag')
        if len(pick_types(my_info, meg=False, eeg=True, eog=False,
                          ref_meg=False, exclude='bads')) == 0:
            _safe_del_key(reject, 'eeg')
        if len(pick_types(my_info, meg=False, eeg=False, eog=True,
                          ref_meg=False, exclude='bads')) == 0:
            _safe_del_key(reject, 'eog')
    if flat is not None:  # make sure they didn't pass None
        if len(pick_types(my_info, meg='grad', eeg=False, eog=False,
                          ref_meg=False, exclude='bads')) == 0:
            _safe_del_key(flat, 'grad')
        if len(pick_types(my_info, meg='mag', eeg=False, eog=False,
                          ref_meg=False, exclude='bads')) == 0:
            _safe_del_key(flat, 'mag')
        if len(pick_types(my_info, meg=False, eeg=True, eog=False,
                          ref_meg=False, exclude='bads')) == 0:
            _safe_del_key(flat, 'eeg')
        if len(pick_types(my_info, meg=False, eeg=False, eog=True,
                          ref_meg=False, exclude='bads')) == 0:
            _safe_del_key(flat, 'eog')

    # exclude bad channels from projection
    # keep reference channels if compensation channels are present
    ref_meg = len(my_info['comps']) > 0
    picks = pick_types(my_info, meg=True, eeg=True, eog=True, ecg=True,
                       ref_meg=ref_meg, exclude='bads')

    raw.filter(l_freq, h_freq, picks=picks, filter_length=filter_length,
               n_jobs=n_jobs, method=filter_method, iir_params=iir_params,
               l_trans_bandwidth=0.5, h_trans_bandwidth=0.5,
               phase='zero-double', fir_design='firwin2')

    epochs = Epochs(raw, events, None, tmin, tmax, baseline=None, preload=True,
                    picks=picks, reject=reject, flat=flat, proj=True)

    drop_log = epochs.drop_log
    if epochs.events.shape[0] < 1:
        warn('No good epochs found, returning None for projs')
        return (None, events) + ((drop_log,) if return_drop_log else ())

    if average:
        evoked = epochs.average()
        ev_projs = compute_proj_evoked(evoked, n_grad=n_grad, n_mag=n_mag,
                                       n_eeg=n_eeg)
    else:
        ev_projs = compute_proj_epochs(epochs, n_grad=n_grad, n_mag=n_mag,
                                       n_eeg=n_eeg, n_jobs=n_jobs)

    for p in ev_projs:
        p['desc'] = mode + "-" + p['desc']

    projs.extend(ev_projs)
    logger.info('Done.')
    return (projs, events) + ((drop_log,) if return_drop_log else ())


@verbose
def compute_proj_ecg(raw, raw_event=None, tmin=-0.2, tmax=0.4,
                     n_grad=2, n_mag=2, n_eeg=2, l_freq=1.0, h_freq=35.0,
                     average=None, filter_length='10s', n_jobs=1,
                     ch_name=None, reject=dict(grad=2000e-13, mag=3000e-15,
                                               eeg=50e-6, eog=250e-6),
                     flat=None, bads=[], avg_ref=False,
                     no_proj=False, event_id=999, ecg_l_freq=5, ecg_h_freq=35,
                     tstart=0., qrs_threshold='auto', filter_method='fft',
                     iir_params=None, copy=True, return_drop_log=False,
                     verbose=None):
    """Compute SSP/PCA projections for ECG artifacts.

    .. note:: raw data will be loaded if it is not already.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw input file.
    raw_event : mne.io.Raw or None
        Raw file to use for event detection (if None, raw is used).
    tmin : float
        Time before event in seconds.
    tmax : float
        Time after event in seconds.
    n_grad : int
        Number of SSP vectors for gradiometers.
    n_mag : int
        Number of SSP vectors for magnetometers.
    n_eeg : int
        Number of SSP vectors for EEG.
    l_freq : float | None
        Filter low cut-off frequency for the data channels in Hz.
    h_freq : float | None
        Filter high cut-off frequency for the data channels in Hz.
    average : bool
        Compute SSP after averaging. Default is False in 0.16 but will change
        to True in 0.17.
    filter_length : str | int | None
        Number of taps to use for filtering.
    n_jobs : int
        Number of jobs to run in parallel.
    ch_name : string (or None)
        Channel to use for ECG detection (Required if no ECG found).
    reject : dict | None
        Epoch rejection configuration (see Epochs).
    flat : dict | None
        Epoch flat configuration (see Epochs).
    bads : list
        List with (additional) bad channels.
    avg_ref : bool
        Add EEG average reference proj.
    no_proj : bool
        Exclude the SSP projectors currently in the fiff file.
    event_id : int
        ID to use for events.
    ecg_l_freq : float
        Low pass frequency applied to the ECG channel for event detection.
    ecg_h_freq : float
        High pass frequency applied to the ECG channel for event detection.
    tstart : float
        Start artifact detection after tstart seconds.
    qrs_threshold : float | str
        Between 0 and 1. qrs detection threshold. Can also be "auto" to
        automatically choose the threshold that generates a reasonable
        number of heartbeats (40-160 beats / min).
    filter_method : str
        Method for filtering ('iir' or 'fft').
    iir_params : dict | None
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details. If iir_params
        is None and method="iir", 4th order Butterworth will be used.
    copy : bool
        If False, filtering raw data is done in place. Defaults to True.
    return_drop_log : bool
        If True, return the drop log.

        .. versionadded:: 0.15
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    proj : list
        Computed SSP projectors.
    ecg_events : ndarray
        Detected ECG events.
    drop_log : list
        The drop log, if requested.

    See Also
    --------
    find_ecg_events
    create_ecg_epochs

    Notes
    -----
    Filtering is applied to the ECG channel while finding events using
    ``ecg_l_freq`` and ``ecg_h_freq``, and then to the ``raw`` instance
    using ``l_freq`` and ``h_freq`` before creation of the epochs used to
    create the projectors.
    """
    return _compute_exg_proj(
        'ECG', raw, raw_event, tmin, tmax, n_grad, n_mag, n_eeg,
        l_freq, h_freq, average, filter_length, n_jobs, ch_name, reject, flat,
        bads, avg_ref, no_proj, event_id, ecg_l_freq, ecg_h_freq, tstart,
        qrs_threshold, filter_method, iir_params, return_drop_log, copy,
        verbose)


@verbose
def compute_proj_eog(raw, raw_event=None, tmin=-0.2, tmax=0.2,
                     n_grad=2, n_mag=2, n_eeg=2, l_freq=1.0, h_freq=35.0,
                     average=None, filter_length='10s', n_jobs=1,
                     reject=dict(grad=2000e-13, mag=3000e-15, eeg=500e-6,
                                 eog=np.inf), flat=None, bads=[],
                     avg_ref=False, no_proj=False, event_id=998, eog_l_freq=1,
                     eog_h_freq=10, tstart=0., filter_method='fft',
                     iir_params=None, ch_name=None, copy=True,
                     return_drop_log=False, verbose=None):
    """Compute SSP/PCA projections for EOG artifacts.

    .. note:: raw data must be preloaded.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw input file.
    raw_event : mne.io.Raw or None
        Raw file to use for event detection (if None, raw is used).
    tmin : float
        Time before event in seconds.
    tmax : float
        Time after event in seconds.
    n_grad : int
        Number of SSP vectors for gradiometers.
    n_mag : int
        Number of SSP vectors for magnetometers.
    n_eeg : int
        Number of SSP vectors for EEG.
    l_freq : float | None
        Filter low cut-off frequency for the data channels in Hz.
    h_freq : float | None
        Filter high cut-off frequency for the data channels in Hz.
    average : bool
        Compute SSP after averaging. Default is False in 0.16 but will change
        to True in 0.17.
    filter_length : str | int | None
        Number of taps to use for filtering.
    n_jobs : int
        Number of jobs to run in parallel.
    reject : dict | None
        Epoch rejection configuration (see Epochs).
    flat : dict | None
        Epoch flat configuration (see Epochs).
    bads : list
        List with (additional) bad channels.
    avg_ref : bool
        Add EEG average reference proj.
    no_proj : bool
        Exclude the SSP projectors currently in the fiff file.
    event_id : int
        ID to use for events.
    eog_l_freq : float
        Low pass frequency applied to the E0G channel for event detection.
    eog_h_freq : float
        High pass frequency applied to the EOG channel for event detection.
    tstart : float
        Start artifact detection after tstart seconds.
    filter_method : str
        Method for filtering ('iir' or 'fft').
    iir_params : dict | None
        Dictionary of parameters to use for IIR filtering.
        See mne.filter.construct_iir_filter for details. If iir_params
        is None and method="iir", 4th order Butterworth will be used.
    ch_name: str | None
        If not None, specify EOG channel name.
    copy : bool
        If False, filtering raw data is done in place. Defaults to True.
    return_drop_log : bool
        If True, return the drop log.

        .. versionadded:: 0.15
    verbose : bool, str, int, or None
        If not None, override default verbose level (see :func:`mne.verbose`
        and :ref:`Logging documentation <tut_logging>` for more).

    Returns
    -------
    proj: list
        Computed SSP projectors.
    eog_events: ndarray
        Detected EOG events.
    drop_log : list
        The drop log, if requested.

    See Also
    --------
    find_eog_events
    create_eog_epochs

    Notes
    -----
    Filtering is applied to the EOG channel while finding events using
    ``eog_l_freq`` and ``eog_h_freq``, and then to the ``raw`` instance
    using ``l_freq`` and ``h_freq`` before creation of the epochs used to
    create the projectors.
    """
    return _compute_exg_proj(
        'EOG', raw, raw_event, tmin, tmax, n_grad, n_mag, n_eeg,
        l_freq, h_freq, average, filter_length, n_jobs, ch_name, reject, flat,
        bads, avg_ref, no_proj, event_id, eog_l_freq, eog_h_freq, tstart,
        'auto', filter_method, iir_params, return_drop_log, copy, verbose)
