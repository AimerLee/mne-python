"""
======================================
Plotting sensor layouts of EEG Systems
======================================

Show sensor layouts of different EEG systems.

XXX: things to refer properly:
:ref:`example_eeg_sensors_on_the_scalp`
:ref:`tut_erp`
"""
# Author: Joan Massich <mailsik@gmail.com>
#
# License: BSD (3-clause)

import mne
from mne.datasets import sample
from mne.channels.montage import _set_montage, get_builtin_montages
from mne.viz import plot_alignment
from mayavi import mlab

# print(__doc__)

data_path = mne.datasets.sample.data_path()
subjects_dir = data_path + '/subjects'

###############################################################################
# check all montages
#

current_montage = get_builtin_montages()[0]
# for current_montage in get_builtin_montages():
montage = mne.channels.read_montage(current_montage)
info = mne.create_info(ch_names=montage.ch_names,
                       sfreq=1,
                       ch_types='eeg',
                       montage=montage)

plot_alignment(info, trans=None, subject='fsaverage', dig=False,
               eeg=['projected'], meg=[],
               coord_frame='head', subjects_dir=subjects_dir)

###############################################################################
# Questions I've
#
# 1 - What happens with `_set_montage` and therefore `create_info` when
#     len(info.ch_names) != len(montage.ch_names)
#
# 2 - 


###############################################################################
# TODO

# trans = mne.read_trans(data_path + '/MEG/sample/sample_audvis_raw-trans.fif')
# raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif')
# raw.load_data()
# raw.pick_types(meg=False, eeg=True, eog=True)
# print(raw.info['chs'][0]['loc'])

# # Nothing
# fig = plot_alignment(raw.info, trans=None, subject='sample', dig=False,
#                      eeg=['original', 'projected'], meg=[],
#                      coord_frame='head', subjects_dir=subjects_dir)
# mlab.view(135, 80)

# fig = plot_alignment(info, trans=None, subject='fsaverage', dig=False,
#                      eeg=['projected'], meg=[],
#                      coord_frame='head', subjects_dir=subjects_dir)
# mlab.view(135, 80)

# # With montage
# montage = mne.channels.read_montage('standard_1020')
# # raw.set_montage(montage, set_dig=True)

# _set_montage(raw.info, montage, update_ch_names=True, set_dig=True)
# fig = plot_alignment(raw.info, trans, subject='sample', dig=False,
#                      eeg=['original', 'projected'], meg=[],
#                      coord_frame='head', subjects_dir=subjects_dir)
# mlab.view(135, 80)

# # with a name
# # raw.set_montage('mgh60')  # test loading with string argument
# montage = mne.channels.read_montage('standard_1020')
# _set_montage(raw.info, montage, update_ch_names=True, set_dig=True)
# fig = plot_alignment(raw.info, trans, subject='sample', dig=False,
#                      eeg=['original', 'projected'], meg=[],
#                      coord_frame='head', subjects_dir=subjects_dir)
# mlab.view(135, 80)
