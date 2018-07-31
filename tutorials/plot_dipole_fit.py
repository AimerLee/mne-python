# -*- coding: utf-8 -*-
"""
============================================================
Source localization with equivalent current dipole (ECD) fit
============================================================

This shows how to fit a dipole using mne-python.

For a comparison of fits between MNE-C and mne-python, see:

    https://gist.github.com/Eric89GXL/ca55f791200fe1dc3dd2

"""

from os import path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
from mne.forward import make_forward_dipole
from mne.evoked import combine_evoked
from mne.simulation import simulate_evoked

data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
fname_ave = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
fname_cov = op.join(data_path, 'MEG', 'sample', 'sample_audvis-cov.fif')
fname_bem = op.join(subjects_dir, 'sample', 'bem', 'sample-5120-bem-sol.fif')
fname_trans = op.join(data_path, 'MEG', 'sample',
                      'sample_audvis_raw-trans.fif')
fname_surf_lh = op.join(subjects_dir, 'sample', 'surf', 'lh.white')

###############################################################################
# Let's localize the N100m (using MEG only)
evoked = mne.read_evokeds(fname_ave, condition='Right Auditory',
                          baseline=(None, 0))
evoked.pick_types(meg=True, eeg=False)
evoked_full = evoked.copy()
evoked.crop(0.07, 0.08)

# Fit a dipole
dip = mne.fit_dipole(evoked, fname_cov, fname_bem, fname_trans)[0]

# Plot the result in 3D brain with the MRI image.
dip.plot_locations(fname_trans, 'sample', subjects_dir, mode='orthoview')

# xxxxxxxx

# Finally, explore several points in time
field_map = evoked_dip.plot_field(maps, time=t_max)
field_map.scene.x_minus_view()
mlab.savefig(fig_folder + '/%s_ica_comp_topo.png' % subject)

dip, _ = mne.fit_dipole(evoked_dip,
                               noise_cov, bem_fname, trans_fname)
fig = dip.plot_locations(trans_fname, subject=subject, subjects_dir=subjects_dir)
fig.savefig(fig_folder + '/%s_dip_fit.png' % subject)

trans = mne.read_trans(trans_fname)
mni_pos = mne.head_to_mni(dip.pos, mri_head_t=trans,
                          subject=subject, subjects_dir=subjects_dir)

mri_pos = mne.transforms.apply_trans(trans, dip.pos) * 1e3
t1_fname = os.path.join(subjects_dir, subject, 'mri', 'T1.mgz')
t1 = nib.load(t1_fname)
vox2ras_tkr = t1.header.get_vox2ras_tkr()
ras2vox_tkr = linalg.inv(vox2ras_tkr)
vox2ras = t1.header.get_vox2ras()
mri_pos = mne.transforms.apply_trans(ras2vox_tkr, mri_pos)  # in vox
mri_pos = mne.transforms.apply_trans(vox2ras, mri_pos)  # in RAS

from nilearn.plotting import plot_anat
from nilearn.datasets import load_mni152_template

fig = plot_anat(t1_fname, cut_coords=mri_pos[0], title='Subj: %s' % subject)
fig.savefig(fig_folder + '/%s_mri_coord.png' % subject)

template = load_mni152_template()
fig = plot_anat(template, cut_coords=mni_pos[0], title='Subj: %s (MNI Space)' % subject)
fig.savefig(fig_folder + '/%s_mni_coord.png' % subject)

###############################################################################
# Calculate and visualise magnetic field predicted by dipole with maximum GOF
# and compare to the measured data, highlighting the ipsilateral (right) source
fwd, stc = make_forward_dipole(dip, fname_bem, evoked.info, fname_trans)
pred_evoked = simulate_evoked(fwd, stc, evoked.info, cov=None, nave=np.inf)

# find time point with highest GOF to plot
best_idx = np.argmax(dip.gof)
best_time = dip.times[best_idx]
print('Highest GOF %0.1f%% at t=%0.1f ms with confidence volume %0.1f cm^3'
      % (dip.gof[best_idx], best_time * 1000,
         dip.conf['vol'][best_idx] * 100 ** 3))
# remember to create a subplot for the colorbar
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=[10., 3.4])
vmin, vmax = -400, 400  # make sure each plot has same colour range

# first plot the topography at the time of the best fitting (single) dipole
plot_params = dict(times=best_time, ch_type='mag', outlines='skirt',
                   colorbar=False, time_unit='s')
evoked.plot_topomap(time_format='Measured field', axes=axes[0], **plot_params)

# compare this to the predicted field
pred_evoked.plot_topomap(time_format='Predicted field', axes=axes[1],
                         **plot_params)

# Subtract predicted from measured data (apply equal weights)
diff = combine_evoked([evoked, -pred_evoked], weights='equal')
plot_params['colorbar'] = True
diff.plot_topomap(time_format='Difference', axes=axes[2], **plot_params)
plt.suptitle('Comparison of measured and predicted fields '
             'at {:.0f} ms'.format(best_time * 1000.), fontsize=16)

###############################################################################
# Estimate the time course of a single dipole with fixed position and
# orientation (the one that maximized GOF) over the entire interval
dip_fixed = mne.fit_dipole(evoked_full, fname_cov, fname_bem, fname_trans,
                           pos=dip.pos[best_idx], ori=dip.ori[best_idx])[0]
dip_fixed.plot(time_unit='s')
