"""
.. _tut-forward:

Head model and forward computation
==================================

The aim of this tutorial is to be a getting started for forward
computation.

For more extensive details and presentation of the general
concepts for forward modeling. See :ref:`ch_forward`.

"""

import os.path as op
import mne
from mne.datasets import sample
data_path = sample.data_path()

# the raw file containing the channel location + types
raw_fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
# The paths to Freesurfer reconstructions
subjects_dir = data_path + '/subjects'
subject = 'sample'

###############################################################################
# Computing the forward operator
# ------------------------------
#
# To compute a forward operator we need:
#
#    - a ``-trans.fif`` file that contains the coregistration info.
#    - a source space
#    - the :term:`BEM` surfaces

###############################################################################
# Compute and visualize BEM surfaces
# ----------------------------------
#
# The :term:`BEM` surfaces are the triangulations of the interfaces between
# different tissues needed for forward computation. These surfaces are for
# example the inner skull surface, the outer skull surface and the outer skin
# surface, a.k.a. scalp surface.
#
# Computing the BEM surfaces requires FreeSurfer and makes use of either of
# the two following command line tools:
#
#   - :ref:`gen_mne_watershed_bem`
#   - :ref:`gen_mne_flash_bem`
#
# Or by calling in a Python script one of the functions
# :func:`mne.bem.make_watershed_bem` or :func:`mne.bem.make_flash_bem`.
#
# Here we'll assume it's already computed. It takes a few minutes per subject.
#
# For EEG we use 3 layers (inner skull, outer skull, and skin) while for
# MEG 1 layer (inner skull) is enough.
#
# Let's look at these surfaces. The function :func:`mne.viz.plot_bem`
# assumes that you have the the *bem* folder of your subject FreeSurfer
# reconstruction the necessary files.

mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                 brain_surfaces='white', orientation='coronal')

###############################################################################
# Visualization the coregistration
# --------------------------------
#
# The coregistration is operation that allows to position the head and the
# sensors in a common coordinate system. In the MNE software the transformation
# to align the head and the sensors in stored in a so-called **trans file**.
# It is a FIF file that ends with ``-trans.fif``. It can be obtained with
# :func:`mne.gui.coregistration` (or its convenient command line
# equivalent :ref:`gen_mne_coreg`), or mrilab if you're using a Neuromag
# system.
#
# For the Python version see :func:`mne.gui.coregistration`
#
# Here we assume the coregistration is done, so we just visually check the
# alignment with the following code.

# The transformation file obtained by coregistration
trans = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'

info = mne.io.read_info(raw_fname)
# Here we look at the dense head, which isn't used for BEM computations but
# is useful for coregistration.
mne.viz.plot_alignment(info, trans, subject=subject, dig=True,
                       meg=['helmet', 'sensors'], subjects_dir=subjects_dir,
                       surfaces='head-dense')

###############################################################################
# .. _plot_forward_source_space:
#
# Compute Source Space
# --------------------
#
# The source space defines the position and orientation of the candidate source
# locations. There are two types of source spaces:
#
# - **source-based** source space when the candidates are confined to a
#   surface.
#
# - **volumetric or discrete** source space when the candidates are discrete,
#   arbitrarily located source points bounded by the surface.
#
# **Source-based** source space is computed using
# :func:`mne.setup_source_space`, while **volumetric** source space is computed
# using :func:`mne.setup_volume_source_space`.
#
# We will now compute a source-based source space with an OCT-6 resolution.
# See :ref:`setting_up_source_space` for details on source space definition
# and spacing parameter.

src = mne.setup_source_space(subject, spacing='oct6',
                             subjects_dir=subjects_dir, add_dist=False)
print(src)

###############################################################################
# The surface based source space ``src`` contains two parts, one for the left
# hemisphere (4098 locations) and one for the right hemisphere
# (4098 locations). Sources can be visualized on top of the BEM surfaces
# in purple.

mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                 brain_surfaces='white', src=src, orientation='coronal')

###############################################################################
# To compute a volume based source space defined with a grid of candidate
# dipoles inside a sphere of radius 90mm centered at (0.0, 0.0, 40.0)
# you can use the following code.
# Obviously here, the sphere is not perfect. It is not restricted to the
# brain and it can miss some parts of the cortex.

sphere = (0.0, 0.0, 40.0, 90.0)
vol_src = mne.setup_volume_source_space(subject, subjects_dir=subjects_dir,
                                        sphere=sphere)
print(vol_src)

mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                 brain_surfaces='white', src=vol_src, orientation='coronal')

###############################################################################
# To compute a volume based source space defined with a grid of candidate
# dipoles inside the brain (requires the :term:`BEM` surfaces) you can use the
# following.

surface = op.join(subjects_dir, subject, 'bem', 'inner_skull.surf')
vol_src = mne.setup_volume_source_space(subject, subjects_dir=subjects_dir,
                                        surface=surface)
print(vol_src)

mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,
                 brain_surfaces='white', src=vol_src, orientation='coronal')

###############################################################################
# With the surface-based source space only sources that lie in the plotted MRI
# slices are shown. Let's see how to view all sources in 3D.

fig = mne.viz.plot_alignment(subject=subject, subjects_dir=subjects_dir,
                             surfaces='white', coord_frame='head',
                             src=src)
mne.viz.set_3d_view(fig, azimuth=173.78, elevation=101.75,
                    distance=0.30, focalpoint=(-0.03, -0.01, 0.03))

###############################################################################
# .. _plot_forward_compute_forward_solution:
#
# Compute forward solution
# ------------------------
#
# We can now compute the forward solution.
# To reduce computation we'll just compute a single layer BEM (just inner
# skull) that can then be used for MEG (not EEG).
#
# We specify if we want a one-layer or a three-layer BEM using the
# conductivity parameter.
#
# The BEM solution requires a BEM model which describes the geometry
# of the head the conductivities of the different tissues.

conductivity = (0.3,)  # for single layer
# conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(subject='sample', ico=4,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

###############################################################################
# Note that the :term:`BEM` does not involve any use of the trans file. The BEM
# only depends on the head geometry and conductivities.
# It is therefore independent from the MEG data and the head position.
#
# Let's now compute the forward operator, commonly referred to as the
# gain or leadfield matrix.
#
# See :func:`mne.make_forward_solution` for details on parameters meaning.

fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                meg=True, eeg=False, mindist=5.0, n_jobs=2)
print(fwd)

###############################################################################
# We can explore the content of fwd to access the numpy array that contains
# the gain matrix.

leadfield = fwd['sol']['data']
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

###############################################################################
# To extract the numpy array containing the forward operator corresponding to
# the source space `fwd['src']` with cortical orientation constraint
# we can use the following:

fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                         use_cps=True)
leadfield = fwd_fixed['sol']['data']
print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

###############################################################################
# This is equivalent to the following code that explicitly applies the
# forward operator to a source estimate composed of the identity operator:

import numpy as np  # noqa

n_dipoles = leadfield.shape[1]
vertices = [src_hemi['vertno'] for src_hemi in fwd_fixed['src']]
stc = mne.SourceEstimate(1e-9 * np.eye(n_dipoles), vertices, tmin=0., tstep=1)
leadfield = mne.apply_forward(fwd_fixed, stc, info).data / 1e-9

###############################################################################
# To save to disk a forward solution you can use
# :func:`mne.write_forward_solution` and to read it back from disk
# :func:`mne.read_forward_solution`. Don't forget that FIF files containing
# forward solution should end with *-fwd.fif*.
#
# To get a fixed-orientation forward solution, use
# :func:`mne.convert_forward_solution` to convert the free-orientation
# solution to (surface-oriented) fixed orientation.

###############################################################################
# Exercise
# --------
#
# By looking at
# :ref:`sphx_glr_auto_examples_forward_plot_forward_sensitivity_maps.py`
# plot the sensitivity maps for EEG and compare it with the MEG, can you
# justify the claims that:
#
#   - MEG is not sensitive to radial sources
#   - EEG is more sensitive to deep sources
#
# How will the MEG sensitivity maps and histograms change if you use a free
# instead if a fixed/surface oriented orientation?
#
# Try this changing the mode parameter in :func:`mne.sensitivity_map`
# accordingly. Why don't we see any dipoles on the gyri?
