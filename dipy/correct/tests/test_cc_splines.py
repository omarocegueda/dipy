from __future__ import print_function
import os
import pickle
import numpy as np
import nibabel as nib
import dipy.align.imwarp as imwarp
from dipy.align import VerbosityLevels
from dipy.align.imaffine import MutualInformationMetric, AffineRegistration
from dipy.align.transforms import regtransforms
import dipy.correct.gradients as gr
from dipy.correct.splines import CubicSplineField

import dipy.viz.regtools as rt
from experiments.registration.regviz import overlay_slices
import experiments.registration.dataset_info as info

from dipy.epicor import (OppositeBlips_CC,
                         OffResonanceFieldEstimator)

floating = np.float64
data_dir = '/home/omar/data/topup_example/'
up_fname = data_dir + "b0_blipup.nii"
#up_fname = info.get_scil(1, 'b0_up_strip')
down_fname = data_dir + "b0_blipdown.nii"
up_unwarped_fname = data_dir+'b0_blipup_unwarped.nii'
t1_fname = info.get_scil(1, 't1_strip')



def dipy_align(static, static_grid2world, moving, moving_grid2world):
    r''' Full rigid registration with Dipy's imaffine module

    Here we implement an extra optimization heuristic: move the geometric
    centers of the images to the origin. Imaffine does not do this by default
    because we want to give the user as much control of the optimization
    process as possible.

    '''
    # Bring the center of the moving image to the origin
    c_moving = tuple(0.5 * np.array(moving.shape, dtype=np.float64))
    c_moving = moving_grid2world.dot(c_moving+(1,))
    correction_moving = np.eye(4, dtype=np.float64)
    correction_moving[:3,3] = -1 * c_moving[:3]
    centered_moving_aff = correction_moving.dot(moving_grid2world)

    # Bring the center of the static image to the origin
    c_static = tuple(0.5 * np.array(static.shape, dtype=np.float64))
    c_static = static_grid2world.dot(c_static+(1,))
    correction_static = np.eye(4, dtype=np.float64)
    correction_static[:3,3] = -1 * c_static[:3]
    centered_static_aff = correction_static.dot(static_grid2world)

    dim = len(static.shape)
    metric = MutualInformationMetric(nbins=32, sampling_proportion=0.3)
    level_iters = [10000, 1000, 100]
    affr = AffineRegistration(metric=metric, level_iters=level_iters)
    affr.verbosity = VerbosityLevels.DEBUG
    #metric.verbosity = VerbosityLevels.DEBUG

    # Registration schedule: center-of-mass then translation, then rigid and then affine
    prealign = 'mass'
    transforms = ['TRANSLATION', 'RIGID', 'AFFINE']

    sol = np.eye(dim + 1)
    for transform_name in transforms:
        transform = regtransforms[(transform_name, dim)]
        print('Optimizing: %s'%(transform_name,))
        x0 = None
        sol = affr.optimize(static, moving, transform, x0,
                              centered_static_aff, centered_moving_aff, starting_affine = prealign)
        prealign = sol.affine.copy()

    # Now bring the geometric centers back to their original location
    fixed = np.linalg.inv(correction_moving).dot(sol.affine.dot(correction_static))
    sol.set_affine(fixed)
    sol.domain_grid2world = static_grid2world
    sol.codomain_grid2world = moving_grid2world

    return sol


def test_epicor_api_cc():
    # Load images
    up_nib = nib.load(up_fname)
    up_affine = up_nib.get_affine()
    direction, spacings = imwarp.get_direction_and_spacings(up_affine, 3)
    up = up_nib.get_data().squeeze().astype(np.float64)

    down_nib = nib.load(down_fname)
    down = down_nib.get_data().squeeze().astype(np.float64)

    radius = 4

    #use_extend_volume = True
    #if use_extend_volume:
    #    up = extend_volume(up, 2 * radius + 1)
    #    down = extend_volume(down, 2 * radius + 1)


    # Preprocess intensities
    up /= up.mean()
    down /= down.mean()
    max_subsampling = 2
    in_shape = np.array(up.shape, dtype=np.int32)
    # This makes no sense, it should be + (max_subsampling - regrid_shape%max_subsampling)%max_subsampling
    regrid_shape = in_shape + max_subsampling
    regrided_spacings = ((in_shape - 1) * spacings) / regrid_shape
    factors = regrided_spacings / spacings
    regrided_up = np.array(gr.regrid(up, factors, regrid_shape)).astype(np.float64)
    regrided_down = np.array(gr.regrid(down, factors, regrid_shape)).astype(np.float64)

    # Configure and run orfield estimation
    pedir_up = np.array((0,1,0), dtype=np.float64)
    pedir_down = np.array((0,-1,0), dtype=np.float64)
    distortion_model = OppositeBlips_CC(radius=radius)
    level_iters = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    #lambdas = np.array([1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e1])/5000.0
    level_iters = [200, 200, 200, 200, 200, 200, 200, 200, 200]
    #level_iters = None
    lambdas = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                       0.5, 0.05, 0.05])*300
    fwhm = np.array([8, 6, 4, 3, 3, 2, 1, 0, 0])
    estimator = OffResonanceFieldEstimator(distortion_model, level_iters=level_iters, lambdas=lambdas, fwhm=fwhm)
    #estimator = OffResonanceFieldEstimator(distortion_model)
    #orfield = estimator.optimize_with_ss(regrided_down, pedir_down, regrided_up, pedir_up, regrided_spacings)
    orfield = estimator.optimize(regrided_down, pedir_down, regrided_up, pedir_up, regrided_spacings)

    # Warp and modulte images
    b  = np.array(orfield.get_volume((0, 0, 0)))
    db = np.array(orfield.get_volume((0, 1, 0)))
    shape = np.array(regrided_up.shape, dtype=np.int32)
    w_up, _m = gr.warp_with_orfield(regrided_up, b, pedir_up, None,
                                    None, None, shape)
    w_down, _m = gr.warp_with_orfield(regrided_down, b, pedir_down, None,
                                      None, None, shape)
    rt.plot_slices(b)
    overlay_slices(w_down, w_up, slice_type=2)
    overlay_slices(w_down*(1.0-db), w_up*(1+db), slice_type=2)


def test_epicor_api_cc_with_scale_space():
    # Load images
    up_nib = nib.load(up_fname)
    up_affine = up_nib.get_affine()
    direction, spacings = imwarp.get_direction_and_spacings(up_affine, 3)
    up = up_nib.get_data().squeeze().astype(np.float64)

    down_nib = nib.load(down_fname)
    down_affine = down_nib.get_affine()
    down = down_nib.get_data().squeeze().astype(np.float64)

    radius = 4

    #use_extend_volume = True
    #if use_extend_volume:
    #    up = extend_volume(up, 2 * radius + 1)
    #    down = extend_volume(down, 2 * radius + 1)


    # Preprocess intensities
    up /= up.mean()
    down /= down.mean()
    max_subsampling = 0
    in_shape = np.array(up.shape, dtype=np.int32)
    # This makes no sense, it should be + (max_subsampling - regrid_shape%max_subsampling)%max_subsampling
    regrid_shape = in_shape + max_subsampling
    regrided_spacings = ((in_shape - 1) * spacings) / regrid_shape
    factors = regrided_spacings / spacings
    regrided_up = np.array(gr.regrid(up, factors, regrid_shape)).astype(np.float64)
    regrided_down = np.array(gr.regrid(down, factors, regrid_shape)).astype(np.float64)

    # Configure and run orfield estimation
    pedir_up = np.array((0,1,0), dtype=np.float64)
    pedir_down = np.array((0,-1,0), dtype=np.float64)
    distortion_model = OppositeBlips_CC(radius=radius)
    level_iters = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    #lambdas = np.array([1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e1])/5000.0
    level_iters = [200, 200, 200, 200, 200, 200, 200, 200, 200]
    #level_iters = None
    lambdas = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                       0.5, 0.05, 0.05])*300
    fwhm = np.array([8, 6, 4, 3, 3, 2, 1, 0, 0])
    estimator = OffResonanceFieldEstimator(distortion_model, level_iters=level_iters, lambdas=lambdas, fwhm=fwhm)
    #estimator = OffResonanceFieldEstimator(distortion_model)
    #orfield = estimator.optimize_with_ss(regrided_down, pedir_down, regrided_up, pedir_up, regrided_spacings)
    orfield = estimator.optimize_with_ss(regrided_down, down_affine, pedir_down, regrided_up, up_affine, pedir_up, regrided_spacings)

    # Warp and modulte images
    b  = np.array(orfield.get_volume((0, 0, 0)))
    db = np.array(orfield.get_volume((0, 1, 0)))
    shape = np.array(regrided_up.shape, dtype=np.int32)
    w_up, _m = gr.warp_with_orfield(regrided_up, b, pedir_up, None,
                                    None, None, shape)
    w_down, _m = gr.warp_with_orfield(regrided_down, b, pedir_down, None,
                                      None, None, shape)
    rt.plot_slices(b)
    overlay_slices(w_down, w_up, slice_type=2)
    overlay_slices(w_down*(1.0-db), w_up*(1+db), slice_type=2)
    return


    up_nib = nib.load(up_fname)
    up_affine = up_nib.get_affine()
    direction, spacings = imwarp.get_direction_and_spacings(up_affine, 3)
    up = up_nib.get_data().squeeze().astype(np.float64)

    down_nib = nib.load(down_fname)
    down_affine = down_nib.get_affine()
    down = down_nib.get_data().squeeze().astype(np.float64)

    radius = 4

    # Preprocess intensities
    up /= up.mean()
    down /= down.mean()

    # Configure and run orfield estimation
    pedir_up = np.array((0,1,0), dtype=np.float64)
    pedir_down = np.array((0,-1,0), dtype=np.float64)
    distortion_model = OppositeBlips_CC(radius=radius)
    level_iters = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    #lambdas = np.array([1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e1])/5000.0
    level_iters = [200, 200, 200, 200, 200, 200, 200, 200, 200]
    #level_iters = None
    lambdas = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                       0.5, 0.05, 0.05])*300
    fwhm = np.array([8, 6, 4, 3, 3, 2, 1, 0, 0])
    estimator = OffResonanceFieldEstimator(distortion_model, level_iters=level_iters, lambdas=lambdas, fwhm=fwhm)
    #estimator = OffResonanceFieldEstimator(distortion_model)
    orfield_coef_fname = 'orfield_coef_ss.p'
    orfield = None
    if os.path.isfile(orfield_coef_fname):
        coef = pickle.load(open(orfield_coef_fname, 'r'))
        kspacing = np.round(estimator.warp_res[-1]/spacings)
        kspacing = kspacing.astype(np.int32)
        kspacing[kspacing < 1] = 1
        orfield = CubicSplineField(up.shape, kspacing)
        orfield.copy_coefficients(coef)
    else:
        orfield = estimator.optimize_with_ss(down, down_affine, pedir_down, up, up_affine, pedir_up, regrided_spacings)
        pickle.dump(np.array(orfield.coef), open(orfield_coef_fname, 'w'))


    # Warp and modulte images
    b  = np.array(orfield.get_volume((0, 0, 0)))
    db = np.array(orfield.get_volume((0, 1, 0)))
    shape = np.array(up.shape, dtype=np.int32)
    w_up, _m = gr.warp_with_orfield(up, b, pedir_up, None,
                                    None, None, shape)
    w_down, _m = gr.warp_with_orfield(down, b, pedir_down, None,
                                      None, None, shape)
    rt.overlay_slices(down, up, slice_type=2);
    rt.plot_slices(b);
    overlay_slices(w_down, w_up, slice_type=2);
    overlay_slices(w_down*(1.0-db), w_up*(1+db), slice_type=2);






