from __future__ import print_function
import os
import pickle
import numpy as np
import numpy.linalg as npl
import nibabel as nib
import dipy.align.imwarp as imwarp
from dipy.align import VerbosityLevels
from dipy.align.imaffine import (MutualInformationMetric,
                                 AffineRegistration,
                                 AffineMap)
from dipy.align.transforms import regtransforms
import dipy.correct.gradients as gr
from dipy.correct.splines import CubicSplineField

import dipy.viz.regtools as rt
from experiments.registration.regviz import overlay_slices
import experiments.registration.dataset_info as info

from dipy.correct.epicor import (OppositeBlips_CC,
                                 OppositeBlips_CC_Motion,
                                 SingleEPI_CC,
                                 SingleEPI_ECC,
                                 OffResonanceFieldEstimator)

floating = np.float64
data_dir = '/home/omar/data/topup_example/'
up_fname = data_dir + "b0_blipup.nii"
up_strip_fname = info.get_scil(1, 'b0_up_strip')
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


def test_epicor_OPPOSITE_BLIPS_CC_REGRID():
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


def test_epicor_OPPOSITE_BLIPS_CC_SS():
    # Load images
    up_nib = nib.load(up_fname)
    up_affine = up_nib.get_affine()
    direction, spacings = imwarp.get_direction_and_spacings(up_affine, 3)
    up = up_nib.get_data().squeeze().astype(np.float64)

    down_nib = nib.load(down_fname)
    down_affine = down_nib.get_affine()
    down = down_nib.get_data().squeeze().astype(np.float64)

    radius = 4

    spacings[:] = 1

    up_affine = np.diag([spacings[0], spacings[1], spacings[2], 1.0])
    down_affine = np.diag([spacings[0], spacings[1], spacings[2], 1.0])

    # Preprocess intensities
    up /= up.mean()
    down /= down.mean()

    # Configure and run orfield estimation
    pedir_up = np.array((0,1,0), dtype=np.float64)
    pedir_down = np.array((0,-1,0), dtype=np.float64)
    distortion_model = OppositeBlips_CC(radius=radius)
    #level_iters = [200, 200, 200, 200, 200, 200, 200, 200, 200]
    #level_iters = [100, 1, 1, 1, 1, 1, 1, 1, 1]
    level_iters = [100, 100, 100, 100, 50, 25, 25, 20, 10]
    lambdas = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                       0.5, 0.05, 0.05])*400
    fwhm = np.array([8, 6, 4, 3, 3, 2, 1, 0, 0])
    estimator = OffResonanceFieldEstimator(distortion_model, level_iters=level_iters, lambdas=lambdas, fwhm=fwhm)

    orfield_coef_fname = 'orfield_coef_new_ss_fix_sp.p'
    orfield = None
    if os.path.isfile(orfield_coef_fname):
        coef = pickle.load(open(orfield_coef_fname, 'r'))
        kspacing = np.round(estimator.warp_res[-1]/spacings)
        kspacing = kspacing.astype(np.int32)
        kspacing[kspacing < 1] = 1
        orfield = CubicSplineField(up.shape, kspacing)
        orfield.copy_coefficients(coef)
    else:
        orfield = estimator.optimize_with_ss(down, down_affine, pedir_down, up, up_affine, pedir_up, spacings)
        pickle.dump(np.array(orfield.coef), open(orfield_coef_fname, 'w'))

    # Warp and modulte images
    b  = np.array(orfield.get_volume((0, 0, 0)))
    db = np.array(orfield.get_volume((0, 1, 0)))
    shape = np.array(up.shape, dtype=np.int32)
    w_up, _m = gr.warp_with_orfield(up, b, pedir_up, None,
                                    None, None, shape)
    w_down, _m = gr.warp_with_orfield(down, b, pedir_down, None,
                                      None, None, shape)
    rt.plot_slices(b)
    overlay_slices(w_down, w_up, slice_type=2)

    Jdown = (1.0-db)
    Jup = (1.0+db)
    Jdown[Jdown<0] = 0
    Jup[Jup<0] = 0
    rt.overlay_slices(w_down*Jdown, w_up*Jup, slice_type=2);

    #overlay_slices(w_down*(1.0-db), w_up*(1+db), slice_type=2)


def extend(vol):
    margin = 8
    half = margin // 2
    new_shape = np.array(vol.shape) + margin
    new_vol = np.zeros(shape = new_shape, dtype=np.float64)

    new_vol.shape
    new_vol[half:-half, half:-half, half:-half] = vol
    return new_vol




def test_epicor_OPPOSITE_BLIPS_CC_SS_MOTION():
    # Load images
    up_nib = nib.load(up_fname)
    up_affine = up_nib.get_affine()
    up = up_nib.get_data().squeeze().astype(np.float64)
    up = extend(up)
    up /= up.mean()

    down_nib = nib.load(down_fname)
    down_affine = down_nib.get_affine()
    down = down_nib.get_data().squeeze().astype(np.float64)
    down = extend(down)
    down /= down.mean()

    pedir_up = np.array((0,1,0), dtype=np.float64)
    pedir_down = np.array((0,-1,0), dtype=np.float64)

    dir_up, spacings_up = imwarp.get_direction_and_spacings(up_affine, 3)
    dir_down, spacings_down = imwarp.get_direction_and_spacings(down_affine, 3)

    radius = 4

    spacings = spacings_down.copy()

    # Configure and run orfield estimation
    distortion_model = OppositeBlips_CC_Motion(radius=radius)
    level_iters = np.array([300, 300, 300, 300,
                            250, 200, 100], dtype=np.int32)
    lambdas = np.array([0.0, 0.0, 0.0, 0.0,
                        0.01, 0.01, 0.005])*((radius+1)**3)*0.5
    fwhm = np.array([8, 6, 4, 3,
                     2, 1, 0], dtype=np.float64)
    step_lengths = np.array([0.1, 0.05, 0.05, 0.5,
                             0.05, 0.05, 0.05])
    warp_res = np.array([20, 16, 14, 12,
                         6, 4, 4], dtype=np.float64)
    subsampling = [2, 2, 2, 2,
                   1, 1, 1]
    estimator = OffResonanceFieldEstimator(distortion_model,
                                           level_iters=level_iters,
                                           lambdas=lambdas,
                                           fwhm=fwhm,
                                           step_lengths=step_lengths,
                                           warp_res=warp_res,
                                           subsampling=subsampling)

    orfield_coef_fname = 'orfield_coef_trans_spsq_r4_postbest.p'
    orfield = None
    if os.path.isfile(orfield_coef_fname):
        coef, R = pickle.load(open(orfield_coef_fname, 'r'))
        kspacing = np.round(estimator.warp_res[-1]/spacings)
        kspacing = kspacing.astype(np.int32)
        kspacing[kspacing < 1] = 1
        orfield = CubicSplineField(up.shape, kspacing)
        orfield.copy_coefficients(coef)
    else:
        orfield, R = estimator.optimize_with_ss_motion(down, down_affine, pedir_down, up, up_affine, pedir_up, spacings)
        pickle.dump(tuple([np.array(orfield.coef), R]), open(orfield_coef_fname, 'w'))

    # Warp and modulte images
    b  = np.array(orfield.get_volume((0, 0, 0)))

    shape = np.array(down.shape, dtype=np.int32)

    Ain = None
    Aout = npl.inv(up_affine).dot(down_affine).dot(R)
    Adisp = Aout

    w_up, _m = gr.warp_with_orfield(up, b, pedir_up, Ain,
                                        Aout, Adisp, shape)

    Ain = None
    Aout = npl.inv(down_affine).dot(down_affine)
    Adisp = Aout

    w_down, _m = gr.warp_with_orfield(down, b, pedir_down, Ain,
                                          Aout, Adisp, shape)

    if True:
        rt.plot_slices(b)
        overlay_slices(w_down, w_up, slice_type=2)
        gb = np.zeros(shape=b.shape + (3,), dtype=np.float64)
        gb[...,0] = orfield.get_volume((1,0,0))
        gb[...,1] = orfield.get_volume((0,1,0))
        gb[...,2] = orfield.get_volume((0,0,1))


        Jdown = gb[...,0]*pedir_down[0] + gb[...,1]*pedir_down[1] + gb[...,2]*pedir_down[2] + 1
        Jup = gb[...,0]*pedir_up[0] + gb[...,1]*pedir_up[1] + gb[...,2]*pedir_up[2] + 1
        overlay_slices(w_down*Jdown, w_up*Jup, slice_type=2)

    if False:
        d0 = estimator.f1_ss.get_image(6)
        u0 = estimator.f2_ss.get_image(6)
        Ain = None
        Aout = npl.inv(up_affine).dot(down_affine).dot(R)
        Adisp = Aout
        w_up, _m = gr.warp_with_orfield(u0, b, pedir_up, Ain,
                                            Aout, Adisp, shape)
        Ain = None
        Aout = npl.inv(down_affine).dot(down_affine)
        Adisp = Aout

        w_down, _m = gr.warp_with_orfield(d0, b, pedir_down, Ain,
                                              Aout, Adisp, shape)
        overlay_slices(w_down*Jdown, w_up*Jup, slice_type=2)





def test_epicor_SINGLE_CC_SS():
    # Load images
    epi_fname = up_fname
    nonepi_fname = up_unwarped_fname

    epi_nib = nib.load(epi_fname)
    epi_affine = epi_nib.get_affine()
    direction, spacings = imwarp.get_direction_and_spacings(epi_affine, 3)
    epi = epi_nib.get_data().squeeze().astype(np.float64)

    nonepi_nib = nib.load(nonepi_fname)
    nonepi_affine = nonepi_nib.get_affine()
    nonepi = nonepi_nib.get_data().squeeze().astype(np.float64)

    radius = 4

    # Preprocess intensities
    epi /= epi.mean()
    nonepi /= nonepi.mean()

    # Configure and run orfield estimation
    pedir_epi = np.array((0,1,0), dtype=np.float64)
    pedir_nonepi = np.array((0,0,0), dtype=np.float64)
    distortion_model = SingleEPI_CC(radius=radius)
    level_iters = np.array([200, 200, 200, 200, 200, 200, 200, 200, 200])
    lambdas = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                       0.5, 0.05, 0.05])*300
    fwhm = np.array([8, 6, 4, 3, 3, 2, 1, 0, 0])
    estimator = OffResonanceFieldEstimator(distortion_model, level_iters=level_iters, lambdas=lambdas, fwhm=fwhm)

    orfield_coef_fname = 'orfield_coef_single_cc_ss.p'
    orfield = None
    if os.path.isfile(orfield_coef_fname):
        coef = pickle.load(open(orfield_coef_fname, 'r'))
        kspacing = np.round(estimator.warp_res[-1]/spacings)
        kspacing = kspacing.astype(np.int32)
        kspacing[kspacing < 1] = 1
        orfield = CubicSplineField(epi.shape, kspacing)
        orfield.copy_coefficients(coef)
    else:
        orfield = estimator.optimize_with_ss(epi, epi_affine, pedir_epi,
                                             nonepi, nonepi_affine, pedir_nonepi,
                                             spacings)
        pickle.dump(np.array(orfield.coef), open(orfield_coef_fname, 'w'))

    # Warp and modulte images
    b  = np.array(orfield.get_volume((0, 0, 0)))
    db = np.array(orfield.get_volume((0, 1, 0)))
    shape = np.array(epi.shape, dtype=np.int32)
    w_epi, _m = gr.warp_with_orfield(epi, b, pedir_epi, None,
                                    None, None, shape)
    rt.plot_slices(b)
    overlay_slices(nonepi, w_epi, slice_type=2)
    overlay_slices(nonepi, w_epi*(1+db), slice_type=2)



def test_epicor_SINGLE_ECC_SS():
    # Load images
    epi_fname = up_strip_fname
    #epi_fname = up_unwarped_fname
    nonepi_fname = t1_fname

    epi_nib = nib.load(epi_fname)
    epi_affine = epi_nib.get_affine()
    direction, spacings = imwarp.get_direction_and_spacings(epi_affine, 3)
    epi = epi_nib.get_data().squeeze().astype(np.float64)

    nonepi_nib = nib.load(nonepi_fname)
    nonepi_affine = nonepi_nib.get_affine()
    nonepi = nonepi_nib.get_data().squeeze().astype(np.float64)
    nonepi /=nonepi.max()

    radius = 4

    # Preprocess intensities
    epi /= epi.mean()

    # Configure and run orfield estimation
    epi_pedir = np.array((0,1,0), dtype=np.float64)
    nonepi_pedir = np.array((0,0,0), dtype=np.float64)
    distortion_model = SingleEPI_ECC(radius=radius, q_levels=256)
    level_iters = [200, 200, 200, 200, 200, 200, 200, 200, 200]
    lambdas = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                       0.5, 0.05, 0.05])*300
    fwhm = np.array([8, 6, 4, 3, 3, 2, 1, 0, 0])
    estimator = OffResonanceFieldEstimator(distortion_model, level_iters=level_iters, lambdas=lambdas, fwhm=fwhm)

    orfield = estimator.optimize_with_ss(epi, epi_affine, epi_pedir,
                                         nonepi, nonepi_affine, nonepi_pedir,
                                         spacings)


    b  = np.array(orfield.get_volume((0, 0, 0)))
    db = np.array(orfield.get_volume((0, 1, 0)))
    shape = np.array(epi.shape, dtype=np.int32)
    w_epi, _m = gr.warp_with_orfield(epi, b, epi_pedir, None,
                                    None, None, shape)

    # Resample nonepi on top of epi
    affmap = AffineMap(None, epi.shape, epi_affine, nonepi.shape, nonepi_affine)
    nonepi_resampled = affmap.transform(nonepi)
    rt.plot_slices(b)
    overlay_slices(nonepi_resampled, epi, slice_type=2)
    overlay_slices(nonepi_resampled, w_epi*(1+db), slice_type=2)


def compare_gradients():
    from dipy.align.scalespace import IsotropicScaleSpace
    # Load images
    up_nib = nib.load(up_fname)
    up_affine = up_nib.get_affine()
    direction, spacings = imwarp.get_direction_and_spacings(up_affine, 3)
    up = up_nib.get_data().squeeze().astype(np.float64)

    down_nib = nib.load(down_fname)
    down_affine = down_nib.get_affine()
    down = down_nib.get_data().squeeze().astype(np.float64)

    radius = 4



    #up_affine = np.diag([spacings[0], spacings[1], spacings[2], 1.0])
    #down_affine = np.diag([spacings[0], spacings[1], spacings[2], 1.0])
    #up_affine = np.eye(4)
    #down_affine = np.eye(4)
    #spacings[:]=1

    # Preprocess intensities
    #up /= up.mean()
    #down /= down.mean()

    # Configure and run orfield estimation
    pedir_up = np.array((0,1,0), dtype=np.float64)
    pedir_down = np.array((0,-1,0), dtype=np.float64)
    level_iters = [200, 200, 200, 200, 200, 200, 200, 200, 200]
    #level_iters = [5, 1, 1, 1, 1, 1, 1, 1, 1]
    lambdas = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                       0.5, 0.05, 0.05])*400
    fwhm = np.array([8, 6, 4, 3, 3, 2, 1, 0, 0], dtype=np.float64)
    step_lengths = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*4
    subsampling = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1])
    warp_res = np.array([20, 16, 14, 12, 10, 6, 4, 4, 4])*1.7
    nstages = len(subsampling)


    motion = OppositeBlips_CC_Motion(radius=radius)
    nomotion = OppositeBlips_CC(radius=radius)

    # Prepare one iteration from scratch

    f1 = down
    f1_affine = down_affine
    f1_pedir = pedir_down
    f2 = up
    f2_affine = up_affine
    f2_pedir = pedir_up

    fwhm2sigma = (np.sqrt(8.0 * np.log(2)))
    f1_ss = IsotropicScaleSpace(f1, subsampling, fwhm / fwhm2sigma,
                                f1_affine, spacings, False)
    f2_ss = IsotropicScaleSpace(f2, subsampling, fwhm / fwhm2sigma,
                                f2_affine, spacings, False)
    field = None
    b = None
    b_coeff = None
    stage = 0
    if True:
        scale = nstages - 1 - stage
        step_length = step_lengths[stage]

        shape1 = f1_ss.get_domain_shape(scale)
        affine1 = f1_ss.get_affine(scale)
        f1_smooth = f1_ss.get_image(scale)
        f1_mask = (f1_smooth>0).astype(np.int32)

        shape2 = f2_ss.get_domain_shape(scale)
        affine2 = f2_ss.get_affine(scale)
        f2_smooth = f2_ss.get_image(scale)
        f2_mask = (f2_smooth>0).astype(np.int32)


        resampled_sp = subsampling[stage] * spacings
        tps_lambda = lambdas[stage]

        # get the spline resolution from millimeters to voxels
        kspacing = np.round(warp_res[stage]/resampled_sp)
        kspacing = kspacing.astype(np.int32)
        kspacing[kspacing < 1] = 1

        if field is None:
            field = CubicSplineField(shape1, kspacing)
            b_coeff = np.zeros(field.num_coefficients())
            field.copy_coefficients(b_coeff)


        if True:
            it = 0
            energy_nomotion, grad_nomotion = nomotion.energy_and_gradient(
                            f1_smooth, f1_affine, f1_pedir,
                            f2_smooth, f2_affine, f2_pedir,
                            field, affine1,
                            f1_mask, f2_mask)
            rt.plot_slices(grad_nomotion)

            theta = motion.transform.get_identity_parameters()
            energy_motion, grad_motion, dtheta_motion = motion.energy_and_gradient(
                                    f1_smooth, f1_affine, f1_pedir, spacings,
                                    f2_smooth, f2_affine, f2_pedir, spacings, theta,
                                    field, affine1,
                                    f1_mask, f2_mask)
            rt.plot_slices(grad_motion)

            #rt.overlay_slices(motion.w_down, nomotion.w_down, slice_type=2)
            print(np.abs(motion.w_down - nomotion.w_down).max())
            #rt.overlay_slices(motion.w_up, nomotion.w_up, slice_type=2)
            print(np.abs(motion.w_up - nomotion.w_up).max())

            print(nomotion.dw_down.shape)
            print(motion.wgrad_down[...,1].shape)
            rt.overlay_slices(motion.wgrad_down[...,1], nomotion.dw_down, slice_type=2)
            print(np.abs(motion.wgrad_down[...,1]-nomotion.dw_down).max())
            rt.overlay_slices(motion.wgrad_up[...,1], nomotion.dw_up, slice_type=2)
            print(np.abs(motion.wgrad_up[...,1]-nomotion.dw_up).max())

            print(motion.down_pedir, "\n", motion.up_pedir)
            print(nomotion.pedir_factor)


            print(np.abs(motion.gkernel[...,1]-nomotion.dkernel).max())
            print(motion.kspacing)
            print(nomotion.kspacing)
            print(motion.field_shape)
            print(nomotion.field_shape)

            print(np.abs(grad_motion - grad_nomotion).max())





            #dtheta[1] = 0 # do not move along the pe-dir
            if energy is None or grad is None:
                break
            grad = np.array(gr.unwrap_scalar_field(grad))

            bending_energy, bending_grad = field.get_bending_gradient()
            bending_grad = tps_lambda * np.array(bending_grad)
            print("Bending grad. range: [%f, %f]"%(bending_grad.min(), bending_grad.max()))
            print("Sim. grad. range: [%f, %f]"%(grad.min(), grad.max()))

            bending_energy *= tps_lambda
            total_energy = energy + bending_energy
            self.energy_list.append(total_energy)
            #print("Energy: %f [data] + %f [reg] = %f"%(energy, bending_energy, total_energy))
            step = -1 * (grad + bending_grad)



            #print(">>>>>>>>>>>>>>>>>> ", np.abs(step).max())
            #if np.abs(step).max() > 1:
            if True:
                step = step_length * (step/np.abs(step).max())
            else:
                step = step_length * step

            if np.abs(dtheta).max() > 1:
                theta_step = -0.05 * step_length * (dtheta/np.abs(dtheta).max())
            else:
                theta_step = -0.05 * step_length * dtheta

            theta += theta_step



test_epicor_OPPOSITE_BLIPS_CC_SS_MOTION()