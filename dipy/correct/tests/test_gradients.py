from __future__ import print_function
import numpy as np
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises)
import matplotlib.pyplot as plt
import dipy.align.imwarp as imwarp
import dipy.align.metrics as metrics
import dipy.align.vector_fields as vfu
from dipy.data import get_data
from dipy.align import floating
import nibabel as nib
import nibabel.eulerangles as eulerangles
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align import VerbosityLevels
import dipy.viz.regtools as rt
import scipy as sp
import scipy
import scipy.sparse
import scipy.sparse.linalg
import dipy.correct.gradients as gr
import dipy.correct.splines as splines


def get_diag_affine(sp):
    A = np.eye(4)
    A[np.diag_indices(3)] = sp
    return A


def extend_volume(vol, margin):
    dims = np.array(vol.shape)
    dims += 2*margin
    new_vol = np.zeros(tuple(dims))
    new_vol[margin:-margin, margin:-margin, margin:-margin] = vol[...]
    return new_vol


def get_preprocessed_data(levels, use_extend_volume = True):
    up_nib = nib.load("b0_blipup.nii")
    down_nib = nib.load("b0_blipdown.nii")
    up = up_nib.get_data().squeeze().astype(np.float64)
    down = down_nib.get_data().squeeze().astype(np.float64)

    if use_extend_volume:
        up = extend_volume(up, 16)
        down = extend_volume(down, 16)

    # ====Simplify sampling transforms====
    up_dir, up_spacings = np.eye(4), np.ones(3)*1.8
    down_dir, down_spacings = np.eye(4), np.ones(3)*1.8
    up_affine = up_dir
    up_affine[:3, :3] *= 1.8
    up_affine_inv = np.linalg.inv(up_affine)
    down_affine = down_dir
    down_affine[:3, :3] *= 1.8
    down_affine_inv = np.linalg.inv(down_affine)
    # ====================================

    ss_sigma_factor = 0.2

    up_ss = imwarp.ScaleSpace(up, levels, up_affine, up_spacings, ss_sigma_factor, False)
    down_ss = imwarp.ScaleSpace(down, levels, down_affine, down_spacings, ss_sigma_factor, False)

    return up_ss, down_ss, up_affine, up_affine_inv, down_affine, down_affine_inv




def test_andersson_new_subsample():
    up_nib = nib.load("b0_blipup.nii")
    down_nib = nib.load("b0_blipdown.nii")
    up = up_nib.get_data().squeeze().astype(floating)
    down = down_nib.get_data().squeeze().astype(floating)
    up *= 2.0/up.mean()
    down *= 2.0/down.mean()

    up_affine = up_nib.get_affine()
    up_affine_inv = np.linalg.inv(up_affine)
    down_affine = down_nib.get_affine()
    down_affine_inv = np.linalg.inv(down_affine)

    direction, spacings = imwarp.get_direction_and_spacings(up_affine, 3)

    # Compute the regridding factors
    max_subsampling = 2
    in_shape = np.array(up.shape, dtype=np.int32)

    # This makes no sense, it should be + (max_subsampling - regrid_shape%max_subsampling)%max_subsampling
    regrid_shape = in_shape + max_subsampling

    # This are the new spacings
    reg_sp = ((in_shape - 1) * spacings) / regrid_shape
    regrid_affine = get_diag_affine(reg_sp)
    regrid_affine_inv = np.linalg.inv(regrid_affine)
    factors = reg_sp / spacings

    resampled_up = np.array(gr.regrid(up, factors, regrid_shape))
    resampled_down = np.array(gr.regrid(down, factors, regrid_shape))
    resampled_sp = 2.0 * reg_sp
    resampled_affine = get_diag_affine(resampled_sp)



    #subsample by 2
    sub_up = vfu.downsample_scalar_field_3d(resampled_up.astype(floating))
    sub_down = vfu.downsample_scalar_field_3d(resampled_down.astype(floating))

    d_up = np.array([0, 1, 0], dtype=np.float64)
    d_down = np.array([0, -1, 0], dtype=np.float64)

    l1 = 0.01
    l2 = 0.01
    max_iter = 2000
    it = 0
    epsilon = 0.000001
    nrm = 1 + epsilon
    energy = None
    prev_energy=None

    # Instanciate a spline field to fit the reference volume grid
    kspacing = np.array([6,6,6], dtype=np.int32)
    field = gr.SplineField(sub_up.shape, kspacing)
    #field = splines.CubicSplineField(sub_up.shape, kspacing)
    b_coeff = np.zeros(field.num_coefficients())
    field.copy_coefficients(b_coeff)
    b = field.get_volume()
    b=b.astype(floating)

    #smooth_params = [8.0, 6.0, 4.0, 3.0]
    smooth_params = [8.0]
    #if True:
    for it in range(5 * len(smooth_params)):
        #d = b/current_sp[1]
        d = b

        if it % 5 == 0:
            # Smooth and derive
            fwhm = np.ones(3) * smooth_params[it//5]
            sigma_mm = fwhm / (np.sqrt(8.0 * np.log(2)))
            sigma_vox = sigma_mm/resampled_sp

            current_up = sp.ndimage.filters.gaussian_filter(sub_up, sigma_vox)
            current_down = sp.ndimage.filters.gaussian_filter(sub_down, sigma_vox)
            dcurrent_up = gr.der_y(current_up)
            dcurrent_down = gr.der_y(current_down)

            current_shape = np.array(current_up.shape, dtype=np.int32)
            current_sp = resampled_sp
            current_affine = get_diag_affine(current_sp)
            current_affine_inv = np.linalg.inv(current_affine)

        w_up = gr.warp_with_orfield(current_up, d, d_up, None, None, None, current_shape)
        w_down = gr.warp_with_orfield(current_down, d, d_down, None, None, None, current_shape)
        w_up = np.array(w_up)
        w_down = np.array(w_down)
        if it == 0: # Plot initial state
            rt.overlay_slices(w_up, w_down, slice_type=2)

        dw_up = gr.warp_with_orfield(dcurrent_up, d, d_up, None, None, None, current_shape)
        dw_down = gr.warp_with_orfield(dcurrent_down, d, d_down, None, None, None, current_shape)
        db = field.get_volume((0,1,0))
        db = np.array(db).astype(floating)


        kernel = field.splines[(0,0,0)].spline
        dkernel = field.splines[(0,1,0)].spline
        # Get the linear system
        Jth, data, indices, indptr= \
            gr.gauss_newton_system_andersson(w_up, w_down, dw_up, dw_down,
                                             kernel, dkernel, db, field.kspacing,
                                             field.grid_shape, l1, l2)

        Jth = np.array(Jth)
        data = np.array(data)
        indices = np.array(indices)
        indptr = np.array(indptr)

        ncoeff = field.num_coefficients()
        JtJ = sp.sparse.csr_matrix((data, indices, indptr), shape=(ncoeff, ncoeff))

        x = sp.sparse.linalg.spsolve(JtJ, -1.0*Jth)
        if b_coeff is None:
            b_coeff = x
        else:
            b_coeff += x

        field.copy_coefficients(b_coeff)
        b = field.get_volume()
        b=b.astype(floating)


    rt.overlay_slices(w_up*(1+db), w_down*(1-db), slice_type=2)










def test_andersson():
    levels = 4
    up_ss, down_ss, up_affine, up_affine_inv, down_affine, down_affine_inv = get_preprocessed_data(levels, True)

    ref_shape = up_ss.get_domain_shape(levels-1)
    ref_affine = up_ss.get_affine(levels-1)
    ref_affine_inv = up_ss.get_affine_inv(levels-1)

    d_up = np.array([0, 1, 0], dtype=np.float64)
    d_down = np.array([0, -1, 0], dtype=np.float64)

    level = levels-1
    current_up = up_ss.get_image(level)
    current_down = down_ss.get_image(level)
    dcurrent_up = gr.der_y(current_up)
    dcurrent_down = gr.der_y(current_down)

    # warp up
    S = ref_affine
    Rinv = ref_affine_inv
    Tinv = up_affine_inv

    affine_idx_in = Rinv.dot(S)
    affine_idx_out = Tinv.dot(S)
    affine_disp = Tinv

    # Warp down
    S = ref_affine
    Rinv = ref_affine_inv
    Tinv = down_affine_inv

    affine_idx_in = Rinv.dot(S)
    affine_idx_out = Tinv.dot(S)
    affine_disp = Tinv

    l1 = 0.01
    l2 = 0.01
    max_iter = 2000
    it = 0
    epsilon = 0.000001
    nrm = 1 + epsilon
    energy = None
    prev_energy=None

    # Instanciate a spline field to fit the reference volume grid
    kspacing = np.array([2,2,2], dtype=np.int32)
    field = gr.SplineField(ref_shape, kspacing)
    b_coeff = np.zeros(field.num_coefficients())
    field.copy_coefficients(b_coeff)
    b = field.get_volume()
    b=b.astype(floating)
    #for it in range(1000):

    current_up *= 0.05 / current_up.mean()
    current_down *= 0.05 / current_down.mean()

    if True:
        it = 0
        w_up = gr.warp_with_orfield(current_up, b, d_up, affine_idx_in, affine_idx_out,
                                    affine_disp, ref_shape)
        w_down = gr.warp_with_orfield(current_down, b, d_down, affine_idx_in, affine_idx_out,
                                      affine_disp, ref_shape)
        w_up = np.array(w_up)
        w_down = np.array(w_down)

        dw_up = gr.warp_with_orfield(dcurrent_up, b, d_up, affine_idx_in, affine_idx_out,
                                     affine_disp, ref_shape)
        dw_down = gr.warp_with_orfield(dcurrent_down, b, d_down, affine_idx_in, affine_idx_out,
                                       affine_disp, ref_shape)
        db = field.get_volume((0,1,0))
        db = np.array(db).astype(floating)


        kernel = field.splines[(0,0,0)].spline
        dkernel = field.splines[(0,1,0)].spline
        # Get the linear system
        Jth, data, indices, indptr= \
            gr.gauss_newton_system_andersson(w_up, w_down, dw_up, dw_down,
                                             kernel, dkernel, db, field.kspacing,
                                             field.grid_shape, l1, l2)

        Jth = np.array(Jth)
        data = np.array(data)
        indices = np.array(indices)
        indptr = np.array(indptr)

        ncoeff = field.num_coefficients()
        JtJ = sp.sparse.csr_matrix((data, indices, indptr), shape=(ncoeff, ncoeff))

        #Jth_test, JtJ_test, Jt_test = gr.test_gauss_newton_andersson(w_up, w_down, dw_up, dw_down,
        #                                kernel, dkernel, db, field.kspacing,
        #                                field.grid_shape, l1, l2)
        #Jt_test = np.array(Jt_test)
        #assert_almost_equal(np.abs(JtJ-JtJ_test).max(), 0.0)
        #assert_almost_equal(np.abs(Jth-Jth_test).max(), 0.0)

        x = sp.sparse.linalg.spsolve(JtJ, -1.0*Jth)
        if b_coeff is None:
            b_coeff = x
        else:
            b_coeff += x

        field.copy_coefficients(b_coeff)
        b = field.get_volume()
        b=b.astype(floating)


    rt.overlay_slices(w_up*(1+db), w_down*(1-db), slice_type=2)














def test_holland():
    levels = 4
    up_ss, down_ss, up_affine, up_affine_inv, down_affine, down_affine_inv = get_preprocessed_data(levels, True)

    ref_shape = up_ss.get_domain_shape(levels-1)
    ref_affine = up_ss.get_affine(levels-1)
    ref_affine_inv = up_ss.get_affine_inv(levels-1)
    b = np.zeros(shape=tuple(ref_shape), dtype=floating)
    d_up = np.array([0, -1, 0], dtype=np.float64)
    d_down = np.array([0, 1, 0], dtype=np.float64)

    level = levels-1
    current_up = up_ss.get_image(level)
    current_down = down_ss.get_image(level)
    dcurrent_up = gr.der_y(current_up)
    dcurrent_down = gr.der_y(current_down)

    # warp up
    S = ref_affine
    Rinv = ref_affine_inv
    Tinv = up_affine_inv

    affine_idx_in = Rinv.dot(S)
    affine_idx_out = Tinv.dot(S)
    affine_disp = Tinv

    # Warp down
    S = ref_affine
    Rinv = ref_affine_inv
    Tinv = down_affine_inv

    affine_idx_in = Rinv.dot(S)
    affine_idx_out = Tinv.dot(S)
    affine_disp = Tinv

    l1 = 0.01
    l2 = 0.01
    max_iter = 2000
    it = 0
    epsilon = 0.000001
    nrm = 1 + epsilon
    energy = None
    prev_energy=None

    current_up *= 0.05 / current_up.mean()
    current_down *= 0.05 / current_down.mean()


    for it in range(1000):
        w_up = gr.warp_with_orfield(current_up, b, d_up, affine_idx_in, affine_idx_out,
                                    affine_disp, ref_shape)
        w_down = gr.warp_with_orfield(current_down, b, d_down, affine_idx_in, affine_idx_out,
                                      affine_disp, ref_shape)
        w_up = np.array(w_up)
        w_down = np.array(w_down)

        dw_up = gr.warp_with_orfield(dcurrent_up, b, d_up, affine_idx_in, affine_idx_out,
                                     affine_disp, ref_shape)
        dw_down = gr.warp_with_orfield(dcurrent_down, b, d_down, affine_idx_in, affine_idx_out,
                                       affine_disp, ref_shape)
        db = np.array(gr.der_y(b))

        Jth, data, indices, indptr = gr.gauss_newton_system_holland(w_up, w_down, dw_up, dw_down, db, l1, l2)
        Jth = np.array(Jth)
        data = np.array(data)
        indices = np.array(indices)
        indptr = np.array(indptr)

        JtJ = sp.sparse.csr_matrix((data, indices, indptr), shape=(w_up.size, w_up.size))
        x = sp.sparse.linalg.spsolve(JtJ, -1.0*Jth)
        step = gr.wrap_scalar_field(x, np.array(w_up.shape, dtype=np.int32))
        step = np.array(step)
        b += step

    rt.overlay_slices(w_up*(1+db), w_down*(1-db), slice_type=2)

    #Jth_test, JtJ_test = gr.test_gauss_newton_holland(w_up, w_down, dw_up, dw_down, db, l1, l2)
    #dd = np.abs(JtJ - JtJ_test)
    #dd.max()
