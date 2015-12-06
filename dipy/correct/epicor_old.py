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

from dipy.correct.cc_splines import (cc_splines_gradient,
                                     cc_splines_gradient_epicor)


def load_topup_results():
    topup_corrected_fname = data_dir + 'results_topup_mdesco/unwarped_images.nii.gz'
    topup_corrected_nib = nib.load(topup_corrected_fname)
    topup_corrected = topup_corrected_nib.get_data().squeeze()
    tc0 = topup_corrected[...,0][::-1,:,:]
    tc1 = topup_corrected[...,1][::-1,:,:]
    overlay_slices(tc0, tc1, slice_type=2)
    tp_field_fname = data_dir + 'results_topup_mdesco/field.nii.gz'
    tp_field_nib = nib.load(tp_field_fname)
    tp_field = tp_field_nib.get_data().squeeze()
    overlay_slices(tp_field, tp_field, slice_type=2)

def read_topup_field(fname):
    fname = 'results_fieldcoef.nii'
    coef_nib = nib.load(fname)
    coef = coef_nib.get_data()
    coef_shape = np.array(coef.shape)
    h = coef_nib.get_header()
    knot_spacings = np.round(np.array(h.get_zooms()) + 0.5).astype(np.int32)
    vox_size = np.array([h._structarr['intent_p'+str(i + 1)] for i in range(3)])
    vox_size = vox_size.astype(np.float64)
    vol_shape = coef_nib.get_affine()[:3,3].astype(np.int32)

    sfield = gr.SplineField(vol_shape, knot_spacings)
    sfield.copy_coefficients(coef.astype(np.float64))
    vol = sfield.get_volume()

def get_diag_affine(sp):
    A = np.eye(4)
    A[np.diag_indices(3)] = sp
    return A


def extend_volume(vol, margin):
    dims = np.array(vol.shape)
    dims += 2 * margin
    new_vol = np.zeros(tuple(dims))
    new_vol[margin:-margin, margin:-margin, margin:-margin] = vol[...]
    return new_vol.astype(vol.dtype)

def test_cc_splines_old():
    # Prameters
    radius = 4  # CC radius
    nstages = 1  # Multi-resolution stages
    fwhm = np.array([8, 6, 4, 3, 3, 2, 1, 0, 0], dtype=np.float64)
    #fwhm = np.array([4, 3, 2, 1.5, 1.5, 1, 0.5, 0, 0], dtype=np.float64)
    warp_res = np.array([20, 16, 14, 12, 10, 6, 4, 4, 4], dtype=np.float64)
    #subsampling = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1], dtype=np.int32)
    subsampling = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1], dtype=np.int32)
    lambda1 = np.array([1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2])*2
    lambda2 = np.array([1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e1, 1])
    #max_iter = np.array([200, 40, 20, 10, 10, 10, 10, 10, 10], dtype=np.int32)
    max_iter = np.array([100, 50, 25, 20, 10, 10, 10, 10, 10], dtype=np.int32)
    #max_iter = np.array([10, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    step_sizes = np.array([0.3, 0.25, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float64)*0+0.35
    #max_iter = np.array([2, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    pedir_factor = 1.0
    pedir_vector = np.array((0.0, pedir_factor, 0.0))

    # Read and scale data
    up_nib = nib.load(up_fname)
    up_affine = up_nib.get_affine()
    direction, spacings = imwarp.get_direction_and_spacings(up_affine, 3)
    up_affine_inv = np.linalg.inv(up_affine)
    up = up_nib.get_data().squeeze().astype(np.float64)
    up /= up.mean()

    down_nib = nib.load(down_fname)
    down_affine = down_nib.get_affine()
    down_affine_inv = np.linalg.inv(down_affine)
    down = down_nib.get_data().squeeze().astype(np.float64)
    down /= down.mean()

    unwarped_nib = nib.load(up_unwarped_fname)
    unwarped_affine = unwarped_nib.get_affine()
    unwarped_affine_inv = np.linalg.inv(unwarped_affine)
    unwarped = unwarped_nib.get_data().squeeze().astype(np.float64)
    unwarped /= unwarped.mean()

    print('up range: %f, %f'%(up.min(), up.max()))
    print('dn range: %f, %f'%(down.min(), down.max()) )

    t1_nib = nib.load(t1_fname)
    t1_affine = t1_nib.get_affine()
    t1 = t1_nib.get_data().squeeze()

    aff_map_name = data_dir + 'unwarped_to_t1_map.p'
    if os.path.isfile(aff_map_name):
        aff_map = pickle.load(open(aff_map_name))
    else:
        aff_map = dipy_align(unwarped, unwarped_affine, t1, t1_affine)
        pickle.dump(aff_map, open(aff_map_name,'w'))
    t1_on_b0 = aff_map.transform(t1)




    # Resample the input images
    max_subsampling = subsampling.max()
    in_shape = np.array(up.shape, dtype=np.int32)
    # This makes no sense, it should be + (max_subsampling - regrid_shape%max_subsampling)%max_subsampling
    regrid_shape = in_shape + max_subsampling
    reg_sp = ((in_shape - 1) * spacings) / regrid_shape
    regrid_affine = get_diag_affine(reg_sp)
    regrid_affine_inv = np.linalg.inv(regrid_affine)
    factors = reg_sp / spacings
    resampled_up = np.array(gr.regrid(up, factors, regrid_shape)).astype(np.float64)
    resampled_down = np.array(gr.regrid(down, factors, regrid_shape)).astype(np.float64)
    resampled_unwarped = np.array(gr.regrid(unwarped, factors, regrid_shape)).astype(np.float64)
    resampled_t1 = np.array(gr.regrid(t1_on_b0, factors, regrid_shape)).astype(np.float64)

    # Apply transfer
    num_labels = 128
    t1_lab, levels, hist = quantize_positive_3d(resampled_t1, num_labels)
    t1_lab = np.array(t1_lab)
    t1_mask = (t1_lab > 0).astype(np.int32)
    up_mask = (resampled_up > 0).astype(np.int32)
    means, variances = compute_masked_class_stats_3d(t1_mask*up_mask, resampled_up, num_labels, t1_lab)
    means, variances = np.array(means), np.array(variances)
    resampled_ft1 = means[t1_lab]




    # Choose static and moving
    #static = resampled_unwarped
    #static = resampled_t1
    #static = resampled_ft1
    static = resampled_down
    moving = resampled_up
    warp_both = True

    field = None

    #if True:
    #    stage = 0
    for stage in range(nstages):
        print("Stage: %d / %d"%(stage + 1, nstages))
        #subsample by 2, if required
        if subsampling[stage] > 1:
            sub_static = vfu.downsample_scalar_field_3d(static)
            sub_moving = vfu.downsample_scalar_field_3d(moving)
        else:
            sub_static = static
            sub_moving = moving

        resampled_sp = subsampling[stage] * reg_sp
        resampled_affine = get_diag_affine(resampled_sp)

        l1 = lambda1[stage]
        l2 = lambda2[stage]
        # get the spline resolution from millimeters to voxels
        kspacing = np.round(warp_res[stage]/resampled_sp).astype(np.int32)
        kspacing[kspacing<1] = 1

        sigma_mm = fwhm[stage] / (np.sqrt(8.0 * np.log(2)))
        sigma_vox = sigma_mm/resampled_sp
        print("kspacing:",kspacing)
        print(">>>resampled_sp:",resampled_sp)
        print("sigma_vox:",sigma_vox)


        # Create, rescale or keep field as needed
        if field is None:
            # The field has not been created, this must be the first stage
            print ("Creating field")
            field = CubicSplineField(sub_static.shape, kspacing)
            b_coeff = np.zeros(field.num_coefficients())
            field.copy_coefficients(b_coeff)
            b = field.get_volume()
            b=b.astype(np.float64)
        elif not np.all(sub_static.shape == field.vol_shape) or not np.all(kspacing == field.kspacing):
            # We need to reshape the field
            new_field = CubicSplineField(sub_static.shape, kspacing)
            resample_affine = subsampling[stage] * np.eye(4) / subsampling[stage-1]
            resample_affine[3,3] = 1
            new_b = vfu.transform_3d_affine(b.astype(floating),
                                            np.array(sub_static.shape, dtype=np.int32),
                                            resample_affine)
            new_b = np.array(new_b, dtype=np.float64)
            # Scale to new voxel size
            new_b *= 1.0* subsampling[stage-1] / subsampling[stage]
            # Compute the coefficients associated with the resampled field
            coef = new_field.spline3d.fit_to_data(new_b, 0.0)
            new_field.copy_coefficients(coef)

            field = new_field
            b = field.get_volume()
            b=b.astype(floating)
            b_coeff = gr.unwrap_scalar_field(coef)
        else:
            print ("Keeping field as is")


        # smooth out images and compute spatial derivatives
        current_static = sp.ndimage.filters.gaussian_filter(sub_static, sigma_vox)
        current_moving = sp.ndimage.filters.gaussian_filter(sub_moving, sigma_vox)
        dcurrent_static = gr.der_y(current_static)
        dcurrent_moving = gr.der_y(current_moving)
        current_shape = np.array(current_static.shape, dtype=np.int32)
        current_sp = resampled_sp
        current_affine = get_diag_affine(current_sp)
        current_affine_inv = np.linalg.inv(current_affine)

        print('>>> kspacing:',kspacing)

        # Start gradient descent
        for it in range(max_iter[stage]):
            print("Iter: %d / %d"%(it + 1, max_iter[stage]))
            d = b.astype(np.float64)
            rt.plot_slices(d)
            print(">>> b range: %f, %f"%(d.min(), d.max()))
            # We want to sample at a grid with current_shape shape,
            # The grid-to-space transform is diag(current_sp)
            curr_aff = get_diag_affine(current_sp)
            disp_aff = None
            # We want to sample: f[ curr_aff^{-1} * curr_aff*x + curr_aff^{-1}*b[curr_aff^{-1}*(curr_aff*x)] ]
            # which is: f[ x + curr_aff^{-1}*b[x] ], the displacement affine is the inverse of curr_aff

            w_moving, mask_moving = gr.warp_with_orfield(current_moving, d, pedir_vector, None, None, disp_aff, current_shape)
            mask_moving[...] = 1
            w_moving = np.array(w_moving)
            if warp_both:
                w_static, mask_static = gr.warp_with_orfield(current_static, d, -1.0*pedir_vector, None, None, disp_aff, current_shape)
                w_static = np.array(w_static)
            else:
                w_static = current_static

            #if it == 0: # Plot initial state
            #    if b.max() > b.min():
            #        rt.plot_slices(b)
            #    rt.overlay_slices(w_up, w_down, slice_type=2)

            dw_moving, dmask_moving = gr.warp_with_orfield(dcurrent_moving, d, pedir_vector, None, None, None, current_shape)
            dw_moving = np.array(dw_moving)
            if warp_both:
                dw_static, dmask_static = gr.warp_with_orfield(dcurrent_static, d, -1.0*pedir_vector, None, None, None, current_shape)
                dw_static = np.array(dw_static)
            else:
                dw_static = dcurrent_static

            db = field.get_volume((0,1,0))
            db = np.array(db).astype(floating)


            #db /= current_sp[1]

            # Get spline kernel and its derivative. Evaluate full derivative
            kernel = field.spline3d.get_kernel_grid((0,0,0))
            dkernel = field.spline3d.get_kernel_grid((0,1,0))
            dfield = field.get_volume((0,1,0))
            dfield = np.array(dfield, dtype=np.float64)

            # Specify phase encode direction. Create buffer to hold gradient
            kcoef_grad = np.zeros_like(field.coef)

            if warp_both:
                step_function = cc_splines_gradient_epicor
            else:
                step_function = cc_splines_gradient

            energy = step_function(w_static, w_moving,
                                   dw_static, dw_moving,
                                   pedir_factor,
                                   None, None,
                                   kernel, dkernel,
                                   dfield, field.kspacing,
                                   field.grid_shape, radius,
                                   kcoef_grad)

            bending_energy, bending_grad = field.get_bending_gradient()
            bending_grad = (l2/5000.0) * np.array(bending_grad)

            print("Energy: %f"%(energy,))
            step = np.array(gr.unwrap_scalar_field(kcoef_grad)) - bending_grad
            step = step_sizes[stage] * (step/np.abs(step).max())
            if b_coeff is None:
                b_coeff = step
            else:
                b_coeff += step

            field.copy_coefficients(b_coeff)
            b = field.get_volume()
            b=b.astype(floating)



up_nib = nib.load("b0_blipup.nii")
down_nib = nib.load("b0_blipdown.nii")
up = up_nib.get_data().squeeze().astype(np.float64)
down = down_nib.get_data().squeeze().astype(np.float64)


use_extend_volume = True
if use_extend_volume:
    up = extend_volume(up, 16)
    down = extend_volume(down, 16)

#up_dir, up_spacings = imwarp.get_direction_and_spacings(up_nib.get_affine(), 3)
#down_dir, down_spacings = imwarp.get_direction_and_spacings(down_nib.get_affine(), 3)

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

levels = 4
ss_sigma_factor = 0.2

up_ss = imwarp.ScaleSpace(up, levels, up_affine, up_spacings, ss_sigma_factor, False)
down_ss = imwarp.ScaleSpace(down, levels, down_affine, down_spacings, ss_sigma_factor, False)

ref_shape = up_ss.get_domain_shape(levels-1)
ref_affine = up_ss.get_affine(levels-1)
ref_affine_inv = up_ss.get_affine_inv(levels-1)
b = np.zeros(shape=tuple(ref_shape), dtype=floating)
d_up = np.array([0, -1, 0], dtype=np.float64)
d_down = np.array([0, 1, 0], dtype=np.float64)





#one level
#for level in range(levels-1, levels-4, -1):
if True:
    level = levels-1
    current_up = up_ss.get_image(level)
    current_down = down_ss.get_image(level)
    dcurrent_up = gr.der_y(current_up)
    dcurrent_down = gr.der_y(current_down)
    print("Range up: %f %f"%(current_up.min(), current_up.max()))
    print("Range down: %f %f"%(current_down.min(), current_down.max()))
    print("Means:",current_up.mean(), current_down.mean())
    #current_up *= 10.0/current_up.mean()
    #current_down *= 10.0/current_down.mean()
    print("New range up: %f %f"%(current_up.min(), current_up.max()))
    print("New range down: %f %f"%(current_down.min(), current_down.max()))
    #current_up/=10000.0
    #current_down/=10000.0

    if level<levels-1:
        rt.plot_slices(b)
        new_shape = up_ss.get_domain_shape(level)
        factors = up_ss.get_expand_factors(level + 1, level)
        new_b = gr.resample_orfield(b, factors, new_shape)

        b = np.array(new_b)
        ref_shape = new_shape
        ref_affine = up_ss.get_affine(level)
        ref_affine_inv = up_ss.get_affine_inv(level)
        rt.plot_slices(b)

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

    test_holland_hessian = True
    if test_holland_hessian:
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


    while it<max_iter and nrm > epsilon:# and (prev_energy is None or prev_energy>=energy):
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

        gh = np.array(gr.grad_holland(w_up, w_down, dw_up, dw_down, b, db, l1, l2))


        #w_up *= ( 1.0 - db)
        #w_down *= ( 1.0 + db)

        prev_energy = energy
        energy = np.sum((w_up - w_down)**2)
        #if prev_energy is not None:
        #    print("Decreased energy:%f", prev_energy - energy)

        nrm = np.abs(gh).max()
        print("Level: %d. Iter: %d/%d. Grad norm: %f. Energy: %f"%(level, it+1, max_iter, nrm,energy))

        if it % 200 == 0:
            rt.overlay_slices(w_up, w_down, slice_type=2)

        tau = 0.1
        if nrm > tau:
            gh = gh * (tau / nrm)
        b -= gh

        it +=1
    rt.overlay_slices(w_up, w_down, slice_type=2)





