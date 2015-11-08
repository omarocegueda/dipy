from __future__ import print_function
import numpy as np
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises)
import dipy.align.imwarp as imwarp
import dipy.align.metrics as metrics
import dipy.align.vector_fields as vfu
from dipy.data import get_data
import nibabel as nib
import nibabel.eulerangles as eulerangles
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align import VerbosityLevels
import scipy as sp
import scipy
import scipy.sparse
import scipy.sparse.linalg
import dipy.correct.gradients as gr
import dipy.correct.splines as splines
import dipy.viz.regtools as rt
from dipy.align.imaffine import MutualInformationMetric, AffineRegistration
from dipy.align.transforms import regtransforms
from experiments.registration.regviz import overlay_slices
import mayavi
import mayavi.mlab as mlab
from tvtk.tools import visual
from tvtk.api import tvtk
from inverse.dfinverse_3d import warp_points_3d, compute_jacobian_3d
from dipy.correct.cc_splines import cc_splines_gradient

def get_diag_affine(sp):
    A = np.eye(4)
    A[np.diag_indices(3)] = sp
    return A


def extend_volume(vol, margin):
    dims = np.array(vol.shape)
    dims += 2*margin
    new_vol = np.zeros(tuple(dims))
    new_vol[margin:-margin, margin:-margin, margin:-margin] = vol[...]
    return new_vol.astype(vol.dtype)


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

def test_cc_splines():
    from dipy.correct.splines import CubicSplineField
    # Prameters
    #data_dir = 'D:/opt/registration/data/topup_example/'
    data_dir = './'
    up_fname = data_dir + "b0_blipup.nii"
    down_fname = data_dir + "b0_blipdown.nii"
    d_up = np.array([0, 1, 0], dtype=np.float64)
    d_down = np.array([0, -1, 0], dtype=np.float64)
    
    radius = 4  # CC radius

    nstages = 9
    fwhm = np.array([8, 6, 4, 3, 3, 2, 1, 0, 0], dtype=np.float64)
    warp_res = np.array([20, 16, 14, 12, 10, 6, 4, 4, 4], dtype=np.float64)
    subsampling = np.array([2, 2, 2, 2, 2, 1, 1, 1, 1], dtype=np.int32)
    lambda1 = np.array([1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2])*2
    lambda2 = np.array([1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e2, 1e1, 1])
    #lambda2 = np.array([10, 200, 5, 5, 5, 5, 5, 5e-1, 5e-1, 5e-1])
    #max_iter = np.array([10, 10, 10, 10, 10, 10, 10, 10, 10], dtype=np.int32)
    max_iter = np.array([30, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int32)
    #max_iter = np.array([5, 5, 5, 5, 5, 10, 10, 20, 20], dtype=np.int32)
    #max_iter = np.array([10, 10, 10, 10, 10, 10, 10, 20, 20], dtype=np.int32)

    # Read and scale data
    up_nib = nib.load(up_fname)
    down_nib = nib.load(down_fname)
    up = up_nib.get_data().squeeze().astype(np.float64)
    down = down_nib.get_data().squeeze().astype(np.float64)
    
    up /= up.mean()
    down /= down.mean()
    
    print('up range: %f, %f'%(up.min(), up.max()) )
    print('dn range: %f, %f'%(down.min(), down.max()) )
    up_affine = up_nib.get_affine()
    up_affine_inv = np.linalg.inv(up_affine)
    down_affine = down_nib.get_affine()
    down_affine_inv = np.linalg.inv(down_affine)

    direction, spacings = imwarp.get_direction_and_spacings(up_affine, 3)
    
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

    field = None

    if True:
        stage = 0
    #for stage in range(nstages):
        print("Stage: %d / %d"%(stage + 1, nstages))
        #subsample by 2, if required
        if subsampling[stage] > 1:
            sub_up = vfu.downsample_scalar_field_3d(resampled_up)
            sub_down = vfu.downsample_scalar_field_3d(resampled_down)
        else:
            sub_up = resampled_up
            sub_down = resampled_down

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
        print("sigma_vox:",sigma_vox)


        # Create, rescale or keep field as needed
        if field is None:
            print ("Creating field")
            # The field has not been created, this must be the first stage
            field = CubicSplineField(sub_up.shape, kspacing)
            b_coeff = np.zeros(field.num_coefficients())
            field.copy_coefficients(b_coeff)
            b = field.get_volume()
            b=b.astype(np.float64)
            
        current_up = sp.ndimage.filters.gaussian_filter(sub_up, sigma_vox)
        current_down = sp.ndimage.filters.gaussian_filter(sub_down, sigma_vox)
        dcurrent_up = gr.der_y(current_up)
        dcurrent_down = gr.der_y(current_down)

        current_shape = np.array(current_up.shape, dtype=np.int32)
        current_sp = resampled_sp
        current_affine = get_diag_affine(current_sp)
        current_affine_inv = np.linalg.inv(current_affine)
        
        kernel = field.spline3d.get_kernel_grid((0,0,0))
        dkernel = field.spline3d.get_kernel_grid((0,1,0))
        dfield = field.get_volume((0,1,0))
        dfield = np.array(dfield).astype(np.float64)
        
        pedir_factor = 1.0
        kcoef_grad = np.zeros_like(field.coef)
        cc_splines_gradient(current_up, current_down,
                        dcurrent_up, dcurrent_down,
                        pedir_factor,
                        None, None,
                        kernel, dkernel,
                        dfield, field.kspacing,
                        field.grid_shape, radius,
                        kcoef_grad)
    
    