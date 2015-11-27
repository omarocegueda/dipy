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
from dipy.align import floating
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
from dipy.correct.splines import CubicSplineField

def draw_deformed_grid(field, zlist=None, npoints=75, tube_radius=0.05, x0=None, x1=None, y0=None, y1=None):
    if x0 is None:
        x0 = 0
    if y0 is None:
        y0 = 0
    if x1 is None:
        x1 = field.shape[0]-1
    if y1 is None:
        y1 = field.shape[1]-1
    if zlist is None:
        zlist = [field.shape[1]//2]
    x = np.linspace(x0, x1, npoints)
    y = np.linspace(y0, y1, npoints)

    mlab.figure(bgcolor=(1,1,1))
    for z0 in zlist:
        hlines = []
        for i in range(npoints):
            hline = [(x[i], y[j], z0) for j in range(npoints)]
            hlines.append(np.array(hline))
        vlines = []
        for i in range(npoints):
            vline = [(x[j], y[i], z0) for j in range(npoints)]
            vlines.append(np.array(vline))

        for line  in hlines:
            warped = np.array(warp_points_3d(line, field))
            mlab.plot3d(warped[:,0], warped[:,1], warped[:,2], color=(0,0,0), tube_radius=tube_radius)
        for line  in vlines:
            warped = np.array(warp_points_3d(line, field))
            mlab.plot3d(warped[:,0], warped[:,1], warped[:,2], color=(0,0,0), tube_radius=tube_radius)

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


def test_hcp():
    #
    from experiments.registration.images2gif import *
    from experiments.registration.regviz import overlay_slices
    from dipy.align.imaffine import AffineMap
    from dipy.correct.gradients import warp_with_orfield
    from dipy.align.vector_fields import compute_jacobian_3d, warp_3d
    from inverse.inverse_3d import invert_vf_full_box_3d
    from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
    from dipy.align.metrics import CCMetric
    import os
    import pickle
    import experiments.registration.dataset_info as info
    import dipy.data as dpd
    
    
    use_preproc = True
    if use_preproc:
        data_dir = 'D:/opt/registration/data/HCP/mgh_1010/diff/preproc/mri/'
    else:
        data_dir = 'D:/opt/registration/data/HCP/mgh_1010/diff/raw/mri/'
        
    # Names
    t1_name = 'D:/opt/registration/data/HCP/mgh_1010/anat/T1/T1.nii.gz'
    t1seg_name = 'D:/opt/registration/data/topup_example/T1seg.nii.gz'
    t1seg_mask_name = 'T1seg_mask.nii.gz'
    t2_name = 'D:/opt/registration/data/HCP/mgh_1010/anat/T2/T2.nii.gz'
    b0_name = 'b_0.nii.gz'
    b0_strip_name = 'D:/opt/registration/data/HCP/mgh_1010/diff/raw/mri/b0_strip.nii.gz'
    b0_brain_mask_name = 'D:/opt/registration/data/HCP/mgh_1010/diff/raw/mri/b0_strip_mask.nii.gz'
    
    
    # Load data
    b0_strip_nib = nib.load(b0_strip_name)
    b0_strip = b0_strip_nib.get_data().squeeze()
    b0_brain_mask_nib = nib.load(b0_brain_mask_name)
    b0_brain_mask = b0_brain_mask_nib.get_data().squeeze()
    b0_nib = nib.load(data_dir + b0_name)
    b0_affine = b0_nib.get_affine()
    b0_full = b0_nib.get_data().squeeze().astype(np.float64)
    b0 = np.mean(b0_full,3)
    t1_nib = nib.load(t1_name)
    t1 = t1_nib.get_data().squeeze()
    t1_affine = t1_nib.get_affine()
    t1seg_nib = nib.load(t1seg_name)
    t1_seg = t1seg_nib.get_data().squeeze()
    t1seg_mask_nib = nib.load(t1seg_mask_name)
    t1seg_mask = t1seg_nib.get_data().squeeze()
    t2_nib = nib.load(t2_name)
    t2 = t2_nib.get_data().squeeze()
    t2_affine = t2_nib.get_affine()
    
    # Resampling
    b0_towards_t1_affinemap = AffineMap(None, t1.shape, t1_affine, b0.shape, b0_affine)
    
    t1_t2_affinemap = AffineMap(None, t1.shape, t1_affine, t2.shape, t2_affine)
    t2_resampled_t1 = t1_t2_affinemap.transform(t2)
    t1_resampled_t2 = t1_t2_affinemap.transform_inverse(t1)
    #create_dwi_gif('b0s.gif', b0_full, axis=2, duration=0.1)
    #Register brainweb template to T1 for brain extraction
    metric = CCMetric(3)
    level_iters = [100, 100, 10]
    sdr = SymmetricDiffeomorphicRegistration(metric, level_iters)
    diffmap_name = 'bwt1_towards_t1_diffmap.p'
    if os.path.isfile(diffmap_name):
        bwt1_towards_t1_diffmap = pickle.load(open(diffmap_name,'r'))
    else:
        MNI_T2 = dpd.read_mni_template()
        MNI_T2_data = MNI_T2.get_data()
        MNI_T2_affine = MNI_T2.get_affine()
        
        
        bwt1_name = info.get_brainweb('t1','strip')
        bwt1_nib = nib.load(bwt1_name)
        bwt1 = bwt1_nib.get_data().squeeze()
        bwt1_affine = bwt1_nib.get_affine()
        affmap_bwt1_towards_t1 = dipy_align(t1, t1_affine, bwt1, bwt1_affine)
        bwt1_aligned_t1 = affmap_bwt1_towards_t1.transform(bwt1)
        
        affmap = bwt1_t1 = AffineMap(None, bwt1.shape, bwt1_affine, t1.shape, t1_affine)
        t1_resampled_bwt1 = affmap.transform(t1)
        
        bwt1_towards_t1_diffmap = sdr.optimize(t2, b0_strip, t2_affine, b0_affine)
        b0_towards_t2_diffmap.forward = np.array(b0_towards_t2_diffmap.forward)
        b0_towards_t2_diffmap.backward = np.array(b0_towards_t2_diffmap.backward)
        pickle.dump(b0_towards_t2_diffmap, open(diffmap_name,'w'))
        
    t2_brain_mask = b0_towards_t2_diffmap.transform(b0_brain_mask, 'nearest')
    b0_warped_t2 = b0_towards_t2_diffmap.transform(b0, 'nearest')
    t2_strip = t2*t2_brain_mask
    
    # The T1 and T2 are aligned, we can use the same transform to get the T1 mask
    t1_brain_mask = b0_towards_t2_diffmap.transform(b0_brain_mask, 'nearest', out_shape=t1.shape, out_grid2world=t1_affine)
    b0_warped_t1 = b0_towards_t2_diffmap.transform(b0, out_shape=t1.shape, out_grid2world=t1_affine)
    t1_strip = t1*t1_brain_mask
    
    
    b0_t1_affinemap = AffineMap(None, b0.shape, b0_affine, t1.shape, t1_affine)
    t1_resampled_b0 = b0_t1_affinemap.transform(t1)
    overlay_slices(b0, t1_resampled_b0, slice_type=0)
    b0_resampled_t1 = b0_t1_affinemap.transform_inverse(b0)
    overlay_slices(b0_resampled_t1, t1, slice_type=0)

    b0_t2_affinemap = AffineMap(None, b0.shape, b0_affine, t2.shape, t2_affine)
    t2_resampled_b0 = b0_t2_affinemap.transform(t2)
    overlay_slices(b0, t2_resampled_b0, slice_type=0)
    b0_resampled_t2 = b0_t2_affinemap.transform_inverse(b0)
    overlay_slices(b0_resampled_t2, t2, slice_type=0)
    
    
    #create simple field
    sigma = 10
    max_d = 10
    pe_dir = np.array([0,-1,0], dtype=np.float64)   
    
    field = np.zeros(b0.shape)
    X, Y, Z, = np.meshgrid(range(field.shape[0]), range(field.shape[1]), range(field.shape[2]))
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    Z = Z.astype(np.float64)
    x0 = 0
    y0 = Y.shape[1]//2 
    z0 = Z.shape[2]//2
    field = np.exp(-1*((X-x0)**2 + (Y-y0)**2 + (Z-z0)**2)/(2*sigma**2))
    field/=field.max()
    field *= max_d
    dfield = np.zeros(field.shape+(3,))
    dfield[...,1] = pe_dir[1]*field
    
    inv_field = invert_vf_full_box_3d(dfield)   
    inv_field = np.array(inv_field)
    #deform B0
    deformed = warp_3d(b0, inv_field)
    jacobian = compute_jacobian_3d(inv_field)
    jacobian = np.array(jacobian)
    
    #Jacobian modulation
    def_mod = deformed*jacobian
    
    overlay_slices(deformed, def_mod, slice_type=2)
    

    
    
    
    
    
    

def topup():
    # Prameters
    #data_dir = 'D:/opt/registration/data/topup_example/'
    data_dir = '/home/omar/data/topup_example/'
    up_fname = data_dir + "b0_blipup.nii"
    down_fname = data_dir + "b0_blipdown.nii"
    d_up = np.array([0, 1, 0], dtype=np.float64)
    d_down = np.array([0, -1, 0], dtype=np.float64)

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
    up = up_nib.get_data().squeeze().astype(floating)
    down = down_nib.get_data().squeeze().astype(floating)

    #up = extend_volume(up, 10)
    #down = extend_volume(down, 10)

    #up *= 1.0/up.mean()
    #down *= 1.0/down.mean()

    # Create a sphere at the center of the images
    r = np.min(up.shape) // 4
    sphere = vfu.create_sphere(up.shape[0], up.shape[1], up.shape[2], r)
    sphere = np.array(sphere, dtype=np.int32)
    up_nz = up[sphere>0]
    dn_nz = down[sphere>0]

    try_new_scaling = False
    if try_new_scaling:
        mup = up_nz.mean()
        sup = mup - up.min()
        mdown = dn_nz.mean()
        sdown = mdown - down.min()
        if mup < mdown:
            up = sdown*(up - mup)/sup + mdown
            #down = (down - mdown)/sdown + mdown
        else:
            #up = (up - mup)/sup + mup
            down = sup*(down - mdown)/sdown + mup
        up /= 100
        down /= 100
        #up *= 1.0/up_nz.mean()
        #down *= 1.0/
    else:
        up *= 1.0/up.mean()
        down *= 1.0/down.mean()



    print('up range: %f, %f'%(up.min(), up.max()) )
    print('dn range: %f, %f'%(down.min(), down.max()) )
    up_affine = up_nib.get_affine()
    up_affine_inv = np.linalg.inv(up_affine)
    down_affine = down_nib.get_affine()
    down_affine_inv = np.linalg.inv(down_affine)

    direction, spacings = imwarp.get_direction_and_spacings(up_affine, 3)

    # Resample
    max_subsampling = subsampling.max()
    in_shape = np.array(up.shape, dtype=np.int32)
    # This makes no sense, it should be + (max_subsampling - regrid_shape%max_subsampling)%max_subsampling
    regrid_shape = in_shape + max_subsampling
    reg_sp = ((in_shape - 1) * spacings) / regrid_shape
    regrid_affine = get_diag_affine(reg_sp)
    regrid_affine_inv = np.linalg.inv(regrid_affine)
    factors = reg_sp / spacings

    resampled_up = np.array(gr.regrid(up, factors, regrid_shape)).astype(floating)
    resampled_down = np.array(gr.regrid(down, factors, regrid_shape)).astype(floating)

    field = None
    #if True:
    #    stage = 0
    for stage in range(nstages):
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
            b=b.astype(floating)
        elif not np.all(sub_up.shape == field.vol_shape) or not np.all(kspacing == field.kspacing):
            # We need to reshape the field
            print ("Reshaping field")
            new_field = CubicSplineField(sub_up.shape, kspacing)
            resample_affine = subsampling[stage] * np.eye(4) / subsampling[stage-1]
            resample_affine[3,3] = 1
            new_b = vfu.transform_3d_affine(b.astype(floating),
                                            np.array(sub_up.shape, dtype=np.int32),
                                            resample_affine)

            new_b = np.array(new_b, dtype=np.float64)
            # Scale to new voxel size
            new_b *= 1.0* subsampling[stage-1] / subsampling[stage]
            coef = new_field.spline3d.fit_to_data(new_b, 0.0)
            new_field.copy_coefficients(coef)

            field = new_field
            b = field.get_volume()
            b=b.astype(floating)
            b_coeff = gr.unwrap_scalar_field(coef)
        else:
            print ("Keeping field as is")

        print("Field coef:", field.coef.shape)
        print("Vol shape:", field.vol_shape)
        # Preprocess subsamled images


        current_up = sp.ndimage.filters.gaussian_filter(sub_up, sigma_vox)
        current_down = sp.ndimage.filters.gaussian_filter(sub_down, sigma_vox)
        dcurrent_up = gr.der_y(current_up)
        dcurrent_down = gr.der_y(current_down)

        current_shape = np.array(current_up.shape, dtype=np.int32)
        current_sp = resampled_sp
        current_affine = get_diag_affine(current_sp)
        current_affine_inv = np.linalg.inv(current_affine)

        # Scale the derivatives to mm^{-1}
        #dcurrent_up /= current_sp[1]
        #dcurrent_down /= current_sp[1]


        # Iterate
        #if True:
        #    it = 0
        for it in range(max_iter[stage]):
            print("Iter: %d / %d"%(it + 1, max_iter[stage]))
            d = b
            # We want to sample at a grid with current_shape shape,
            # The grid-to-space transform is diag(current_sp)
            curr_aff = get_diag_affine(current_sp)
            #The shape of the displacement field grid is the same as current_shape
            b_aff = curr_aff
            # current images' grids are also the same
            f_aff = curr_aff
            disp_aff = np.linalg.inv(curr_aff)
            # We want to sample: f[ curr_aff^{-1} * curr_aff*x + curr_aff^{-1}*b[curr_aff^{-1}*(curr_aff*x)] ]
            # which is: f[ x + curr_aff^{-1}*b[x] ], the displacement affine is the inverse of curr_aff
            disp_aff = None
            w_up, mask_up= gr.warp_with_orfield(current_up, d, d_up, None, None, disp_aff, current_shape)
            w_down, mask_down = gr.warp_with_orfield(current_down, d, d_down, None, None, disp_aff, current_shape)
            mask_up[...] = 1
            mask_down[...] = 1
            w_up = np.array(w_up)
            w_down = np.array(w_down)
            #if it == 0: # Plot initial state
            #    if b.max() > b.min():
            #        rt.plot_slices(b)
            #    rt.overlay_slices(w_up, w_down, slice_type=2)

            dw_up, dmask_up = gr.warp_with_orfield(dcurrent_up, d, d_up, None, None, None, current_shape)
            dw_down, dmask_down = gr.warp_with_orfield(dcurrent_down, d, d_down, None, None, None, current_shape)
            db = field.get_volume((0,1,0))
            db = np.array(db).astype(floating)
            #db /= current_sp[1]

            kernel = field.spline3d.get_kernel_grid((0,0,0))
            dkernel = field.spline3d.get_kernel_grid((0,1,0))
            #dkernel /= current_sp[1]
            # Get the linear system

            Jth, data, indices, indptr, energy= \
                gr.gauss_newton_system_andersson(w_up, w_down, dw_up, dw_down,
                                                 mask_up, mask_down,
                                                 kernel, dkernel, db, field.kspacing,
                                                 field.grid_shape, l1, l2)

            Jth = np.array(Jth)
            data = np.array(data)
            indices = np.array(indices)
            indptr = np.array(indptr)

            # Divide by n
            n = current_shape[0] * current_shape[1] * current_shape[2]
            #Jth /= n
            #data /= n
            print("Energy: %f"%(energy,))


            ncoeff = field.num_coefficients()
            JtJ = sp.sparse.csr_matrix((data, indices, indptr), shape=(ncoeff, ncoeff))
            #print(">>>%f, %f"%(JtJ.min(), JtJ.max()) )

            # Add the bending energy
            bgrad, bdata, bindices, bindptr = field.spline3d.get_bending_system(field.coef, current_sp)
            bgrad = np.array(bgrad)
            bdata = np.array(bdata)
            bindices = np.array(bindices)
            bindptr = np.array(bindptr)
            bhessian = sp.sparse.csr_matrix((bdata, bindices, bindptr), shape=(ncoeff, ncoeff))
            #print(">>>%f, %f"%(bhessian.min(), bhessian.max()))

            Jth += bgrad * l2
            JtJ += bhessian * l2


            x, info = sp.sparse.linalg.cg(JtJ, -1.0*Jth, tol=1e-3, maxiter=500)
            if info < 0:
                raise ValueError("Illegal input or breakdown")
            elif info > 0:
                print("Did not converge.")

            if b_coeff is None:
                b_coeff = x
            else:
                b_coeff += x

            field.copy_coefficients(b_coeff)
            b = field.get_volume()
            b=b.astype(floating)

        #rt.overlay_slices(w_up, w_down, slice_type=2)
        #rt.plot_slices(b)
    return field, w_up, w_down








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
    #field = gr.SplineField(sub_up.shape, kspacing)
    field = splines.CubicSplineField(sub_up.shape, kspacing)
    b_coeff = np.zeros(field.num_coefficients())
    field.copy_coefficients(b_coeff)
    b = field.get_volume()
    b=b.astype(floating)

    #smooth_params = [8.0, 6.0, 4.0, 3.0]
    smooth_params = [8.0]
    if True:
    #for it in range(5 * len(smooth_params)):
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

        w_up, mask_up = gr.warp_with_orfield(current_up, d, d_up, None, None, None, current_shape)
        w_down, mask_down= gr.warp_with_orfield(current_down, d, d_down, None, None, None, current_shape)
        w_up = np.array(w_up)
        w_down = np.array(w_down)
        if it == 0: # Plot initial state
            rt.overlay_slices(w_up, w_down, slice_type=2)

        dw_up, dmask_up= gr.warp_with_orfield(dcurrent_up, d, d_up, None, None, None, current_shape)
        dw_down, dmask_down = gr.warp_with_orfield(dcurrent_down, d, d_down, None, None, None, current_shape)
        db = field.get_volume((0,1,0))
        db = np.array(db).astype(floating)


        kernel = field.spline3d.get_kernel_grid((0,0,0))
        dkernel = field.spline3d.get_kernel_grid((0,1,0))
        # Get the linear system

        Jth, data, indices, indptr, energy= \
            gr.gauss_newton_system_andersson(w_up, w_down, dw_up, dw_down,
                                             mask_up, mask_down,
                                             kernel, dkernel, db, field.kspacing,
                                             field.grid_shape, l1, l2)
        print("Energy: %f"%(energy,))

        Jth = np.array(Jth)
        data = np.array(data)
        indices = np.array(indices)
        indptr = np.array(indptr)

        ncoeff = field.num_coefficients()
        JtJ = sp.sparse.csr_matrix((data, indices, indptr), shape=(ncoeff, ncoeff))

        #timeit x = sp.sparse.linalg.spsolve(JtJ, -1.0*Jth)
        #1 loops, best of 3: 3min 16s per loop
        #timeit x = sp.sparse.linalg.bicg(JtJ, -1.0*Jth)
        #1 loops, best of 3: 28.4 s per loop
        #timeit x = sp.sparse.linalg.bicgstab(JtJ, -1.0*Jth)
        #1 loops, best of 3: 13.9 s per loop
        #timeit x = sp.sparse.linalg.cgs(JtJ, -1.0*Jth)
        #1 loops, best of 3: 12.4 s per loop
        #timeit x = sp.sparse.linalg.gmres(JtJ, -1.0*Jth)
        #1 loops, best of 3: 2min 19s per loop
        #timeit x = sp.sparse.linalg.lgmres(JtJ, -1.0*Jth)
        #1 loops, best of 3: 14.9 s per loop
        x, info = sp.sparse.linalg.cg(JtJ, -1.0*Jth)
        if info < 0:
            raise ValueError("Illegal input or breakdown")

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
        w_up, mask_up = gr.warp_with_orfield(current_up, b, d_up, affine_idx_in, affine_idx_out,
                                    affine_disp, ref_shape)
        w_down, mask_down = gr.warp_with_orfield(current_down, b, d_down, affine_idx_in, affine_idx_out,
                                      affine_disp, ref_shape)
        w_up = np.array(w_up)
        w_down = np.array(w_down)

        dw_up, dmask_up = gr.warp_with_orfield(dcurrent_up, b, d_up, affine_idx_in, affine_idx_out,
                                     affine_disp, ref_shape)
        dw_down, dmask_down = gr.warp_with_orfield(dcurrent_down, b, d_down, affine_idx_in, affine_idx_out,
                                       affine_disp, ref_shape)
        db = field.get_volume((0,1,0))
        db = np.array(db).astype(floating)


        kernel = field.splines[(0,0,0)].spline
        dkernel = field.splines[(0,1,0)].spline
        # Get the linear system
        Jth, data, indices, indptr, energy= \
            gr.gauss_newton_system_andersson(w_up, w_down, dw_up, dw_down,
                                             mask_up, mask_down,
                                             kernel, dkernel, db, field.kspacing,
                                             field.grid_shape, l1, l2)
        print("Energy: %f"%(energy,))
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
        w_up, mask_up= gr.warp_with_orfield(current_up, b, d_up, affine_idx_in, affine_idx_out,
                                    affine_disp, ref_shape)
        w_down, mask_down = gr.warp_with_orfield(current_down, b, d_down, affine_idx_in, affine_idx_out,
                                      affine_disp, ref_shape)
        w_up = np.array(w_up)
        w_down = np.array(w_down)

        dw_up, dmask_up = gr.warp_with_orfield(dcurrent_up, b, d_up, affine_idx_in, affine_idx_out,
                                     affine_disp, ref_shape)
        dw_down, dmask_down = gr.warp_with_orfield(dcurrent_down, b, d_down, affine_idx_in, affine_idx_out,
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

if __name__ =="__main__":
    topup()
