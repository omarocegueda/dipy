from __future__ import print_function
import os
import pickle
import numpy as np
import numpy.linalg as npl
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import nibabel as nib
import dipy.align.imwarp as imwarp
import dipy.align.vector_fields as vfu
from dipy.align import VerbosityLevels
from dipy.align.imaffine import MutualInformationMetric, AffineRegistration
from dipy.align.transforms import regtransforms
import dipy.correct.gradients as gr
import dipy.correct.splines as splines
from dipy.correct.splines import CubicSplineField
from dipy.correct.cc_splines import (cc_splines_gradient,
                                     cc_splines_gradient_epicor)
import dipy.viz.regtools as rt
from experiments.registration.regviz import overlay_slices
from inverse.dfinverse_3d import warp_points_3d, compute_jacobian_3d
import experiments.registration.dataset_info as info
from dipy.align.expectmax import (quantize_positive_3d,
                                  compute_masked_class_stats_3d)
#from dipy.align import floating
from experiments.registration.regviz import (overlay_slices,
                                             overlay_slices_with_contours)
from dipy.align.scalespace import IsotropicScaleSpace
#import mayavi
#import mayavi.mlab as mlab
#from tvtk.tools import visual
#from tvtk.api import tvtk
floating = np.float64
data_dir = '/home/omar/data/topup_example/'
up_fname = data_dir + "b0_blipup.nii"
#up_fname = info.get_scil(1, 'b0_up_strip')
down_fname = data_dir + "b0_blipdown.nii"
up_unwarped_fname = data_dir+'b0_blipup_unwarped.nii'
t1_fname = info.get_scil(1, 't1_strip')

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


class EPIDistortionModel(object):
    pass


class OppositeBlips_SSD(EPIDistortionModel):
    pass


class OppositeBlips_CC(EPIDistortionModel):
    def __init__(self,
                 radius=4):
        r""" OppositeBlips_CC

        EPI distortion model based on the Normalized Cross Correlation
        metric.

        Parameters
        ----------
        radius : int (optional)
            radius of the local window of the CC metric. Default is 4.
        """
        self.radius = radius

    def energy_and_gradient(self, down, down_pedir, up, up_pedir, field):
        current_shape = np.array(up.shape, dtype=np.int32)
        b  = np.array(field.get_volume((0,0,0)))
        b = b.astype(np.float64)

        db = np.array(field.get_volume((0,1,0))) #dtype=np.float64
        kernel = np.array(field.spline3d.get_kernel_grid((0,0,0)))
        dkernel = np.array(field.spline3d.get_kernel_grid((0,1,0)))
        kspacing = field.kspacing
        field_shape = field.grid_shape
        kcoef_grad = np.zeros_like(field.coef)

        # Numerical derivatives of input images
        dup = gr.der_y(up)
        ddown = gr.der_y(down)

        # Warp images and their derivatives
        w_up, _m = gr.warp_with_orfield(up, b, up_pedir, None,
                                        None, None, current_shape)
        dw_up, _m = gr.warp_with_orfield(dup, b, up_pedir, None,
                                         None, None, current_shape)
        w_down, _m = gr.warp_with_orfield(down, b, down_pedir, None,
                                          None, None, current_shape)
        dw_down, _m = gr.warp_with_orfield(ddown, b, down_pedir,
                                           None, None, None, current_shape)

        dw_up = gr.der_y(w_up)
        dw_down = gr.der_y(w_down)
        # Convert to numpy arrays
        w_up = np.array(w_up)
        w_down = np.array(w_down)
        dw_up = np.array(dw_up)
        dw_down = np.array(dw_down)
        pedir_factor = up_pedir[1]

        # This should consider more general PE directions, but for now
        # it assumes it's along the y axis
        energy = cc_splines_gradient_epicor(w_down, w_up, dw_down, dw_up,
                                            pedir_factor,
                                            None, None, kernel, dkernel,
                                            db, kspacing, field_shape,
                                            self.radius, kcoef_grad)
        return energy, kcoef_grad


class SingleEPI_CC(EPIDistortionModel):
    def __init__(self,
                 radius=4):
        r""" SingleEPI_CC
        """
        self.radius = radius

    def energy_and_gradient(self, up, up_pedir, down, down_pedir, field):
        return None


class SingleEPI_ECC(EPIDistortionModel):
    def __init__(self,
                 radius=4):
        r""" SingleEPI_ECC
        """
        self.radius = radius

    def energy_and_gradient(self, up, up_pedir, down, down_pedir, field):
        return None


class OffResonanceFieldEstimator(object):
    def __init__(self,
                 distortion_model,
                 level_iters=None,
                 subsampling=None,
                 warp_res=None,
                 fwhm=None,
                 lambdas=None,
                 step_lengths=None):
        r""" OffResonanceFieldEstimator

        Estimates the off-resonance field causing geometric distortions and
        intensity modulations on echo-planar images.

        Parameters
        ----------
        distortion_model : object derived from EPIDistortionModel
            an object defining the similarity measure used for estimation
            of the off-resonance field
        level_iters : sequence of ints (optional)
            the maximum number of iteration per stage. If None (default),
            the level iters are set to [200, 100, 50, 25, 12, 10, 10, 5, 5]
        subsampling : sequence of ints (optional)
            subsampling factor in each resolution. If None (default),
            the subsampling factors are set to [2, 2, 2, 2, 2, 1, 1, 1, 1]
        warp_res : sequence of floats (optional)
            separation (in millimeters) between the spline knots
            parameterizing the off-resonance field in each resolution
            stage. If Nont (default), the separations are set to
            [20, 16, 14, 12, 10, 6, 4, 4, 4]
        fwhm : sequence of floats (optional)
            full width at half magnitude of the smoothing kernel used at
            each resolution. If None (default), the smoothig parameters
            are set to [8, 6, 4, 3, 3, 2, 1, 0, 0].
        lambdas : sequence of floats (optional)
            regularization parameter of the thin-plate spline model per
            stage. If None (default), the regularization parameters are set
            to
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.005, 0.0005]
        step_lengths : sequence of floats
            the step length in the gradient descent optimization for each
            resolution. The step length is the maximum spline coefficient
            variation at each iteration
        """
        self.distortion_model = distortion_model
        if level_iters is None:
            level_iters = [100, 50, 50, 25, 100, 50, 25, 15, 10]
        if subsampling is None:
            subsampling = [2, 2, 2, 2, 2, 1, 1, 1, 1]
        if warp_res is None:
            warp_res = [20, 16, 14, 12, 10, 6, 4, 4, 4]
        if fwhm is None:
            fwhm = [8, 6, 4, 3, 3, 2, 1, 0, 0]
        if lambdas is None:
            lambdas = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                       0.5, 0.5, 0.05]
        if step_lengths is None:
            step_lengths = [0.2, 0.2, 0.2, 0.2, 0.2,
                            0.35, 0.35, 0.35, 0.35]
        self._set_multires_params(level_iters, subsampling, warp_res,
                                  fwhm, lambdas, step_lengths)
        self.energy_list = []
        self.energy_window = 12

    def _approximate_derivative_direct(self, x, y):
        r"""Derivative of the degree-2 polynomial fit of the given x, y pairs

        Directly computes the derivative of the least-squares-fit quadratic
        function estimated from (x[...],y[...]) pairs.

        Parameters
        ----------
        x : array, shape (n,)
            increasing array representing the x-coordinates of the points to
            be fit
        y : array, shape (n,)
            array representing the y-coordinates of the points to be fit

        Returns
        -------
        y0 : float
            the estimated derivative at x0 = 0.5*len(x)
        """
        x = np.asarray(x)
        y = np.asarray(y)
        X = np.row_stack((x**2, x, np.ones_like(x)))
        XX = (X).dot(X.T)
        b = X.dot(y)
        beta = npl.solve(XX, b)
        x0 = 0.5 * len(x)
        y0 = 2.0 * beta[0] * (x0) + beta[1]
        return y0


    def _get_energy_derivative(self):
        r"""Approximate derivative of the energy profile

        Returns the derivative of the estimated energy as a function of "time"
        (iterations) at the last iteration
        """
        n_iter = len(self.energy_list)
        if n_iter < self.energy_window:
            raise ValueError('Not enough data to fit the energy profile')
        x = range(self.energy_window)
        y = self.energy_list[(n_iter - self.energy_window):n_iter]
        ss = sum(y)
        if(ss > 0):
            ss *= -1
        y = [v / ss for v in y]
        der = self._approximate_derivative_direct(x, y)
        return der

    def _set_nstages_from_sequence(self, seq):
        r""" Sets the number of stages of the multiresolution strategy

        Verifies that the length of the array is consistent with the
        number of stages previously set.

        Parameters
        ----------
        seq : sequence
            the number of stages will be set as the length of the sequence.
            If None, the number of stages is set to None. If the number of
            stages previously set is not None and it's different from the
            length of the sequence then an exception is raised (the input
            sequence is considered inconsistent with some previously set
            sequence).
        """
        if seq is None:
            self.nstages = None
        elif self.nstages is None:
            self.nstages = len(seq)
        elif self.nstages != len(seq):
            raise ValueError('Inconsistent multiresolution settings')

    def _set_multires_params(self, level_iters=None, subsampling=None,
                             warp_res=None, fwhm=None, lambdas=None,
                             step_lengths=None):
        r""" Verify that the multi-resolution parameters are consistent

        Parameters
        ----------
        level_iters : sequence of ints (optional)
            the maximum number of iteration per stage. If None (default),
            the previous sequence is left unchanged.
        subsampling : sequence of ints
            subsampling factor in each resolution. If None (default),
            the previous sequence is left unchanged.
        warp_res : sequence of floats
            separation (in millimeters) between the spline knots
            parameterizing the off-resonance field in each resolution
            stage. If None (default), the previous sequence is left
            unchanged.
        fwhm : sequence of floats
            full width at half magnitude of the smoothing kernel used at
            each resolution. If None (default), the previous sequence is
            left unchanged.
        lambdas : sequence of floats
            regularization parameter of the thin-plate spline model per
            stage. If None (default), the previous sequence is left
            unchanged.
        """
        self._set_nstages_from_sequence(None)
        if level_iters is not None:
            self.level_iters = np.copy(level_iters)
        self._set_nstages_from_sequence(self.level_iters)
        if subsampling is not None:
            self.subsampling = np.copy(subsampling)
        self._set_nstages_from_sequence(self.subsampling)
        if warp_res is not None:
            self.warp_res = np.copy(warp_res)
        self._set_nstages_from_sequence(self.warp_res)
        if fwhm is not None:
            self.fwhm = np.copy(fwhm)
        self._set_nstages_from_sequence(self.fwhm)
        if lambdas is not None:
            self.lambdas = np.copy(lambdas)
        self._set_nstages_from_sequence(self.lambdas)
        if step_lengths is not None:
            self.step_lengths = np.copy(step_lengths)
        self._set_nstages_from_sequence(self.step_lengths)

    def optimize(self, f1, f1_pedir, f2, f2_pedir, spacings):
        r""" Start off-resonance field estimation
        """
        try_isotropic_ss = False
        if try_isotropic_ss:
            fwhm2sigma = (np.sqrt(8.0 * np.log(2)))
            self.f1_ss = IsotropicScaleSpace(f1,
                                             self.subsampling,
                                             self.fwhm / fwhm2sigma,
                                             None,
                                             spacings,
                                             False)
            self.f2_ss = IsotropicScaleSpace(f2,
                                             self.subsampling,
                                             self.fwhm / fwhm2sigma,
                                             None,
                                             spacings,
                                             False)
        field = None
        b = None
        b_coeff = None
        self.fields = []
        self.images = []
        for stage in range(self.nstages):
            step_length = self.step_lengths[stage]
            print("Stage: %d / %d"%(stage + 1, self.nstages))
            if self.subsampling[stage] > 1:
                sub1 = vfu.downsample_scalar_field_3d(f1)
                sub2 = vfu.downsample_scalar_field_3d(f2)
            else:
                sub1 = f1
                sub2 = f2

            resampled_sp = self.subsampling[stage] * spacings
            tps_lambda = self.lambdas[stage]

            # get the spline resolution from millimeters to voxels
            kspacing = np.round(self.warp_res[stage]/resampled_sp)
            kspacing = kspacing.astype(np.int32)
            kspacing[kspacing < 1] = 1

            # Scale space smoothing sigma
            sigma_mm = self.fwhm[stage] / (np.sqrt(8.0 * np.log(2)))
            sigma_vox = sigma_mm/resampled_sp
            print(">>>kspacing:",kspacing)
            print(">>>resampled_sp:",resampled_sp)
            print(">>>sigma_vox:",sigma_vox)

            # Create, rescale or keep field as needed
            if field is None:
                # The field has not been created, this must be the first stage
                print ("Creating field")
                field = CubicSplineField(sub1.shape, kspacing)
                b_coeff = np.zeros(field.num_coefficients())
                field.copy_coefficients(b_coeff)
            elif (not np.all(sub1.shape == field.vol_shape) or
                  not np.all(kspacing == field.kspacing)):
                b = field.get_volume()
                # We need to reshape the field
                new_field = CubicSplineField(sub1.shape, kspacing)
                resample_affine = (self.subsampling[stage] * np.eye(4) /
                                   self.subsampling[stage-1])
                resample_affine[3,3] = 1.0
                print ("Resampling field:",resample_affine)
                new_b = vfu.transform_3d_affine(b.astype(np.float64),
                                                np.array(sub1.shape, dtype=np.int32),
                                                resample_affine)
                new_b = np.array(new_b, dtype=np.float64)
                # Scale to new voxel size
                new_b *= ((1.0 * self.subsampling[stage-1]) /
                          self.subsampling[stage])
                # Compute the coefficients associated with the resampled field
                coef = new_field.spline3d.fit_to_data(new_b, 0.0)
                new_field.copy_coefficients(coef)
                field = new_field
                b_coeff = gr.unwrap_scalar_field(coef)
            else:
                print ("Keeping field as is")

            # smooth out images and compute spatial derivatives
            current1 = sp.ndimage.filters.gaussian_filter(sub1, sigma_vox)
            current2 = sp.ndimage.filters.gaussian_filter(sub2, sigma_vox)
            self.images.append([current1, current2])
            #rt.overlay_slices(current1, current2, slice_type=2)
            #current1 = self.f1_ss.get_image(self.nstages - 1 - stage)
            #current2 = self.f2_ss.get_image(self.nstages - 1 - stage)

            # Start gradient descent
            self.energy_list = []
            tolerance = 1e-6
            for it in range(self.level_iters[stage]):
                print("Iter: %d / %d"%(it + 1, self.level_iters[stage]))

                energy, grad = self.distortion_model.energy_and_gradient(
                            current1, f1_pedir, current2, f2_pedir, field)
                grad = np.array(gr.unwrap_scalar_field(grad))

                bending_energy, bending_grad = field.get_bending_gradient()
                bending_grad = tps_lambda * np.array(bending_grad)

                bending_energy *= tps_lambda
                total_energy = energy + bending_energy
                self.energy_list.append(total_energy)
                #print("Energy: %f [data] + %f [reg] = %f"%(energy, bending_energy, total_energy))
                step = -1 * (grad + bending_grad)
                step = step_length * (step/np.abs(step).max())
                if b_coeff is None:
                    b_coeff = step
                else:
                    b_coeff += step
                field.copy_coefficients(b_coeff)
                if len(self.energy_list)>=self.energy_window:
                    der = self._get_energy_derivative()
                    if der < tolerance:
                        break
                else:
                    der = np.inf
                print("Energy: %f. [%f]"%(total_energy, der))
            self.fields.append(field.get_volume())
        return field

def extend_volume(vol, margin):
    dims = np.array(vol.shape)
    dims += 2 * margin
    new_vol = np.zeros(tuple(dims))
    new_vol[margin:-margin, margin:-margin, margin:-margin] = vol[...]
    return new_vol.astype(vol.dtype)

def test_epicor_api_cc():
    # Load images
    up_nib = nib.load(up_fname)
    up_affine = up_nib.get_affine()
    direction, spacings = imwarp.get_direction_and_spacings(up_affine, 3)
    up = up_nib.get_data().squeeze().astype(np.float64)

    down_nib = nib.load(down_fname)
    down = down_nib.get_data().squeeze().astype(np.float64)

    radius = 4

    use_extend_volume = True
    if use_extend_volume:
        up = extend_volume(up, 2 * radius + 1)
        down = extend_volume(down, 2 * radius + 1)


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





