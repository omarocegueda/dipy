import numpy as np
import numpy.linalg as npl
import scipy as sp
import dipy.viz.regtools as rt
import dipy.correct.gradients as gr
from dipy.align import expectmax as em
from dipy.align import crosscorr as cc
from dipy.align import vector_fields as vfu
from dipy.align.imaffine import AffineMap
from dipy.align.scalespace import IsotropicScaleSpace
from dipy.correct.splines import CubicSplineField
from dipy.correct.cc_splines import (cc_splines_gradient_epicor,
                                     cc_splines_gradient)

floating = np.float64

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

    def energy_and_gradient(self, down, down_pedir, up, up_pedir, field,
                            mask_down=None, mask_up=None):
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
        EPI distortion model based on a single EPI image and a T2 non-EPI
        image (undistorted). The metric is the modulated CC metric.
        """
        self.radius = radius

    def energy_and_gradient(self, nonepi, nonepi_pedir, epi, epi_pedir, field,
                            mask_nonepi=None, mask_epi=None):
        current_shape = np.array(epi.shape, dtype=np.int32)
        b  = np.array(field.get_volume((0,0,0)))
        b = b.astype(np.float64)

        db = np.array(field.get_volume((0,1,0))) #dtype=np.float64
        kernel = np.array(field.spline3d.get_kernel_grid((0,0,0)))
        dkernel = np.array(field.spline3d.get_kernel_grid((0,1,0)))
        kspacing = field.kspacing
        field_shape = field.grid_shape
        kcoef_grad = np.zeros_like(field.coef)

        # Numerical derivatives of input images
        depi = gr.der_y(epi)
        dnonepi = gr.der_y(nonepi)

        # Warp images and their derivatives
        w_epi, _m = gr.warp_with_orfield(epi, b, epi_pedir, None,
                                        None, None, current_shape)
        dw_epi, _m = gr.warp_with_orfield(depi, b, epi_pedir, None,
                                         None, None, current_shape)

        dw_epi = gr.der_y(w_epi)
        # Convert to numpy arrays
        w_epi = np.array(w_epi)
        dw_epi = np.array(dw_epi)
        pedir_factor = epi_pedir[1]

        # This should consider more general PE directions, but for now
        # it assumes it's along the y axis
        energy = cc_splines_gradient(nonepi, w_epi, dnonepi, dw_epi,
                                     pedir_factor,
                                     None, None, kernel, dkernel,
                                     db, kspacing, field_shape,
                                     self.radius, kcoef_grad)
        return energy, kcoef_grad


class SingleEPI_ECC(EPIDistortionModel):
    def __init__(self, radius=4, q_levels=256):
        r""" SingleEPI_ECC
        """
        self.radius = radius
        self.q_levels = q_levels

        self.nonepi_mask = None
        self.epi_mask = None
        self.nonepiq_means_field = None
        self.eqiq_means_field = None
        self.epiq_levels = None
        self.nonepiq_levels = None

        self.precompute_factors = cc.precompute_cc_factors_3d
        self.compute_forward_step = cc.compute_cc_forward_step_3d
        self.compute_backward_step = cc.compute_cc_backward_step_3d
        self.reorient_vector_field = vfu.reorient_vector_field_3d
        self.quantize = em.quantize_positive_3d
        self.compute_stats = em.compute_masked_class_stats_3d

    def initialize_iteration(self, epi, pedir, nonepi, field,
                             epi_mask=None, nonepi_mask=None):
        r"""
        Pre-computes the cross-correlation factors
        """
        ##################################################
        #Estimate the hidden variables (EM-initialization)
        ##################################################
        self.epi = epi
        self.nonepi = nonepi
        self.epi_mask = epi_mask
        self.nonepi_mask = nonepi_mask

        #Use only the foreground intersection for estimation
        sampling_mask = np.array(self.nonepi_mask*self.epi_mask, dtype = np.int32)
        self.sampling_mask = sampling_mask

        #Process the non-epi quantization
        nonepiq, self.nonepiq_levels, hist = self.quantize(self.nonepi,
                                                           self.q_levels)
        nonepiq = np.array(nonepiq, dtype=np.int32)
        self.nonepiq_levels = np.array(self.nonepiq_levels)
        nonepiq_means, nonepiq_variances = self.compute_stats(sampling_mask,
                                                              self.epi,
                                                              self.q_levels,
                                                              nonepiq)
        nonepiq_means[0] = 0
        self.nonepiq_means = np.array(nonepiq_means)
        self.nonepiq_variances = np.array(nonepiq_variances)
        self.nonepiq_variances[np.isinf(self.nonepiq_variances)] = self.nonepiq_variances.max()
        self.nonepiq_sigma_sq_field = self.nonepiq_variances[nonepiq]
        self.nonepiq_means_field = self.nonepiq_means[nonepiq]

        #Process the epi quantization
        epiq, self.epiq_levels, hist = self.quantize(self.epi,
                                                           self.q_levels)
        epiq = np.array(epiq, dtype=np.int32)
        self.epiq_levels = np.array(self.epiq_levels)
        epiq_means, epiq_variances = self.compute_stats(
            sampling_mask, self.nonepi, self.q_levels, epiq)
        epiq_means[0] = 0
        self.epiq_means = np.array(epiq_means)
        self.epiq_variances = np.array(epiq_variances)
        self.epiq_variances[np.isinf(self.epiq_variances)] = self.epiq_variances.max()
        self.epiq_sigma_sq_field = self.epiq_variances[epiq]
        self.epiq_means_field = self.epiq_means[epiq]

        ##################################################
        #Compute the CC factors (CC-initialization)
        ##################################################

        #self.nonepiq_factors = self.precompute_factors(self.nonepiq_means_field,
        #                                     self.epi,
        #                                     self.radius)
        #self.epiq_factors = self.precompute_factors(self.nonepi,
        #                                     self.epiq_means_field,
        #                                     self.radius)
        #
        #self.nonepiq_factors = np.array(self.nonepiq_factors)
        #self.epiq_factors = np.array(self.epiq_factors)

        ##################################################
        #Compute the gradients (common initialization)
        ##################################################
        self.gradient_epi = np.empty(
                shape=(self.epi.shape)+(3,), dtype=np.float64)
        for i, grad in enumerate(sp.gradient(self.epi)):
            self.gradient_epi[..., i] = grad

        self.gradient_nonepi = np.empty(
                shape=(self.nonepi.shape)+(3,), dtype=np.float64)
        for i, grad in enumerate(sp.gradient(self.nonepi)):
            self.gradient_nonepi[..., i] = grad

        # Note: the gradient needs to be in grid-space to estimate the
        # off-resonance field, but it must be in physical space to
        # align the non-epi image


    def energy_and_gradient_single(self, epi, pedir, nonepi, field,
                                   epi_mask=None, nonepi_mask=None):
        self.initialize_iteration(epi, pedir, nonepi, field, epi_mask, nonepi_mask)
        rt.overlay_slices(self.epi, self.nonepiq_means_field, slice_type=2)
        #rt.overlay_slices(self.epiq_means_field, self.nonepi, slice_type=2)
        return None, None

    def energy_and_gradient(self, epi, epi_pedir, non_epi, non_epi_pedir,
                            field, epi_mask=None, nonepi_mask=None):
        r"""
        Ignore `non_epi_pedir` parameter
        """
        return self.energy_and_gradient_single(epi, epi_pedir, non_epi, field,
                                               epi_mask, nonepi_mask)




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
            #continue
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


    def optimize_with_ss(self, f1, f1_affine, f1_pedir,
                               f2, f2_affine, f2_pedir,
                               spacings):
        fwhm2sigma = (np.sqrt(8.0 * np.log(2)))
        self.f1_ss = IsotropicScaleSpace(f1,
                                         self.subsampling,
                                         self.fwhm / fwhm2sigma,
                                         f1_affine,
                                         spacings,
                                         False)
        self.f2_ss = IsotropicScaleSpace(f2,
                                         self.subsampling,
                                         self.fwhm / fwhm2sigma,
                                         f2_affine,
                                         spacings,
                                         False)

        field = None
        b = None
        b_coeff = None
        self.fields = []
        self.images = []
        for stage in range(self.nstages):
            scale = self.nstages - 1 - stage
            step_length = self.step_lengths[stage]
            print("Stage: %d / %d"%(stage + 1, self.nstages))

            # Resample first image
            shape1 = self.f1_ss.get_domain_shape(scale)
            affine1 = self.f1_ss.get_affine(scale)
            f1_smooth = self.f1_ss.get_image(scale)
            aff = AffineMap(None, shape1, affine1, f1.shape, f1_affine)
            current1 = aff.transform(f1_smooth)
            current1_mask = (aff.transform(f1>0)).astype(np.int32)

            # Resample second image
            # In the EPI vs. Non-EPI case, this is the non-epi image,
            # which is going to be rigidly aligned towards f1
            shape2 = self.f2_ss.get_domain_shape(scale)
            affine2 = self.f2_ss.get_affine(scale)
            f2_smooth = self.f2_ss.get_image(scale)
            # We must warp f2 towards [shape1, affine1]
            aff = AffineMap(None, shape1, affine1, f2.shape, f2_affine)
            current2 = aff.transform(f2_smooth)
            current2_mask = (aff.transform(f2>0)).astype(np.int32)

            self.images.append([current1, current2])
            #continue

            resampled_sp = self.subsampling[stage] * spacings
            tps_lambda = self.lambdas[stage]

            # get the spline resolution from millimeters to voxels
            kspacing = np.round(self.warp_res[stage]/resampled_sp)
            kspacing = kspacing.astype(np.int32)
            kspacing[kspacing < 1] = 1

            # Scale space smoothing sigma
            print(">>>kspacing:",kspacing)
            print(">>>resampled_sp:",resampled_sp)
            # Create, rescale or keep field as needed
            if field is None:
                # The field has not been created, this must be the first stage
                print("Creating field")
                field = CubicSplineField(current1.shape, kspacing)
                b_coeff = np.zeros(field.num_coefficients())
                field.copy_coefficients(b_coeff)
            elif (not np.all(current1.shape == field.vol_shape) or
                  not np.all(kspacing == field.kspacing)):
                b = field.get_volume()
                # We need to reshape the field
                new_field = CubicSplineField(current1.shape, kspacing)
                resample_affine = (self.subsampling[stage] * np.eye(4) /
                                   self.subsampling[stage-1])
                resample_affine[3,3] = 1.0
                print ("Resampling field:",resample_affine)
                new_b = vfu.transform_3d_affine(b.astype(np.float64),
                                                np.array(current1.shape, dtype=np.int32),
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

            # Start gradient descent
            self.energy_list = []
            tolerance = 1e-6
            for it in range(self.level_iters[stage]):
                print("Iter: %d / %d"%(it + 1, self.level_iters[stage]))

                energy, grad = self.distortion_model.energy_and_gradient(
                            current1, f1_pedir, current2, f2_pedir, field,
                            current1_mask, current2_mask)
                if energy is None or grad is None:
                    break
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
