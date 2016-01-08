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
                                     cc_splines_grad_epicor_motion,
                                     cc_splines_gradient,
                                     cc_splines_grad_epicor_general)
from dipy.align.transforms import (TranslationTransform3D,
                                   RigidTransform3D)
from dipy.align.imwarp import get_direction_and_spacings

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
        self.cnt = 0

    def energy_and_gradient(self,
                            down, down_grid2world, down_pedir,
                            up, up_grid2world, up_pedir,
                            field, field_grid2world,
                            mask_down=None, mask_up=None):
        b  = np.array(field.get_volume((0,0,0)))
        b = b.astype(np.float64)
        current_shape = np.array(b.shape, dtype=np.int32)

        db = np.array(field.get_volume((0,1,0))) #dtype=np.float64
        kernel = np.array(field.spline3d.get_kernel_grid((0,0,0)))
        dkernel = np.array(field.spline3d.get_kernel_grid((0,1,0)))
        kspacing = field.kspacing
        field_shape = field.grid_shape
        kcoef_grad = np.zeros_like(field.coef)
        # Compute gradients

        grad_up = np.empty(shape=(up.shape)+(3,), dtype=np.float64)
        grad_down = np.empty(shape=(down.shape)+(3,), dtype=np.float64)

        for i, grad in enumerate(sp.gradient(up)):
            grad_up[..., i] = grad

        for i, grad in enumerate(sp.gradient(down)):
            grad_down[..., i] = grad

        # Warp images and their derivatives
        Ain = None
        Aout = npl.inv(up_grid2world).dot(field_grid2world)
        Adisp = Aout

        w_up, _m = gr.warp_with_orfield(up, b, up_pedir, Ain,
                                        Aout, Adisp, current_shape)
        wgrad_up = np.zeros(shape=tuple(current_shape)+(3,), dtype=np.float64)
        gr.resample_vector_field(grad_up, b, up_pedir, Ain, Aout, Adisp, current_shape, wgrad_up)

        Ain = None
        Aout = npl.inv(down_grid2world).dot(field_grid2world)
        Adisp = Aout
        w_down, _m = gr.warp_with_orfield(down, b, down_pedir, Ain,
                                          Aout, Adisp, current_shape)

        wgrad_down = np.zeros(shape=tuple(current_shape)+(3,), dtype=np.float64)
        gr.resample_vector_field(grad_down, b, down_pedir, Ain, Aout, Adisp, current_shape, wgrad_down)

        dw_up = wgrad_up[...,1].copy()
        dw_down = wgrad_down[...,1].copy()

        if self.cnt % 10 == -1:
            #rt.overlay_slices(w_down, w_up, slice_type=2)
            #rt.overlay_slices(dw_down, dw_up, slice_type=2)
            #rt.plot_slices(b)
            rt.plot_slices(db)
        self.cnt += 1

        #rt.overlay_slices(dw_down, dw_up, slice_type=2)

        #print("[%f, %f], [%f, %f]"%(np.min(dw_down), np.max(dw_down), np.min(dw_up), np.max(dw_up)))
        #print("[%f, %f]"%(np.min(b), np.max(b)))


        #dw_up = gr.der_y(w_up)
        #dw_down = gr.der_y(w_down)
        # Convert to numpy arrays
        w_up = np.array(w_up)
        w_down = np.array(w_down)
        dw_up = np.array(dw_up)
        dw_down = np.array(dw_down)
        pedir_factor = down_pedir[1]

        # This should consider more general PE directions, but for now
        # it assumes it's along the y axis
        energy = cc_splines_gradient_epicor(w_down, w_up,
                                            dw_down, dw_up,
                                            pedir_factor,
                                            None, None, kernel, dkernel,
                                            db, kspacing, field_shape,
                                            self.radius, kcoef_grad)

        remember_last_iter = True
        if remember_last_iter:
            self.pedir_factor = pedir_factor
            self.w_down = w_down
            self.w_up = w_up
            self.dw_down = dw_down
            self.dw_up = dw_up
            self.kernel = kernel
            self.dkernel = dkernel
            self.db = db
            self.kspacing = kspacing
            self.field_shape = field_shape


        return energy, kcoef_grad


class OppositeBlips_CC_Motion(EPIDistortionModel):
    def __init__(self,
                 radius=4):
        r""" OppositeBlips_CC_Motion

        EPI distortion model based on the Normalized Cross Correlation
        metric, with motion estimation.

        Parameters
        ----------
        radius : int (optional)
            radius of the local window of the CC metric. Default is 4.
        """
        self.radius = radius
        self.transform = RigidTransform3D()
        #self.transform = TranslationTransform3D()
        self.iter_count = 0

    def energy_and_gradient(self,
                            down, down_grid2world, down_pedir, down_spacings,
                            up, up_grid2world, up_pedir, up_spacings, prealign,
                            theta, field, field_grid2world,
                            mask_down=None, mask_up=None):
        r""" The phase encode directions must be normalized and
        given in physical space coordinates
        """
        # Evaluate the field and its gradient on the curret grid
        b  = np.array(field.get_volume((0,0,0))).astype(np.float64)
        gb = np.empty(b.shape + (3,), dtype=np.float64)
        gb[...,0] = field.get_volume((1,0,0))
        gb[...,1] = field.get_volume((0,1,0))
        gb[...,2] = field.get_volume((0,0,1))
        # Now we know the shape of the field's grid
        current_shape = np.array(b.shape, dtype=np.int32)

        # Evaluate the kernel and its gradient
        kernel = np.array(field.spline3d.get_kernel_grid((0,0,0)))
        gkernel = np.empty(kernel.shape + (3,), dtype=np.float64)
        gkernel[...,0] = field.spline3d.get_kernel_grid((1,0,0))
        gkernel[...,1] = field.spline3d.get_kernel_grid((0,1,0))
        gkernel[...,2] = field.spline3d.get_kernel_grid((0,0,1))
        # This gradient is already in physical space coordinate system
        # because we did not interpolate any image but only evaluate
        # the analytical expression of the splines
        kspacing = field.kspacing
        field_shape = field.grid_shape

        # Compute gradients
        grad_up = np.empty(shape=(up.shape)+(3,), dtype=np.float64)
        grad_down = np.empty(shape=(down.shape)+(3,), dtype=np.float64)

        for i, grad in enumerate(np.gradient(up)):
            grad_up[..., i] = grad

        for i, grad in enumerate(np.gradient(down)):
            grad_down[..., i] = grad

        # Get the warped images considering motion and off-resonance field
        if theta is not None:
            R = self.transform.param_to_matrix(theta)
            # The center of rotation is the geometric center of the image
            c = np.array([current_shape[0], current_shape[1], current_shape[2]]) * 0.5
            R[:3, 3] += (c - R[:3,:3].dot(c))
        else:
            R = np.eye(4)

        if prealign is not None:
            R = prealign.dot(R)

        Ain = None
        Aout = npl.inv(up_grid2world).dot(field_grid2world).dot(R)
        Adisp = Aout

        #print("Aout:\n", Aout)
        #Adisp = np.eye(4)
        #_dir, _sp = get_direction_and_spacings(Aout, 3)
        #Adisp[:3,:3] = _dir[:,:]

        w_up, _m = gr.warp_with_orfield(up, b, up_pedir, Ain,
                                        Aout, Adisp, current_shape)
        wgrad_up = np.zeros(shape=tuple(current_shape)+(3,), dtype=np.float64)
        gr.resample_vector_field(grad_up, b, up_pedir, Ain, Aout, Adisp, current_shape, wgrad_up)

        Ain = None
        Aout = npl.inv(down_grid2world).dot(field_grid2world)
        Adisp = Aout
        Jpremult = npl.inv(down_grid2world).dot(field_grid2world)
        if prealign is not None:
            Jpremult = Jpremult.dot(prealign)
        #Adisp = np.eye(4)
        #_dir, _sp = get_direction_and_spacings(Aout, 3)
        #Adisp[:3,:3] = _dir[:,:]

        w_down, _m = gr.warp_with_orfield(down, b, down_pedir, Ain,
                                          Aout, Adisp, current_shape)
        wgrad_down = np.zeros(shape=tuple(current_shape)+(3,), dtype=np.float64)
        gr.resample_vector_field(grad_down, b, down_pedir, Ain, Aout, Adisp, current_shape, wgrad_down)


        #if self.iter_count%10 == 0 :
        #    rt.overlay_slices(w_down, w_up, slice_type=2)

        # Compute image gradients
        #out2up = npl.inv(up_grid2world).dot(field_grid2world).dot(R)
        #wgrad_up = np.zeros(shape=up.shape+(3,), dtype=np.float64)
        #gr.gradient_warped_3d(up, b, up_pedir, out2up, wgrad_up)

        #out2down = npl.inv(down_grid2world).dot(field_grid2world)
        #wgrad_down = np.zeros(shape=down.shape+(3,), dtype=np.float64)
        #gr.gradient_warped_3d(down, b, down_pedir, out2down, wgrad_down)

        #wgrad_up = np.empty(shape=(b.shape)+(3,), dtype=np.float64)
        #wgrad_down = np.empty(shape=(b.shape)+(3,), dtype=np.float64)

        #for i, grad in enumerate(sp.gradient(w_up)):
        #    wgrad_up[..., i] = grad

        #for i, grad in enumerate(sp.gradient(w_down)):
        #    wgrad_down[..., i] = grad

        # Allocate space for gradients and call
        kcoef_grad = np.zeros_like(field.coef)
        n = self.transform.get_number_of_parameters()
        dtheta = np.zeros(n, dtype=np.float64)
        energy = cc_splines_grad_epicor_general(w_down, w_up,
                                                down_grid2world, up_grid2world,
                                                down_pedir, up_pedir,
                                                wgrad_down, wgrad_up,
                                                None, None,
                                                kernel, gkernel,
                                                gb, field_grid2world,
                                                kspacing, field_shape,
                                                self.radius, Jpremult,
                                                self.transform, theta,
                                                kcoef_grad, dtheta)
        self.iter_count += 1

        remember_last_iter = False
        if remember_last_iter:
            self.reorientation_matrix = np.linalg.inv(down_grid2world).T
            self.w_down = w_down
            self.w_up = w_up
            self.down_grid2world = down_grid2world
            self.up_grid2world = up_grid2world
            self.down_pedir = down_pedir
            self.up_pedir = up_pedir
            self.wgrad_down = wgrad_down
            self.wgrad_up = wgrad_up
            self.kernel = kernel
            self.gkernel = gkernel
            self.gb = gb
            self.field_grid2world = field_grid2world
            self.kspacing = kspacing
            self.field_shape = field_shape

        return energy, kcoef_grad, dtheta



class SingleEPI_CC(EPIDistortionModel):
    def __init__(self,
                 radius=4):
        r""" SingleEPI_CC
        EPI distortion model based on a single EPI image and a T2 non-EPI
        image (undistorted). The metric is the modulated CC metric.
        """
        self.radius = radius

    def energy_and_gradient(self,
                            epi, epi_grid2world, epi_pedir,
                            nonepi, nonepi_grid2world, nonepi_pedir,
                            field, field_grid2world,
                            mask_epi=None, mask_nonepi=None):
        b  = np.array(field.get_volume((0,0,0)))
        b = b.astype(np.float64)
        current_shape = np.array(b.shape, dtype=np.int32)

        db = np.array(field.get_volume((0,1,0))) #dtype=np.float64
        kernel = np.array(field.spline3d.get_kernel_grid((0,0,0)))
        dkernel = np.array(field.spline3d.get_kernel_grid((0,1,0)))
        kspacing = field.kspacing
        field_shape = field.grid_shape
        kcoef_grad = np.zeros_like(field.coef)

        # Warp images and their derivatives
        Ain = None
        Aout = npl.inv(epi_grid2world).dot(field_grid2world)
        Adisp = Aout
        w_epi, _m = gr.warp_with_orfield(epi, b, epi_pedir, Ain,
                                        Aout, Adisp, current_shape)
        # align nonepi to the field grid
        affmap = AffineMap(None, current_shape, field_grid2world,
                           nonepi.shape, nonepi_grid2world)
        aligned_nonepi = affmap.transform(nonepi)

        dw_epi = gr.der_y(w_epi)
        dnonepi = gr.der_y(aligned_nonepi)
        # Convert to numpy arrays
        w_epi = np.array(w_epi)
        dw_epi = np.array(dw_epi)
        pedir_factor = epi_pedir[1]

        # This should consider more general PE directions, but for now
        # it assumes it's along the y axis
        energy = cc_splines_gradient(aligned_nonepi, w_epi, dnonepi, dw_epi,
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

    def energy_and_gradient(self,
                            epi, epi_grid2world, epi_pedir,
                            nonepi, nonepi_grid2world, nonepi_pedir,
                            field, field_grid2world,
                            epi_mask, nonepi_mask):
        b  = np.array(field.get_volume((0,0,0)))
        b = b.astype(np.float64)
        current_shape = np.array(b.shape, dtype=np.int32)

        db = np.array(field.get_volume((0,1,0))) #dtype=np.float64
        kernel = np.array(field.spline3d.get_kernel_grid((0,0,0)))
        dkernel = np.array(field.spline3d.get_kernel_grid((0,1,0)))
        kspacing = field.kspacing
        field_shape = field.grid_shape
        kcoef_grad = np.zeros_like(field.coef)

        # Resample input images at field's grid
        Ain = None
        Aout = npl.inv(epi_grid2world).dot(field_grid2world)
        Adisp = Aout
        w_epi, _m = gr.warp_with_orfield(epi, b, epi_pedir, Ain,
                                        Aout, Adisp, current_shape)
        w_epi_mask, _m = gr.warp_with_orfield_nn(epi_mask, b, epi_pedir, Ain,
                                        Aout, Adisp, current_shape)
        # align nonepi to the field grid
        affmap = AffineMap(None, current_shape, field_grid2world,
                           nonepi.shape, nonepi_grid2world)
        aligned_nonepi = affmap.transform(nonepi)
        aligned_nonepi_mask = affmap.transform(nonepi_mask, interp='nearest')

        ##################################################
        #Estimate the hidden variables (EM-initialization)
        ##################################################
        mult_by_mask = True
        if mult_by_mask:
            self.epi = np.array(w_epi) * np.array(w_epi_mask)
            self.nonepi = np.array(aligned_nonepi) * np.array(aligned_nonepi_mask)
        else:
            self.epi = np.array(w_epi)
            self.nonepi = np.array(aligned_nonepi)
        self.depi = gr.der_y(self.epi)
        self.dnonepi = gr.der_y(self.nonepi)
        self.epi_mask = w_epi_mask
        self.nonepi_mask = aligned_nonepi_mask

        #Use only the foreground intersection for estimation
        sampling_mask = np.array(self.nonepi_mask * self.epi_mask, dtype = np.int32)
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
        #rt.overlay_slices(self.epi, self.nonepi, slice_type=2)
        #rt.overlay_slices(self.epi_mask, self.nonepi_mask, slice_type=2)
        #rt.overlay_slices(self.epi, self.nonepiq_means_field, slice_type=2)

        self.dnonepiq_means_field = gr.der_y(self.nonepiq_means_field)
        pedir_factor = epi_pedir[1]
        energy = cc_splines_gradient(self.nonepiq_means_field, self.epi,
                                     self.dnonepiq_means_field, self.depi,
                                     pedir_factor,
                                     None, None, kernel, dkernel,
                                     db, kspacing, field_shape,
                                     self.radius, kcoef_grad)
        return energy, kcoef_grad



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
            raise ValueError('Inconsistent multiresolution settings. Expected: %d. Actual: %d'%(self.nstages, len(seq)))

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
                Id = np.eye(4)
                energy, grad = self.distortion_model.energy_and_gradient(
                            current1, Id, f1_pedir, current2, Id, f2_pedir, field)
                grad = np.array(gr.unwrap_scalar_field(grad))

                bending_energy, bending_grad = field.get_bending_gradient(resampled_sp)
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
            f1_mask = (f1_smooth>0).astype(np.int32)
            #f1_mask = (f1>0).astype(np.int32)
            #aff = AffineMap(None, shape1, affine1, f1.shape, f1_affine)
            #current1 = aff.transform(f1_smooth)


            # Resample second image
            # In the EPI vs. Non-EPI case, this is the non-epi image,
            # which is going to be rigidly aligned towards f1
            shape2 = self.f2_ss.get_domain_shape(scale)
            affine2 = self.f2_ss.get_affine(scale)
            f2_smooth = self.f2_ss.get_image(scale)
            f2_mask = (f2_smooth>0).astype(np.int32)
            #f2_mask = (f2>0).astype(np.int32)
            # We must warp f2 towards [shape1, affine1]
            #aff = AffineMap(None, shape1, affine1, f2.shape, f2_affine)
            #current2 = aff.transform(f2_smooth)


            #self.images.append([current1, current2])
            #continue

            resampled_sp = self.subsampling[stage] * spacings
            tps_lambda = self.lambdas[stage]

            # get the spline resolution from millimeters to voxels
            kspacing = np.round(self.warp_res[stage]/resampled_sp)
            kspacing = kspacing.astype(np.int32)
            kspacing[kspacing < 1] = 1

            # Scale space smoothing sigma
            #print(">>>kspacing:",kspacing)
            #print(">>>resampled_sp:",resampled_sp)
            # Create, rescale or keep field as needed
            if field is None:
                # The field has not been created, this must be the first stage
                print("Creating field")
                field = CubicSplineField(shape1, kspacing)
                b_coeff = np.zeros(field.num_coefficients())
                field.copy_coefficients(b_coeff)
            elif (not np.all(shape1 == field.vol_shape) or
                  not np.all(kspacing == field.kspacing)):
                b = field.get_volume()
                # We need to reshape the field
                new_field = CubicSplineField(shape1, kspacing)
                resample_affine = (self.subsampling[stage] * np.eye(4) /
                                   self.subsampling[stage-1])
                resample_affine[3,3] = 1.0
                print ("Resampling field:",resample_affine)
                new_b = vfu.transform_3d_affine(b.astype(np.float64),
                                                np.array(shape1, dtype=np.int32),
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
                            f1_smooth, f1_affine, f1_pedir,
                            f2_smooth, f2_affine, f2_pedir,
                            field, affine1,
                            f1_mask, f2_mask)
                if energy is None or grad is None:
                    break
                grad = np.array(gr.unwrap_scalar_field(grad))

                bending_energy, bending_grad = field.get_bending_gradient(resampled_sp)
                bending_grad = tps_lambda * np.array(bending_grad)
                print("Bending grad. range: [%f, %f]"%(bending_grad.min(), bending_grad.max()))
                print("Sim. grad. range: [%f, %f]"%(grad.min(), grad.max()))

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


    def optimize_with_ss_motion(self, f1, f1_affine, f1_pedir,
                               f2, f2_affine, f2_pedir,
                               spacings):

        fwhm2sigma = (np.sqrt(8.0 * np.log(2)))
        sigmas = (self.fwhm / fwhm2sigma)
        sigmas /= np.max(spacings)
        sigmas = np.sqrt(sigmas)
        self.f1_ss = IsotropicScaleSpace(f1,
                                         self.subsampling,
                                         sigmas,
                                         f1_affine,
                                         spacings,
                                         False)
        self.f2_ss = IsotropicScaleSpace(f2,
                                         self.subsampling,
                                         sigmas,
                                         f2_affine,
                                         spacings,
                                         False)
        field = None
        b = None
        b_coeff = None
        self.fields = []
        self.images = []
        prealign = np.eye(4)
        theta = self.distortion_model.transform.get_identity_parameters()
        for stage in range(self.nstages):
            scale = self.nstages - 1 - stage
            step_length = self.step_lengths[stage]
            print("Stage: %d / %d"%(stage + 1, self.nstages))

            # Resample first image
            shape1 = self.f1_ss.get_domain_shape(scale)
            affine1 = self.f1_ss.get_affine(scale)
            f1_smooth = self.f1_ss.get_image(scale)
            f1_mask = (f1_smooth>0).astype(np.int32)

            # Resample second image
            # In the EPI vs. Non-EPI case, this is the non-epi image,
            # which is going to be rigidly aligned towards f1
            shape2 = self.f2_ss.get_domain_shape(scale)
            affine2 = self.f2_ss.get_affine(scale)
            f2_smooth = self.f2_ss.get_image(scale)
            f2_mask = (f2_smooth>0).astype(np.int32)

            #rt.overlay_slices(f1_smooth, f2_smooth, slice_type=2)

            resampled_sp = self.subsampling[stage] * spacings
            tps_lambda = self.lambdas[stage]

            # get the spline resolution from millimeters to voxels
            kspacing = np.round(self.warp_res[stage]/resampled_sp)
            kspacing = kspacing.astype(np.int32)
            kspacing[kspacing < 1] = 1
            #kspacing = np.array([self.warp_res[stage]] * 3, np.int32)

            # Create, rescale or keep field as needed
            if field is None:
                # The field has not been created, this must be the first stage
                print("Creating field")
                field = CubicSplineField(shape1, kspacing)
                b_coeff = np.zeros(field.num_coefficients())
                field.copy_coefficients(b_coeff)
            elif (not np.all(shape1 == field.vol_shape) or
                  not np.all(kspacing == field.kspacing)):
                b = field.get_volume()
                # We need to reshape the field
                new_field = CubicSplineField(shape1, kspacing)

                prev_aff_inv = self.f1_ss.get_affine_inv(scale+1)
                cur_aff = self.f1_ss.get_affine(scale)
                resample_affine = prev_aff_inv.dot(cur_aff)
                new_b = vfu.transform_3d_affine(b.astype(np.float64),
                                                np.array(shape1, dtype=np.int32),
                                                resample_affine)
                new_b = np.array(new_b, dtype=np.float64)
                new_b *= ((1.0 * self.subsampling[stage-1]) /
                          self.subsampling[stage])
                # Compute the coefficients associated with the resampled field
                coef = new_field.spline3d.fit_to_data(new_b, 0.1)
                new_field.copy_coefficients(coef)
                field = new_field
                b_coeff = gr.unwrap_scalar_field(coef)
                # prealign is defined on the previous scale, we need to
                # change coordinates to this scale
                prealign =  npl.inv(resample_affine).dot(prealign).dot(resample_affine)

            # Start gradient descent
            self.energy_list = []
            tolerance = 0
            #if self.subsampling[stage] > 1:
            if False:
                # Do not estimate motion at full resolution
                theta = None
            elif theta is None:
                theta = self.distortion_model.transform.get_identity_parameters()

            best_coeff = None
            best_theta = None
            best_energy = None
            for it in range(self.level_iters[stage]):
                print("Iter: %d / %d"%(it + 1, self.level_iters[stage]))
                # f1 may be regarded as being moved by the rigid transform
                # in physical space
                #print('Prealign:\n',prealign)
                energy, grad, dtheta = self.distortion_model.energy_and_gradient(
                            f1_smooth, f1_affine, f1_pedir, spacings,
                            f2_smooth, f2_affine, f2_pedir, spacings, prealign, theta,
                            field, affine1,
                            f1_mask, f2_mask)
                #dtheta[1] = 0 # do not move along the pe-dir
                if energy is None or grad is None:
                    break
                grad = np.array(gr.unwrap_scalar_field(grad))

                bending_energy, bending_grad = field.get_bending_gradient(resampled_sp)
                bending_grad = tps_lambda * np.array(bending_grad)
                #print("Bending grad. range: [%f, %f]"%(bending_grad.min(), bending_grad.max()))
                #print("Sim. grad. range: [%f, %f]"%(grad.min(), grad.max()))

                bending_energy *= tps_lambda
                total_energy = energy + bending_energy
                if best_energy is None or (total_energy < best_energy):
                    best_energy = total_energy
                    best_coeff = b_coeff
                    best_theta = theta
                #print("Energy: %f [data] + %f [reg] = %f"%(energy, bending_energy, total_energy))
                step = -1 * (grad + bending_grad)

                #print(">>>>>>>>>>>>>>>>>> ", np.abs(step).max())
                #if np.abs(step).max() > 1:
                if True:
                    step = step_length * (step/np.abs(step).max())
                else:
                    step = step_length * step

                if theta is not None:
                    has_rotation = len(dtheta)>3
                    if has_rotation:
                        max_theta = np.arcsin(0.05/(2.0 * np.max(shape1)))
                        theta_step_length = np.array([max_theta, max_theta, max_theta, 0.05, 0.05, 0.05])
                        dtheta[:3] /= np.abs(dtheta[:3]).max()
                        dtheta[3:] /= np.abs(dtheta[3:]).max()
                        #dtheta /= np.abs(dtheta).max()
                    else:
                        theta_step_length = np.array([0.01, 0.0, 0.001])
                        dtheta /= np.abs(dtheta).max()
                    theta_step = -1.0 * theta_step_length * dtheta
                    theta += theta_step
                    print("Theta:", theta)

                if b_coeff is None:
                    b_coeff = step
                else:
                    b_coeff += step

                self.energy_list.append(total_energy)
                field.copy_coefficients(b_coeff)
                if len(self.energy_list)>=self.energy_window:
                    der = self._get_energy_derivative()
                    if der < tolerance:
                        break
                else:
                    der = np.inf

                print("Energy: %f (%f + %f). [%f]"%(total_energy, energy, bending_energy, der))
            if best_coeff is not None:
                b_coeff = best_coeff
                field.copy_coefficients(b_coeff)
                theta = best_theta
            # Get matrix representation of theta, according to transform
            # Prevent theta from being applied again, now `prealign` is
            # the full rigid transform at this scale.
            if theta is not None:
                R = self.distortion_model.transform.param_to_matrix(theta)
                # move the center of rotation to the geometric center of the image
                c = np.array([shape1[0], shape1[1], shape1[2]]) * 0.5
                R[:3, 3] += (c - R[:3,:3].dot(c))

                # Make prealign contain the full rigid transform
                prealign = prealign.dot(R)
                theta = self.distortion_model.transform.get_identity_parameters()

            self.fields.append(field.get_volume())

            # Evaluate current bending energy
            if False:
                bending_energy, bending_grad = field.get_bending_gradient(resampled_sp)
                print(">>> Field Shape:", field.coef.shape)
                print(">>> Bending energy:", bending_energy)
                rt.plot_slices(field.coef)
        return field, prealign
