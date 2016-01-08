from __future__ import print_function
import numpy as np
import scipy as sp
from scipy import ndimage
from scipy import stats
from dipy.align.comparison import count_masked_values
from .metrics import SimilarityMetric
from . import vector_fields as vfu
from . import floating
from . import expectmax as em
from . import sumsqdiff as ssd
import dipy.viz.regtools as rt

class PolynomialTransfer(SimilarityMetric):
    def __init__(self,
                 dim,
                 degree=9,
                 smooth=1.0,
                 q_levels=256,
                 cprop=0.95,
                 drop_zeros=False):
        r"""PolynomialTransfer Metric

        Parameters
        ----------
        dim : int (either 2 or 3)
            the dimension of the image domain
        smooth : float
            smoothness parameter, the larger the value the smoother the
            deformation field
        q_levels : int
            number of quantization levels
        cprop : float in (0,1]
            proportion of samples for estimating the transfer
        drop_zeros : boolean
            disregard background (zeros) for fitting the transfer
        """
        super(PolynomialTransfer, self).__init__(dim)
        self.degree = degree
        self.smooth = smooth
        self.q_levels = q_levels
        self.static_image_mask = None
        self.moving_image_mask = None
        self.staticq_transferred = None
        self.movingq_transferred = None
        self.movingq_levels = None
        self.staticq_levels = None
        self.cprop = cprop
        self.drop_zeros = drop_zeros
        self._connect_functions()

    def _connect_functions(self):
        r"""Assign the methods to be called according to the image dimension

        Assigns the appropriate functions to be called for image quantization,
        statistics computation and multi-resolution iterations according to the
        dimension of the input images
        """
        if self.dim == 2:
            self.quantize = em.quantize_positive_2d
            self.compute_stats = em.compute_masked_class_stats_2d
            self.reorient_vector_field = vfu.reorient_vector_field_2d
        elif self.dim == 3:
            self.quantize = em.quantize_positive_3d
            self.compute_stats = em.compute_masked_class_stats_3d
            self.reorient_vector_field = vfu.reorient_vector_field_3d
        else:
            raise ValueError('Polynomial Metric not defined for dim. %d' % (self.dim))

    def estimate_polynomial_transfer(self, x, y, theta=None, tolerance=1e-9):
        n = len(x)
        c = int(self.cprop * n)
        p = self.degree

        if n < 10 * p:
            raise ValueError('Not enough data to fit polynomial of degree %d: %d given'%(p, n))

        # Initialize parameters with random sample
        if theta is None:
            sel = np.random.choice(range(n), c, replace=False)
            xsel = x[sel]
            ysel = y[sel]
            theta = np.polyfit(xsel, ysel, deg=p)
        else:
            y_hat = np.polyval(theta, x)
            residual_vector = (y - y_hat)**2
            sel = np.argsort(residual_vector)[:c]
            xsel = x[sel]
            ysel = y[sel]


        # Start LTS
        prev_residual = np.inf
        ysel_hat = np.polyval(theta, xsel)
        residual = np.sum((ysel-ysel_hat)**2)
        cnt = 0
        while tolerance < (prev_residual - residual):
            cnt += 1
            y_hat = np.polyval(theta, x)
            residual_vector = (y - y_hat)**2

            # Select the c smallest residuals
            sel = np.argsort(residual_vector)[:c]
            xsel = x[sel]
            ysel = y[sel]
            theta = np.polyfit(xsel, ysel, deg=p)
            ysel_hat = np.polyval(theta, xsel)
            prev_residual = residual
            residual = np.sum((ysel - ysel_hat)**2) # this is the sum of the rho's, eq. (9) in Guimod 2001
        # Start RLS
        # get the 0.5 + c/2N quantile of the standard Gaussian distribution
        q = 0.5 + c/(2.0*n)
        alpha = sp.stats.norm.ppf(q, 0, 1)

        # Numerically integrate the second non-central moment of the Gaussian distribution within [-alpha, alpha]
        integral = sp.integrate.quad(lambda x: (x**2)*sp.stats.norm.pdf(x, loc=0.0, scale=1), -alpha, alpha)
        K = (integral[0]*n) / c
        # Estimate standard deviation
        sigma_hat = np.sqrt(residual/(K*n))

        # Select all samples whose residual is within 3*sigma_hat
        y_hat = np.polyval(theta, x)
        rho = (y - y_hat)**2
        sel = rho <= 3*sigma_hat
        xsel = x[sel]
        ysel = y[sel]
        theta = np.polyfit(xsel, ysel, deg=p)

        return theta, sigma_hat, sel

    def polynomial_fit(self, x, y, x_eval, y_eval, theta0=None, theta1=None):
        # Allocate count vectors
        #print("Unique values: %f %%"%(100.0*(len(np.unique(y))-1)/np.sum(y!=0)))
        nvalues = 1 + np.max([np.max(x), np.max(x_eval)])
        n0 = np.empty(nvalues, dtype=np.int32)
        n1 = np.empty(nvalues, dtype=np.int32)

        # Estimate first transfer
        theta0, sigma0, sel0 = self.estimate_polynomial_transfer(x, y, theta0)
        # Estimate second transfer
        x1 = x[~sel0]
        y1 = y[~sel0]
        try:
            theta1, sigma1, sel1 = self.estimate_polynomial_transfer(x1, y1, theta1)
        except(ValueError):
            #print('Using mono-functional dependence model')
            yhat = np.polyval(theta0, x_eval)
            return yhat, theta0, None


        # Estimate marginal probabilities
        count_masked_values(x, sel0.astype(np.int32), n0)
        count_masked_values(x1, sel1.astype(np.int32), n1)
        total = n0 + n1
        pi0 = n0.astype(np.float64)
        pi1 = n1.astype(np.float64)

        pi0[total>0] /= total[total>0]
        pi1[total>0] /= total[total>0]

        pi0[total==0] = 0.5
        pi1[total==0] = 0.5

        # Estimate sigma
        n0 = sel0.sum()
        n1 = sel1.sum()
        p0 = (n0)/float(n0+n1)
        p1 = (n1)/float(n0+n1)
        sigma = p0*sigma0 + p1*sigma1

        #Gaussian distribution evaluated at all residuals w.r.t. each transfer
        yhat0 = np.polyval(theta0, x_eval)
        yhat1 = np.polyval(theta1, x_eval)
        d0 = yhat0 - y_eval
        d1 = yhat1 - y_eval
        G0 = sp.stats.norm.pdf(d0, loc=0, scale=sigma)
        G1 = sp.stats.norm.pdf(d1, loc=0, scale=sigma)

        #Final weights
        den = pi0[x_eval] * G0 + pi1[x_eval] * G1
        P0 = (pi0[x_eval] * G0) / den
        P1 = (pi1[x_eval] * G1) / den

        P0[den==0] = 0.5
        P1[den==0] = 0.5

        yhat = yhat0*P0 + yhat1*P1
        return yhat, theta0, theta1

    def initialize_iteration(self):
        r"""Prepares the metric to compute one displacement field iteration.

        Pre-computes the transfer functions. Also pre-computes the gradient of both
        input images. Note that once the images are transformed to the opposite
        modality, the gradient of the transformed images can be used with the
        gradient of the corresponding modality in the same fashion as
        diff-demons does for mono-modality images.
        """
        sampling_mask = self.static_image_mask*self.moving_image_mask
        self.sampling_mask = sampling_mask

        # Quantize static image
        staticq, self.staticq_levels, hist = self.quantize(self.static_image, self.q_levels)
        staticq = np.array(staticq, dtype=np.int32)
        self.staticq_levels = np.array(self.staticq_levels)

        # Quantize moving image
        movingq, self.movingq_levels, hist = self.quantize(self.moving_image, self.q_levels)
        movingq = np.array(movingq, dtype=np.int32)
        self.movingq_levels = np.array(self.movingq_levels)


        x_data_staticq = None
        y_data_staticq = None
        x_data_movingq = None
        y_data_movingq = None
        if self.drop_zeros:
            x_data_staticq = staticq[sampling_mask != 0]
            y_data_staticq = self.moving_image[sampling_mask != 0]
            x_data_movingq = movingq[sampling_mask != 0]
            y_data_movingq = self.static_image[sampling_mask != 0]
        else:
            x_data_staticq = staticq.reshape(-1)
            y_data_staticq = self.moving_image.reshape(-1)
            x_data_movingq = movingq.reshape(-1)
            y_data_movingq = self.static_image.reshape(-1)

        #rt.plot_slices(sampling_mask)
        #print(x_data_staticq.shape, y_data_staticq.shape, x_data_movingq.shape, y_data_movingq.shape)
        #print((x_data_staticq!=0).sum(), x_data_staticq.shape[0])
        #print((x_data_movingq!=0).sum(), x_data_movingq.shape[0])
        #print(sampling_mask.sum(), sampling_mask.size)

        # Compute polynomial transfer from static to moving intensities
        self.staticq_transferred, theta0_staticq, theta1_staticq =\
            self.polynomial_fit(x_data_staticq,
                                y_data_staticq,
                                staticq.reshape(-1),
                                self.moving_image.reshape(-1))

        self.staticq_transferred = self.staticq_transferred.reshape(staticq.shape)

        # Compute polynomial transfer from moving to static intensities
        self.movingq_transferred, theta0_movingq, theta1_movingq =\
            self.polynomial_fit(x_data_movingq,
                                y_data_movingq,
                                movingq.reshape(-1),
                                self.static_image.reshape(-1))

        self.movingq_transferred = self.movingq_transferred.reshape(movingq.shape)

        if self.drop_zeros:
            self.staticq_transferred[sampling_mask == 0] = 0
            self.movingq_transferred[sampling_mask == 0] = 0


        # Compute moving image's gradient and reorient to physical space
        self.gradient_moving = np.empty(
            shape=(self.moving_image.shape)+(self.dim,), dtype=floating)

        for i, grad in enumerate(sp.gradient(self.moving_image)):
            self.gradient_moving[..., i] = grad

        if self.moving_spacing is not None:
            self.gradient_moving /= self.moving_spacing
        if self.moving_direction is not None:
            self.reorient_vector_field(self.gradient_moving, self.moving_direction)

        # Compute static image's gradient and reorient to physical space
        self.gradient_static = np.empty(
            shape=(self.static_image.shape)+(self.dim,), dtype=floating)

        for i, grad in enumerate(sp.gradient(self.static_image)):
            self.gradient_static[..., i] = grad

        if self.static_spacing is not None:
            self.gradient_static /= self.static_spacing
        if self.static_direction is not None:
            self.reorient_vector_field(self.gradient_static, self.static_direction)

    def free_iteration(self):
        r"""
        Frees the resources allocated during initialization
        """
        del self.sampling_mask
        del self.staticq_levels
        del self.movingq_levels
        del self.staticq_transferred
        del self.movingq_transferred
        del self.gradient_moving
        del self.gradient_static

    def compute_forward(self):
        """Computes one step bringing the reference image towards the static.

        Computes the forward update field to register the moving image towards
        the static image in a gradient-based optimization algorithm
        """
        return self.compute_demons_step(True)

    def compute_backward(self):
        r"""Computes one step bringing the static image towards the moving.

        Computes the update displacement field to be used for registration of
        the static image towards the moving image
        """
        return self.compute_demons_step(False)

    def compute_demons_step(self, forward_step=True):
        r"""Demons step for PolynomialTransfer metric

        Parameters
        ----------
        forward_step : boolean
            if True, computes the Demons step in the forward direction
            (warping the moving towards the static image). If False,
            computes the backward step (warping the static image to the
            moving image)

        Returns
        -------
        displacement : array, shape (R, C, 2) or (S, R, C, 3)
            the Demons step
        """
        sigma_reg_2 = np.sum(self.static_spacing**2)/self.dim

        if forward_step:
            gradient = self.gradient_static
            delta_field = (self.static_image - self.movingq_transferred).astype(floating)
        else:
            gradient = self.gradient_moving
            delta_field = (self.moving_image - self.staticq_transferred).astype(floating)

        if self.dim == 2:
            step, self.energy = ssd.compute_ssd_demons_step_2d(delta_field,
                                                               gradient,
                                                               sigma_reg_2,
                                                               None)
        else:
            step, self.energy = ssd.compute_ssd_demons_step_3d(delta_field,
                                                               gradient,
                                                               sigma_reg_2,
                                                               None)
        for i in range(self.dim):
            step[..., i] = ndimage.filters.gaussian_filter(step[..., i],
                                                           self.smooth)
        return step

    def get_energy(self):
        r"""The numerical value assigned by this metric to the current image pair

        """
        return self.energy

    def use_static_image_dynamics(self, original_static_image, transformation):
        r"""This is called by the optimizer just after setting the static image.

        PolynomialTransfer metric takes advantage of the image dynamics by
        computing the current static image mask from the
        original_static_image mask (warped by nearest neighbor interpolation)

        Parameters
        ----------
        original_static_image : array, shape (R, C) or (S, R, C)
            the original static image from which the current static image was
            generated, the current static image is the one that was provided
            via 'set_static_image(...)', which may not be the same as the
            original static image but a warped version of it (even the static
            image changes during Symmetric Normalization, not only the moving
            one).
        transformation : DiffeomorphicMap object
            the transformation that was applied to the original_static_image
            to generate the current static image
        """
        self.static_image_mask = (original_static_image > 0).astype(np.int32)
        if transformation is None:
            return
        shape = np.array(self.static_image.shape, dtype=np.int32)
        affine = self.static_affine
        self.static_image_mask = transformation.transform(
            self.static_image_mask, 'nearest', None, shape, affine)

    def use_moving_image_dynamics(self, original_moving_image, transformation):
        r"""This is called by the optimizer just after setting the moving image.

        PolynomialTransfer metric takes advantage of the image dynamics by
        computing the current moving image mask from the
        original_moving_image mask (warped by nearest neighbor interpolation)

        Parameters
        ----------
        original_moving_image : array, shape (R, C) or (S, R, C)
            the original moving image from which the current moving image was
            generated, the current moving image is the one that was provided
            via 'set_moving_image(...)', which may not be the same as the
            original moving image but a warped version of it.
        transformation : DiffeomorphicMap object
            the transformation that was applied to the original_moving_image
            to generate the current moving image
        """
        self.moving_image_mask = (original_moving_image > 0).astype(np.int32)
        if transformation is None:
            return
        shape = np.array(self.moving_image.shape, dtype=np.int32)
        affine = self.moving_affine
        self.moving_image_mask = transformation.transform(
            self.moving_image_mask, 'nearest', None, shape, affine)