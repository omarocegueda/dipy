import numpy as np
import numpy.linalg as npl
import scipy as sp
import scipy.ndimage as ndimage
from dipy.align import floating
import dipy.align.vector_fields as vf
from dipy.align.mattes import MattesBase, sample_domain_regular
from dipy.core.optimize import Optimizer
import matplotlib.pyplot as plt
from dipy.align.transforms import (Transform)
from dipy.align.imwarp import (get_direction_and_spacings, ScaleSpace)
from dipy.align.scalespace import IsotropicScaleSpace


class MattesMIMetric(MattesBase):
    def __init__(self, nbins=32, sampling_proportion=None):
        r""" Initializes an instance of the Mattes MI metric

        This class implements the methods required by Optimizer to drive the
        registration process by making calls to the low level methods defined
        in MattesBase.

        Parameters
        ----------
        nbins : int
            the number of bins to be used for computing the intensity histograms
        sampling_proportion : int in (0,100]
            the percentage of voxels to be used for estimating the (joint and marginal)
            intensity histograms. If None, dense sampling is used.
        """
        super(MattesMIMetric, self).__init__(nbins)
        self.sampling_proportion = sampling_proportion

    def setup(self, transform, static, moving, moving_spacing,
              static_grid2space=None, moving_grid2space=None, prealign=None):
        r""" Prepares the metric to compute intensity densities and gradients

        The histograms will be setup to compute probability densities of
        intensities within the minimum and maximum values of `static` and
        `moving`


        Parameters
        ----------
        transform: instance of Transform
            the transformation with respect to whose parameters the gradient
            must be computed
        static : array, shape (S, R, C) or (R, C)
            static image
        moving : array, shape (S', R', C') or (R', C')
            moving image
        moving_spacing : array, shape (3,) or (2,)
            the spacing between neighboring voxels along each dimension
        static_grid2space : array (dim+1, dim+1)
            the grid-to-space transform of the static image
        moving_grid2space : array (dim+1, dim+1)
            the grid-to-space transform of the moving image
        prealign : array, shape (dim+1, dim+1)
            the pre-aligning matrix (an affine transform) that roughly aligns
            the moving image towards the static image. If None, no pre-alignment
            is performed. If a pre-alignment matrix is available, it is
            recommended to directly provide the transform to the MattesMIMetric
            instead of manually warping the moving image and provide None or
            identity as prealign. This way, the metric avoids performing more
            than one interpolation.
        """
        self.dim = len(static.shape)
        self.transform = transform
        self.static = np.array(static).astype(np.float64)
        self.moving = np.array(moving).astype(np.float64)
        self.static_grid2space = static_grid2space
        self.static_space2grid = npl.inv(static_grid2space)
        self.moving_grid2space = moving_grid2space
        self.moving_space2grid = npl.inv(moving_grid2space)
        self.moving_spacing = moving_spacing
        self.prealign = prealign
        self.param_scales = None

        T = np.eye(self.dim + 1)
        if self.prealign is not None:
            T = T.dot(self.prealign)

        if self.dim == 2:
            self.interp_method = vf.interpolate_scalar_2d
        else:
            self.interp_method = vf.interpolate_scalar_3d

        if self.sampling_proportion is None:
            self.warped = aff_warp(self.static, self.static_grid2space, self.moving,
                                   self.moving_grid2space, T).astype(np.float64)
            self.samples = None
            self.ns = 0
        else:
            self.warped = None
            k = 100/self.sampling_proportion
            shape = np.array(static.shape, dtype=np.int32)
            samples = np.array(sample_domain_regular(k, shape, static_grid2space))
            ns = samples.shape[0]
            # Add a column of ones (homogeneous coordinates)
            samples = np.hstack((samples, np.ones(ns)[:,None]))
            # Sample the static image
            points_on_static = (self.static_space2grid.dot(samples.T).T)[...,:self.dim]
            static_vals, inside = self.interp_method(self.static.astype(np.float32),
                                                     points_on_static)
            static_vals = np.array(static_vals, dtype=np.float64)
            # Sample the moving image
            sp_to_moving = self.moving_space2grid.dot(T)
            points_on_moving = (sp_to_moving.dot(samples.T).T)[...,:self.dim]
            moving_vals, inside = self.interp_method(self.moving.astype(np.float32),
                                                     points_on_moving)
            # Store relevant information
            self.samples = samples
            self.ns = ns
            self.static_vals = static_vals
            self.moving_vals = moving_vals

        MattesBase.setup(self, self.static, self.moving)

    def _update_dense(self, xopt, update_gradient = True):
        r""" Updates marginal and joint distributions and the joint gradient

        The distributions are updated according to the static and warped
        images. The warped image is precisely the moving image after
        transforming it by the transform defined by the xopt parameters.

        The gradient of the joint PDF is computed only if update_gradient
        is True.

        Parameters
        ----------
        xopt : array, shape (n,)
            the parameter vector of the transform currently used by the metric
            (the transform name is provided when self.setup is called), n is
            the number of parameters of the transform
        update_gradient : Boolean
            if true, the gradient of the joint PDF will also be computed,
            otherwise, only the marginal and joint PDFs will be computed
        """
        # Get the matrix associated to the xopt parameter vector
        T = self.transform.param_to_matrix(xopt)
        if self.prealign is not None:
            T = T.dot(self.prealign)

        # Warp the moving image
        self.warped = aff_warp(self.static, self.static_grid2space, self.moving,
                               self.moving_grid2space, T).astype(np.float64)

        # Update the joint and marginal intensity distributions
        self.update_pdfs_dense(self.static, self.warped, None, None)

        # Compute the gradient of the joint PDF w.r.t. parameters
        if update_gradient:
            if self.static_grid2space is None:
                grid_to_space = T
            else:
                grid_to_space = T.dot(self.static_grid2space)

            # Compute the gradient of the moving image at the current transform
            grid_to_space = T.dot(self.static_grid2space)
            self.grad_w, inside = vf.gradient(self.moving.astype(floating), self.moving_space2grid,
                                      self.moving_spacing, self.static.shape, grid_to_space)

            # Update the gradient of the metric
            self.update_gradient_dense(xopt, self.transform, self.static,
                                       self.warped, grid_to_space, self.grad_w,
                                       None, None)

        # Evaluate the mutual information and its gradient
        # The results are in self.metric_val and self.metric_grad
        # ready to be returned from 'distance' and 'gradient'
        self.update_mi_metric(update_gradient)


    def _update_sparse(self, xopt, update_gradient = True):
        r""" Updates the marginal and joint distributions and the joint gradient

        The distributions are updated according to the samples taken from the
        static and moving images. The samples are points in physical space,
        so the static intensities are always the same, but the corresponding
        points in the moving image depend on the transform defined by xopt.

        The gradient of the joint PDF is computed only if update_gradient
        is True.

        Parameters
        ----------
        xopt : array, shape (n,)
            the parameter vector of the transform currently used by the metric
            (the transform name is provided when self.setup is called), n is
            the number of parameters of the transform
        update_gradient : Boolean
            if true, the gradient of the joint PDF will also be computed,
            otherwise, only the marginal and joint PDFs will be computed
        """
        # Get the matrix associated to the xopt parameter vector
        T = self.transform.param_to_matrix(xopt)
        if self.prealign is not None:
            T = T.dot(self.prealign)

        # Sample the moving image
        sp_to_moving = self.moving_space2grid.dot(T)
        points_on_moving = (sp_to_moving.dot(self.samples.T).T)[...,:self.dim]
        self.moving_vals, inside = self.interp_method(self.moving.astype(np.float32),
                                                      points_on_moving)
        self.moving_vals = np.array(self.moving_vals, dtype=np.float64)

        # Update the joint and marginal intensity distributions
        self.update_pdfs_sparse(self.static_vals, self.moving_vals)

        # Compute the gradient of the joint PDF w.r.t. parameters
        if update_gradient:
            # Compute the gradient of the moving image at the current transform
            mgrad, inside = vf.sparse_gradient(self.moving.astype(np.float32),
                                               sp_to_moving,
                                               self.moving_spacing,
                                               self.samples)
            self.update_gradient_sparse(xopt, self.transform, self.static_vals,
                                        self.moving_vals,
                                        self.samples[...,:self.dim],
                                        mgrad)

        # Evaluate the mutual information and its gradient
        # The results are in self.metric_val and self.metric_grad
        # ready to be returned from 'distance' and 'gradient'
        self.update_mi_metric(update_gradient)


    def distance(self, xopt):
        r""" Numeric value of the negative Mutual Information

        We need to change the sign so we can use standard minimization
        algorithms.

        Parameters
        ----------
        xopt : array, shape (n,)
            the parameter vector of the transform currently used by the metric
            (the transform name is provided when self.setup is called), n is
            the number of parameters of the transform
        """
        if self.samples is None:
            self._update_dense(xopt, False)
        else:
            self._update_sparse(xopt, False)
        return -1 * self.metric_val

    def gradient(self, xopt):
        r""" Numeric value of the metric's gradient at the given parameters

        Parameters
        ----------
        xopt : array, shape (n,)
            the parameter vector of the transform currently used by the metric
            (the transform name is provided when self.setup is called), n is
            the number of parameters of the transform
        """
        if self.samples is None:
            self._update_dense(xopt, True)
        else:
            self._update_dense(xopt, True)
        return self.metric_grad * (-1)



    def value_and_gradient(self, xopt):
        r""" Numeric value of the metric and its gradient at given parameters

        Parameters
        ----------
        xopt : array, shape (n,)
            the parameter vector of the transform currently used by the metric
            (the transform name is provided when self.setup is called), n is
            the number of parameters of the transform
        """
        if self.samples is None:
            self._update_dense(xopt, True)
        else:
            self._update_sparse(xopt, True)
        return -1 * self.metric_val, self.metric_grad * (-1)


class AffineRegistration(object):
    def __init__(self,
                 metric=None,
                 level_iters=None,
                 opt_tol=1e-5,
                 sigmas = None,
                 factors = None,
                 method = 'BFGS',
                 ss_sigma_factor=None,
                 options=None):
        r""" Initializes an instance of the AffineRegistration class

        Parameters
        ----------
        metric : object
            an instance of a metric
        level_iters : list
            the number of iterations at each level of the Gaussian pyramid.
            level_iters[0] corresponds to the finest level, level_iters[n-1] the
            coarsest, where n is the length of the list
        opt_tol : float
            tolerance parameter for the optimizer
        sigmas : list of floats
            custom smoothing parameter to build the scale space (one parameter
            for each scale)
        factors : list of floats
            custom scale factors to build the scale space (one factor for each
            scale)
        method : string
            optimization method to be used
        ss_sigma_factor : float
            parameter of the scale-space smoothing kernel. For example, the
            std. dev. of the kernel will be factor*(2^i) in the isotropic case
            where i = 0, 1, ..., n_scales is the scale
        options : None or dict,
            extra optimization options.
        """

        self.metric = metric

        if self.metric is None:
            self.metric = MattesMIMetric()

        if level_iters is None:
            level_iters = [10000, 10000, 2500]
        self.level_iters = level_iters
        self.levels = len(level_iters)
        if self.levels == 0:
            raise ValueError('The iterations list cannot be empty')

        self.opt_tol = opt_tol
        

        self.options = options
        self.method = method
        
        if ss_sigma_factor is not None:
            self.use_isotropic = False
            self.ss_sigma_factor = ss_sigma_factor
        else:
            self.use_isotropic = True
            if factors is None:
                factors = [4, 2, 1]
            if sigmas is None:
                sigmas = [3, 1, 0]
            self.factors = factors
            self.sigmas = sigmas


    def _init_optimizer(self, static, moving, transform, x0,
                        static_grid2space, moving_grid2space, prealign):
        r"""Initializes the registration optimizer

        Initializes the optimizer by computing the scale space of the input
        images

        Parameters
        ----------
        static: array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization.
        moving: array, shape (S, R, C) or (R, C)
            the image to be used as "moving" during optimization. It is
            necessary to pre-align the moving image to ensure its domain
            lies inside the domain of the deformation fields. This is assumed to
            be accomplished by "pre-aligning" the moving image towards the
            static using an affine transformation given by the 'prealign' matrix
        transform: instance of Transform
            the transformation with respect to whose parameters the gradient
            must be computed
        x0: array, shape (n,)
            parameters from which to start the optimization. If None, the
            optimization will start at the identity transform. n is the
            number of parameters of the specified transformation.
        static_grid2space: array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated to the static image
        moving_grid2space: array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated to the moving image
        prealign: string, or matrix, or None
            If string:
                'mass': align centers of gravity
                'origins': align physical coordinates of voxel (0,0,0)
                'centers': align physical coordinates of central voxels
            If matrix:
                array, shape (dim+1, dim+1)
            If None:
                Start from identity
        """
        self.dim = len(static.shape)
        self.transform = transform
        self.nparams = transform.get_number_of_parameters()

        if x0 is None:
            x0 = self.transform.get_identity_parameters()
        self.x0 = x0
        if prealign is None:
            self.prealign = np.eye(self.dim + 1)
        elif prealign == 'mass':
            self.prealign = aff_centers_of_mass(static, static_grid2space, moving,
                                                moving_grid2space)
        elif prealign == 'origins':
            self.prealign = aff_origins(static, static_grid2space, moving,
                                        moving_gris2space)
        elif prealign == 'centers':
            self.prealign = aff_geometric_centers(static, static_grid2space, moving,
                                                  moving_grid2space)
        else:
            self.prealign = prealign
        #Extract information from the affine matrices to create the scale space
        static_direction, static_spacing = \
            get_direction_and_spacings(static_grid2space, self.dim)
        moving_direction, moving_spacing = \
            get_direction_and_spacings(moving_grid2space, self.dim)

        static = ((static.astype(np.float64) - static.min())/
                 (static.max() - static.min()))
        moving = ((moving.astype(np.float64) - moving.min())/
                  (moving.max() - moving.min()))
        #Build the scale space of the input images

        if self.use_isotropic:
            self.moving_ss = IsotropicScaleSpace(moving, self.factors, self.sigmas,
                                    moving_grid2space, moving_spacing, False)

            self.static_ss = IsotropicScaleSpace(static, self.factors, self.sigmas,
                                        static_grid2space, static_spacing, False)
        else:
            self.moving_ss = ScaleSpace(moving, self.levels, moving_grid2space,
                                    moving_spacing, self.ss_sigma_factor,
                                    False)

            self.static_ss = ScaleSpace(static, self.levels, static_grid2space,
                                    static_spacing, self.ss_sigma_factor,
                                    False)


    def optimize(self, static, moving, transform, x0, static_grid2space=None,
                 moving_grid2space=None, prealign=None):
        r'''
        Parameters
        ----------
        static: array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization.
        moving: array, shape (S, R, C) or (R, C)
            the image to be used as "moving" during optimization. It is
            necessary to pre-align the moving image to ensure its domain
            lies inside the domain of the deformation fields. This is assumed to
            be accomplished by "pre-aligning" the moving image towards the
            static using an affine transformation given by the 'prealign' matrix
        transform: string
        x0: array, shape (n,)
            parameters from which to start the optimization. If None, the
            optimization will start at the identity transform. n is the
            number of parameters of the specified transformation.
        static_grid2space: array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated to the static image
        moving_grid2space: array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated to the moving image
        prealign: string, or matrix, or None
            If string:
                'mass': align centers of gravity
                'origins': align physical coordinates of voxel (0,0,0)
                'centers': align physical coordinates of central voxels
            If matrix:
                array, shape (dim+1, dim+1)
            If None:
                Start from identity
        '''
        self._init_optimizer(static, moving, transform, x0, static_grid2space,
                             moving_grid2space, prealign)
        del prealign # now we must refer to self.prealign

        # Multi-resolution iterations
        original_static_grid2space = self.static_ss.get_affine(0)
        original_moving_grid2space = self.moving_ss.get_affine(0)
        original_moving_spacing = self.moving_ss.get_spacing(0)

        for level in range(self.levels - 1, -1, -1):
            self.current_level = level
            max_iter = self.level_iters[level]
            print('Optimizing level %d [max iter: %d]'%(level, max_iter))

            # Resample the smooth static image to the shape of this level
            smooth_static = self.static_ss.get_image(level)
            current_static_shape = self.static_ss.get_domain_shape(level)
            current_static_grid2space = self.static_ss.get_affine(level)

            current_static = aff_warp(tuple(current_static_shape),
                                      current_static_grid2space, smooth_static,
                                      original_static_grid2space, None, False)

            # The moving image is full resolution
            current_moving_grid2space = original_moving_grid2space
            current_moving_spacing = self.moving_ss.get_spacing(level)

            current_moving = self.moving_ss.get_image(level)

            # Prepare the metric for iterations at this resolution
            self.metric.setup(transform, current_static, current_moving,
                              current_moving_spacing, current_static_grid2space,
                              current_moving_grid2space, self.prealign)

            #optimize this level
            if self.options is None:
                self.options = {'gtol':1e-4, 'maxiter': max_iter, 'disp':False}

            opt = Optimizer(self.metric.value_and_gradient, self.x0,
                            method=self.method, jac = True,
                            options=self.options)
            xopt = opt.xopt

            # Update prealign matrix with optimal parameters
            T = self.transform.param_to_matrix(xopt)
            self.prealign = T.dot(self.prealign)

            # Start next iteration at identity
            self.x0 = self.transform.get_identity_parameters()

            # Update the metric to the current solution
            self.metric._update_dense(xopt, False)
        return self.prealign


def aff_warp(static, static_grid2space, moving, moving_grid2space, transform, nn=False):
    r""" Warps the moving image towards the static using the given transform

    Parameters
    ----------
    static: array, shape(S, R, C)
        static image: it will provide the grid and grid-to-space transform for
        the warped image
    static_grid2space:
        grid-to-space transform associated to the static image
    moving: array, shape(S', R', C')
        moving image
    moving_grid2space:
        grid-to-space transform associated to the moving image

    Returns
    -------
    warped: array, shape (S, R, C)
    """
    if type(static) is tuple:
        dim = len(static)
        shape = np.array(static, dtype=np.int32)
    else:
        dim = len(static.shape)
        shape = np.array(static.shape, dtype=np.int32)
    if nn:
        input = np.array(moving,dtype=np.int32)
        if dim == 2:
            warp_method = vf.warp_2d_affine_nn
        elif dim == 3:
            warp_method = vf.warp_3d_affine_nn
    else:
        input = np.array(moving,dtype=floating)
        if dim == 2:
            warp_method = vf.warp_2d_affine
        elif dim == 3:
            warp_method = vf.warp_3d_affine

    m_space2grid = npl.inv(moving_grid2space)
    if transform is None:
        composition = m_space2grid.dot(static_grid2space)
    else:
        composition = m_space2grid.dot(transform.dot(static_grid2space))

    warped = warp_method(input, shape, composition)

    return np.array(warped)


def aff_centers_of_mass(static, static_grid2space, moving, moving_grid2space):
    r""" Transformation to align the center of mass of the input images

    Parameters
    ----------
    static: array, shape(S, R, C)
        static image
    static_grid2space: array, shape (4, 4)
        the voxel-to-space transformation of the static image
    moving: array, shape(S, R, C)
        moving image
    moving_grid2space: array, shape (4, 4)
        the voxel-to-space transformation of the moving image

    Returns
    -------
    transform : array, shape(4, 4)
        the affine transformation (translation only, in this case) aligning
        the center of mass of the moving image towards the one of the static
        image
    """
    dim = len(static.shape)
    c_static = ndimage.measurements.center_of_mass(np.array(static))
    c_static = static_grid2space.dot(c_static+(1,))
    c_moving = ndimage.measurements.center_of_mass(np.array(moving))
    c_moving = moving_grid2space.dot(c_moving+(1,))
    transform = np.eye(dim + 1)
    transform[:dim,dim] = (c_moving - c_static)[:dim]
    return transform


def aff_geometric_centers(static, static_grid2space, moving, moving_grid2space):
    r""" Transformation to align the geometric center of the input images

    With "geometric center" of a volume we mean the physical coordinates of
    its central voxel

    Parameters
    ----------
    static: array, shape(S, R, C)
        static image
    static_grid2space: array, shape (4, 4)
        the voxel-to-space transformation of the static image
    moving: array, shape(S, R, C)
        moving image
    moving_grid2space: array, shape (4, 4)
        the voxel-to-space transformation of the moving image

    Returns
    -------
    transform : array, shape(4, 4)
        the affine transformation (translation only, in this case) aligning
        the geometric center of the moving image towards the one of the static
        image
    """
    dim = len(static.shape)
    c_static = tuple((np.array(static.shape, dtype = np.float64))*0.5)
    c_static = static_grid2space.dot(c_static+(1,))
    c_moving = tuple((np.array(moving.shape, dtype = np.float64))*0.5)
    c_moving = moving_grid2space.dot(c_moving+(1,))
    transform = np.eye(dim + 1)
    transform[:dim,dim] = (c_moving - c_static)[:dim]
    return transform


def aff_origins(static, static_grid2space, moving, moving_grid2space):
    r""" Transformation to align the origins of the input images

    With "origin" of a volume we mean the physical coordinates of
    voxel (0,0,0)

    Parameters
    ----------
    static: array, shape(S, R, C)
        static image
    static_grid2space: array, shape (4, 4)
        the voxel-to-space transformation of the static image
    moving: array, shape(S, R, C)
        moving image
    moving_grid2space: array, shape (4, 4)
        the voxel-to-space transformation of the moving image

    Returns
    -------
    transform : array, shape(4, 4)
        the affine transformation (translation only, in this case) aligning
        the origin of the moving image towards the one of the static
        image
    """
    dim = len(static.shape)
    c_static = static_grid2space[:dim, dim]
    c_moving = moving_grid2space[:dim, dim]
    transform = np.eye(dim + 1)
    transform[:dim,dim] = (c_moving - c_static)[:dim]
    return transform
