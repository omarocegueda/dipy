import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
from dipy.align import floating
import dipy.align.vector_fields as vf
import dipy.align.mattes as mattes
from dipy.align.mattes import MattesBase
from dipy.core.optimize import Optimizer
import matplotlib.pyplot as plt
from dipy.align.transforms import (transform_type,
                                   number_of_parameters,
                                   param_to_matrix,
                                   get_identity_parameters)
from dipy.align.imwarp import (get_direction_and_spacings,
                               ScaleSpace)

def estimate_param_scales(transform_type, dim, samples):
    nsamples = samples.shape[0]
    epsilon = 0.01

    n = number_of_parameters[transform_type, dim]
    theta = np.empty(n)
    T = np.ndarray((dim + 1, dim + 1))

    scales = np.zeros(n)
    for i in range(n):
        get_identity_parameters(transform_type, dim, theta)
        theta[i] += epsilon
        param_to_matrix(transform_type, dim, theta, T)
        transformed = samples.dot(T.transpose())
        max_shift_sq = np.sum((transformed - samples)**2, 1).max()
        scales[i] = max_shift_sq

    scales[scales==0] = scales[scales>0].min()
    scales /= (epsilon*epsilon)
    return scales


class MattesMIMetric(MattesBase):
    def __init__(self, nbins=32, padding=2):
        super(MattesMIMetric, self).__init__(nbins, padding)

    def setup(self, transform, static, moving, static_aff=None, moving_aff=None, smask=None, mmask=None, prealign=None):
        MattesBase.setup(self, static, moving, smask, mmask)
        self.dim = len(static.shape)
        self.transform = transform_type[transform]
        self.static = np.array(static).astype(np.float64)
        self.moving = np.array(moving).astype(np.float64)
        self.static_aff = static_aff
        self.moving_aff = moving_aff
        self.smask = smask
        self.mmask = mmask
        self.prealign = prealign
        self.param_scales = None

    def _update_dense(self, xopt):
        # Get the matrix associated to the xopt parameter vector
        T = np.empty(shape=(self.dim + 1, self.dim + 1))
        param_to_matrix(self.transform, self.dim, xopt, T)
        if self.prealign is not None:
            T = T.dot(self.prealign)

        # Warp the moving image
        self.warped = aff_warp(self.static, self.static_aff, self.moving, self.moving_aff, T).astype(np.float64)

        # Get the warped mask.
        # Note: we should warp mmask with nearest neighbor interpolation instead
        self.wmask = aff_warp(self.static, self.static_aff, self.mmask, self.moving_aff, T, True).astype(np.int32)

        # Compute the gradient of the moving image at the current transform (i.e. warped)
        self.grad_w = np.empty(shape=(self.warped.shape)+(self.dim,))
        for i, grad in enumerate(sp.gradient(self.warped)):
            self.grad_w[..., i] = grad

        # Update the joint and marginal intensity distributions
        self.update_pdfs_dense(self.static, self.warped, self.smask, self.wmask)
        # Compute the gradient of the joint PDF w.r.t. parameters
        self.update_gradient_dense(xopt, self.transform, self.static, self.warped,
                                self.static_aff, self.grad_w, self.smask, self.wmask)
        # Evaluate the mutual information and its gradient
        # The results are in self.metric_val and self.metric_grad
        # ready to be returned from 'distance' and 'gradient'
        self.update_mi_metric(True)

    def distance(self, xopt):
        self._update_dense(xopt)
        return self.metric_val

    def gradient(self, xopt):
        self._update_dense(xopt)
        if self.param_scales is not None:
            self.metric_grad /= self.param_scales
        return self.metric_grad

    def value_and_gradient(self, xopt):
        self._update_dense(xopt)
        if self.param_scales is not None:
            self.metric_grad /= self.param_scales
        return self.metric_val, self.metric_grad


class AffineRegistration(object):
    def __init__(self,
                 metric=None,
                 level_iters=None,
                 opt_tol=1e-5,
                 ss_sigma_factor=1.0,
                 bounds=None,
                 verbose=True,
                 options=None,
                 evolution=False):

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
        self.ss_sigma_factor = ss_sigma_factor

        self.bounds = bounds
        self.verbose = verbose
        self.options = options
        self.evolution = evolution
        self.method = 'CG'


    def _init_optimizer(self, static, moving, transform, x0,
                        static_affine, moving_affine, prealign):
        r"""Initializes the registration optimizer

        Initializes the optimizer by computing the scale space of the input
        images

        Parameters
        ----------
        static: array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization.
        moving: array, shape (S, R, C) or (R, C)
            the image to be used as "moving" during optimization. It is necessary
            to pre-align the moving image to ensure its domain
            lies inside the domain of the deformation fields. This is assumed to
            be accomplished by "pre-aligning" the moving image towards the
            static using an affine transformation given by the 'prealign' matrix
        transform: string
            the name of the transformation to be used, must be one of
            {'TRANSLATION', 'ROTATION', 'SCALING', 'AFFINE'}
        x0: array, shape (n,)
            parameters from which to start the optimization. If None, the
            optimization will start at the identity transform. n is the
            number of parameters of the specified transformation.
        static_affine: array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated to the static image
        moving_affine: array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated to the moving image
        prealign: array, shape (dim+1, dim+1)
            the affine transformation (operating on the physical space)
            pre-aligning the moving image towards the static

        """
        self.dim = len(static.shape)
        self.transform_type = transform_type[transform]
        self.nparams = number_of_parameters[(self.transform_type, self.dim)]

        # If x0 was not provided, assume that a zero parameter vector maps to identity
        if x0 is None:
            x0 = np.empty(self.nparams, dtype=np.float64)
            get_identity_parameters(self.transform_type, self.dim, x0)
        self.x0 = x0
        if prealign is None:
            self.prealign = np.eye(self.dim + 1)
        elif prealign == 'mass':
            self.prealign = aff_centers_of_mass(static, static_affine, moving, moving_affine)
        elif prealign == 'origins':
            self.prealign = aff_origins(static, static_affine, moving, moving_affine)
        elif prealign == 'centers':
            self.prealign = aff_geometric_centers(static, static_affine, moving, moving_affine)
        #Extract information from the affine matrices to create the scale space
        static_direction, static_spacing = \
            get_direction_and_spacings(static_affine, self.dim)
        moving_direction, moving_spacing = \
            get_direction_and_spacings(moving_affine, self.dim)

        static = (static - static.min())/(static.max() - static.min())
        moving = (moving - moving.min())/(moving.max() - moving.min())
        #Build the scale space of the input images
        self.moving_ss = ScaleSpace(moving, self.levels, moving_affine,
                                    moving_spacing, self.ss_sigma_factor,
                                    False)

        self.static_ss = ScaleSpace(static, self.levels, static_affine,
                                    static_spacing, self.ss_sigma_factor,
                                    False)

        # Sample the static domain
        self.nsamples = 1000
        self.samples = np.empty((self.nsamples, self.dim + 1))
        self.samples[:,self.dim] = 1
        #mask = (static>0).astype(np.int32)
        mask = np.ones_like(static, dtype=np.int32)
        if self.dim == 2:
            mattes.sample_domain_2d(np.array(static.shape, dtype=np.int32), self.nsamples, self.samples, mask)
        else:
            mattes.sample_domain_3d(np.array(static.shape, dtype=np.int32), self.nsamples, self.samples, mask)
        if static_affine is not None:
            self.samples = self.samples.dot(static_affine.transpose()) # now samples are in physical space


    def optimize(self, static, moving, transform, x0, static_affine=None, moving_affine=None,
                 smask=None, mmask=None, prealign=None):
        r'''
        Parameters
        ----------
        transform: string
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
        self._init_optimizer(static, moving, transform, x0, static_affine, moving_affine, prealign)
        del prealign # now we must refer to self.prealign

        # Multi-resolution iterations
        original_static_affine = self.static_ss.get_affine(0)
        original_moving_affine = self.moving_ss.get_affine(0)

        if smask is None:
            smask = np.ones_like(self.static_ss.get_image(0), dtype=np.int32)
        if mmask is None:
            mmask = np.ones_like(self.moving_ss.get_image(0), dtype=np.int32)

        original_smask = smask
        original_mmask = mmask

        for level in range(self.levels - 1, -1, -1):
            self.current_level = level
            print('Optimizing level %d [%d iterations maximum]'%(self.current_level, self.level_iters[level]))

            # Resample the smooth static image to the shape of this level
            smooth_static = self.static_ss.get_image(level)
            current_static_shape = self.static_ss.get_domain_shape(level)
            current_static_aff = self.static_ss.get_affine(level)
            current_static = aff_warp(tuple(current_static_shape), current_static_aff, smooth_static, original_static_affine, None, False)
            current_smask = aff_warp(tuple(current_static_shape), current_static_aff, original_smask, original_static_affine, None, True)

            # The moving image is full resolution
            current_moving_aff = original_moving_affine
            current_moving = self.moving_ss.get_image(level)
            current_mmask = original_mmask

            # Prepare the metric for iterations at this resolution
            self.metric.setup(transform, current_static, current_moving, current_static_aff, current_moving_aff, current_smask, current_mmask, self.prealign)
            scales = estimate_param_scales(self.transform_type, self.dim, self.samples)
            self.metric.param_scales = scales

            #optimize this level
            if self.options is None:
                self.options = {'maxiter': self.level_iters[self.current_level]}

            opt = Optimizer(self.metric.value_and_gradient, self.x0, method=self.method, jac = True,
                            options=self.options, evolution=self.evolution)

            # Update prealign matrix with optimal parameters
            T = np.empty(shape=(self.dim + 1, self.dim + 1))
            param_to_matrix(self.metric.transform, self.dim, opt.xopt, T)
            self.prealign = T.dot(self.prealign)

            # Start next iteration at identity
            get_identity_parameters(self.transform_type, self.dim, self.x0)

            print("Metric value: %f"%(self.metric.metric_val,))

        return self.prealign


def aff_warp(static, static_affine, moving, moving_affine, transform, nn=False):
    r""" Warps the moving image towards the static using the given transform

    Parameters
    ----------
    static: array, shape(S, R, C)
        static image: it will provide the grid and grid-to-space transform for
        the warped image
    static_affine:
        grid-to-space transform associated to the static image
    moving: array, shape(S', R', C')
        moving image
    moving_affine:
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

    m_aff_inv = np.linalg.inv(moving_affine)
    if transform is None:
        composition = m_aff_inv.dot(static_affine)
    else:
        composition = m_aff_inv.dot(transform.dot(static_affine))

    warped = warp_method(input, shape, composition)

    return np.array(warped)


def aff_centers_of_mass(static, static_affine, moving, moving_affine):
    r""" Transformation to align the center of mass of the input images

    Parameters
    ----------
    static: array, shape(S, R, C)
        static image
    static_affine: array, shape (4, 4)
        the voxel-to-space transformation of the static image
    moving: array, shape(S, R, C)
        moving image
    moving_affine: array, shape (4, 4)
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
    c_static = static_affine.dot(c_static+(1,))
    c_moving = ndimage.measurements.center_of_mass(np.array(moving))
    c_moving = moving_affine.dot(c_moving+(1,))
    transform = np.eye(dim + 1)
    transform[:dim,dim] = (c_moving - c_static)[:dim]
    return transform


def aff_geometric_centers(static, static_affine, moving, moving_affine):
    r""" Transformation to align the geometric center of the input images

    With "geometric center" of a volume we mean the physical coordinates of
    its central voxel

    Parameters
    ----------
    static: array, shape(S, R, C)
        static image
    static_affine: array, shape (4, 4)
        the voxel-to-space transformation of the static image
    moving: array, shape(S, R, C)
        moving image
    moving_affine: array, shape (4, 4)
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
    c_static = static_affine.dot(c_static+(1,))
    c_moving = tuple((np.array(moving.shape, dtype = np.float64))*0.5)
    c_moving = moving_affine.dot(c_moving+(1,))
    transform = np.eye(dim + 1)
    transform[:dim,dim] = (c_moving - c_static)[:dim]
    return transform


def aff_origins(static, static_affine, moving, moving_affine):
    r""" Transformation to align the origins of the input images

    With "origin" of a volume we mean the physical coordinates of
    voxel (0,0,0)

    Parameters
    ----------
    static: array, shape(S, R, C)
        static image
    static_affine: array, shape (4, 4)
        the voxel-to-space transformation of the static image
    moving: array, shape(S, R, C)
        moving image
    moving_affine: array, shape (4, 4)
        the voxel-to-space transformation of the moving image

    Returns
    -------
    transform : array, shape(4, 4)
        the affine transformation (translation only, in this case) aligning
        the origin of the moving image towards the one of the static
        image
    """
    dim = len(static.shape)
    c_static = static_affine[:dim, dim]
    c_moving = moving_affine[:dim, dim]
    transform = np.eye(dim + 1)
    transform[:dim,dim] = (c_moving - c_static)[:dim]
    return transform