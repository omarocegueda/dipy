import numpy as np
import scipy as sp
import scipy.ndimage as ndimage
from dipy.align import floating
import dipy.align.vector_fields as vf
from dipy.align.mattes import MattesBase
from dipy.core.optimize import Optimizer
from dipy.align.transforms import (transform_type,
                                   number_of_parameters,
                                   param_to_matrix)

class MattesMIMetric(MattesBase):
    def __init__(self, nbins=32, padding=2):
        super(MattesMIMetric, self).__init__(nbins, padding)

    def setup(self, transform, static, moving, static_aff=None, moving_aff=None, smask=None, mmask=None, prealign=None):
        MattesBase.setup(self, static, moving, smask, mmask)
        self.dim = len(static.shape)
        self.transform = transform_type[transform]
        self.static = static
        self.moving = moving
        self.static_aff = static_aff
        self.moving_aff = moving_aff
        self.smask = smask
        self.mmask = mmask
        self.prealign = prealign

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
        self.wmask = (self.warped>0).astype(np.int32)

        # Compute the gradient of the moving image at the current transform (i.e. warped)
        self.grad_w = np.empty(shape=(self.warped.shape)+(self.dim,))
        for i, grad in enumerate(sp.gradient(self.moving)):
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
        return self.metric_grad


class AffineRegistration(object):
    def __init__(self, metric=None, method='CG',
                 bounds=None, verbose=True, options=None,
                 evolution=False):

        self.metric = metric

        if self.metric is None:
            self.metric = MattesMIMetric()

        self.verbose = verbose
        self.method = method
        if self.method not in ['CG']:
            raise ValueError('Only Conjugate Gradient can be used')
        self.bounds = bounds
        self.options = options
        self.evolution = evolution

    def optimize(self, static, moving, transform, x0, static_affine=None, moving_affine=None,
                 smask=None, mmask=None, prealign=None):
        r'''
        Parameters
        ----------
        transform:

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
        self.dim = len(static.shape)
        self.transform_type = transform_type[transform]
        self.nparams = number_of_parameters[(self.transform_type, self.dim)]

        # If x0 was not provided, assume that a zero parameter vector maps to identity
        if x0 is None:
            x0 = np.zeros(self.nparams)
        if prealign is None:
            self.prealign = np.eye(self.dim + 1)
        elif prealign == 'mass':
            self.prealign = aff_centers_of_mass(static, static_affine, moving, moving_affine)
        elif prealign == 'origins':
            self.prealign = aff_origins(static, static_affine, moving, moving_affine)
        elif prealign == 'centers':
            self.prealign = aff_geometric_centers(static, static_affine, moving, moving_affine)

        self.metric.setup(transform, static, moving, static_affine, moving_affine, smask, mmask, prealign)

        if self.options is None:
            self.options = {'xtol': 1e-6, 'ftol': 1e-6, 'maxiter': 1e6}

        print('Starting optimization...')
        opt = Optimizer(self.metric.distance, x0, method=self.method, jac = self.metric.gradient,
                        options=self.options, evolution=self.evolution)
        print('Finished optimization...')
        if self.verbose:
            opt.print_summary()

        self.xopt = opt.xopt
        return self.xopt


def aff_warp(static, static_affine, moving, moving_affine, transform):
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
    dim = len(static.shape)
    if dim == 2:
        warp_method = vf.warp_2d_affine
    elif dim == 3:
        warp_method = vf.warp_3d_affine
    shape = np.array(static.shape, dtype=np.int32)
    m_aff_inv = np.linalg.inv(moving_affine)
    composition = m_aff_inv.dot(transform.dot(static_affine))
    warped = warp_method(np.array(moving,dtype=floating),
                         shape, composition)
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