import numpy as np
import scipy.ndimage as ndimage
from dipy.align import floating
import dipy.align.vector_fields as vf
from dipy.align.mattes import MattesBase

class MattesMIMetric(MattesBase):
    def __init__():
        pass


class AffineRegistration(object):
    def __init__(self, metric=None, x0="rigid", method='L-BFGS-B',
                 bounds=None, verbose=False, options=None,
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
                 prealign=None):
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
        if dim == 2:
            nparam = {'translation':2, 'rotation':1, 'iso':1, 'aniso':2, 'affine':6}
        elif dim == 3:
            nparam = {'translation':3, 'rotation':3, 'iso':1, 'aniso':3, 'affine':12}
        else:
            raise ValueError('Unsuported image dimension: '+str(dim))

        self.nparam = nparam
        #Assume that zero parameter vector maps to identity
        if x0 is None:
            x0 = np.zeros(self.nparam)
        if prealign is None:
            prealign = np.eye(dim + 1)
        elif prealign == 'mass':
            prealign = aff_centers_of_mass(static, static_affine, moving, moving_affine)
        elif prealign == 'origins':
            prealign = aff_origins(static, static_affine, moving, moving_affine)
        elif prealign == 'centers':
            prealign = aff_geometric_centers(static, static_affine, moving, moving_affine)

        self.metric.setup(prealign)#Provide the pre-align matrix. The metric must store it
        distance_method = self.metric.distance
        gradient_method = self.metric.gradient

        opt = Optimizer(gradient_method, x0, method=self.method,
                        options=self.options, evolution=self.evolution)
        if self.verbose:
            opt.print_summary()


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