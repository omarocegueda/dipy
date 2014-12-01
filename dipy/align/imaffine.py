import numpy as np
import scipy.ndimage as ndimage
from dipy.align import floating
import dipy.align.vector_fields as vf

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
    shape = np.array(static.shape, dtype=np.int32)
    m_aff_inv = np.linalg.inv(moving_affine)
    composition = m_aff_inv.dot(transform.dot(static_affine))
    warped = vf.warp_3d_affine(np.array(moving,dtype=floating), 
                               shape,
                               composition)
    return np.array(warped)


def aff_centers_of_mass_3d(static, static_affine, moving, moving_affine):
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
    c_static = ndimage.measurements.center_of_mass(np.array(static))
    c_static = static_affine.dot(c_static+(1,))
    c_moving = ndimage.measurements.center_of_mass(np.array(moving))
    c_moving = moving_affine.dot(c_moving+(1,))
    transform = np.eye(4)
    transform[:3,3] = (c_moving - c_static)[:3]
    return transform


def aff_geometric_centers_3d(static, static_affine, moving, moving_affine):
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
    c_static = tuple((np.array(static.shape, dtype = np.float64))*0.5)
    c_static = static_affine.dot(c_static+(1,))
    c_moving = tuple((np.array(moving.shape, dtype = np.float64))*0.5)
    c_moving = moving_affine.dot(c_moving+(1,))
    transform = np.eye(4)
    transform[:3,3] = (c_moving - c_static)[:3]
    return transform
   

def aff_origins_3d(static, static_affine, moving, moving_affine):
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
    c_static = static_affine[:3, 3]
    c_moving = moving_affine[:3, 3]
    transform = np.eye(4)
    transform[:3,3] = (c_moving - c_static)[:3]
    return transform


