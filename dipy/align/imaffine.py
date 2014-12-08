import numpy as np
import scipy.ndimage as ndimage
from dipy.align import floating
import dipy.align.vector_fields as vf

def aff_warp_2d(static, static_affine, moving, moving_affine, transform):
    r""" Warps the moving image towards the static using the given transform

    Parameters
    ----------
    static: array, shape(R, C)
        static image: it will provide the grid and grid-to-space transform for
        the warped image
    static_affine:
        grid-to-space transform associated to the static image
    moving: array, shape(R', C')
        moving image
    moving_affine:
        grid-to-space transform associated to the moving image

    Returns
    -------
    warped: array, shape (R, C)
    """
    shape = np.array(static.shape, dtype=np.int32)
    m_aff_inv = np.linalg.inv(moving_affine)
    composition = m_aff_inv.dot(transform.dot(static_affine))
    warped = vf.warp_2d_affine(np.array(moving,dtype=floating),
                               shape,
                               composition)
    return np.array(warped)

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


def mattes_mi_gradient(joint, joint_gradient, mmarginal):
    EPSILON = 2.2204460492503131e-016
    n = joint_gradient.shape[2]
    mi_gradient = np.zeros(n)
    for i in range(nrows):
        for j in range(ncols):
            if joint[i,j] < EPSILON or mmarginal[j] < EPSILON:
                continue
            for k in range(n):
                mi_gradient[k] -= joint_gradient[i, j, k] * np.log(joint[i,j] / mmarginal[j])
    return mi_gradient


def align_mattes_mi():
    import experiments.registration.dataset_info as info
    import nibabel as nib
    import dipy.align.mattes as mattes
    import scipy as sp
    i1_name = info.get_ibsr(1, 'strip')
    i2_name = info.get_ibsr(2, 'strip')
    i1_nib = nib.load(i1_name)
    i2_nib = nib.load(i2_name)
    i1_vol = i1_nib.get_data().squeeze()
    i2_vol = i2_nib.get_data().squeeze()
    i1_aff = i1_nib.get_affine()
    i2_aff = i2_nib.get_affine()

    #Acquire static and moving images
    dim = 2
    static = i1_vol[:, 64, :].transpose()[::-1, ::-1].astype(np.float64)
    static_aff = np.eye(1+dim)
    moving = i2_vol[:, 64, :].transpose()[::-1, ::-1].astype(np.float64)
    moving_aff = np.eye(1+dim)

    smask = (static>0).astype(np.int32)

    #Warp moving image to the common reference
    warped = aff_warp_2d(static, static_aff, moving, moving_aff, np.eye(1+dim)).astype(np.float64)
    wmask = (warped>0).astype(np.int32)

    #Gradient of the warped image
    grad_w = np.empty(shape=(warped.shape)+(dim,))
    for i, grad in enumerate(sp.gradient(moving)):
        grad_w[..., i] = grad



    #Initialize distribution estimation
    pdf = mattes.MattesPDF(32, static, warped)

    pdf.update_pdfs_dense(static, warped, smask, wmask)

    #Compute gradient
    theta = np.array([0.0, 0.0])
    pdf.update_gradient_dense(theta, 'translation', static, warped, static_aff, grad_w, smask, wmask)
    mattes_mi_gradient(pdf.joint, pdf.joint_grad, pdf.mmarginal)

    theta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    pdf.update_gradient_dense(theta, 'affine', static, warped, static_aff, grad_w, smask, wmask)

    theta = np.array([0.0])
    pdf.update_gradient_dense(theta, 'rotation', static, warped, static_aff, grad_w, smask, wmask)

    theta = np.array([0.0])
    pdf.update_gradient_dense(theta, 'scale', static, warped, static_aff, grad_w, smask, wmask)



    max_iter = 10
    for i in range(max_iter):
        pass
