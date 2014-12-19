#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython

cdef extern from "dpy_math.h" nogil:
    double cos(double)
    double sin(double)
    double log(double)

#This dictionary allows us to get the appropriate transform index from a string
transform_type = {'TRANSLATION':TRANSLATION,
                  'ROTATION':ROTATION,
                  'RIGID':RIGID,
                  'SCALING':SCALING,
                  'AFFINE':AFFINE}

#This dictionary allows us to get the number of parameters of the available transforms
number_of_parameters = {(TRANSLATION,2): 2,
                        (TRANSLATION,3): 3,
                        (ROTATION,2): 1,
                        (ROTATION,3): 3,
                        (RIGID,2): 3,
                        (RIGID,3): 6,
                        (SCALING,2): 1,
                        (SCALING,3): 1,
                        (AFFINE,2): 6,
                        (AFFINE,3): 12}

def eval_jacobian_function(int transform_type, int dim, double[:] theta, double[:] x, double[:,:] J):
    r""" Compute the Jacobian of a transformation with given parameters at x
    """
    with nogil:
        get_jacobian_function(transform_type, dim)(theta, x, J)


def param_to_matrix(int transform_type, int dim, double[:] theta, double[:,:] T):
    r""" Compute the matrix associated to the given transform and parameters
    """
    with nogil:
        get_param_to_matrix_function(transform_type, dim)(theta, T)


def get_identity_parameters(int transform_type, int dim, double[:] theta):
    r""" Gets the parameters corresponding to the identity transform
    """
    with nogil:
        if dim == 2:
            if transform_type == TRANSLATION:
                theta[:2] = 0
            elif transform_type == ROTATION:
                theta[0] = 0
            elif transform_type == RIGID:
                theta[:3] = 0
            elif transform_type == SCALING:
                theta[0] = 1
            elif transform_type == AFFINE:
                theta[0], theta[1], theta[2] = 1, 0, 0
                theta[3], theta[4], theta[5] = 0, 1, 0
        elif dim == 3:
            if transform_type == TRANSLATION:
                theta[:3] = 0
            elif transform_type == ROTATION:
                theta[:3] = 0
            elif transform_type == RIGID:
                theta[:6] = 0
            elif transform_type == SCALING:
                theta[0] = 1
            elif transform_type == AFFINE:
                theta[0], theta[1], theta[2], theta[3] = 1, 0, 0, 0
                theta[4], theta[5], theta[6], theta[7] = 0, 1, 0, 0
                theta[8], theta[9], theta[10], theta[11] = 0, 0, 1, 0


cdef jacobian_function get_jacobian_function(int transform_type, int dim) nogil:
    r""" Jacobian function corresponding to the given transform and dimension
    """
    if dim == 2:
        if transform_type == TRANSLATION:
            return _translation_jacobian_2d
        elif transform_type == ROTATION:
            return _rotation_jacobian_2d
        elif transform_type == RIGID:
            return _rigid_jacobian_2d
        elif transform_type == SCALING:
            return _scaling_jacobian_2d
        elif transform_type == AFFINE:
            return _affine_jacobian_2d
    elif dim == 3:
        if transform_type == TRANSLATION:
            return _translation_jacobian_3d
        elif transform_type == ROTATION:
            return _rotation_jacobian_3d
        elif transform_type == RIGID:
            return _rigid_jacobian_3d
        elif transform_type == SCALING:
            return _scaling_jacobian_3d
        elif transform_type == AFFINE:
            return _affine_jacobian_3d
    return NULL


cdef param_to_matrix_function get_param_to_matrix_function(int transform_type, int dim) nogil:
    r""" Param-to-Matrix function of a given transform and dimension
    """
    if dim == 2:
        if transform_type == TRANSLATION:
            return _translation_matrix_2d
        elif transform_type == ROTATION:
            return _rotation_matrix_2d
        elif transform_type == RIGID:
            return _rigid_matrix_2d
        elif transform_type == SCALING:
            return _scaling_matrix_2d
        elif transform_type == AFFINE:
            return _affine_matrix_2d
    elif dim == 3:
        if transform_type == TRANSLATION:
            return _translation_matrix_3d
        elif transform_type == ROTATION:
            return _rotation_matrix_3d
        elif transform_type == RIGID:
            return _rigid_matrix_3d
        elif transform_type == SCALING:
            return _scaling_matrix_3d
        elif transform_type == AFFINE:
            return _affine_matrix_3d
    return NULL


cdef void _translation_matrix_2d(double[:] theta, double[:,:] R) nogil:
    R[0,0], R[0,1], R[0, 2] = 1, 0, theta[0]
    R[1,0], R[1,1], R[1, 2] = 0, 1, theta[1]
    R[2,0], R[2,1], R[2, 2] = 0, 0, 1


cdef void _translation_matrix_3d(double[:] theta, double[:,:] R) nogil:
    R[0,0], R[0,1], R[0,2], R[0,3] = 1, 0, 0, theta[0]
    R[1,0], R[1,1], R[1,2], R[1,3] = 0, 1, 0, theta[1]
    R[2,0], R[2,1], R[2,2], R[2,3] = 0, 0, 1, theta[2]
    R[3,0], R[3,1], R[3,2], R[3,3] = 0, 0, 0, 1


cdef int _translation_jacobian_2d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r""" Jacobian matrix of the 2D translation transform
    The transformation is given by:

    T(x) = (T1(x), T2(x)) = (x0 + t0, x1 + t1)

    The derivative w.r.t. t1 and t2 is given by

    T'(x) = [[1, 0], # derivatives of [T1, T2] w.r.t. t0
             [0, 1]] # derivatives of [T1, T2] w.r.t. t1
    """
    J[0,0], J[0, 1] = 1.0, 0.0
    J[1,0], J[1, 1] = 0.0, 1.0
    # This Jacobian does not depend on x (it's constant): return 1
    return 1


cdef int _translation_jacobian_3d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r""" Jacobian matrix of the 3D translation transform
    The transformation is given by:

    T(x) = (T1(x), T2(x), T3(x)) = (x0 + t0, x1 + t1, x2 + t2)

    The derivative w.r.t. t1, t2 and t3 is given by

    T'(x) = [[1, 0, 0], # derivatives of [T1, T2, T3] w.r.t. t0
             [0, 1, 0], # derivatives of [T1, T2, T3] w.r.t. t1
             [0, 0, 1]] # derivatives of [T1, T2, T3] w.r.t. t2
    """
    J[0,0], J[0,1], J[0,2] = 1.0, 0.0, 0.0
    J[1,0], J[1,1], J[1,2] = 0.0, 1.0, 0.0
    J[2,0], J[2,1], J[2,2] = 0.0, 0.0, 1.0
    # This Jacobian does not depend on x (it's constant): return 1
    return 1


cdef void _rotation_matrix_2d(double[:] theta, double[:,:] R) nogil:
    cdef:
        double ct = cos(theta[0])
        double st = sin(theta[0])
    R[0,0], R[0,1], R[0,2] = ct, -st, 0
    R[1,0], R[1,1], R[1,2] = st, ct, 0
    R[2,0], R[2,1], R[2,2] = 0, 0, 1


cdef int _rotation_jacobian_2d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r''' Jacobian matrix of a 3D rotation transform with parameters theta, at x

    T1[t] = x cost - y sint
    T2[t] = x sint + y cost

    dT1/dt = -x sint - y cost
    dT2/dt = x cost - y sint
    '''
    cdef:
        double st = sin(theta[0])
        double ct = cos(theta[0])
        double px = x[0], py = x[1]

    J[0, 0] = -px * st - py * ct
    J[1, 0] = px * ct - py * st
    # This Jacobian depends on x (it's not constant): return 0
    return 0


cdef void _rotation_matrix_3d(double[:] theta, double[:,:] R) nogil:
    r""" Product of rotation matrices around canonical axes

    Product of rotation matrices of angles theta[0], theta[1],
    theta[2] around axes x, y, z applied in the following order: y, x, z.
    This order was chosen for consistency with ANTS.

    Parameters
    ----------
    theta : array, shape(3,)
        theta[0] : rotation angle around x axis
        theta[1] : rotation angle around y axis
        theta[2] : rotation angle around z axis
    R : array, shape(3, 3)
        array to write the rotation matrix
    """
    cdef:
        double sa = sin(theta[0])
        double ca = cos(theta[0])
        double sb = sin(theta[1])
        double cb = cos(theta[1])
        double sc = sin(theta[2])
        double cc = cos(theta[2])

    with nogil:
        R[0,0], R[0,1], R[0,2], R[0, 3] = cc*cb-sc*sa*sb, -sc*ca, cc*sb+sc*sa*cb, 0
        R[1,0], R[1,1], R[1,2], R[1, 3] = sc*cb+cc*sa*sb, cc*ca, sc*sb-cc*sa*cb, 0
        R[2,0], R[2,1], R[2,2], R[2, 3] = -ca*sb, sa, ca*cb, 0
        R[3,0], R[3,1], R[3,2], R[3, 3] = 0, 0, 0, 1


cdef int _rotation_jacobian_3d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r''' Jacobian matrix of a 3D rotation transform with parameters theta, at x
    '''
    cdef:
        double sa = sin(theta[0])
        double ca = cos(theta[0])
        double sb = sin(theta[1])
        double cb = cos(theta[1])
        double sc = sin(theta[2])
        double cc = cos(theta[2])
        double px = x[0], py = x[1], pz = x[2]

    J[0, 0] = ( -sc * ca * sb ) * px + ( sc * sa ) * py + ( sc * ca * cb ) * pz
    J[1, 0] = ( cc * ca * sb ) * px + ( -cc * sa ) * py + ( -cc * ca * cb ) * pz
    J[2, 0] = ( sa * sb ) * px + ( ca ) * py + ( -sa * cb ) * pz

    J[0, 1] = ( -cc * sb - sc * sa * cb ) * px + ( cc * cb - sc * sa * sb ) * pz
    J[1, 1] = ( -sc * sb + cc * sa * cb ) * px + ( sc * cb + cc * sa * sb ) * pz
    J[2, 1] = ( -ca * cb ) * px + ( -ca * sb ) * pz

    J[0, 2] = ( -sc * cb - cc * sa * sb ) * px + ( -cc * ca ) * py + \
              ( -sc * sb + cc * sa * cb ) * pz
    J[1, 2] = ( cc * cb - sc * sa * sb ) * px + ( -sc * ca ) * py + \
              ( cc * sb + sc * sa * cb ) * pz
    J[2, 2] = 0
    # This Jacobian depends on x (it's not constant): return 0
    return 0


cdef void _rigid_matrix_2d(double[:] theta, double[:,:] R) nogil:
    cdef:
        double ct = cos(theta[0])
        double st = sin(theta[0])
    R[0,0], R[0,1], R[0,2] = ct, -st, theta[1]
    R[1,0], R[1,1], R[1,2] = st, ct, theta[2]
    R[2,0], R[2,1], R[2,2] = 0, 0, 1


cdef int _rigid_jacobian_2d(double[:] theta, double[:] x, double[:,:] J) nogil:
    cdef:
        double st = sin(theta[0])
        double ct = cos(theta[0])
        double px = x[0], py = x[1]

    J[0, 0], J[0, 1], J[0, 2] = -px * st - py * ct, 1, 0
    J[1, 0], J[1, 1], J[1, 2] = px * ct - py * st, 0, 1
    # This Jacobian depends on x (it's not constant): return 0
    return 0


cdef void _rigid_matrix_3d(double[:] theta, double[:,:] R) nogil:
    cdef:
        double sa = sin(theta[0])
        double ca = cos(theta[0])
        double sb = sin(theta[1])
        double cb = cos(theta[1])
        double sc = sin(theta[2])
        double cc = cos(theta[2])

    R[0,0], R[0,1], R[0,2], R[0,3] = cc*cb-sc*sa*sb, -sc*ca, cc*sb+sc*sa*cb, theta[3]
    R[1,0], R[1,1], R[1,2], R[1,3] = sc*cb+cc*sa*sb, cc*ca, sc*sb-cc*sa*cb, theta[4]
    R[2,0], R[2,1], R[2,2], R[2,3] = -ca*sb, sa, ca*cb, theta[5]
    R[3,0], R[3,1], R[3,2], R[3,3] = 0, 0, 0, 1


cdef int _rigid_jacobian_3d(double[:] theta, double[:] x, double[:,:] J) nogil:
    cdef:
        double sa = sin(theta[0])
        double ca = cos(theta[0])
        double sb = sin(theta[1])
        double cb = cos(theta[1])
        double sc = sin(theta[2])
        double cc = cos(theta[2])
        double px = x[0], py = x[1], pz = x[2]

    J[0, 0] = ( -sc * ca * sb ) * px + ( sc * sa ) * py + ( sc * ca * cb ) * pz
    J[1, 0] = ( cc * ca * sb ) * px + ( -cc * sa ) * py + ( -cc * ca * cb ) * pz
    J[2, 0] = ( sa * sb ) * px + ( ca ) * py + ( -sa * cb ) * pz

    J[0, 1] = ( -cc * sb - sc * sa * cb ) * px + ( cc * cb - sc * sa * sb ) * pz
    J[1, 1] = ( -sc * sb + cc * sa * cb ) * px + ( sc * cb + cc * sa * sb ) * pz
    J[2, 1] = ( -ca * cb ) * px + ( -ca * sb ) * pz

    J[0, 2] = ( -sc * cb - cc * sa * sb ) * px + ( -cc * ca ) * py + \
              ( -sc * sb + cc * sa * cb ) * pz
    J[1, 2] = ( cc * cb - sc * sa * sb ) * px + ( -sc * ca ) * py + \
              ( cc * sb + sc * sa * cb ) * pz
    J[2, 2] = 0

    J[0,3:6] = 0
    J[1,3:6] = 0
    J[2,3:6] = 0
    J[0,3], J[1,4], J[2,5] = 1, 1, 1
    # This Jacobian depends on x (it's not constant): return 0
    return 0


cdef void _scaling_matrix_2d(double[:] theta, double[:,:] R) nogil:
    R[0,0], R[0,1], R[0, 2] = theta[0], 0, 0
    R[1,0], R[1,1], R[1, 2] = 0, theta[0], 0
    R[2,0], R[2,1], R[2, 2] = 0, 0, 1


cdef void _scaling_matrix_3d(double[:] theta, double[:,:] R) nogil:
    R[0,0], R[0,1], R[0,2], R[0,3] = theta[0], 0, 0, 0
    R[1,0], R[1,1], R[1,2], R[1,3] = 0, theta[0], 0, 0
    R[2,0], R[2,1], R[2,2], R[2,3] = 0, 0, theta[0], 0
    R[3,0], R[3,1], R[3,2], R[3,3] = 0, 0, 0, 1


cdef int _scaling_jacobian_2d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r""" Jacobian matrix of the isotropic 2D scale transform
    The transformation is given by:

    T(x) = (s*x0, s*x1)

    The derivative w.r.t. s is T'(x) = [x0, x1]
    """
    J[0,0], J[1,0] = x[0], x[1]
    # This Jacobian depends on x (it's not constant): return 0
    return 0

cdef int _scaling_jacobian_3d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r""" Jacobian matrix of the isotropic 3D scale transform
    The transformation is given by:

    T(x) = (s*x0, s*x1, s*x2)

    The derivative w.r.t. s is T'(x) = [x0, x1, x2]
    """
    J[0,0], J[1,0], J[2,0]= x[0], x[1], x[3]
    # This Jacobian depends on x (it's not constant): return 0
    return 0


cdef void _affine_matrix_2d(double[:] theta, double[:,:] R) nogil:
    R[0,0], R[0,1], R[0, 2] = theta[0], theta[1], theta[2]
    R[1,0], R[1,1], R[1, 2] = theta[3], theta[4], theta[5]
    R[2,0], R[2,1], R[2, 2] = 0, 0, 1


cdef void _affine_matrix_3d(double[:] theta, double[:,:] R) nogil:
    R[0,0], R[0,1], R[0,2], R[0,3] = theta[0], theta[1], theta[2], theta[3]
    R[1,0], R[1,1], R[1,2], R[1,3] = theta[4], theta[5], theta[6], theta[7]
    R[2,0], R[2,1], R[2,2], R[2,3] = theta[8], theta[9], theta[10], theta[11]
    R[3,0], R[3,1], R[3,2], R[3,3] = 0, 0, 0, 1


cdef int _affine_jacobian_2d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r""" Jacobian matrix of the 2D affine transform
    The transformation is given by:

    T(x) = |a0, a1, a2 |   |x0|   | T1(x) |   |a0*x0 + a1*x1 + a2|
           |a3, a4, a5 | * |x1| = | T2(x) | = |a3*x0 + a4*x1 + a5|
                           | 1|

    The derivatives w.r.t. each parameter are given by

    T'(x) = [[x0,  0], #derivatives of [T1, T2] w.r.t a0
             [x1,  0], #derivatives of [T1, T2] w.r.t a1
             [ 1,  0], #derivatives of [T1, T2] w.r.t a2
             [ 0, x0], #derivatives of [T1, T2] w.r.t a3
             [ 0, x1], #derivatives of [T1, T2] w.r.t a4
             [ 0,  1]] #derivatives of [T1, T2, T3] w.r.t a5

    The Jacobian matrix is the transpose of the above matrix.
    """
    J[0,:] = 0
    J[1,:] = 0

    J[0, :2] = x[:]
    J[0, 2] = 1
    J[1, 3:5] = x[:]
    J[1, 5] = 1
    # This Jacobian depends on x (it's not constant): return 0
    return 0


cdef int _affine_jacobian_3d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r""" Jacobian matrix of the 3D affine transform
    The transformation is given by:

    T(x) = |a0, a1, a2,  a3 |   |x0|   | T1(x) |   |a0*x0 + a1*x1 + a2*x2 + a3|
           |a4, a5, a6,  a7 | * |x1| = | T2(x) | = |a4*x0 + a5*x1 + a6*x2 + a7|
           |a8, a9, a10, a11|   |x2|   | T3(x) |   |a8*x0 + a9*x1 + a10*x2+a11|
                                | 1|

    The derivatives w.r.t. each parameter are given by

    T'(x) = [[x0,  0,  0], #derivatives of [T1, T2, T3] w.r.t a0
             [x1,  0,  0], #derivatives of [T1, T2, T3] w.r.t a1
             [x2,  0,  0], #derivatives of [T1, T2, T3] w.r.t a2
             [ 1,  0,  0], #derivatives of [T1, T2, T3] w.r.t a3
             [ 0, x0,  0], #derivatives of [T1, T2, T3] w.r.t a4
             [ 0, x1,  0], #derivatives of [T1, T2, T3] w.r.t a5
             [ 0, x2,  0], #derivatives of [T1, T2, T3] w.r.t a6
             [ 0,  1,  0], #derivatives of [T1, T2, T3] w.r.t a7
             [ 0,  0, x0], #derivatives of [T1, T2, T3] w.r.t a8
             [ 0,  0, x1], #derivatives of [T1, T2, T3] w.r.t a9
             [ 0,  0, x2], #derivatives of [T1, T2, T3] w.r.t a10
             [ 0,  0,  1]] #derivatives of [T1, T2, T3] w.r.t a11

    The Jacobian matrix is the transpose of the above matrix.
    """
    cdef:
        cnp.npy_intp j

    for j in range(3):
        J[j,:] = 0
    J[0, :3] = x[:]
    J[0, 3] = 1
    J[1, 4:7] = x[:]
    J[1, 7] = 1
    J[2, 8:11] = x[:]
    J[2, 11] = 1
    # This Jacobian depends on x (it's not constant): return 0
    return 0

