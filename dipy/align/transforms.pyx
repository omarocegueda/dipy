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
                  'SCALING':SCALING,
                  'AFFINE':AFFINE}

def eval_jacobian_function(int transform_type, int dim, double[:] theta, double[:] x, double[:,:] J):
    r""" Compute the Jacobian of a transformation with given parameters at x

    """
    with nogil:
        get_jacobian_function(transform_type, dim)(theta, x, J)


cdef jacobian_function get_jacobian_function(int transform_type, int dim) nogil:
    r""" Jacobian function corresponding to the given transform and dimension
    """
    if dim == 2:
        if transform_type == TRANSLATION:
            return _translation_jacobian_2d
        elif transform_type == ROTATION:
            return _rotation_jacobian_2d
        elif transform_type == SCALING:
            return _scaling_jacobian_2d
        elif transform_type == AFFINE:
            return _affine_jacobian_2d
    elif dim == 3:
        if transform_type == TRANSLATION:
            return _translation_jacobian_3d
        elif transform_type == ROTATION:
            return _rotation_jacobian_3d
        elif transform_type == SCALING:
            return _scaling_jacobian_3d
        elif transform_type == AFFINE:
            return _affine_jacobian_3d
    return NULL


cdef param_to_matrix_function get_param_to_matrix_function(int transform_type, int dim):
    r""" Param-to-Matrix function of a given transform and dimension
    """
    if dim == 2:
        if transform_type == TRANSLATION:
            return _translation_matrix_2d
        elif transform_type == ROTATION:
            return _rotation_matrix_2d
        elif transform_type == SCALING:
            return _scaling_matrix_2d
        elif transform_type == AFFINE:
            return _affine_matrix_2d
    elif dim == 3:
        if transform_type == TRANSLATION:
            return _translation_matrix_3d
        elif transform_type == ROTATION:
            return _rotation_matrix_3d
        elif transform_type == SCALING:
            return _scaling_matrix_3d
        elif transform_type == AFFINE:
            return _affine_matrix_3d
    return NULL


cdef void _rotation_matrix_2d(double[:] theta, double[:,:] R):
    cdef:
        double ct = cos(theta[0])
        double st = sin(theta[0])
    R[0,0], R[0,1] = ct, -st
    R[1,0], R[1,1] = st, ct


cdef void _rotation_matrix_3d(double[:] theta, double[:,:] R):
    cdef:
        double sa = sin(theta[0])
        double ca = cos(theta[0])
        double sb = sin(theta[1])
        double cb = cos(theta[1])
        double sc = sin(theta[2])
        double cc = cos(theta[2])
        double[:,:] rot_a = np.ndarray(shape=(3,3), dtype = np.float64)
        double[:,:] rot_b = np.ndarray(shape=(3,3), dtype = np.float64)
        double[:,:] rot_c = np.ndarray(shape=(3,3), dtype = np.float64)
        double[:,:] temp = np.ndarray(shape=(3,3), dtype = np.float64)

    with nogil:
        rot_a[0,0], rot_a[0, 1], rot_a[0, 2] = 1.0, 0.0, 0.0
        rot_a[1,0], rot_a[1, 1], rot_a[1, 2] = 0.0,  ca, -sa
        rot_a[2,0], rot_a[2, 1], rot_a[2, 2] = 0.0,  sa,  ca

        rot_b[0,0], rot_b[0, 1], rot_b[0, 2] =  cb, 0.0,  sb
        rot_b[1,0], rot_b[1, 1], rot_b[1, 2] = 0.0, 1.0, 0.0
        rot_b[2,0], rot_b[2, 1], rot_b[2, 2] = -sb, 0.0,  cb

        rot_c[0,0], rot_c[0, 1], rot_c[0, 2] =  cc, -sc, 0.0
        rot_c[1,0], rot_c[1, 1], rot_c[1, 2] =  sc,  cc, 0.0
        rot_c[2,0], rot_c[2, 1], rot_c[2, 2] = 0.0, 0.0, 1.0

        # Compute rot_c * rot_a * rot_b
        _mult_mat_3d(rot_a, rot_b, temp)
        _mult_mat_3d(rot_c, temp, R)


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


cdef void _scaling_matrix_2d(double[:] theta, double[:,:] R):
    pass

cdef void _scaling_matrix_3d(double[:] theta, double[:,:] R):
    pass


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


cdef void _translation_matrix_2d(double[:] theta, double[:,:] R):
    pass


cdef void _translation_matrix_3d(double[:] theta, double[:,:] R):
    pass


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


cdef void _affine_matrix_2d(double[:] theta, double[:,:] R):
    pass


cdef void _affine_matrix_3d(double[:] theta, double[:,:] R):
    pass


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

    
############### Not sure if this is the appropriate place for this helper#####
cdef inline void _mult_mat_3d(double[:,:] A, double[:,:] B, double[:,:] C) nogil:
    r''' Multiplies two 3x3 matrices A, B and writes the product in C
    '''
    cdef:
        cnp.npy_intp i, j

    for i in range(3):
        for j in range(3):
            C[i, j] = A[i, 0]*B[0, j] + A[i, 1]*B[1, j] + A[i, 2]*B[2, j]
            