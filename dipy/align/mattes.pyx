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


cdef inline double _apply_affine_3d_x0(double x0, double x1, double x2,
                                       double h, double[:, :] aff) nogil:
    r"""Multiplies aff by (x0, x1, x2, h), returns the 1st element of product

    Returns the first component of the product of the homogeneous matrix aff by
    (x0, x1, x2, h)
    """
    return aff[0, 0] * x0 + aff[0, 1] * x1 + aff[0, 2] * x2 + h*aff[0, 3]


cdef inline double _apply_affine_3d_x1(double x0, double x1, double x2,
                                       double h, double[:, :] aff) nogil:
    r"""Multiplies aff by (x0, x1, x2, h), returns the 2nd element of product

    Returns the first component of the product of the homogeneous matrix aff by
    (x0, x1, x2, h)
    """
    return aff[1, 0] * x0 + aff[1, 1] * x1 + aff[1, 2] * x2 + h*aff[1, 3]


cdef inline double _apply_affine_3d_x2(double x0, double x1, double x2,
                                       double h, double[:, :] aff) nogil:
    r"""Multiplies aff by (x0, x1, x2, h), returns the 3d element of product

    Returns the first component of the product of the homogeneous matrix aff by
    (x0, x1, x2, h)
    """
    return aff[2, 0] * x0 + aff[2, 1] * x1 + aff[2, 2] * x2 + h*aff[2, 3]


cdef inline double _apply_affine_2d_x0(double x0, double x1, double h,
                                       double[:, :] aff) nogil:
    r"""Multiplies aff by (x0, x1, h), returns the 1st element of product
    Returns the first component of the product of the homogeneous matrix aff by
    (x0, x1, h)
    """
    return aff[0, 0] * x0 + aff[0, 1] * x1 + h*aff[0, 2]


cdef inline double _apply_affine_2d_x1(double x0, double x1, double h,
                                       double[:, :] aff) nogil:
    r"""Multiplies aff by (x0, x1, h), returns the 2nd element of product

    Returns the first component of the product of the homogeneous matrix aff by
    (x0, x1, h)
    """
    return aff[1, 0] * x0 + aff[1, 1] * x1 + h*aff[1, 2]


ctypedef int (*jacobian_function)(double[:], double[:], double[:,:]) nogil
r""" Type of a function that computes the Jacobian of a transform.
Jacobian functions receive a vector containing the current parameters
of the transformation, the coordinates of a point to compute the
Jacobian at, and the Jacobian matrix to write the result in. The
shape of the resulting Jacobian must be a dxn matrix, where d is the
dimension of the transform, and n is the number of parameters of the
transformation.

If the Jacobian is CONSTANT along its domain, the corresponding
jacobian_function must RETURN 1. Otherwise it must RETURN 0. This
information is used by the optimizer to avoid making unnecessary
function calls
"""


cdef inline void _mult_mat_3d(double[:,:] A, double[:,:] B, double[:,:] C) nogil:
    r''' Multiplies two 3x3 matrices A, B and writes the product in C
    '''
    cdef:
        cnp.npy_intp i, j

    for i in range(3):
        for j in range(3):
            C[i, j] = A[i, 0]*B[0, j] + A[i, 1]*B[1, j] + A[i, 2]*B[2, j]


cdef void _rotation_matrix_2d(double theta, double[:,:] R):
    cdef:
        double ct = cos(theta)
        double st = sin(theta)
    R[0,0], R[0,1] = ct, -st
    R[1,0], R[1,1] = st, ct


cdef void _rotation_matrix_3d(double a, double b, double c, double[:,:] R):
    cdef:
        double sa = sin(a)
        double ca = cos(a)
        double sb = sin(b)
        double cb = cos(b)
        double sc = sin(c)
        double cc = cos(c)
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


cdef int _scale_jacobian_2d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r""" Jacobian matrix of the isotropic 2D scale transform
    The transformation is given by:

    T(x) = (s*x0, s*x1)

    The derivative w.r.t. s is T'(x) = [x0, x1]
    """
    J[0,0], J[1,0] = x[0], x[1]
    # This Jacobian depends on x (it's not constant): return 0
    return 0

cdef int _scale_jacobian_3d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r""" Jacobian matrix of the isotropic 3D scale transform
    The transformation is given by:

    T(x) = (s*x0, s*x1, s*x2)

    The derivative w.r.t. s is T'(x) = [x0, x1, x2]
    """
    J[0,0], J[1,0], J[2,0]= x[0], x[1], x[3]
    # This Jacobian depends on x (it's not constant): return 0
    return 0


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


cdef inline double _bin_normalize(double x, double mval, double delta) nogil:
    r''' Normalizes intensity x to the range covered by the Parzen histogram
    We assume that mval was computed as:

    (1) mval = xmin / delta - padding

    where xmin is the minimum observed image intensity and delta is the
    bin size, computed as:

    (2) delta = (xmax - xmin)/(nbins - 2 * padding)

    If the minimum and maximum intensities were assigned to the first and last
    bins (with no padding), it could be possible that samples at the first and
    last bins contribute to "non-existing" bins beyond the boundary (because
    the support of the Parzen window may be larger than one bin). The padding
    bins are used to collect such contributions (i.e. the probability of
    observing a value beyond the minimum and maximum observed intensities may
    be assigned a positive value).

    The normalized intensity is (from eq(1) ):

    (3) nx = (x - xmin) / delta + padding = x/delta - mval

    This means that the observed intensity x must be between
    bins padding and nbins-1-padding, although it may affect bins 0 to nbins-1.
    '''
    return x / delta - mval


cdef inline cnp.npy_intp _bin_index(double normalized, int nbins, int padding) nogil:
    r''' Index of the bin that normalized intensity lies in
    The intensity is assumed to have been normalized to the range of intensities
    covered by the histogram: the bin index is the integer part of the argument,
    which must be within the interval [padding, nbins - 1 - padding].
    '''
    cdef:
        cnp.npy_intp bin

    bin = <cnp.npy_intp>(normalized)
    if bin < padding:
        return padding
    if bin > nbins - 1 - padding:
        return nbins - 1 - padding
    return bin


cdef inline double _cubic_spline(double x) nogil:
    r''' Cubic B-Spline
    See eq. (3) of [1].

    [1] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., & Eubank, W.
    PET-CT image registration in the chest using free-form deformations.
    IEEE Transactions on Medical Imaging, 22(1), 120–8, 2003.
    '''
    cdef:
        double absx = -x if x < 0.0 else x
        double sqrx = x * x

    if absx < 1.0:
        return ( 4.0 - 6.0 * sqrx + 3.0 * sqrx * absx ) / 6.0
    elif absx < 2.0:
        return ( 8.0 - 12 * absx + 6.0 * sqrx - sqrx * absx ) / 6.0
    return 0.0


cdef inline double _cubic_spline_derivative(double x) nogil:
    r''' Derivative of cubic B-Spline
    See eq. (3) of [1].

    [1] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., & Eubank, W.
    PET-CT image registration in the chest using free-form deformations.
    IEEE Transactions on Medical Imaging, 22(1), 120–8, 2003.
    '''
    cdef:
        double absx = -x if x < 0.0 else x
        double sqrx = x * x
    if absx < 1.0:
        if x >= 0.0:
            return -2.0 * x + 1.5 * x * x
        else:
            return -2.0 * x - 1.5 * x * x
    elif absx < 2.0:
        if x >= 0:
            return -2.0 + 2.0 * x - 0.5 * x * x
        else:
            return 2.0 + 2.0 * x + 0.5 * x * x
    return 0.0


cdef _compute_pdfs_dense_2d(double[:,:] static, double[:,:] moving,
                       int[:,:] smask, int[:,:] mmask,
                       double smin, double sdelta,
                       double mmin, double mdelta,
                       int nbins, int padding, double[:,:] joint,
                       double[:] smarginal, double[:] mmarginal):
    r''' Joint Probability Density Function of intensities of two 2D images

    Parameters
    ----------
    static: array, shape (R, C)
        static image
    moving: array, shape (R, C)
        moving image
    smask: array, shape (R, C)
        mask of static object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    mmask: array, shape (R, C)
        mask of moving object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    smin: float
        the minimum observed intensity associated to the static image, which
        was used to define the joint PDF
    sdelta: float
        bin size associated to the intensities of the static image
    mmin: float
        the minimum observed intensity associated to the moving image, which
        was used to define the joint PDF
    mdelta: float
        bin size associated to the intensities of the moving image
    nbins: int
        number of histogram bins
    padding: int
        number of bins used as padding (the total bins used for padding at both sides
        of the histogram is actually 2*padding)
    joint: array, shape (nbins, nbins)
        the array to write the joint PDF
    smarginal: array, shape (nbins,)
        the array to write the marginal PDF associated to the static image
    mmarginal: array, shape (nbins,)
        the array to write the marginal PDF associated to the moving image
    '''
    cdef:
        int nrows = static.shape[0]
        int ncols = static.shape[1]
        int offset, valid_points
        cnp.npy_intp i, j, r, c
        double rn, cn
        double val, spline_arg, sum

    joint[...] = 0
    sum = 0
    valid_points = 0
    with nogil:
        smarginal[:] = 0
        for i in range(nrows):
            for j in range(ncols):
                if smask is not None and smask[i, j] == 0:
                    continue
                if mmask is not None and mmask[i, j] == 0:
                    continue
                valid_points += 1
                rn = _bin_normalize(static[i, j], smin, sdelta)
                r = _bin_index(rn, nbins, padding)
                cn = _bin_normalize(moving[i, j], mmin, mdelta)
                c = _bin_index(cn, nbins, padding)
                spline_arg = (c-1) - cn

                smarginal[r] += 1
                for offset in range(-1,3):
                    val = _cubic_spline(spline_arg)
                    joint[r, c + offset] += val
                    sum += val
                    spline_arg += 1.0

        if sum > 0:
            for i in range(nbins):
                for j in range(nbins):
                    joint[i, j] /= sum

            for i in range(nbins):
                smarginal[i] /= valid_points

            for j in range(nbins):
                mmarginal[j] = 0
                for i in range(nbins):
                    mmarginal[j] += joint[i, j]



cdef _compute_pdfs_dense_3d(double[:,:,:] static, double[:,:,:] moving,
                       int[:,:,:] smask, int[:,:,:] mmask,
                       double smin, double sdelta,
                       double mmin, double mdelta,
                       int nbins, int padding, double[:,:] joint,
                       double[:] smarginal, double[:] mmarginal):
    r''' Joint Probability Density Function of intensities of two 3D images

    Parameters
    ----------
    static: array, shape (S, R, C)
        static image
    moving: array, shape (S, R, C)
        moving image
    smask: array, shape (S, R, C)
        mask of static object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    mmask: array, shape (S, R, C)
        mask of moving object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    smin: float
        the minimum observed intensity associated to the static image, which
        was used to define the joint PDF
    sdelta: float
        bin size associated to the intensities of the static image
    mmin: float
        the minimum observed intensity associated to the moving image, which
        was used to define the joint PDF
    mdelta: float
        bin size associated to the intensities of the moving image
    nbins: int
        number of histogram bins
    padding: int
        number of bins used as padding (the total bins used for padding at both sides
        of the histogram is actually 2*padding)
    joint: array, shape(nbins, nbins)
        the array to write the joint PDF to
    smarginal: array, shape (nbins,)
        the array to write the marginal PDF associated to the static image
    mmarginal: array, shape (nbins,)
        the array to write the marginal PDF associated to the moving image
    '''
    cdef:
        int nslices = static.shape[0]
        int nrows = static.shape[1]
        int ncols = static.shape[2]
        int offset, valid_points
        cnp.npy_intp k, i, j, r, c
        double rn, cn
        double val, spline_arg, sum

    joint[...] = 0
    sum = 0
    with nogil:
        smarginal[:] = 0
        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if smask is not None and smask[k, i, j] == 0:
                        continue
                    if mmask is not None and mmask[k, i, j] == 0:
                        continue
                    valid_points += 1
                    rn = _bin_normalize(static[k, i, j], smin, sdelta)
                    r = _bin_index(rn, nbins, padding)
                    cn = _bin_normalize(moving[k, i, j], mmin, mdelta)
                    c = _bin_index(cn, nbins, padding)
                    spline_arg = (c-1) - cn

                    smarginal[r] += 1
                    for offset in range(-1,3):
                        val = _cubic_spline(spline_arg)
                        joint[r, c + offset] += val
                        sum += val
                        spline_arg += 1.0

        if sum > 0:
            for i in range(nbins):
                for j in range(nbins):
                    joint[i, j] /= sum

            for i in range(nbins):
                smarginal[i] /= valid_points

            for j in range(nbins):
                mmarginal[j] = 0
                for i in range(nbins):
                    mmarginal[j] += joint[i, j]


cdef _compute_pdfs_sparse(double[:] sval, double[:] mval, double smin,
                       double sdelta, double mmin, double mdelta,
                       int nbins, int padding, double[:,:] joint,
                       double[:] smarginal, double[:] mmarginal):
    r''' Probability Density Functions of paired intensities

    Parameters
    ----------
    sval: array, shape (n,)
        sampled intensities from the static image at sampled_points
    mval: array, shape (n,)
        sampled intensities from the moving image at sampled_points
    smin: float
        the minimum observed intensity associated to the static image, which
        was used to define the joint PDF
    sdelta: float
        bin size associated to the intensities of the static image
    mmin: float
        the minimum observed intensity associated to the moving image, which
        was used to define the joint PDF
    mdelta: float
        bin size associated to the intensities of the moving image
    nbins: int
        number of histogram bins
    padding: int
        number of bins used as padding (the total bins used for padding at both sides
        of the histogram is actually 2*padding)
    joint: array, shape(nbins, nbins)
        the array to write the joint PDF to
    smarginal: array, shape (nbins,)
        the array to write the marginal PDF associated to the static image
    mmarginal: array, shape (nbins,)
        the array to write the marginal PDF associated to the moving image
    '''
    cdef:
        int n = sval.shape[0]
        int offset, valid_points
        cnp.npy_intp i, r, c
        double rn, cn
        double val, spline_arg, sum

    joint[...] = 0
    sum = 0
    valid_points = 0
    with nogil:
        smarginal[:] = 0
        for i in range(n):
            valid_points += 1
            rn = _bin_normalize(sval[i], smin, sdelta)
            r = _bin_index(rn, nbins, padding)
            cn = _bin_normalize(mval[i], mmin, mdelta)
            c = _bin_index(cn, nbins, padding)
            spline_arg = (c-1) - cn

            smarginal[r] += 1
            for offset in range(-1, 3):
                val = _cubic_spline(spline_arg)
                joint[r, c + offset] += val
                sum += val
                spline_arg += 1.0

        if sum > 0:
            for i in range(nbins):
                for j in range(nbins):
                    joint[i, j] /= sum

            for i in range(nbins):
                smarginal[i] /= valid_points

            for j in range(nbins):
                mmarginal[j] = 0
                for i in range(nbins):
                    mmarginal[j] += joint[i, j]


cdef _joint_pdf_gradient_dense_2d(double[:] theta, jacobian_function jacobian,
                                  double[:,:] static, double[:,:] moving,
                                  double[:,:] grid_to_space, double[:,:,:] mgradient,
                                  int[:,:] smask, int[:,:] mmask,
                                  double smin, double sdelta,
                                  double mmin, double mdelta,
                                  int nbins, int padding, double[:,:,:] grad_pdf):
    r''' Gradient of the joint PDF w.r.t. transform parameters theta

    Computes the vector of partial derivatives of the joint histogram w.r.t.
    each transformation parameter. The transformation itself is not necessary
    to compute the gradient, but only its Jacobian.

    Parameters
    ----------
    theta: array, shape (n,)
        parameters of the transformation to compute the gradient from
    jacobian: jacobian_function
        function that computes the Jacobian matrix of a transformation
    static: array, shape (R, C)
        static image
    moving: array, shape (R, C)
        moving image
    grid_to_space: array, shape (3, 3)
        the grid-to-space transform associated to images static and moving (
        we assume that both images have already been sampled at a common grid)
    mgradient: array, shape (R, C, 2)
        the gradient of the moving image
    smask: array, shape (R, C)
        mask of static object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    mmask: array, shape (R, C)
        mask of moving object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    smin: float
        the minimum observed intensity associated to the static image, which
        was used to define the joint PDF
    sdelta: float
        bin size associated to the intensities of the static image
    mmin: float
        the minimum observed intensity associated to the moving image, which
        was used to define the joint PDF
    mdelta: float
        bin size associated to the intensities of the moving image
    nbins: int
        number of histogram bins
    padding: int
        number of bins used as padding (the total bins used for padding at both sides
        of the histogram is actually 2*padding)
    grad_pdf: array, shape (nbins, nbins, len(theta))
        the array to write the gradient to
    '''
    cdef:
        int nrows = static.shape[0]
        int ncols = static.shape[1]
        int n = theta.shape[0]
        int offset, valid_points, constant_jacobian=0
        cnp.npy_intp k, i, j, r, c
        double rn, cn
        double val, spline_arg
        double[:,:] J = np.ndarray(shape=(2, n), dtype=np.float64)
        double[:] prod = np.ndarray(shape=(n,), dtype=np.float64)
        double[:] x = np.ndarray(shape=(2,), dtype=np.float64)

    grad_pdf[...] = 0
    with nogil:
        valid_points = 0
        for i in range(nrows):
            for j in range(ncols):
                if smask is not None and smask[i, j] == 0:
                    continue
                if mmask is not None and mmask[i, j] == 0:
                    continue
                
                valid_points += 1
                x[0] = _apply_affine_2d_x0(i, j, 1, grid_to_space)
                x[1] = _apply_affine_2d_x1(i, j, 1, grid_to_space)

                if constant_jacobian == 0:
                    constant_jacobian = jacobian(theta, x, J)

                for k in range(n):
                    prod[k] = J[0,k] * mgradient[i,j,0] +\
                              J[1,k] * mgradient[i,j,1]

                rn = _bin_normalize(static[i, j], smin, sdelta)
                r = _bin_index(rn, nbins, padding)
                cn = _bin_normalize(moving[i, j], mmin, mdelta)
                c = _bin_index(cn, nbins, padding)
                spline_arg = (c-1) - cn

                for offset in range(-1,3):
                    val = _cubic_spline_derivative(spline_arg)
                    for k in range(n):
                        grad_pdf[r, c + offset,k] -= val * prod[k]
                    spline_arg += 1.0

        if valid_points * mdelta > 0:
            for i in range(nbins):
                for j in range(nbins):
                    for k in range(n):
                        grad_pdf[i, j, k] /= (valid_points * mdelta)
                    


cdef _joint_pdf_gradient_dense_3d(double[:] theta, jacobian_function jacobian,
                                  double[:,:,:] static, double[:,:,:] moving,
                                  double[:,:] grid_to_space, double[:,:,:,:] mgradient,
                                  int[:,:,:] smask, int[:,:,:] mmask,
                                  double smin, double sdelta,
                                  double mmin, double mdelta,
                                  int nbins, int padding, double[:,:,:] grad_pdf):
    r''' Gradient of the joint PDF w.r.t. transform parameters theta

    Computes the vector of partial derivatives of the joint histogram w.r.t.
    each transformation parameter. The transformation itself is not necessary
    to compute the gradient, but only its Jacobian.

    Parameters
    ----------
    theta: array, shape (n,)
        parameters of the transformation to compute the gradient from
    jacobian: jacobian_function
        function that computes the Jacobian matrix of a transformation
    static: array, shape (S, R, C)
        static image
    moving: array, shape (S, R, C)
        moving image
    grid_to_space: array, shape (4, 4)
        the grid-to-space transform associated to images static and moving (
        we assume that both images have already been sampled at a common grid)
    mgradient: array, shape (S, R, C, 3)
        the gradient of the moving image
    smask: array, shape (S, R, C)
        mask of static object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    mmask: array, shape (S, R, C)
        mask of moving object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    smin: float
        the minimum observed intensity associated to the static image, which
        was used to define the joint PDF
    sdelta: float
        bin size associated to the intensities of the static image
    mmin: float
        the minimum observed intensity associated to the moving image, which
        was used to define the joint PDF
    mdelta: float
        bin size associated to the intensities of the moving image
    nbins: int
        number of histogram bins
    padding: int
        number of bins used as padding (the total bins used for padding at both sides
        of the histogram is actually 2*padding)
    grad_pdf: array, shape (nbins, nbins, len(theta))
        the array to write the gradient to
    '''
    cdef:
        int nslices = static.shape[0]
        int nrows = static.shape[1]
        int ncols = static.shape[2]
        int n = theta.shape[0]
        int offset, constant_jacobian=0
        cnp.npy_intp l, k, i, j, r, c
        double rn, cn
        double val, spline_arg
        double[:,:] J = np.ndarray(shape=(3, n), dtype=np.float64)
        double[:] prod = np.ndarray(shape=(n,), dtype=np.float64)
        double[:] x = np.ndarray(shape=(3,), dtype=np.float64)

    grad_pdf[...] = 0
    with nogil:

        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if smask is not None and smask[k, i, j] == 0:
                        continue
                    if mmask is not None and mmask[k, i, j] == 0:
                        continue
                    x[0] = _apply_affine_3d_x0(k, i, j, 1, grid_to_space)
                    x[1] = _apply_affine_3d_x1(k, i, j, 1, grid_to_space)
                    x[2] = _apply_affine_3d_x2(k, i, j, 1, grid_to_space)

                    if constant_jacobian == 0:
                        constant_jacobian = jacobian(theta, x, J)

                    for l in range(n):
                        prod[l] = J[0,l] * mgradient[k,i,j,0] +\
                                  J[1,l] * mgradient[k,i,j,1] +\
                                  J[2,l] * mgradient[k,i,j,2]

                    rn = _bin_normalize(static[k, i, j], smin, sdelta)
                    r = _bin_index(rn, nbins, padding)
                    cn = _bin_normalize(moving[k, i, j], mmin, mdelta)
                    c = _bin_index(cn, nbins, padding)
                    spline_arg = (c-1) - cn

                    for offset in range(-1,3):
                        val = _cubic_spline_derivative(spline_arg)
                        for l in range(n):
                            grad_pdf[r, c + offset,l] -= val * prod[l]
                        spline_arg += 1.0


cdef _joint_pdf_gradient_sparse_2d(double[:] theta, jacobian_function jacobian,
                                  double[:] sval, double[:] mval,
                                  double[:,:] sample_points, double[:,:] mgradient,
                                  double smin, double sdelta,
                                  double mmin, double mdelta,
                                  int nbins, int padding, double[:,:,:] grad_pdf):
    r''' Gradient of the joint PDF w.r.t. transform parameters theta

    Computes the vector of partial derivatives of the joint histogram w.r.t.
    each transformation parameter. The transformation itself is not necessary
    to compute the gradient, but only its Jacobian.

    Parameters
    ----------
    theta: array, shape (n,)
        parameters to compute the gradient at
    jacobian: jacobian_function
        function that computes the Jacobian matrix of a transformation
    sval: array, shape (m,)
        sampled intensities from the static image at sampled_points
    mval: array, shape (m,)
        sampled intensities from the moving image at sampled_points
    sample_points: array, shape (m, 2)
        coordinates (in physical space) of the points the images were sampled at
    mgradient: array, shape (m, 2)
        the gradient of the moving image at the sample points
    smin: float
        the minimum observed intensity associated to the static image, which
        was used to define the joint PDF
    sdelta: float
        bin size associated to the intensities of the static image
    mmin: float
        the minimum observed intensity associated to the moving image, which
        was used to define the joint PDF
    mdelta: float
        bin size associated to the intensities of the moving image
    nbins: int
        number of histogram bins
    padding: int
        number of bins used as padding (the total bins used for padding at both sides
        of the histogram is actually 2*padding)
    grad_pdf: array, shape (nbins, nbins, len(theta))
        the array to write the gradient to
    '''
    cdef:
        int n = theta.shape[0]
        int m = sval.shape[0]
        int offset, constant_jacobian=0
        cnp.npy_intp i, j, r, c
        double rn, cn
        double val, spline_arg
        double[:,:] J = np.ndarray(shape=(2, n), dtype=np.float64)
        double[:] prod = np.ndarray(shape=(n,), dtype=np.float64)

    grad_pdf[...] = 0
    with nogil:

        for i in range(m):
            if constant_jacobian == 0:
                constant_jacobian = jacobian(theta, sample_points[i], J)

            for j in range(n):
                prod[j] = J[0,j] * mgradient[i,0] +\
                          J[1,j] * mgradient[i,1]

            rn = _bin_normalize(sval[i], smin, sdelta)
            r = _bin_index(rn, nbins, padding)
            cn = _bin_normalize(mval[i], mmin, mdelta)
            c = _bin_index(cn, nbins, padding)
            spline_arg = (c-1) - cn

            for offset in range(-1,3):
                val = _cubic_spline_derivative(spline_arg)
                for j in range(n):
                    grad_pdf[r, c + offset,j] -= val * prod[j]
                spline_arg += 1.0


cdef _joint_pdf_gradient_sparse_3d(double[:] theta, jacobian_function jacobian,
                                  double[:] sval, double[:] mval,
                                  double[:,:] sample_points, double[:,:] mgradient,
                                  double smin, double sdelta,
                                  double mmin, double mdelta,
                                  int nbins, int padding, double[:,:,:] grad_pdf):
    r''' Gradient of the joint PDF w.r.t. transform parameters theta

    Computes the vector of partial derivatives of the joint histogram w.r.t.
    each transformation parameter. The transformation itself is not necessary
    to compute the gradient, but only its Jacobian.

    Parameters
    ----------
    theta: array, shape (n,)
        parameters to compute the gradient at
    jacobian: jacobian_function
        function that computes the Jacobian matrix of a transformation
    sval: array, shape (m,)
        sampled intensities from the static image at sampled_points
    mval: array, shape (m,)
        sampled intensities from the moving image at sampled_points
    sample_points: array, shape (m, 3)
        coordinates (in physical space) of the points the images were sampled at
    mgradient: array, shape (m, 3)
        the gradient of the moving image at the sample points
    smin: float
        the minimum observed intensity associated to the static image, which
        was used to define the joint PDF
    sdelta: float
        bin size associated to the intensities of the static image
    mmin: float
        the minimum observed intensity associated to the moving image, which
        was used to define the joint PDF
    mdelta: float
        bin size associated to the intensities of the moving image
    nbins: int
        number of histogram bins
    padding: int
        number of bins used as padding (the total bins used for padding at both sides
        of the histogram is actually 2*padding)
    grad_pdf: array, shape (nbins, nbins, len(theta))
        the array to write the gradient to
    '''
    cdef:
        int n = theta.shape[0]
        int m = sval.shape[0]
        int offset, constant_jacobian=0
        cnp.npy_intp i, j, r, c
        double rn, cn
        double val, spline_arg
        double[:,:] J = np.ndarray(shape=(3, n), dtype=np.float64)
        double[:] prod = np.ndarray(shape=(n,), dtype=np.float64)

    grad_pdf[...] = 0
    with nogil:

        for i in range(m):
            if constant_jacobian == 0:
                constant_jacobian = jacobian(theta, sample_points[i], J)

            for j in range(n):
                prod[j] = J[0,j] * mgradient[i,0] +\
                          J[1,j] * mgradient[i,1] +\
                          J[2,j] * mgradient[i,2]

            rn = _bin_normalize(sval[i], smin, sdelta)
            r = _bin_index(rn, nbins, padding)
            cn = _bin_normalize(mval[i], mmin, mdelta)
            c = _bin_index(cn, nbins, padding)
            spline_arg = (c-1) - cn

            #Sweep the bins affected by a Parzen window centered at (rn, cn)
            #I think it should be range (-2, 3), but this is how it's
            #implemented in ANTS (the contribution for bin c-2 would be very
            #small, though)
            for offset in range(-1,3):
                val = _cubic_spline_derivative(spline_arg)
                for j in range(n):
                    grad_pdf[r, c + offset,j] -= val * prod[j]
                spline_arg += 1.0





def joint_pdf_dense_3d(static, moving, int nbins):
    r''' Joint Probability Density Function
    Computes the Joint Probability Density Function (PDF) of intensity levels
    of the given images (assuming they're aligned) as a squared histogram of
    [nbins x nbins] bins.

    Parameters
    ----------
    static : array, shape (S, R, C)
        Static image
    moving :  array, shape (S, R, C)
        Moving image
    '''
    cdef:
        int padding = 2
        double smin, smax, mmin, mmax
        double sdelta, mdelta

    smask = np.array(static > 0).astype(np.int32)
    mmask = np.array(moving > 0).astype(np.int32)

    smin = np.min(static[static>0])
    smax = np.max(static[static>0])
    mmin = np.min(moving[moving>0])
    mmax = np.max(moving[moving>0])

    sdelta = (smax - smin)/(nbins - 2 * padding)
    mdelta = (mmax - mmin)/(nbins - 2 * padding)
    smin = smin/sdelta - padding
    mmin = mmin/sdelta - padding

    joint = np.ndarray(shape = (nbins, nbins), dtype = np.float64)
    smarginal = np.ndarray(shape = (nbins,), dtype = np.float64)
    mmarginal = np.ndarray(shape = (nbins,), dtype = np.float64)
    energy = _compute_pdfs_dense_3d(static, moving, smask, mmask,
                                 smin, sdelta, mmin, mdelta,
                                 nbins, padding, joint, smarginal, mmarginal)
    return joint


cdef inline int _int_max(int a, int b) nogil:
    r"""
    Returns the maximum of a and b
    """
    return a if a >= b else b


cdef inline int _int_min(int a, int b) nogil:
    r"""
    Returns the minimum of a and b
    """
    return a if a <= b else b

cdef enum:
    SI = 0
    SI2 = 1
    SJ = 2
    SJ2 = 3
    SIJ = 4
    CNT = 5

def compute_cc_residuals(double[:,:,:] I, double[:,:,:] J, int radius):
    cdef:
        int ns = I.shape[0]
        int nr = I.shape[1]
        int nc = I.shape[2]
        double s1, s2, wx, p, t
        int i, j, k, s, r, c
        int start_k, end_k, start_i, end_i, start_j, end_j

        double[:,:,:] residuals = np.zeros((ns, nr, nc))

    with nogil:
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):

                    #Affine fit
                    s1 = 0
                    s2 = 0
                    wx = 0
                    p = 0
                    t = 0
                    start_k = _int_max(0, s - radius)
                    end_k = _int_min(ns, 1 + s + radius)
                    for k in range(start_k, end_k):

                        start_i = _int_max(0, r - radius)
                        end_i = _int_min(nr, 1 + r + radius)
                        for i in range(start_i, end_i):

                            start_j = _int_max(0, c - radius)
                            end_j = _int_min(nc, 1 + c + radius)
                            for j in range(start_j, end_j):

                                s1 += I[k, i, j]
                                s2 += I[k, i, j] * I[k, i, j]
                                wx += 1
                                p += I[k, i, j] * J[k, i, j]
                                t += J[k, i, j]

                    if s2 < 1e-9:
                        alpha = 0
                        beta = t/wx
                    else:
                        beta = (t - (s1 * p) / s2) / (wx - (s1 * s1) / s2)
                        alpha = (p - beta * s1) / s2

                    #Compute residuals
                    residuals[s, r, c] = 0
                    start_k = _int_max(0, s - radius)
                    end_k = _int_min(ns, 1 + s + radius)
                    for k in range(start_k, end_k):

                        start_i = _int_max(0, r - radius)
                        end_i = _int_min(nr, 1 + r + radius)
                        for i in range(start_i, end_i):

                            start_j = _int_max(0, c - radius)
                            end_j = _int_min(nc, 1 + c + radius)
                            for j in range(start_j, end_j):

                                residuals[s, r, c] += ((alpha * I[k, i, j] + beta) - J[k, i, j]) ** 2
    return residuals


def compute_cc_residuals_noboundary(double[:,:,:] I, double[:,:,:] J, int radius):
    cdef:
        int ns = I.shape[0]
        int nr = I.shape[1]
        int nc = I.shape[2]
        double s1, s2, wx, p, t, ave, worst
        int i, j, k, s, r, c, intersect
        int start_k, end_k, start_i, end_i, start_j, end_j

        double[:,:,:] residuals = np.zeros((ns, nr, nc))

    with nogil:
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):

                    #Affine fit
                    s1 = 0
                    s2 = 0
                    wx = 0
                    p = 0
                    t = 0
                    start_k = _int_max(0, s - radius)
                    end_k = _int_min(ns, 1 + s + radius)
                    for k in range(start_k, end_k):

                        start_i = _int_max(0, r - radius)
                        end_i = _int_min(nr, 1 + r + radius)
                        for i in range(start_i, end_i):

                            start_j = _int_max(0, c - radius)
                            end_j = _int_min(nc, 1 + c + radius)
                            for j in range(start_j, end_j):

                                intersect = (I[k, i, j]>0) * (J[k, i, j]>0)
                                if intersect == 0:
                                    continue
                                s1 += I[k, i, j]
                                s2 += I[k, i, j] * I[k, i, j]
                                wx += 1
                                p += I[k, i, j] * J[k, i, j]
                                t += J[k, i, j]

                    residuals[s, r, c] = 0
                    if wx<3:
                        continue
                    ave = t/wx

                    if s2 < 1e-6:
                        alpha = 0
                        beta = ave
                    else:
                        if s2 * wx - (s1 * s1) < 1e-6 and s2 * wx - (s1 * s1)  > -1e-6:
                            continue
                        beta = (t - (s1 * p) / s2) / (wx - (s1 * s1) / s2)
                        alpha = (p - beta * s1) / s2

                    #Compute residuals
                    worst = 0
                    start_k = _int_max(0, s - radius)
                    end_k = _int_min(ns, 1 + s + radius)
                    for k in range(start_k, end_k):

                        start_i = _int_max(0, r - radius)
                        end_i = _int_min(nr, 1 + r + radius)
                        for i in range(start_i, end_i):

                            start_j = _int_max(0, c - radius)
                            end_j = _int_min(nc, 1 + c + radius)
                            for j in range(start_j, end_j):

                                intersect = (I[k, i, j]>0) * (J[k, i, j]>0)
                                if intersect == 0:
                                    continue

                                worst += (ave - J[k, i, j]) ** 2
                                residuals[s, r, c] += ((alpha * I[k, i, j] + beta) - J[k, i, j]) ** 2
                    if residuals[s, r, c] > worst:
                        residuals[s, r, c] = worst

    return residuals


cdef double _compute_mattes_mi(double[:,:] joint, double[:,:,:] joint_gradient,
                               double[:] smarginal, double[:] mmarginal,
                               double[:] mi_gradient) nogil:
    cdef:
        double epsilon = 2.2204460492503131e-016
        double metric_value
        int nrows = joint_gradient.shape[0]
        int ncols = joint_gradient.shape[1]
        int n = joint_gradient.shape[2]

    mi_gradient[:] = 0
    metric_value = 0
    for i in range(nrows):
        for j in range(ncols):
            if mmarginal[j] < epsilon:
                continue

            factor = log(joint[i,j] / mmarginal[j])

            if mi_gradient is not None:
                for k in range(n):
                    mi_gradient[k] -= joint_gradient[i, j, k] * factor

            if smarginal[i] > epsilon:
                metric_value += joint[i,j] * (factor - log(smarginal[i]))

    return metric_value







class MattesPDF(object):

    def __init__(self, nbins, static, moving, smask=None, mmask=None, padding=2):
        self.nbins = nbins
        self.padding = padding

        if smask is None:
            smask = np.array(static > 0).astype(np.int32)
        if mmask is None:
            mmask = np.array(moving > 0).astype(np.int32)

        self.smin = np.min(static[smask!=0])
        self.smax = np.max(static[smask!=0])
        self.mmin = np.min(moving[mmask!=0])
        self.mmax = np.max(moving[mmask!=0])

        self.sdelta = (self.smax - self.smin)/(nbins - padding)
        self.mdelta = (self.mmax - self.mmin)/(nbins - padding)
        self.smin = self.smin/self.sdelta - padding
        self.mmin = self.mmin/self.sdelta - padding

        self.joint_grad = None
        self.metric_grad = None
        self.metric_val = 0
        self.joint = np.ndarray(shape = (nbins, nbins), dtype = np.float64)
        self.smarginal = np.ndarray(shape = (nbins,), dtype = np.float64)
        self.mmarginal = np.ndarray(shape = (nbins,), dtype = np.float64)


    def update_pdfs_dense(self, static, moving, smask, mmask):
        dim = len(static.shape)
        if dim == 2:
            _compute_pdfs_dense_2d(static, moving, smask, mmask,
                                   self.smin, self.sdelta, self.mmin, self.mdelta,
                                   self.nbins, self.padding, self.joint, self.smarginal, self.mmarginal)
        elif dim == 3:
            _compute_pdfs_dense_3d(static, moving, smask, mmask,
                                   self.smin, self.sdelta, self.mmin, self.mdelta,
                                   self.nbins, self.padding, self.joint, self.smarginal, self.mmarginal)
        else:
            raise ValueError('Only dimensions 2 and 3 are supported. '+str(dim)+' received')


    def update_pdfs_sparse(self, sval, mval):
        energy = _compute_pdfs_sparse(sval, mval, self.smin, self.sdelta,
                                   self.nbins, self.mmin, self.mdelta, self.padding, self.joint, self.smarginal, self.mmarginal)


    def update_gradient_dense(self, theta, transform, static, moving, grid_to_space, mgradient, smask, mmask):
        dim = len(static.shape)
        if (self.joint_grad is None) or (self.joint_grad.shape[2] != theta.shape[0]):
            self.joint_grad = np.ndarray(shape = (self.nbins, self.nbins, theta.shape[0]), dtype = np.float64)
        if transform == 'translation':
            if dim == 2:
                jacobian = _translation_jacobian_2d
            else:
                jacobian = _translation_jacobian_3d
        elif transform == 'scale':
            if dim == 2:
                jacobian = _scale_jacobian_2d
            else:
                jacobian = _scale_jacobian_3d
        elif transform == 'rotation':
            if dim == 2:
                jacobian = _rotation_jacobian_2d
            else:
                jacobian = _rotation_jacobian_3d
        elif transform == 'affine':
            if dim == 2:
                jacobian = _affine_jacobian_2d
            else:
                jacobian = _affine_jacobian_3d
        else:
            raise(ValueError('Unknown transform type: "'+transform+'"'))

        if dim == 2:
            _joint_pdf_gradient_dense_2d(theta, jacobian, static, moving, grid_to_space, mgradient,
                                         smask, mmask, self.smin, self.sdelta, self.mmin, self.mdelta,
                                         self.nbins, self.padding, self.joint_grad)
        elif dim ==3:
            _joint_pdf_gradient_dense_3d(theta, jacobian, static, moving, grid_to_space, mgradient,
                                         smask, mmask, self.smin, self.sdelta, self.mmin, self.mdelta,
                                         self.nbins, self.padding, self.joint_grad)
        else:
            raise ValueError('Only dimensions 2 and 3 are supported. '+str(dim)+' received')


    def update_gradient_sparse(self, dim, theta, transform, sval, mval, sample_points, mgradient):
        dim = sample_points.shape[1]
        if (self.joint_grad is None) or (self.joint_grad.shape[2] != theta.shape[0]):
            self.joint_grad = np.ndarray(shape = (self.nbins, self.nbins, theta.shape[0]), dtype = np.float64)
        if transform == 'translation':
            if dim == 2:
                jacobian = _translation_jacobian_2d
            else:
                jacobian = _translation_jacobian_3d
        elif transform == 'scale':
            if dim == 2:
                jacobian = _scale_jacobian_2d
            else:
                jacobian = _scale_jacobian_3d
        elif transform == 'rotation':
            if dim == 2:
                jacobian = _rotation_jacobian_2d
            else:
                jacobian = _rotation_jacobian_3d
        elif transform == 'affine':
            if dim == 2:
                jacobian = _affine_jacobian_2d
            else:
                jacobian = _affine_jacobian_3d
        else:
            raise(ValueError('Unknown transform type: "'+transform+'"'))

        if dim == 2:
            _joint_pdf_gradient_sparse_2d(theta, jacobian, sval, mval, sample_points, mgradient,
                                          self.smin, self.sdelta, self.mmin, self.mdelta,
                                          self.nbins, self.padding, self.joint_grad)
        elif dim ==3:
            _joint_pdf_gradient_sparse_3d(theta, jacobian, sval, mval, sample_points, mgradient,
                                          self.smin, self.sdelta, self.mmin, self.mdelta,
                                          self.nbins, self.padding, self.joint_grad)
        else:
            raise ValueError('Only dimensions 2 and 3 are supported. '+str(dim)+' received')


    def update_mi_metric(self, update_gradient=True):
        if update_gradient:
            grad_dimension = self.joint_grad.shape[2]
            if (self.metric_grad is None) or (self.metric_grad.shape[0] != grad_dimension):
                self.metric_grad = np.empty(shape=grad_dimension)
            self.metric_val = _compute_mattes_mi(self.joint, self.joint_grad,
                                                 self.smarginal, self.mmarginal,
                                                 self.metric_grad)
        else:
            self.metric_val = _compute_mattes_mi(self.joint, self.joint_grad,
                                                 self.smarginal, self.mmarginal,
                                                 None)











cdef eval_jacobian(double[:]theta, double[:]x, double[:,:] J, jacobian_function jacobian):
    jacobian(theta, x, J)

def test_evals():
    cdef:
        double[:,:] Jscale = np.ndarray(shape = (12,3), dtype=np.float64)
        double[:] theta = np.ndarray(shape = (1, ), dtype=np.float64)
        double[:] x = np.ndarray(shape = (3, ), dtype=np.float64)

    theta[0] = 5.5
    x[0] = 1.5
    x[1] = 2.0
    x[2] = 2.5
    eval_jacobian(theta, x, Jscale, _scale_jacobian_3d)
    print(np.array(Jscale))
