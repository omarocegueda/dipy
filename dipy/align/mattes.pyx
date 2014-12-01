#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython
import imaffine
from libc.math cimport cos, sin

ctypedef int (*jacobian_function)(double[:], double[:], double[:,:])
r""" Type of a function that computes the Jacobian of a transform
Jacobian functions receive a vector containing the current parameters
of the transformation, the coordinates of a point to compute the
Jacobian at, and the Jacobian matrix to write the result in. The
shape of the resulting Jacobian must be an nxd matrix, where n is 
the number of parameters of the transformation, and d is the dimension
of the transform. Note: this is actually the transpose of the Jacobian
matrix in standard notation, we save the matrix this way for efficienncy
"""


cdef inline void mult_mat_3d(double[:,:] A, double[:,:] B, double[:,:] C) nogil:
    cdef:
        cnp.npy_intp i, j

    for i in range(3):
        for j in range(3):
            C[i, j] = A[i, 0]*B[0, j] + A[i, 1]*B[1, j] + A[i, 2]*B[2, j]


cdef void _rotation_matrix(double a, double b, double c, double[:,:] R):
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
        mult_mat_3d(rot_a, rot_b, temp)
        mult_mat_3d(rot_c, temp, R)


cdef void rotation_jacobian_3d(double[:] theta, double[:] x, double[:,:] J):
    cdef:
        double sa = sin(theta[0])
        double ca = cos(theta[0])
        double sb = sin(theta[1])
        double cb = cos(theta[1])
        double sc = sin(theta[2])
        double cc = cos(theta[2])
        double px = x[0], py = x[1], pz = x[2]

    
    J[0, 0] = ( -sc * ca * sb ) * px + ( sc * sa ) * py + ( sc * ca * cb ) * pz
    J[0, 1] = ( cc * ca * sb ) * px + ( -cc * sa ) * py + ( -cc * ca * cb ) * pz
    J[0, 2] = ( sa * sb ) * px + ( ca ) * py + ( -sa * cb ) * pz

    J[1, 0] = ( -cc * sb - sc * sa * cb ) * px + ( cc * cb - sc * sa * sb ) * pz
    J[1, 1] = ( -sc * sb + cc * sa * cb ) * px + ( sc * cb + cc * sa * sb ) * pz
    J[1, 2] = ( -ca * cb ) * px + ( -ca * sb ) * pz

    J[2, 0] = ( -sc * cb - cc * sa * sb ) * px + ( -cc * ca ) * py + \
              ( -sc * sb + cc * sa * cb ) * pz
    J[2, 1] = ( cc * cb - sc * sa * sb ) * px + ( -sc * ca ) * py + \
              ( cc * sb + sc * sa * cb ) * pz
    J[2, 2] = 0

cdef int scale_jacobian_3d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r""" Jacobian matrix of the isotropic scale transform (one parameter)
    The transformation is given by:
    
    T(x) = (s*x0, s*x1, s*x2) 

    The derivative w.r.t. s is T'(x) = [x0, x1, x2]
    """
    J[0,:] = x[:]
    return 0


cdef int translation_jacobian_3d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r""" Jacobian matrix of the translation transform (three parameters)
    The transformation is given by:
    
    T(x) = (T1(x), T2(x), T3(x)) = (x0 + t0, x1 + t1, x2 + t2) 

    The derivative w.r.t. t1, t2 and t3 is given by

    T'(x) = [[1, 0, 0], # derivatives of [T1, T2, T3] w.r.t. t0
             [0, 1, 0], # derivatives of [T1, T2, T3] w.r.t. t1
             [0, 0, 1]] # derivatives of [T1, T2, T3] w.r.t. t2
    """
    cdef:
        cnp.npy_intp i

    for i in range(3):
        J[i,:] = 0
        J[i,i] = 1
    return 1


cdef int affine_jacobian_3d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r""" Jacobian matrix of the affine transform (twelve parameters)
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
    """
    cdef:
        cnp.npy_intp j

    for j in range(3):
        J[:,j] = 0
    J[:3,0] = x[...]
    J[3,0] = 1
    J[4:7,1] = x[...]
    J[7,1] = 1
    J[8:11,2] = x[...]
    J[11,2] = 1
    return 0


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
    eval_jacobian(theta, x, Jscale, scale_jacobian_3d)
    print(np.array(Jscale))


cdef inline double _bin_normalize(double x, double mval, double delta) nogil:
    return x / delta - mval


cdef inline cnp.npy_intp _bin_index(double normalized, int nbins, int padding) nogil:
    cdef:
        cnp.npy_intp bin

    bin = <cnp.npy_intp>(normalized)
    if bin < padding:
        return padding
    if bin > nbins - 1 - padding:
        return nbins - 1 - padding
    return bin


cdef inline double _cubic_spline(double x) nogil:
    cdef:
        double absx = -x if x<0 else x
        double sqrx = x*x

    if absx < 1.0:
        return ( 4.0 - 6.0 * sqrx + 3.0 * sqrx * absx ) / 6.0
    elif absx < 2.0:
        return ( 8.0 - 12 * absx + 6.0 * sqrx - sqrx * absx ) / 6.0
    return 0.0


cdef _joint_pdf_sparse(double[:] sval, double[:] mval, double smin,
                       double sdelta, double mmin, double mdelta,
                       int padding, double[:,:] pdf):
    cdef:
        int n = sval.shape[0]
        int offset, nbins
        cnp.npy_intp i, r, c
        double rn, cn
        double val, spline_arg, sum

    pdf[...] = 0
    sum = 0
    nbins = pdf.shape[0]

    with nogil:

        for i in range(n):
            rn = _bin_normalize(sval[i], smin, sdelta)
            r = _bin_index(rn, nbins, padding)
            cn = _bin_normalize(mval[i], mmin, mdelta)
            c = _bin_index(cn, nbins, padding)
            spline_arg = (c-1) - cn

            for offset in range(-1, 3):
                val = _cubic_spline(spline_arg)
                pdf[r, c + offset] += val
                sum += val
                spline_arg += 1.0

    return sum


cdef _joint_pdf_dense_3d(double[:,:,:] static, double[:,:,:] moving,
                       int[:,:,:] smask, int[:,:,:] mmask,
                       double smin, double sdelta, 
                       double mmin, double mdelta,
                       int padding, double[:,:] pdf):
    cdef:
        int nslices = static.shape[0]
        int nrows = static.shape[1]
        int ncols = static.shape[2]
        int offset, nbins
        cnp.npy_intp k, i, j, r, c
        double rn, cn
        double val, spline_arg, sum

    pdf[...] = 0
    sum = 0
    nbins = pdf.shape[0]
    with nogil:

        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if smask is not None and smask[k, i, j] == 0:
                        continue
                    if mmask is not None and mmask[k, i, j] == 0:
                        continue
                    rn = _bin_normalize(static[k, i, j], smin, sdelta)
                    r = _bin_index(rn, nbins, padding)
                    cn = _bin_normalize(moving[k, i, j], mmin, mdelta)
                    c = _bin_index(cn, nbins, padding)
                    spline_arg = (c-1) - cn

                    for offset in range(-1,3):
                        val = _cubic_spline(spline_arg)
                        pdf[r, c + offset] += val
                        sum += val
                        spline_arg += 1.0
    return sum


def joint_pdf_dense_3d(static, moving, int nbins):
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
    
    sdelta = (smax - smin)/(nbins - padding)
    mdelta = (mmax - mmin)/(nbins - padding)
    smin = smin/sdelta - padding
    mmin = mmin/sdelta - padding

    pdf = np.ndarray(shape = (nbins, nbins), dtype = np.float64)
    energy = _joint_pdf_dense_3d(static, moving, smask, mmask,
                                 smin, sdelta, mmin, mdelta,
                                 padding, pdf)
    if energy > 0:
        pdf /= energy
    return pdf
