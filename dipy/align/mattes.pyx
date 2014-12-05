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
matrix in standard notation, we save the matrix this way for efficiency

If the Jacobian is CONSTANT along its domain, the corresponding
jacobian_function must RETURN 1. Otherwise it must RETURN 0. This
information is used by the optimizer to avoid making unnecessary
function calls
"""


cdef inline void mult_mat_3d(double[:,:] A, double[:,:] B, double[:,:] C) nogil:
    r''' Multiplies two 3x3 matrices A, B and writes the product in C
    '''
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
    r''' Transposed Jacobian matrix of a rotation transform at theta
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
    # This Jacobian depends on x (it's not constant): return 0
    return 0

cdef int scale_jacobian_3d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r""" Transposed Jacobian matrix of the isotropic scale transform
    The transformation is given by:
    
    T(x) = (s*x0, s*x1, s*x2) 

    The derivative w.r.t. s is T'(x) = [x0, x1, x2]
    """
    J[0,:] = x[:]
    # This Jacobian depends on x (it's not constant): return 0
    return 0


cdef int translation_jacobian_3d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r""" Transposed Jacobian matrix of the translation transform
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
    # This Jacobian does not depend on x (it's constant): return 1
    return 1


cdef int affine_jacobian_3d(double[:] theta, double[:] x, double[:,:] J) nogil:
    r""" Transposed Jacobian matrix of the affine transform
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
    # This Jacobian depends on x (it's not constant): return 0
    return 0

    
    
    
    
    
    
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

    
cdef _joint_pdf_gradient_dense_3d(double[:] theta, jacobian_function jacobian, 
                                  double[:,:] grid_to_space, double[:,:,:,:] mgradient,
                                  double[:,:,:] static, double[:,:,:] moving,
                                  int[:,:,:] smask, int[:,:,:] mmask,
                                  double smin, double sdelta, 
                                  double mmin, double mdelta,
                                  int padding, double[:,:,:] grad_pdf):
    cdef:
        int nslices = static.shape[0]
        int nrows = static.shape[1]
        int ncols = static.shape[2]
        int n = theta.shape[0]
        int offset, nbins, constant_jacobian=0
        cnp.npy_intp k, i, j, r, c
        double rn, cn
        double val, spline_arg, sum
        double[:,:] J = np.ndarray(shape=(n, 3), dtype=np.float64)

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
                    
                    if constant_jacobian == 0:
                        constant_jacobian = jacobian(theta, x, J)
                    
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
    
    
    
    
    
class MattesPDF(object):
    
    def __init__(self, nbins, static, moving, smask=None, mmask=None, padding=2):
        self.nbins = nbins
        self.padding = padding

        if smask is None:
            smask = np.array(static > 0).astype(np.int32)
        if mmask is None:
            mmask = np.array(moving > 0).astype(np.int32)

        self.smin = np.min(static[static>0])
        self.smax = np.max(static[static>0])
        self.mmin = np.min(moving[moving>0])
        self.mmax = np.max(moving[moving>0])
        
        self.sdelta = (self.smax - self.smin)/(nbins - padding)
        self.mdelta = (self.mmax - self.mmin)/(nbins - padding)
        self.smin = self.smin/self.sdelta - padding
        self.mmin = self.mmin/self.sdelta - padding

        self.pdf = np.ndarray(shape = (nbins, nbins), dtype = np.float64)
        energy = _joint_pdf_dense_3d(static, moving, smask, mmask,
                                     self.smin, self.sdelta, self.mmin, self.mdelta,
                                     self.padding, self.pdf)
        if energy > 0:
            self.pdf /= energy

    def update_pdf_dense(static, moving, smask, mmask):
        energy = _joint_pdf_dense_3d(static, moving, smask, mmask,
                                     self.smin, self.sdelta, self.mmin, self.mdelta,
                                     self.padding, self.pdf)
        if energy > 0:
            self.pdf /= energy

    def update_pdf_sparse(sval, mval):
        energy = _joint_pdf_sparse(sval, mval, self.smin, self.sdelta, 
                                   self.mmin, self.mdelta, self.padding, self.pdf)
        if energy > 0:
            self.pdf /= energy
    
    def update_gradient_dense(theta, jacobian, mgradient, static, moving, smask, mmask):
        pass

    def update_gradient_dense(theta, jacobian, mgradient, static, moving, smask, mmask):
        pass
    
    
    
    
    
    
    
    
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



    
    
cdef _joint_pdf_gradient_sparse(double[:] theta, jacobian_function jacobian,
                                double[:] sval, double smin, double sdelta,
                                double[:] mval, double mmin, double mdelta,
                                double[:,:] X, double[:,:] grad,
                                double[:,:] pdf, int padding):
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
                val = _cubic_spline_derivative(spline_arg)
                pdf[r, c + offset] += val
                sum += val
                spline_arg += 1.0

    return sum





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
