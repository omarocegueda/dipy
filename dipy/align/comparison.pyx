#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
import numpy as np
cimport cython
cimport numpy as cnp
cdef extern from "dpy_math.h" nogil:
    int dpy_isinf(double)
    double floor(double)

cdef void _polynomial_fit(double[:] T, double[:] S, double[:] theta):

def iterate_polynomial_fit(double[:,:,:] T, double[:,:,:] S, int c,
                           double[:] theta, int initialize)
    r"""
    """
    cdef:
        cnp.npy_intp deg = len(theta)
        cnp.npy_intp n = T.size
    if initialize != 0:
        # Select c random samples
        np.random.choice(range(n))

    else:
        # select the inliers according to theta

