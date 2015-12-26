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

def _compute_densities(int[:] x, double[:] y, int[:] mask):
    cdef:
        int i, n, minval, maxval
        int[:] cnt
    n = x.shape[0]
    minval = x[0]
    maxval = x[0]
    with nogil:
        for i in range(n):
            if x[i]<minval:
                minval = x[i]
            if x[i]>maxval:
                maxval = x[i]
    cnt = np.zeros(1+maxval, dtype=np.int32)
    with nogil:
        for i in range(n):
            cnt[x[i]] += 1
    return cnt

def count_masked_values(int[:] x, int[:] mask, int[:] out):
    cdef:
        int i, n
    n = x.shape[0]
    out[:] = 0
    with nogil:
        for i in range(n):
            if mask[i] != 0:
                out[x[i]] += 1
