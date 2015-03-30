#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport cython
from fused_types cimport floating, number

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

def compute_cc_residuals(double[:,:,:] I, double[:,:,:] J, int radius, int mask0=1):
    cdef:
        int ns = I.shape[0]
        int nr = I.shape[1]
        int nc = I.shape[2]
        double s1, s2, t1, t2, wx, p, t, ave, worst
        double det0, det1, absdet0, absdet1
        double alpha, beta
        int i, j, k, s, r, c, intersect
        int start_k, end_k, start_i, end_i, start_j, end_j
        int regression_reference

        double[:,:,:] residuals = np.zeros((ns, nr, nc))

    with nogil:
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):

                    #Affine fit
                    s1 = 0
                    s2 = 0
                    t1 = 0
                    t2 = 0
                    wx = 0
                    p = 0
                    start_k = _int_max(0, s - radius)
                    end_k = _int_min(ns, 1 + s + radius)
                    for k in range(start_k, end_k):

                        start_i = _int_max(0, r - radius)
                        end_i = _int_min(nr, 1 + r + radius)
                        for i in range(start_i, end_i):

                            start_j = _int_max(0, c - radius)
                            end_j = _int_min(nc, 1 + c + radius)
                            for j in range(start_j, end_j):
                                if mask0 != 0:
                                    intersect = (I[k, i, j]>0) * (J[k, i, j]>0)
                                    if intersect == 0:
                                        continue
                                s1 += I[k, i, j]
                                s2 += I[k, i, j] * I[k, i, j]
                                t1 += J[k, i, j]
                                t2 += J[k, i, j] * J[k, i, j]
                                wx += 1
                                p += I[k, i, j] * J[k, i, j]

                    residuals[s, r, c] = 0
                    if wx<3:
                        continue

                    det0 = s2 * wx - (s1 * s1)
                    det1 = t2 * wx - (t1 * t1)

                    absdet0 = -det0 if det0 < 0 else det0
                    absdet1 = -det1 if det1 < 0 else det1

                    if absdet0 < 1e-6 and absdet1 < 1e-6:
                        continue

                    if absdet0 > absdet1:
                        regression_reference = 0
                        beta = (t1*s2 - (s1 * p)) / det0
                        alpha = (p - beta * s1) / s2
                    else:
                        regression_reference = 1
                        beta = (s1*t2 - (t1 * p)) / det1
                        alpha = (p - beta * t1) / t2

                    #Compute residuals
                    start_k = _int_max(0, s - radius)
                    end_k = _int_min(ns, 1 + s + radius)
                    for k in range(start_k, end_k):

                        start_i = _int_max(0, r - radius)
                        end_i = _int_min(nr, 1 + r + radius)
                        for i in range(start_i, end_i):

                            start_j = _int_max(0, c - radius)
                            end_j = _int_min(nc, 1 + c + radius)
                            for j in range(start_j, end_j):

                                if mask0 != 0:
                                    intersect = (I[k, i, j]>0) * (J[k, i, j]>0)
                                    if intersect == 0:
                                        continue

                                if regression_reference == 0:
                                    residuals[s, r, c] += ((alpha * I[k, i, j] + beta) - J[k, i, j]) ** 2
                                else:
                                    residuals[s, r, c] += ((alpha * J[k, i, j] + beta) - I[k, i, j]) ** 2

                    residuals[s, r, c] /= wx

    return residuals


def affine_fit(double[:] x, double[:] y, int force_ref=-1):
    cdef:
        int n = x.shape[0]
        int m = y.shape[0]
        double s1, s2, t1, t2, p, mse
        double det0, det1, absdet0, absdet1
        int i, regression_reference
        double[:] fit_x = np.ndarray(n, np.float64)
        double[:] fit_y = np.ndarray(n, np.float64)
    if n != m:
        raise ValueError("Arrays must have the same length")
    for i in range(n):
        s1 += x[i]
        s2 += x[i] * x[i]
        t1 += y[i]
        t2 += y[i] * y[i]
        p += x[i] * y[i]

    det0 = s2 * n - (s1 * s1)
    det1 = t2 * n - (t1 * t1)

    absdet0 = -det0 if det0 < 0 else det0
    absdet1 = -det1 if det1 < 0 else det1

    if absdet0 < 1e-6 and absdet1 < 1e-6:
        raise ValueError("Nearly constant arrays")

    if force_ref == 0 or ( force_ref != 1 and absdet0 > absdet1):
        regression_reference = 0
        beta = (t1*s2 - (s1 * p)) / det0
        alpha = (p - beta * s1) / s2
    else:
        regression_reference = 1
        beta = (s1*t2 - (t1 * p)) / det1
        alpha = (p - beta * t1) / t2

    mse = 0
    for i in range(n):
        if regression_reference == 0:
            fit_x[i] = x[i]
            fit_y[i] = alpha * x[i] + beta
            mse += (fit_y[i] - y[i])**2
        else:
            fit_y[i] = y[i]
            fit_x[i] = alpha * y[i] + beta
            mse += (fit_x[i] - x[i])**2
    mse /= n

    return alpha, beta, fit_x, fit_y, regression_reference, mse


def linear_fit(double[:] x, double[:] y, int force_ref=-1):
    cdef:
        int n = x.shape[0]
        int m = y.shape[0]
        double s2, t2, p, mse
        int i, regression_reference
        double[:] fit_x = np.ndarray(n, np.float64)
        double[:] fit_y = np.ndarray(n, np.float64)
    if n != m:
        raise ValueError("Arrays must have the same length")
    for i in range(n):
        s2 += x[i] * x[i]
        t2 += y[i] * y[i]
        p += x[i] * y[i]

    if s2 < 1e-6 and t2 < 1e-6:
        raise ValueError("Nearly constant arrays")

    if force_ref == 0 or ( force_ref != 1 and s2 > t2):
        regression_reference = 0
        alpha = p / s2
    else:
        regression_reference = 1
        alpha = p / t2

    mse = 0
    for i in range(n):
        if regression_reference == 0:
            fit_x[i] = x[i]
            fit_y[i] = alpha * x[i]
            mse += (fit_y[i] - y[i])**2
        else:
            fit_y[i] = y[i]
            fit_x[i] = alpha * y[i]
            mse += (fit_x[i] - x[i])**2
    mse /= n

    return alpha, fit_x, fit_y, regression_reference, mse


def compute_cc_residuals_noboundary(double[:,:,:] I, double[:,:,:] J, int radius):
    cdef:
        int ns = I.shape[0]
        int nr = I.shape[1]
        int nc = I.shape[2]
        double s1, s2, t1, t2, wx, p, t, ave, worst
        double det0, det1, absdet0, absdet1
        double alpha, beta
        int i, j, k, s, r, c, intersect
        int start_k, end_k, start_i, end_i, start_j, end_j
        int regression_reference

        double[:,:,:] residuals = np.zeros((ns, nr, nc))

    with nogil:
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):

                    #Affine fit
                    s1 = 0
                    s2 = 0
                    t1 = 0
                    t2 = 0
                    wx = 0
                    p = 0
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
                                t1 += J[k, i, j]
                                t2 += J[k, i, j] * J[k, i, j]
                                wx += 1
                                p += I[k, i, j] * J[k, i, j]

                    residuals[s, r, c] = 0
                    if wx<3:
                        continue

                    det0 = s2 * wx - (s1 * s1)
                    det1 = t2 * wx - (t1 * t1)

                    absdet0 = -det0 if det0 < 0 else det0
                    absdet1 = -det1 if det1 < 0 else det1

                    if absdet0 < 1e-6 and absdet1 < 1e-6:
                        continue

                    if absdet0 > absdet1:
                        regression_reference = 0
                        beta = (t1*s2 - (s1 * p)) / det0
                        alpha = (p - beta * s1) / s2
                    else:
                        regression_reference = 1
                        beta = (s1*t2 - (t1 * p)) / det1
                        alpha = (p - beta * t1) / t2

                    #Compute residuals
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

                                if regression_reference == 0:
                                    residuals[s, r, c] += ((alpha * I[k, i, j] + beta) - J[k, i, j]) ** 2
                                else:
                                    residuals[s, r, c] += ((alpha * J[k, i, j] + beta) - I[k, i, j]) ** 2

                    residuals[s, r, c] /= wx

    return residuals