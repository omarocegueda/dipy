""" Utility functions used by the Cross Correlation (CC) metric """

import numpy as np
cimport cython
cimport numpy as cnp
from fused_types cimport floating


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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def precompute_cc_factors_3d_old(floating[:, :, :] static, floating[:, :, :] moving,
                             cnp.npy_intp radius):
    r"""Precomputations to quickly compute the gradient of the CC Metric

    Pre-computes the separate terms of the cross correlation metric and image
    norms at each voxel considering a neighborhood of the given radius to
    efficiently compute the gradient of the metric with respect to the
    deformation field [Avants08][Avants11]

    Parameters
    ----------
    static : array, shape (S, R, C)
        the static volume, which also defines the reference registration domain
    moving : array, shape (S, R, C)
        the moving volume (notice that both images must already be in a common
        reference domain, i.e. the same S, R, C)
    radius : the radius of the neighborhood (cube of (2 * radius + 1)^3 voxels)

    Returns
    -------
    factors : array, shape (S, R, C, 5)
        the precomputed cross correlation terms:
        factors[:,:,:,0] : static minus its mean value along the neighborhood
        factors[:,:,:,1] : moving minus its mean value along the neighborhood
        factors[:,:,:,2] : sum of the pointwise products of static and moving
                           along the neighborhood
        factors[:,:,:,3] : sum of sq. values of static along the neighborhood
        factors[:,:,:,4] : sum of sq. values of moving along the neighborhood

    References
    ----------
    [Avants08] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2008)
               Symmetric Diffeomorphic Image Registration with
               Cross-Correlation: Evaluating Automated Labeling of Elderly and
               Neurodegenerative Brain, Med Image Anal. 12(1), 26-41.
    [Avants11] Avants, B. B., Tustison, N., & Song, G. (2011).
               Advanced Normalization Tools ( ANTS ), 1-35.
    """
    cdef:
        cnp.npy_intp side = 2 * radius + 1
        cnp.npy_intp ns = static.shape[0]
        cnp.npy_intp nr = static.shape[1]
        cnp.npy_intp nc = static.shape[2]
        cnp.npy_intp s, r, c, k, i, j, t, q, qq, firstc, lastc, firstr, lastr
        double Imean, Jmean
        floating[:, :, :, :] factors = np.zeros((ns, nr, nc, 5),
                                                dtype=np.asarray(static).dtype)
        double[:, :] lines = np.zeros((6, side), dtype=np.float64)
        double[:] sums = np.zeros((6,), dtype=np.float64)

    with nogil:
        for r in range(nr):
            firstr = _int_max(0, r - radius)
            lastr = _int_min(nr - 1, r + radius)
            for c in range(nc):
                firstc = _int_max(0, c - radius)
                lastc = _int_min(nc - 1, c + radius)
                # compute factors for line [:,r,c]
                for t in range(6):
                    for q in range(side):
                        lines[t,q] = 0

                # Compute all slices and set the sums on the fly
                # compute each slice [k, i={r-radius..r+radius}, j={c-radius,
                # c+radius}]
                for k in range(ns):
                    q = k % side
                    for t in range(6):
                        sums[t] -= lines[t, q]
                        lines[t, q] = 0
                    for i in range(firstr, lastr + 1):
                        for j in range(firstc, lastc + 1):
                            lines[SI, q] += static[k, i, j]
                            lines[SI2, q] += static[k, i, j] * static[k, i, j]
                            lines[SJ, q] += moving[k, i, j]
                            lines[SJ2, q] += moving[k, i, j] * moving[k, i, j]
                            lines[SIJ, q] += static[k, i, j] * moving[k, i, j]
                            lines[CNT, q] += 1

                    for t in range(6):
                        sums[t] = 0
                        for qq in range(side):
                            sums[t] += lines[t, qq]
                    if(k >= radius):
                        # s is the voxel that is affected by the cube with
                        # slices [s - radius..s + radius, :, :]
                        s = k - radius
                        Imean = sums[SI] / sums[CNT]
                        Jmean = sums[SJ] / sums[CNT]
                        factors[s, r, c, 0] = static[s, r, c] - Imean
                        factors[s, r, c, 1] = moving[s, r, c] - Jmean
                        factors[s, r, c, 2] = (sums[SIJ] - Jmean * sums[SI] -
                            Imean * sums[SJ] + sums[CNT] * Jmean * Imean)
                        factors[s, r, c, 3] = (sums[SI2] - Imean * sums[SI] -
                            Imean * sums[SI] + sums[CNT] * Imean * Imean)
                        factors[s, r, c, 4] = (sums[SJ2] - Jmean * sums[SJ] -
                            Jmean * sums[SJ] + sums[CNT] * Jmean * Jmean)
                # Finally set the values at the end of the line
                for s in range(ns - radius, ns):
                    # this would be the last slice to be processed for voxel
                    # [s, r, c], if it existed
                    k = s + radius
                    q = k % side
                    for t in range(6):
                        sums[t] -= lines[t, q]
                    Imean = sums[SI] / sums[CNT]
                    Jmean = sums[SJ] / sums[CNT]
                    factors[s, r, c, 0] = static[s, r, c] - Imean
                    factors[s, r, c, 1] = moving[s, r, c] - Jmean
                    factors[s, r, c, 2] = (sums[SIJ] - Jmean * sums[SI] -
                        Imean * sums[SJ] + sums[CNT] * Jmean * Imean)
                    factors[s, r, c, 3] = (sums[SI2] - Imean * sums[SI] -
                        Imean * sums[SI] + sums[CNT] * Imean * Imean)
                    factors[s, r, c, 4] = (sums[SJ2] - Jmean * sums[SJ] -
                        Jmean * sums[SJ] + sums[CNT] * Jmean * Jmean)
    return factors


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def precompute_cc_factors_3d_test(floating[:, :, :] static,
                                  floating[:, :, :] moving, int radius):
    r"""Precomputations to quickly compute the gradient of the CC Metric

    This version of precompute_cc_factors_3d is for testing purposes, it
    directly computes the local cross-correlation factors without any
    optimization, so it is less error-prone than the accelerated version.
    """
    cdef:
        cnp.npy_intp ns = static.shape[0]
        cnp.npy_intp nr = static.shape[1]
        cnp.npy_intp nc = static.shape[2]
        cnp.npy_intp s, r, c, k, i, j, t, firstc, lastc, firstr, lastr, firsts, lasts
        double Imean, Jmean
        floating[:, :, :, :] factors = np.zeros((ns, nr, nc, 5),
                                                dtype=np.asarray(static).dtype)
        double[:] sums = np.zeros((6,), dtype=np.float64)

    with nogil:
        for s in range(ns):
            firsts = _int_max(0, s - radius)
            lasts = _int_min(ns - 1, s + radius)
            for r in range(nr):
                firstr = _int_max(0, r - radius)
                lastr = _int_min(nr - 1, r + radius)
                for c in range(nc):
                    firstc = _int_max(0, c - radius)
                    lastc = _int_min(nc - 1, c + radius)
                    for t in range(6):
                        sums[t] = 0
                    for k in range(firsts, 1 + lasts):
                        for i in range(firstr, 1 + lastr):
                            for j in range(firstc, 1 + lastc):
                                sums[SI] += static[k, i, j]
                                sums[SI2] += static[k, i,j]**2
                                sums[SJ] += moving[k, i,j]
                                sums[SJ2] += moving[k, i,j]**2
                                sums[SIJ] += static[k,i,j]*moving[k, i,j]
                                sums[CNT] += 1
                    Imean = sums[SI] / sums[CNT]
                    Jmean = sums[SJ] / sums[CNT]
                    factors[s, r, c, 0] = static[s, r, c] - Imean
                    factors[s, r, c, 1] = moving[s, r, c] - Jmean
                    factors[s, r, c, 2] = (sums[SIJ] - Jmean * sums[SI] -
                        Imean * sums[SJ] + sums[CNT] * Jmean * Imean)
                    factors[s, r, c, 3] = (sums[SI2] - Imean * sums[SI] -
                        Imean * sums[SI] + sums[CNT] * Imean * Imean)
                    factors[s, r, c, 4] = (sums[SJ2] - Jmean * sums[SJ] -
                        Jmean * sums[SJ] + sums[CNT] * Jmean * Jmean)
    return factors


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_cc_forward_step_3d(floating[:, :, :, :] grad_static,
                               floating[:, :, :, :] factors,
                               cnp.npy_intp radius):
    r"""Gradient of the CC Metric w.r.t. the forward transformation

    Computes the gradient of the Cross Correlation metric for symmetric
    registration (SyN) [Avants08] w.r.t. the displacement associated to
    the moving volume ('forward' step) as in [Avants11]

    Parameters
    ----------
    grad_static : array, shape (S, R, C, 3)
        the gradient of the static volume
    factors : array, shape (S, R, C, 5)
        the precomputed cross correlation terms obtained via
        precompute_cc_factors_3d
    radius : int
        the radius of the neighborhood used for the CC metric when
        computing the factors. The returned vector field will be
        zero along a boundary of width radius voxels.

    Returns
    -------
    out : array, shape (S, R, C, 3)
        the gradient of the cross correlation metric with respect to the
        displacement associated to the moving volume
    energy : the cross correlation energy (data term) at this iteration

    References
    ----------
    [Avants08] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2008)
               Symmetric Diffeomorphic Image Registration with
               Cross-Correlation: Evaluating Automated Labeling of Elderly and
               Neurodegenerative Brain, Med Image Anal. 12(1), 26-41.
    [Avants11] Avants, B. B., Tustison, N., & Song, G. (2011).
               Advanced Normalization Tools ( ANTS ), 1-35.
    """
    cdef:
        cnp.npy_intp ns = grad_static.shape[0]
        cnp.npy_intp nr = grad_static.shape[1]
        cnp.npy_intp nc = grad_static.shape[2]
        double energy = 0
        cnp.npy_intp s,r,c
        double Ii, Ji, sfm, sff, smm, localCorrelation, temp
        floating[:, :, :, :] out = np.zeros((ns, nr, nc, 3),
                                            dtype=np.asarray(grad_static).dtype)
    with nogil:
        for s in range(radius, ns-radius):
            for r in range(radius, nr-radius):
                for c in range(radius, nc-radius):
                    Ii = factors[s, r, c, 0]
                    Ji = factors[s, r, c, 1]
                    sfm = factors[s, r, c, 2]
                    sff = factors[s, r, c, 3]
                    smm = factors[s, r, c, 4]
                    if(sff == 0.0 or smm == 0.0):
                        continue
                    localCorrelation = 0
                    if(sff * smm > 1e-5):
                        localCorrelation = sfm * sfm / (sff * smm)
                    if(localCorrelation < 1):  # avoid bad values...
                        energy -= localCorrelation
                    temp = 2.0 * sfm / (sff * smm) * (Ji - sfm / sff * Ii)
                    out[s, r, c, 0] -= temp * grad_static[s, r, c, 0]
                    out[s, r, c, 1] -= temp * grad_static[s, r, c, 1]
                    out[s, r, c, 2] -= temp * grad_static[s, r, c, 2]
    return out, energy

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def compute_cc_backward_step_3d(floating[:, :, :, :] grad_moving,
                                floating[:, :, :, :] factors,
                                cnp.npy_intp radius):
    r"""Gradient of the CC Metric w.r.t. the backward transformation

    Computes the gradient of the Cross Correlation metric for symmetric
    registration (SyN) [Avants08] w.r.t. the displacement associated to
    the static volume ('backward' step) as in [Avants11]

    Parameters
    ----------
    grad_moving : array, shape (S, R, C, 3)
        the gradient of the moving volume
    factors : array, shape (S, R, C, 5)
        the precomputed cross correlation terms obtained via
        precompute_cc_factors_3d
    radius : int
        the radius of the neighborhood used for the CC metric when
        computing the factors. The returned vector field will be
        zero along a boundary of width radius voxels.

    Returns
    -------
    out : array, shape (S, R, C, 3)
        the gradient of the cross correlation metric with respect to the
        displacement associated to the static volume
    energy : the cross correlation energy (data term) at this iteration

    References
    ----------
    [Avants08] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2008)
               Symmetric Diffeomorphic Image Registration with
               Cross-Correlation: Evaluating Automated Labeling of Elderly and
               Neurodegenerative Brain, Med Image Anal. 12(1), 26-41.
    [Avants11] Avants, B. B., Tustison, N., & Song, G. (2011).
               Advanced Normalization Tools ( ANTS ), 1-35.
    """
    ftype = np.asarray(grad_moving).dtype
    cdef:
        cnp.npy_intp ns = grad_moving.shape[0]
        cnp.npy_intp nr = grad_moving.shape[1]
        cnp.npy_intp nc = grad_moving.shape[2]
        cnp.npy_intp s,r,c
        double energy = 0
        double Ii, Ji, sfm, sff, smm, localCorrelation, temp
        floating[:, :, :, :] out = np.zeros((ns, nr, nc, 3), dtype=ftype)

    with nogil:

        for s in range(radius, ns-radius):
            for r in range(radius, nr-radius):
                for c in range(radius, nc-radius):
                    Ii = factors[s, r, c, 0]
                    Ji = factors[s, r, c, 1]
                    sfm = factors[s, r, c, 2]
                    sff = factors[s, r, c, 3]
                    smm = factors[s, r, c, 4]
                    if(sff == 0.0 or smm == 0.0):
                        continue
                    localCorrelation = 0
                    if(sff * smm > 1e-5):
                        localCorrelation = sfm * sfm / (sff * smm)
                    if(localCorrelation < 1):  # avoid bad values...
                        energy -= localCorrelation
                    temp = 2.0 * sfm / (sff * smm) * (Ii - sfm / smm * Ji)
                    out[s, r, c, 0] -= temp * grad_moving[s, r, c, 0]
                    out[s, r, c, 1] -= temp * grad_moving[s, r, c, 1]
                    out[s, r, c, 2] -= temp * grad_moving[s, r, c, 2]
    return out, energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def precompute_cc_factors_2d(floating[:, :] static, floating[:, :] moving,
                             cnp.npy_intp radius):
    r"""Precomputations to quickly compute the gradient of the CC Metric

    Pre-computes the separate terms of the cross correlation metric [Avants08]
    and image norms at each voxel considering a neighborhood of the given
    radius to efficiently [Avants11] compute the gradient of the metric with
    respect to the deformation field.

    Parameters
    ----------
    static : array, shape (R, C)
        the static volume, which also defines the reference registration domain
    moving : array, shape (R, C)
        the moving volume (notice that both images must already be in a common
        reference domain, i.e. the same R, C)
    radius : the radius of the neighborhood(square of (2 * radius + 1)^2 voxels)

    Returns
    -------
    factors : array, shape (R, C, 5)
        the precomputed cross correlation terms:
        factors[:,:,0] : static minus its mean value along the neighborhood
        factors[:,:,1] : moving minus its mean value along the neighborhood
        factors[:,:,2] : sum of the pointwise products of static and moving
                           along the neighborhood
        factors[:,:,3] : sum of sq. values of static along the neighborhood
        factors[:,:,4] : sum of sq. values of moving along the neighborhood

    References
    ----------
    [Avants08] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2008)
               Symmetric Diffeomorphic Image Registration with
               Cross-Correlation: Evaluating Automated Labeling of Elderly and
               Neurodegenerative Brain, Med Image Anal. 12(1), 26-41.
    [Avants11] Avants, B. B., Tustison, N., & Song, G. (2011).
               Advanced Normalization Tools ( ANTS ), 1-35.
    """
    ftype = np.asarray(static).dtype
    cdef:
        cnp.npy_intp side = 2 * radius + 1
        cnp.npy_intp nr = static.shape[0]
        cnp.npy_intp nc = static.shape[1]
        cnp.npy_intp r, c, i, j, t, q, qq, firstc, lastc
        double Imean, Jmean
        floating[:, :, :] factors = np.zeros((nr, nc, 5), dtype=ftype)
        double[:, :] lines = np.zeros((6, side), dtype=np.float64)
        double[:] sums = np.zeros((6,), dtype=np.float64)

    with nogil:

        for c in range(nc):
            firstc = _int_max(0, c - radius)
            lastc = _int_min(nc - 1, c + radius)
            # compute factors for row [:,c]
            for t in range(6):
                for q in range(side):
                    lines[t,q] = 0
            # Compute all rows and set the sums on the fly
            # compute row [i, j = {c-radius, c + radius}]
            for i in range(nr):
                q = i % side
                for t in range(6):
                    lines[t, q] = 0
                for j in range(firstc, lastc + 1):
                    lines[SI, q] += static[i, j]
                    lines[SI2, q] += static[i, j] * static[i, j]
                    lines[SJ, q] += moving[i, j]
                    lines[SJ2, q] += moving[i, j] * moving[i, j]
                    lines[SIJ, q] += static[i, j] * moving[i, j]
                    lines[CNT, q] += 1

                for t in range(6):
                    sums[t] = 0
                    for qq in range(side):
                        sums[t] += lines[t, qq]
                if(i >= radius):
                    # r is the pixel that is affected by the cube with slices
                    # [r - radius.. r + radius, :]
                    r = i - radius
                    Imean = sums[SI] / sums[CNT]
                    Jmean = sums[SJ] / sums[CNT]
                    factors[r, c, 0] = static[r, c] - Imean
                    factors[r, c, 1] = moving[r, c] - Jmean
                    factors[r, c, 2] = (sums[SIJ] - Jmean * sums[SI] -
                        Imean * sums[SJ] + sums[CNT] * Jmean * Imean)
                    factors[r, c, 3] = (sums[SI2] - Imean * sums[SI] -
                        Imean * sums[SI] + sums[CNT] * Imean * Imean)
                    factors[r, c, 4] = (sums[SJ2] - Jmean * sums[SJ] -
                        Jmean * sums[SJ] + sums[CNT] * Jmean * Jmean)
            # Finally set the values at the end of the line
            for r in range(nr - radius, nr):
                # this would be the last slice to be processed for pixel
                # [r, c], if it existed
                i = r + radius
                q = i % side
                for t in range(6):
                    sums[t] -= lines[t, q]
                Imean = sums[SI] / sums[CNT]
                Jmean = sums[SJ] / sums[CNT]
                factors[r, c, 0] = static[r, c] - Imean
                factors[r, c, 1] = moving[r, c] - Jmean
                factors[r, c, 2] = (sums[SIJ] - Jmean * sums[SI] -
                    Imean * sums[SJ] + sums[CNT] * Jmean * Imean)
                factors[r, c, 3] = (sums[SI2] - Imean * sums[SI] -
                    Imean * sums[SI] + sums[CNT] * Imean * Imean)
                factors[r, c, 4] = (sums[SJ2] - Jmean * sums[SJ] -
                    Jmean * sums[SJ] + sums[CNT] * Jmean * Jmean)
    return factors


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def precompute_cc_factors_2d_test(floating[:, :] static, floating[:, :] moving,
                                  cnp.npy_intp radius):
    r"""Precomputations to quickly compute the gradient of the CC Metric

    This version of precompute_cc_factors_2d is for testing purposes, it
    directly computes the local cross-correlation without any optimization.
    """
    ftype = np.asarray(static).dtype
    cdef:
        cnp.npy_intp nr = static.shape[0]
        cnp.npy_intp nc = static.shape[1]
        cnp.npy_intp r, c, i, j, t, firstr, lastr, firstc, lastc
        double Imean, Jmean
        floating[:, :, :] factors = np.zeros((nr, nc, 5), dtype=ftype)
        double[:] sums = np.zeros((6,), dtype=np.float64)

    with nogil:

        for r in range(nr):
            firstr = _int_max(0, r - radius)
            lastr = _int_min(nr - 1, r + radius)
            for c in range(nc):
                firstc = _int_max(0, c - radius)
                lastc = _int_min(nc - 1, c + radius)
                for t in range(6):
                    sums[t]=0
                for i in range(firstr, 1 + lastr):
                    for j in range(firstc, 1+lastc):
                        sums[SI] += static[i, j]
                        sums[SI2] += static[i,j]**2
                        sums[SJ] += moving[i,j]
                        sums[SJ2] += moving[i,j]**2
                        sums[SIJ] += static[i,j]*moving[i,j]
                        sums[CNT] += 1
                Imean = sums[SI] / sums[CNT]
                Jmean = sums[SJ] / sums[CNT]
                factors[r, c, 0] = static[r, c] - Imean
                factors[r, c, 1] = moving[r, c] - Jmean
                factors[r, c, 2] = (sums[SIJ] - Jmean * sums[SI] -
                    Imean * sums[SJ] + sums[CNT] * Jmean * Imean)
                factors[r, c, 3] = (sums[SI2] - Imean * sums[SI] -
                    Imean * sums[SI] + sums[CNT] * Imean * Imean)
                factors[r, c, 4] = (sums[SJ2] - Jmean * sums[SJ] -
                    Jmean * sums[SJ] + sums[CNT] * Jmean * Jmean)
    return factors


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def compute_cc_forward_step_2d(floating[:, :, :] grad_static,
                               floating[:, :, :] factors,
                               cnp.npy_intp radius):
    r"""Gradient of the CC Metric w.r.t. the forward transformation

    Computes the gradient of the Cross Correlation metric for symmetric
    registration (SyN) [Avants08] w.r.t. the displacement associated to
    the moving image ('backward' step) as in [Avants11]

    Parameters
    ----------
    grad_static : array, shape (R, C, 2)
        the gradient of the static image
    factors : array, shape (R, C, 5)
        the precomputed cross correlation terms obtained via
        precompute_cc_factors_2d

    Returns
    -------
    out : array, shape (R, C, 2)
        the gradient of the cross correlation metric with respect to the
        displacement associated to the moving image
    energy : the cross correlation energy (data term) at this iteration

    Notes
    -----
    Currently, the gradient of the static image is not being used, but some
    authors suggest that symmetrizing the gradient by including both, the moving
    and static gradients may improve the registration quality. We are leaving
    this parameters as a placeholder for future investigation

    References
    ----------
    [Avants08] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2008)
               Symmetric Diffeomorphic Image Registration with
               Cross-Correlation: Evaluating Automated Labeling of Elderly and
               Neurodegenerative Brain, Med Image Anal. 12(1), 26-41.
    [Avants11] Avants, B. B., Tustison, N., & Song, G. (2011).
               Advanced Normalization Tools ( ANTS ), 1-35.
    """
    cdef:
        cnp.npy_intp nr = grad_static.shape[0]
        cnp.npy_intp nc = grad_static.shape[1]
        double energy = 0
        cnp.npy_intp r,c
        double Ii, Ji, sfm, sff, smm, localCorrelation, temp
        floating[:, :, :] out = np.zeros((nr, nc, 2),
                                         dtype=np.asarray(grad_static).dtype)
    with nogil:

        for r in range(radius, nr-radius):
            for c in range(radius, nc-radius):
                Ii = factors[r, c, 0]
                Ji = factors[r, c, 1]
                sfm = factors[r, c, 2]
                sff = factors[r, c, 3]
                smm = factors[r, c, 4]
                if(sff == 0.0 or smm == 0.0):
                    continue
                localCorrelation = 0
                if(sff * smm > 1e-5):
                    localCorrelation = sfm * sfm / (sff * smm)
                if(localCorrelation < 1):  # avoid bad values...
                    energy -= localCorrelation
                temp = 2.0 * sfm / (sff * smm) * (Ji - sfm / sff * Ii)
                out[r, c, 0] -= temp * grad_static[r, c, 0]
                out[r, c, 1] -= temp * grad_static[r, c, 1]
    return out, energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_cc_backward_step_2d(floating[:, :, :] grad_moving,
                                floating[:, :, :] factors,
                                cnp.npy_intp radius):
    r"""Gradient of the CC Metric w.r.t. the backward transformation

    Computes the gradient of the Cross Correlation metric for symmetric
    registration (SyN) [Avants08] w.r.t. the displacement associated to
    the static image ('forward' step) as in [Avants11]

    Parameters
    ----------
    grad_moving : array, shape (R, C, 2)
        the gradient of the moving image
    factors : array, shape (R, C, 5)
        the precomputed cross correlation terms obtained via
        precompute_cc_factors_2d

    Returns
    -------
    out : array, shape (R, C, 2)
        the gradient of the cross correlation metric with respect to the
        displacement associated to the static image
    energy : the cross correlation energy (data term) at this iteration

    References
    ----------
    [Avants08] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2008)
               Symmetric Diffeomorphic Image Registration with
               Cross-Correlation: Evaluating Automated Labeling of Elderly and
               Neurodegenerative Brain, Med Image Anal. 12(1), 26-41.
    [Avants11] Avants, B. B., Tustison, N., & Song, G. (2011).
               Advanced Normalization Tools ( ANTS ), 1-35.
    """
    ftype = np.asarray(grad_moving).dtype
    cdef:
        cnp.npy_intp nr = grad_moving.shape[0]
        cnp.npy_intp nc = grad_moving.shape[1]
        cnp.npy_intp r,c
        double energy = 0
        double Ii, Ji, sfm, sff, smm, localCorrelation, temp
        floating[:, :, :] out = np.zeros((nr, nc, 2),
                                             dtype=ftype)

    with nogil:

        for r in range(radius, nr-radius):
            for c in range(radius, nc-radius):
                Ii = factors[r, c, 0]
                Ji = factors[r, c, 1]
                sfm = factors[r, c, 2]
                sff = factors[r, c, 3]
                smm = factors[r, c, 4]
                if(sff == 0.0 or smm == 0.0):
                    continue
                localCorrelation = 0
                if(sff * smm > 1e-5):
                    localCorrelation = sfm * sfm / (sff * smm)
                if(localCorrelation < 1):  # avoid bad values...
                    energy -= localCorrelation
                temp = 2.0 * sfm / (sff * smm) * (Ii - sfm / smm * Ji)
                out[r, c, 0] -= temp * grad_moving[r, c, 0]
                out[r, c, 1] -= temp * grad_moving[r, c, 1]
    return out, energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def window_sums(floating[:,:,:] v, int d, int exponent=1):
    ftype = np.asarray(v).dtype
    cdef:
        cnp.npy_intp ns = v.shape[0]
        cnp.npy_intp nr = v.shape[1]
        cnp.npy_intp nc = v.shape[2]
        cnp.npy_intp s, r, c, i, j, k
        floating[:,:,:] out = np.empty((ns, nr, nc), dtype=ftype)
    with nogil:
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):
                    # New corner
                    out[s, r, c] = v[s,r,c]**exponent
                    # Add signed sub-volumes
                    if s>0:
                        out[s, r, c] += out[s-1,r,c]
                        if r>0:
                            out[s,r,c] -= out[s-1,r-1,c]
                            if c>0:
                                out[s,r,c] += out[s-1,r-1,c-1]
                        if c>0:
                            out[s,r,c] -= out[s-1,r,c-1]
                    if(r>0):
                        out[s, r, c] += out[s,r-1,c]
                        if(c>0):
                            out[s, r, c] -= out[s,r-1,c-1]
                    if(c>0):
                        out[s, r, c] += out[s,r,c-1]

                    # Add signed corners
                    if s>=d:
                        out[s,r,c] -= v[s-d,r,c]**exponent
                        if r>=d:
                            out[s,r,c] += v[s-d,r-d,c]**exponent
                            if c>=d:
                                out[s,r,c] -= v[s-d,r-d,c-d]**exponent
                        if c>=d:
                            out[s,r,c] += v[s-d,r,c-d]**exponent
                    if r>=d:
                        out[s,r,c] -= v[s,r-d,c]**exponent
                        if c>=d:
                            out[s,r,c] += v[s,r-d,c-d]**exponent
                    if c>=d:
                        out[s,r,c] -= v[s,r,c-d]**exponent

    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def window_sums_integral(floating[:,:,:] v, int d, int exponent=1):
    ftype = np.asarray(v).dtype
    cdef:
        cnp.npy_intp ns = v.shape[0]
        cnp.npy_intp nr = v.shape[1]
        cnp.npy_intp nc = v.shape[2]
        cnp.npy_intp s, r, c
        floating[:] c1 = np.zeros((nc,), dtype=ftype)
        floating[:,:] c2 = np.zeros((nr, nc), dtype=ftype)
        floating[:,:,:] iv = np.empty((ns, nr, nc), dtype=ftype)
        floating[:,:,:] out = np.empty((ns, nr, nc), dtype=ftype)
    with nogil:
        # Precompute integral volume
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):
                    # New corner
                    c1[c] = v[s, r, c]**exponent
                    # Add to prevous c1 (if it exists)
                    if c>0:
                        c1[c] += c1[c-1]
                    # Initialize c2 with current c1
                    c2[r, c] = c1[c]
                    # Accumulate previous c2 (if it exists)
                    if r>0:
                        c2[r, c] += c2[r-1, c]
                    # Initialize iv with current c2
                    iv[s, r, c] = c2[r, c]
                    # Accumulate with previous iv (if it exists)
                    if s>0:
                        iv[s, r, c] += iv[s-1, r, c]

        # Use the integral volume to compute the integral over rectangles
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):
                    out[s, r, c] = iv[s, r, c]
                    if s >= d:
                        out[s, r, c] -= iv[s-d, r, c]
                        if r >= d:
                            out[s, r, c] += iv[s-d, r-d, c]
                            if c >= d:
                                out[s, r, c] -= iv[s-d, r-d, c-d]
                        if c >= d:
                            out[s, r, c] += iv[s-d, r, c-d]
                    if r >= d:
                        out[s, r, c] -= iv[s, r-d, c]
                        if c >= d:
                            out[s, r, c] += iv[s, r - d, c - d]
                    if c >= d:
                        out[s, r, c] -= iv[s, r, c - d]
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def window_sums_direct(floating[:,:,:] v, int d, int exponent=1):
    ftype = np.asarray(v).dtype
    cdef:
        cnp.npy_intp ns = v.shape[0]
        cnp.npy_intp nr = v.shape[1]
        cnp.npy_intp nc = v.shape[2]
        cnp.npy_intp firstc, firstr, firsts
        cnp.npy_intp s, r, c, i, j, k
        floating[:,:,:] out = np.empty((ns, nr, nc), dtype=ftype)
    with nogil:
        for s in range(ns):
            firsts = _int_max(0, s - d + 1)
            for r in range(nr):
                firstr = _int_max(0, r - d + 1)
                for c in range(nc):
                    firstc = _int_max(0, c - d + 1)
                    out[s, r, c] = 0
                    for k in range(firsts, 1 + s):
                        for i in range(firstr, 1 + r):
                            for j in range(firstc, 1 + c):
                                out[s, r, c] += v[k, i, j]**exponent
    return out



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int _mod(int x, int m)nogil:
    if x<0:
        return x + m
    return x

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void _increment_factors(double[:,:,:,:] factors, floating[:,:,:] moving, floating[:,:,:] static,
                                    int ss, int rr, int cc, int s, int r, int c, int weight)nogil:
    cdef:
        double sval
        double mval
    if s>=moving.shape[0] or r>=moving.shape[1] or c>=moving.shape[2]:
        if weight == 0:
            factors[ss, rr, cc, SI] = 0
            factors[ss, rr, cc, SI2] = 0
            factors[ss, rr, cc, SJ] = 0
            factors[ss, rr, cc, SJ2] = 0
            factors[ss, rr, cc, SIJ] = 0
    else:
        sval = static[s,r,c]
        mval = moving[s,r,c]
        if weight == 0:
            factors[ss, rr, cc, SI] = sval
            factors[ss, rr, cc, SI2] = sval*sval
            factors[ss, rr, cc, SJ] = mval
            factors[ss, rr, cc, SJ2] = mval*mval
            factors[ss, rr, cc, SIJ] = sval*mval
        elif weight == -1:
            factors[ss, rr, cc, SI] -= sval
            factors[ss, rr, cc, SI2] -= sval*sval
            factors[ss, rr, cc, SJ] -= mval
            factors[ss, rr, cc, SJ2] -= mval*mval
            factors[ss, rr, cc, SIJ] -= sval*mval
        elif weight == 1:
            factors[ss, rr, cc, SI] += sval
            factors[ss, rr, cc, SI2] += sval*sval
            factors[ss, rr, cc, SJ] += mval
            factors[ss, rr, cc, SJ2] += mval*mval
            factors[ss, rr, cc, SIJ] += sval*mval

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def precompute_cc_factors_3d(floating[:, :, :] static, floating[:, :, :] moving, cnp.npy_intp radius):
    cdef:
        cnp.npy_intp ns = static.shape[0]
        cnp.npy_intp nr = static.shape[1]
        cnp.npy_intp nc = static.shape[2]
        cnp.npy_intp side = 2 * radius + 1
        cnp.npy_intp firstc, lastc, firstr, lastr, firsts, lasts
        cnp.npy_intp s, r, c, idx, sides, sider, sidec
        double cnt
        cnp.npy_intp ssss, sss, ss, rr, cc, prev_ss, prev_rr, prev_cc
        double Imean, Jmean, IJprods, Isq, Jsq
        double[:, :, :, :] temp = np.zeros((2, nr, nc, 5), dtype=np.float64)
        floating[:, :, :, :] factors = np.zeros((ns, nr, nc, 5), dtype=np.asarray(static).dtype)

    with nogil:
        sss = 1
        for s in range(ns+radius):
            ss = _mod(s - radius, ns)
            sss = 1 - sss
            firsts = _int_max(0, ss - radius)
            lasts = _int_min(ns - 1, ss + radius)
            sides = (lasts - firsts + 1)
            for r in range(nr+radius):
                rr = _mod(r - radius, nr)
                firstr = _int_max(0, rr - radius)
                lastr = _int_min(nr - 1, rr + radius)
                sider = (lastr - firstr + 1)
                for c in range(nc+radius):
                    cc = _mod(c - radius, nc)
                    # New corner
                    _increment_factors(temp, moving, static, sss, rr, cc, s, r, c, 0)
                    # Add signed sub-volumes
                    if s>0:
                        prev_ss = 1 - sss
                        for idx in range(5):
                            temp[sss, rr, cc, idx] += temp[prev_ss, rr, cc, idx]
                        if r>0:
                            prev_rr = _mod(rr-1, nr)
                            for idx in range(5):
                                temp[sss, rr, cc, idx] -= temp[prev_ss, prev_rr, cc, idx]
                            if c>0:
                                prev_cc = _mod(cc-1, nc)
                                for idx in range(5):
                                    temp[sss, rr, cc, idx] += temp[prev_ss, prev_rr, prev_cc, idx]
                        if c>0:
                            prev_cc = _mod(cc-1, nc)
                            for idx in range(5):
                                temp[sss, rr, cc, idx] -= temp[prev_ss, rr, prev_cc, idx]
                    if(r>0):
                        prev_rr = _mod(rr-1, nr)
                        for idx in range(5):
                            temp[sss, rr, cc, idx] += temp[sss, prev_rr, cc, idx]
                        if(c>0):
                            prev_cc = _mod(cc-1, nc)
                            for idx in range(5):
                                temp[sss, rr, cc, idx] -= temp[sss, prev_rr, prev_cc, idx]
                    if(c>0):
                        prev_cc = _mod(cc-1, nc)
                        for idx in range(5):
                            temp[sss, rr, cc, idx] += temp[sss, rr, prev_cc, idx]
                    # Add signed corners
                    if s>=side:
                        _increment_factors(temp, moving, static, sss, rr, cc, s-side, r, c, -1)
                        if r>=side:
                            _increment_factors(temp, moving, static, sss, rr, cc, s-side, r-side, c, 1)
                            if c>=side:
                                _increment_factors(temp, moving, static, sss, rr, cc, s-side, r-side, c-side, -1)
                        if c>=side:
                            _increment_factors(temp, moving, static, sss, rr, cc, s-side, r, c-side, 1)
                    if r>=side:
                        _increment_factors(temp, moving, static, sss, rr, cc, s, r-side, c, -1)
                        if c>=side:
                            _increment_factors(temp, moving, static, sss, rr, cc, s, r-side, c-side, 1)

                    if c>=side:
                        _increment_factors(temp, moving, static, sss, rr, cc, s, r, c-side, -1)
                    # Compute final factors
                    if s>=radius and r>=radius and c>=radius:
                        firstc = _int_max(0, cc - radius)
                        lastc = _int_min(nc - 1, cc + radius)
                        sidec = (lastc - firstc + 1)
                        cnt = sides*sider*sidec
                        Imean = temp[sss, rr, cc, SI] / cnt
                        Jmean = temp[sss, rr, cc, SJ] / cnt
                        IJprods = (temp[sss, rr, cc, SIJ] - Jmean * temp[sss, rr, cc, SI] -
                            Imean * temp[sss, rr, cc, SJ] + cnt * Jmean * Imean)
                        Isq = (temp[sss, rr, cc, SI2] - Imean * temp[sss, rr, cc, SI] -
                            Imean * temp[sss, rr, cc, SI] + cnt * Imean * Imean)
                        Jsq = (temp[sss, rr, cc, SJ2] - Jmean * temp[sss, rr, cc, SJ] -
                            Jmean * temp[sss, rr, cc, SJ] + cnt * Jmean * Jmean)
                        factors[ss, rr, cc, 0] = static[ss, rr, cc] - Imean
                        factors[ss, rr, cc, 1] = moving[ss, rr, cc] - Jmean
                        factors[ss, rr, cc, 2] = IJprods
                        factors[ss, rr, cc, 3] = Isq
                        factors[ss, rr, cc, 4] = Jsq
    return factors


cdef inline double _cc_forward_gradient(double cnt, double Ii, double Ji,
                                 double sfm, double sff, double smm,
                                 double img_gradx, double img_grady,
                                 double img_gradz, double *out)nogil:
    cdef:
        double aux
        double cc
    if not (sff == 0.0 or smm == 0.0):
        cc = 0
        if(sff * smm > 1e-5):
            cc = sfm * sfm / (sff * smm)
        aux = -2.0 * sfm / (sff * smm) * (Ji - sfm / sff * Ii)
        out[0] = aux * img_gradx
        out[1] = aux * img_grady
        out[2] = aux * img_gradz
        return cc
    out[0] = 0.0
    out[1] = 0.0
    out[2] = 0.0
    return 0.0

cdef inline double _cc_backward_gradient(double cnt, double Ii, double Ji,
                                  double sfm, double sff, double smm,
                                  double img_gradx, double img_grady,
                                  double img_gradz, double *out)nogil:
    cdef:
        double aux
        double cc
    if not (sff == 0.0 or smm == 0.0):
        cc = 0
        if(sff * smm > 1e-5):
            cc = sfm * sfm / (sff * smm)
        aux = -2.0 * sfm / (sff * smm) * (Ii - sfm / smm * Ji)
        out[0] = aux * img_gradx
        out[1] = aux * img_grady
        out[2] = aux * img_gradz
        return cc
    out[0] = 0.0
    out[1] = 0.0
    out[2] = 0.0
    return 0.0



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_cc_steps_3d_nofactors(floating[:, :, :] static, floating[:, :, :] moving,
                                  floating[:, :, :, :] grad_static,
                                  floating[:, :, :, :] grad_moving,
                                  cnp.npy_intp radius):
    cdef:
        cnp.npy_intp ns = static.shape[0]
        cnp.npy_intp nr = static.shape[1]
        cnp.npy_intp nc = static.shape[2]
        cnp.npy_intp side = 2 * radius + 1
        cnp.npy_intp firstc, lastc, firstr, lastr, firsts, lasts
        cnp.npy_intp s, r, c, idx, sides, sider, sidec
        double cnt, fwd_energy = 0, bwd_energy = 0
        cnp.npy_intp ssss, sss, ss, rr, cc, prev_ss, prev_rr, prev_cc
        double Imean, Jmean, Ii, Ji, sfm, sff, smm, localCorrelation, aux
        double[:, :, :, :] temp = np.zeros((2, nr, nc, 5), dtype=np.float64)
        double *grad = [0,0,0]
        floating[:, :, :, :] fwd_step = np.zeros((ns, nr, nc, 3),
                                                 dtype=np.asarray(grad_static).dtype)
        floating[:, :, :, :] bwd_step = np.zeros((ns, nr, nc, 3),
                                                 dtype=np.asarray(grad_static).dtype)
    with nogil:
        sss = 1
        for s in range(ns):
            ss = _mod(s - radius, ns)
            sss = 1 - sss
            firsts = _int_max(0, ss - radius)
            lasts = _int_min(ns - 1, ss + radius)
            sides = (lasts - firsts + 1)
            for r in range(nr):
                rr = _mod(r - radius, nr)
                firstr = _int_max(0, rr - radius)
                lastr = _int_min(nr - 1, rr + radius)
                sider = (lastr - firstr + 1)
                for c in range(nc):
                    cc = _mod(c - radius, nc)
                    # New corner
                    _increment_factors(temp, moving, static, sss, rr, cc, s, r, c, 0)
                    # Add signed sub-volumes
                    if s>0:
                        prev_ss = 1 - sss
                        for idx in range(5):
                            temp[sss, rr, cc, idx] += temp[prev_ss, rr, cc, idx]
                        if r>0:
                            prev_rr = _mod(rr-1, nr)
                            for idx in range(5):
                                temp[sss, rr, cc, idx] -= temp[prev_ss, prev_rr, cc, idx]
                            if c>0:
                                prev_cc = _mod(cc-1, nc)
                                for idx in range(5):
                                    temp[sss, rr, cc, idx] += temp[prev_ss, prev_rr, prev_cc, idx]
                        if c>0:
                            prev_cc = _mod(cc-1, nc)
                            for idx in range(5):
                                temp[sss, rr, cc, idx] -= temp[prev_ss, rr, prev_cc, idx]
                    if(r>0):
                        prev_rr = _mod(rr-1, nr)
                        for idx in range(5):
                            temp[sss, rr, cc, idx] += temp[sss, prev_rr, cc, idx]
                        if(c>0):
                            prev_cc = _mod(cc-1, nc)
                            for idx in range(5):
                                temp[sss, rr, cc, idx] -= temp[sss, prev_rr, prev_cc, idx]
                    if(c>0):
                        prev_cc = _mod(cc-1, nc)
                        for idx in range(5):
                            temp[sss, rr, cc, idx] += temp[sss, rr, prev_cc, idx]
                    # Add signed corners
                    if s>=side:
                        _increment_factors(temp, moving, static, sss, rr, cc, s-side, r, c, -1)
                        if r>=side:
                            _increment_factors(temp, moving, static, sss, rr, cc, s-side, r-side, c, 1)
                            if c>=side:
                                _increment_factors(temp, moving, static, sss, rr, cc, s-side, r-side, c-side, -1)
                        if c>=side:
                            _increment_factors(temp, moving, static, sss, rr, cc, s-side, r, c-side, 1)
                    if r>=side:
                        _increment_factors(temp, moving, static, sss, rr, cc, s, r-side, c, -1)
                        if c>=side:
                            _increment_factors(temp, moving, static, sss, rr, cc, s, r-side, c-side, 1)

                    if c>=side:
                        _increment_factors(temp, moving, static, sss, rr, cc, s, r, c-side, -1)
                    # Compute final factors
                    if ss>=radius and ss<ns-radius and rr>=radius and rr<nr-radius and cc>=radius and cc<nc-radius:
                        firstc = _int_max(0, cc - radius)
                        lastc = _int_min(nc - 1, cc + radius)
                        sidec = (lastc - firstc + 1)
                        cnt = sides*sider*sidec

                        Imean = temp[sss, rr, cc, SI] / cnt
                        Jmean = temp[sss, rr, cc, SJ] / cnt
                        Ii = static[ss, rr, cc] - Imean
                        Ji = moving[ss, rr, cc] - Jmean
                        sfm = (temp[sss, rr, cc, SIJ] - Jmean * temp[sss, rr, cc, SI] -
                            Imean * temp[sss, rr, cc, SJ] + cnt * Jmean * Imean)
                        sff = (temp[sss, rr, cc, SI2] - Imean * temp[sss, rr, cc, SI] -
                            Imean * temp[sss, rr, cc, SI] + cnt * Imean * Imean)
                        smm = (temp[sss, rr, cc, SJ2] - Jmean * temp[sss, rr, cc, SJ] -
                            Jmean * temp[sss, rr, cc, SJ] + cnt * Jmean * Jmean)

                        localCorrelation = _cc_forward_gradient(cnt, Ii, Ji, sfm, sff, smm,
                                                                grad_static[ss, rr, cc, 0],
                                                                grad_static[ss, rr, cc, 1],
                                                                grad_static[ss, rr, cc, 2],
                                                                grad)
                        fwd_energy -= localCorrelation
                        fwd_step[ss, rr, cc, 0] = grad[0]
                        fwd_step[ss, rr, cc, 1] = grad[1]
                        fwd_step[ss, rr, cc, 2] = grad[2]
                        localCorrelation = _cc_backward_gradient(cnt, Ii, Ji, sfm, sff, smm,
                                                                 grad_moving[ss, rr, cc, 0],
                                                                 grad_moving[ss, rr, cc, 1],
                                                                 grad_moving[ss, rr, cc, 2],
                                                                 grad)
                        bwd_energy -= localCorrelation
                        bwd_step[ss, rr, cc, 0] = grad[0]
                        bwd_step[ss, rr, cc, 1] = grad[1]
                        bwd_step[ss, rr, cc, 2] = grad[2]
    return fwd_step, bwd_step, fwd_energy, bwd_energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_cc_steps_3d_nofactors_test(floating[:, :, :] static, floating[:, :, :] moving,
                                       floating[:, :, :, :] grad_static,
                                       floating[:, :, :, :] grad_moving,
                                       int radius):
    r"""Precomputations to quickly compute the gradient of the CC Metric

    This version of precompute_cc_factors_3d is for testing purposes, it
    directly computes the local cross-correlation factors without any
    optimization, so it is less error-prone than the accelerated version.
    """
    cdef:
        cnp.npy_intp ns = static.shape[0]
        cnp.npy_intp nr = static.shape[1]
        cnp.npy_intp nc = static.shape[2]
        cnp.npy_intp s, r, c, k, i, j, t, firstc, lastc, firstr, lastr, firsts, lasts
        double Imean, Jmean, Ii, Ji, sfm, sff, smm, localCorrelation
        double[:] sums = np.zeros((6,), dtype=np.float64)
        double *grad = [0,0,0]
        double fwd_energy = 0, bwd_energy = 0
        floating[:, :, :, :] fwd_step = np.zeros((ns, nr, nc, 3),
                                                 dtype=np.asarray(grad_static).dtype)
        floating[:, :, :, :] bwd_step = np.zeros((ns, nr, nc, 3),
                                                 dtype=np.asarray(grad_static).dtype)

    with nogil:
        for s in range(ns):
            firsts = _int_max(0, s - radius)
            lasts = _int_min(ns - 1, s + radius)
            for r in range(nr):
                firstr = _int_max(0, r - radius)
                lastr = _int_min(nr - 1, r + radius)
                for c in range(nc):
                    firstc = _int_max(0, c - radius)
                    lastc = _int_min(nc - 1, c + radius)
                    for t in range(6):
                        sums[t] = 0
                    for k in range(firsts, 1 + lasts):
                        for i in range(firstr, 1 + lastr):
                            for j in range(firstc, 1 + lastc):
                                sums[SI] += static[k, i, j]
                                sums[SI2] += static[k, i,j]**2
                                sums[SJ] += moving[k, i,j]
                                sums[SJ2] += moving[k, i,j]**2
                                sums[SIJ] += static[k,i,j]*moving[k, i,j]
                                sums[CNT] += 1
                    if s<radius or s>=ns-radius or r<radius or r>=nr-radius or c<radius or c>=nc-radius:
                        continue
                    Imean = sums[SI] / sums[CNT]
                    Jmean = sums[SJ] / sums[CNT]
                    Ii = static[s, r, c] - Imean
                    Ji = moving[s, r, c] - Jmean
                    sfm = (sums[SIJ] - Jmean * sums[SI] - Imean * sums[SJ] + sums[CNT] * Jmean * Imean)
                    sff = (sums[SI2] - Imean * sums[SI] - Imean * sums[SI] + sums[CNT] * Imean * Imean)
                    smm = (sums[SJ2] - Jmean * sums[SJ] - Jmean * sums[SJ] + sums[CNT] * Jmean * Jmean)
                    localCorrelation = _cc_forward_gradient(sums[CNT], Ii, Ji, sfm, sff, smm,
                                                            grad_static[s, r, c, 0],
                                                            grad_static[s, r, c, 1],
                                                            grad_static[s, r, c, 2],
                                                            grad)
                    fwd_energy -= localCorrelation
                    fwd_step[s, r, c, 0] = grad[0]
                    fwd_step[s, r, c, 1] = grad[1]
                    fwd_step[s, r, c, 2] = grad[2]
                    localCorrelation = _cc_backward_gradient(sums[CNT], Ii, Ji, sfm, sff, smm,
                                                             grad_moving[s, r, c, 0],
                                                             grad_moving[s, r, c, 1],
                                                             grad_moving[s, r, c, 2],
                                                             grad)
                    bwd_energy -= localCorrelation
                    bwd_step[s, r, c, 0] = grad[0]
                    bwd_step[s, r, c, 1] = grad[1]
                    bwd_step[s, r, c, 2] = grad[2]
    return fwd_step, bwd_step, fwd_energy, bwd_energy



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void _eval_iv_rectangle(double[:,:,:,:] iv, int x, int y, int z, int d, double *out)nogil:
    cdef:
        int idx
    for idx in range(5):
        out[idx] = iv[x, y, z, idx]
        if x>=d:
            out[idx] -= iv[x-d, y, z, idx]
            if y>=d:
                out[idx] += iv[x-d, y-d, z, idx]
                if z>=d:
                    out[idx] -= iv[x-d, y-d, z-d, idx]
            if z>=d:
                out[idx] += iv[x-d, y, z-d, idx]
        if y>=d:
            out[idx] -= iv[x, y-d, z, idx]
            if z>=d:
                out[idx] += iv[x, y-d, z-d, idx]
        if z>=d:
            out[idx] -= iv[x, y, z-d, idx]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_cc_steps_3d_integral(floating[:, :, :] static, floating[:, :, :] moving,
                                 floating[:, :, :, :] grad_static,
                                 floating[:, :, :, :] grad_moving,
                                 cnp.npy_intp radius):
    cdef:
        cnp.npy_intp ns = static.shape[0]
        cnp.npy_intp nr = static.shape[1]
        cnp.npy_intp nc = static.shape[2]
        cnp.npy_intp side = 2 * radius + 1
        cnp.npy_intp firstc, lastc, firstr, lastr, firsts, lasts
        cnp.npy_intp s, r, c, idx, sides, sider, sidec
        double cnt, fwd_energy = 0, bwd_energy = 0
        double Imean, Jmean, Ii, Ji, sfm, sff, smm, localCorrelation
        double *grad = [0,0,0]
        double *iv_eval = [0,0,0,0,0]
        floating[:, :, :, :] fwd_step = np.zeros((ns, nr, nc, 3),
                                                 dtype=np.asarray(grad_static).dtype)
        floating[:, :, :, :] bwd_step = np.zeros((ns, nr, nc, 3),
                                                 dtype=np.asarray(grad_static).dtype)
        double[:,:,:,:] c1 = np.zeros((1, 1, nc, 5), dtype=np.float64)
        double[:,:,:,:] c2 = np.zeros((1, nr, nc, 5), dtype=np.float64)
        double[:,:,:,:] iv = np.zeros((ns, nr, nc, 5), dtype=np.float64)

    with nogil:
        # Precompute integral images
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):
                    # New corner
                    _increment_factors(c1, moving, static, 0, 0, c, s, r, c, 0)
                    # Add to prevous c1 (if it exists)
                    if c>0:
                        for idx in range(5):
                            c1[0, 0, c, idx] += c1[0, 0, c-1, idx]
                    # Initialize c2 with current c1
                    for idx in range(5):
                        c2[0, r, c, idx] = c1[0, 0, c, idx]
                    # Accumulate previous c2 (if it exists)
                    if r>0:
                        for idx in range(5):
                            c2[0, r, c, idx] += c2[0, r-1, c, idx]
                    # Initialize iv with current c2
                    for idx in range(5):
                        iv[s, r, c, idx] = c2[0, r, c, idx]
                    # Accumulate with previous iv (if it exists)
                    if s>0:
                        for idx in range(5):
                            iv[s, r, c, idx] += iv[s-1, r, c, idx]
        # Compute rectangle integrals using iv
        for s in range(radius, ns-radius):
            firsts = _int_max(0, s - radius)
            lasts = _int_min(ns - 1, s + radius)
            sides = (lasts - firsts + 1)
            for r in range(radius, nr-radius):
                firstr = _int_max(0, r - radius)
                lastr = _int_min(nr - 1, r + radius)
                sider = (lastr - firstr + 1)
                for c in range(radius, nc-radius):
                    firstc = _int_max(0, c - radius)
                    lastc = _int_min(nc - 1, c + radius)
                    sidec = (lastc - firstc + 1)
                    cnt = sides*sider*sidec
                    _eval_iv_rectangle(iv, s+radius, r+radius, c+radius, side, iv_eval)
                    Imean = iv_eval[SI] / cnt
                    Jmean = iv_eval[SJ] / cnt
                    Ii = static[s, r, c] - Imean
                    Ji = moving[s, r, c] - Jmean
                    sfm = (iv_eval[SIJ] - Jmean * iv_eval[SI] -
                        Imean * iv_eval[SJ] + cnt * Jmean * Imean)
                    sff = (iv_eval[SI2] - Imean * iv_eval[SI] -
                        Imean * iv_eval[SI] + cnt * Imean * Imean)
                    smm = (iv_eval[SJ2] - Jmean * iv_eval[SJ] -
                        Jmean * iv_eval[SJ] + cnt * Jmean * Jmean)

                    localCorrelation = _cc_forward_gradient(cnt, Ii, Ji, sfm, sff, smm,
                                                            grad_static[s, r, c, 0],
                                                            grad_static[s, r, c, 1],
                                                            grad_static[s, r, c, 2],
                                                            grad)
                    fwd_energy -= localCorrelation
                    fwd_step[s, r, c, 0] = grad[0]
                    fwd_step[s, r, c, 1] = grad[1]
                    fwd_step[s, r, c, 2] = grad[2]
                    localCorrelation = _cc_backward_gradient(cnt, Ii, Ji, sfm, sff, smm,
                                                             grad_moving[s, r, c, 0],
                                                             grad_moving[s, r, c, 1],
                                                             grad_moving[s, r, c, 2],
                                                             grad)
                    bwd_energy -= localCorrelation
                    bwd_step[s, r, c, 0] = grad[0]
                    bwd_step[s, r, c, 1] = grad[1]
                    bwd_step[s, r, c, 2] = grad[2]
    return fwd_step, bwd_step, fwd_energy, bwd_energy
