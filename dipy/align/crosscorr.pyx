""" Utility functions used by the Cross Correlation (CC) metric """

import numpy as np
cimport cython
cimport numpy as cnp
from dipy.align.transforms cimport (Transform)
from fused_types cimport floating
from dipy.align.vector_fields cimport(_apply_affine_3d_x0,
                                      _apply_affine_3d_x1,
                                      _apply_affine_3d_x2,
                                      _apply_affine_2d_x0,
                                      _apply_affine_2d_x1)



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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void compute_coefficients(int s, int r, int c,
                                      floating[:,:,:] static,
                                      floating[:,:,:] moving,
                                      floating[:,:,:,:] factors,
                                      double[:] out)nogil:
    cdef:
        double mu, nu, A, B, C
    mu = static[s, r, c] - factors[s, r, c, 0]
    nu = moving[s, r, c] - factors[s, r, c, 1]
    A = factors[s, r, c, 2]
    B = factors[s, r, c, 3]
    C = factors[s, r, c, 4]
    if(B * C * C > 1e-5):
        out[0] = (2.0 * A) / (B * C)
        out[1] = out[0] * mu
        out[2] = out[0] * (A / C)
        out[3] = out[2] * nu
    else:
        out[0] = 0
        out[1] = 0
        out[2] = 0
        out[3] = 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def cc_val_and_grad(floating[:,:,:] static,
                    floating[:,:,:] moving,
                    floating[:, :, :, :] grad_moving,
                    double[:,:] grid2world,
                    floating[:, :, :, :] factors,
                    Transform transform,
                    double[:] theta,
                    cnp.npy_intp radius,
                    int compute_grad):
    r"""Gradient of the local CC functional w.r.t. parameters
    Computes the gradient of the local Cross Correlation functional w.r.t.
    the parameters of `transform`
    Parameters
    ----------
    grad_moving : array, shape (S, R, C, 3)
        the gradient of the moving volume evaluated at grid-points of the
        static image
    grid2world : array, shape (3, 3)
        the grid-to-space transform associated with the `grad_moving`'s grid
    factors : array, shape (S, R, C, 5)
        the precomputed cross correlation terms obtained via
        precompute_cc_factors_3d
    transform : instance of Transform
        the transformation with respect to whose parameters the gradient
        must be computed
    theta : array, shape (n,)
        parameters of the transformation to compute the gradient from
    radius : int
        the radius of the neighborhood used for the CC metric when
        computing the factors. The returned vector field will be
        zero along a boundary of width radius voxels.
    Returns
    -------
    out : array, shape (n,)
        the gradient of the cross correlation metric with respect to
        parameters of the implicitly specified transform
    energy : the cross correlation energy (data term) at this iteration
    """
    ftype = np.asarray(grad_moving).dtype
    cdef:
        cnp.npy_intp ns = grad_moving.shape[0]
        cnp.npy_intp nr = grad_moving.shape[1]
        cnp.npy_intp nc = grad_moving.shape[2]
        cnp.npy_intp s, r, c, ps, pr, pc, n, i, j, side, constant_jacobian=0
        double Ii, Ji, sfm, sff, smm, factor, local, energy = 0
        double[:,:,:,:] coeffs = None
        double[:] tmp = None
        double[:] x = None
        double[:,:] J = None
        double[:] h = None
        double[:] out = None

    if compute_grad==1:
        n = transform.number_of_parameters
        coeffs = np.zeros((ns, nr, nc, 4), dtype=np.float64)
        tmp = np.zeros((4,), dtype=np.float64)
        x = np.empty(shape=(3,), dtype=np.float64)
        J = np.zeros((3, n), dtype=np.float64)
        h = np.zeros(n, dtype=np.float64)
        out = np.zeros(n, dtype=np.float64)

    side = 2 * radius + 1
    with nogil:
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):
                    # Compute local energy at (s,r,c)
                    Ii = factors[s, r, c, 0]
                    Ji = factors[s, r, c, 1]
                    sfm = factors[s, r, c, 2]
                    sff = factors[s, r, c, 3]
                    smm = factors[s, r, c, 4]
                    local = 0
                    if(sff * smm > 1e-5):
                        local = sfm * sfm / (sff * smm)

                    if compute_grad == 0:
                        energy += local
                        continue

                    # New corner
                    compute_coefficients(s, r, c, static, moving, factors, tmp)
                    for i in range(4):
                        coeffs[s,r,c,i] = tmp[i]
                    # Add signed sub-volumes
                    if s>0:
                        for i in range(4):
                            coeffs[s, r, c,i] += coeffs[s-1,r,c,i]
                        if r>0:
                            for i in range(4):
                                coeffs[s,r,c,i] -= coeffs[s-1,r-1,c,i]
                            if c>0:
                                for i in range(4):
                                    coeffs[s,r,c,i] += coeffs[s-1,r-1,c-1,i]
                        if c>0:
                            for i in range(4):
                                coeffs[s,r,c,i] -= coeffs[s-1,r,c-1,i]
                    if r>0:
                        for i in range(4):
                            coeffs[s, r, c, i] += coeffs[s,r-1,c,i]
                        if c>0:
                            for i in range(4):
                                coeffs[s, r, c,i] -= coeffs[s,r-1,c-1,i]
                    if c>0:
                        for i in range(4):
                            coeffs[s, r, c,i] += coeffs[s,r,c-1,i]
                    # Add signed corners
                    if s>=side:
                        compute_coefficients(s-side, r, c, static, moving, factors, tmp)
                        for i in range(4):
                            coeffs[s,r,c,i] -= tmp[i]
                        if r>=side:
                            compute_coefficients(s-side, r-side, c, static, moving, factors, tmp)
                            for i in range(4):
                                coeffs[s,r,c,i] += tmp[i]
                            if c>=side:
                                compute_coefficients(s-side, r-side, c-side, static, moving, factors, tmp)
                                for i in range(4):
                                    coeffs[s,r,c,i] -= tmp[i]
                        if c>=side:
                            compute_coefficients(s-side, r, c-side, static, moving, factors, tmp)
                            for i in range(4):
                                coeffs[s,r,c,i] += tmp[i]
                    if r>=side:
                        compute_coefficients(s, r-side, c, static, moving, factors, tmp)
                        for i in range(4):
                            coeffs[s,r,c,i] -= tmp[i]
                        if c>=side:
                            compute_coefficients(s, r-side, c-side, static, moving, factors, tmp)
                            for i in range(4):
                                coeffs[s,r,c,i] += tmp[i]
                    if c>=side:
                        compute_coefficients(s, r, c-side, static, moving, factors, tmp)
                        for i in range(4):
                            coeffs[s,r,c,i] -= tmp[i]

                    if s<radius or r<radius or c<radius:
                        continue

                    energy += local
                    # Accumulate gradient at (s,r,c)
                    ps = s-radius
                    pr = r-radius
                    pc = c-radius
                    x[0] = _apply_affine_3d_x0(ps, pr, pc, 1, grid2world)
                    x[1] = _apply_affine_3d_x1(ps, pr, pc, 1, grid2world)
                    x[2] = _apply_affine_3d_x2(ps, pr, pc, 1, grid2world)
                    if constant_jacobian == 0:
                        constant_jacobian = transform._jacobian(theta, x, J)

                    for j in range(n):
                        h[j] = 0
                        for i in range(3):
                            h[j] += grad_moving[ps, pr, pc, i] * J[i,j]

                    factor = (static[ps,pr,pc] * coeffs[s,r,c,0] -
                              coeffs[s,r,c,1] -
                              moving[ps,pr,pc] * coeffs[s,r,c,2] +
                              coeffs[s,r,c,3])

                    for j in range(n):
                        out[j] += h[j] * factor
    return out, energy