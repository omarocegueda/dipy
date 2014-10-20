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
def _precompute_cc_factors_3d(floating[:, :, :] static, floating[:, :, :] moving,
                             cnp.npy_intp radius):
    r"""Precomputations to quickly compute the gradient of the CC Metric

    Pre-computes the separate terms of the cross correlation metric and image
    norms at each voxel considering a neighborhood of the given radius to
    efficiently compute the gradient of the metric with respect to the
    deformation field [Avants09][Avants11]

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
        factors[:,:,:,1] : sum of sq. values of static along the neighborhood
        factors[:,:,:,2] : moving minus its mean value along the neighborhood
        factors[:,:,:,3] : sum of sq. values of moving along the neighborhood
        factors[:,:,:,4] : sum of the pointwise products of static and moving
                           along the neighborhood

    References
    ----------
    [Avants09] Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2009)
               Symmetric Diffeomorphic Image Registration with
               Cross-Correlation: Evaluating Automated Labeling of Elderly and
               Neurodegenerative Brain, 12(1), 26-41.
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
                        factors[s, r, c, 0] = Imean
                        factors[s, r, c, 1] = Jmean
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
                    factors[s, r, c, 0] = Imean
                    factors[s, r, c, 1] = Jmean
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
def _precompute_cc_factors_3d_test(floating[:, :, :] static,
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
                    factors[s, r, c, 0] = Imean
                    factors[s, r, c, 1] = Jmean
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
def fix_cc_factors_3d_test(floating[:, :, :, :] factors, int radius):
    r"""Correction to ANTS CC bug
    """
    cdef:
        cnp.npy_intp ns = factors.shape[0]
        cnp.npy_intp nr = factors.shape[1]
        cnp.npy_intp nc = factors.shape[2]
        cnp.npy_intp s, r, c, k, i, j, t, firstc, lastc, firstr, lastr, firsts, lasts
        double mu, nu, sfm, sff, smm
        floating[:, :, :, :] fixed_factors = np.zeros((ns, nr, nc, 6),
                                                dtype=np.asarray(factors).dtype)
        double[:] sums = np.zeros((5,), dtype=np.float64)

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
                    sums[:] = 0
                    for k in range(firsts, 1 + lasts):
                        for i in range(firstr, 1 + lastr):
                            for j in range(firstc, 1 + lastc):
                                mu = factors[k, i, j, 0]
                                nu = factors[k, i, j, 1]
                                sfm = factors[k, i, j, 2]
                                sff = factors[k, i, j, 3]
                                smm = factors[k, i, j, 4]
                        
                                if sff == 0.0 or smm == 0.0:
                                    continue
                                sums[0] += sfm / (sff * smm)
                                sums[1] += (sfm * sfm) / (sff * sff * smm)
                                sums[2] += (sfm * sfm) / (sff * smm * smm)
                                sums[3] += (nu * sfm) / (sff * smm) - (mu * sfm * sfm) / (sff * sff * smm)
                                sums[4] += (mu * sfm) / (sff * smm) - (nu * sfm * sfm) / (sff * smm * smm)

                    sfm = factors[s, r, c, 2]
                    sff = factors[s, r, c, 3]
                    smm = factors[s, r, c, 4]

                    for i in range(5):
                        fixed_factors[s, r, c, i] = 2 * sums[i]

                    if sff * smm > 1e-5 :
                        fixed_factors[s, r, c, 5] = sfm * sfm / (sff * smm)
                        if fixed_factors[s, r, c, 5] > 1:
                            fixed_factors[s, r, c, 5] = 0
    return fixed_factors


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def fix_cc_factors_3d(floating[:, :, :, :] factors, cnp.npy_intp radius):
    r"""Correction to ANTS CC bug
    """
    cdef:
        cnp.npy_intp side = 2 * radius + 1
        cnp.npy_intp ns = factors.shape[0]
        cnp.npy_intp nr = factors.shape[1]
        cnp.npy_intp nc = factors.shape[2]
        cnp.npy_intp s, r, c, k, i, j, t, q, qq, firstc, lastc, firstr, lastr
        double mu, nu, sfm, sff, smm
        floating[:, :, :, :] fixed_factors = np.zeros((ns, nr, nc, 6),
                                                dtype=np.asarray(factors).dtype)
        double[:, :] lines = np.zeros((5, side), dtype=np.float64)
        double[:] sums = np.zeros((5,), dtype=np.float64)

    with nogil:
        for r in range(nr):
            firstr = _int_max(0, r - radius)
            lastr = _int_min(nr - 1, r + radius)
            for c in range(nc):
                firstc = _int_max(0, c - radius)
                lastc = _int_min(nc - 1, c + radius)
                # compute factors for line [:,r,c]
                for t in range(5):
                    for q in range(side):
                        lines[t,q] = 0

                # Compute all slices and set the sums on the fly
                # compute each slice 
                #[k, i={r-radius..r+radius}, j={c-radius,c+radius}]
                for k in range(ns):
                    q = k % side
                    for t in range(5):
                        sums[t] -= lines[t, q]
                        lines[t, q] = 0
                    for i in range(firstr, lastr + 1):
                        for j in range(firstc, lastc + 1):
                            mu = factors[k, i, j, 0]
                            nu = factors[k, i, j, 1]
                            sfm = factors[k, i, j, 2]
                            sff = factors[k, i, j, 3]
                            smm = factors[k, i, j, 4]
                        
                            if sff == 0.0 or smm == 0.0:
                                continue
                            lines[0, q] += sfm / (sff * smm)
                            lines[1, q] += (sfm * sfm) / (sff * sff * smm)
                            lines[2, q] += (sfm * sfm) / (sff * smm * smm)
                            lines[3, q] += (nu * sfm) / (sff * smm) - (mu * sfm * sfm) / (sff * sff * smm)
                            lines[4, q] += (mu * sfm) / (sff * smm) - (nu * sfm * sfm) / (sff * smm * smm)

                    for t in range(5):
                        sums[t] = 0
                        for qq in range(side):
                            sums[t] += lines[t, qq]
                    if(k >= radius):
                        # s is the voxel that is affected by the cube with
                        # slices [s - radius..s + radius, :, :]
                        s = k - radius
                        for t in range(5):
                            fixed_factors[s, r, c, t] = sums[t]
                        sfm = factors[s, r, c, 2]
                        sff = factors[s, r, c, 3]
                        smm = factors[s, r, c, 4]
                        if sff * smm > 1e-5 :
                            fixed_factors[s, r, c, 5] = sfm * sfm / (sff * smm)
                            if fixed_factors[s, r, c, 5] > 1:
                                fixed_factors[s, r, c, 5] = 0
                # Finally set the values at the end of the line
                for s in range(ns - radius, ns):
                    # this would be the last slice to be processed for voxel
                    # [s, r, c], if it existed
                    k = s + radius
                    q = k % side
                    for t in range(5):
                        sums[t] -= lines[t, q]
                        fixed_factors[s, r, c, t] = sums[t]
                        
                    sfm = factors[s, r, c, 2]
                    sff = factors[s, r, c, 3]
                    smm = factors[s, r, c, 4]
                    if sff * smm > 1e-5 :
                        fixed_factors[s, r, c, 5] = sfm * sfm / (sff * smm)
                        if fixed_factors[s, r, c, 5] > 1:
                            fixed_factors[s, r, c, 5] = 0
    return fixed_factors


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_cc_forward_step_3d(floating[:, :, :] I,
                               floating[:, :, :] J,
                               floating[:, :, :, :] grad_static,
                               floating[:, :, :, :] factors,
                               cnp.npy_intp radius):
    r"""Fixed forward step
    """
    cdef:
        cnp.npy_intp ns = grad_static.shape[0]
        cnp.npy_intp nr = grad_static.shape[1]
        cnp.npy_intp nc = grad_static.shape[2]
        double energy = 0
        cnp.npy_intp s,r,c
        double sa, sb, sd
        floating[:, :, :, :] out = np.zeros((ns, nr, nc, 3),
                                            dtype=np.asarray(grad_static).dtype)
    with nogil:
        for s in range(radius, ns-radius):
            for r in range(radius, nr-radius):
                for c in range(radius, nc-radius):
                    sa = factors[s, r, c, 0]
                    sb = factors[s, r, c, 1]
                    sd = factors[s, r, c, 3]
                    out[s, r, c, 0] -= (sa * J[s, r, c] - sb * I[s, r, c] - sd) * grad_static[s, r, c, 0]
                    out[s, r, c, 1] -= (sa * J[s, r, c] - sb * I[s, r, c] - sd) * grad_static[s, r, c, 1]
                    out[s, r, c, 2] -= (sa * J[s, r, c] - sb * I[s, r, c] - sd) * grad_static[s, r, c, 2]
                    energy -= factors[s, r, c, 5]
    return out, energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_cc_backward_step_3d(floating[:, :, :] I,
                                floating[:, :, :] J,
                                floating[:, :, :, :] grad_moving,
                                floating[:, :, :, :] factors,
                                cnp.npy_intp radius):
    r"""Fixed backward step
    """
    ftype = np.asarray(grad_moving).dtype
    cdef:
        cnp.npy_intp ns = grad_moving.shape[0]
        cnp.npy_intp nr = grad_moving.shape[1]
        cnp.npy_intp nc = grad_moving.shape[2]
        cnp.npy_intp s,r,c
        double energy = 0
        double sa, sc, se
        floating[:, :, :, :] out = np.zeros((ns, nr, nc, 3), dtype=ftype)

    with nogil:

        for s in range(radius, ns-radius):
            for r in range(radius, nr-radius):
                for c in range(radius, nc-radius):
                    sa = factors[s, r, c, 0]
                    sc = factors[s, r, c, 2]
                    se = factors[s, r, c, 4]
                    out[s, r, c, 0] -= (sa * I[s, r, c] - sc * J[s, r, c] - se) * grad_moving[s, r, c, 0]
                    out[s, r, c, 1] -= (sa * I[s, r, c] - sc * J[s, r, c] - se) * grad_moving[s, r, c, 1]
                    out[s, r, c, 2] -= (sa * I[s, r, c] - sc * J[s, r, c] - se) * grad_moving[s, r, c, 2]
                    energy -= factors[s, r, c, 5]
    return out, energy


################################################################
###################################2D###########################
################################################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _precompute_cc_factors_2d(floating[:, :] static, floating[:, :] moving,
                             cnp.npy_intp radius):
    r"""
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
                    factors[r, c, 0] = Imean
                    factors[r, c, 1] = Jmean
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
                factors[r, c, 0] = Imean
                factors[r, c, 1] = Jmean
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
def _precompute_cc_factors_2d_test(floating[:, :] static, floating[:, :] moving,
                                  cnp.npy_intp radius):
    r"""
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
                sums[:] = 0
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
                factors[r, c, 0] = Imean
                factors[r, c, 1] = Jmean
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
def fix_cc_factors_2d_test(floating[:, :, :] factors, int radius):
    r"""Correction to ANTS CC bug
    """
    cdef:
        cnp.npy_intp nr = factors.shape[0]
        cnp.npy_intp nc = factors.shape[1]
        cnp.npy_intp r, c, i, j, t, firstc, lastc, firstr, lastr
        double mu, nu, sfm, sff, smm, Ii, Ji
        floating[:, :, :] fixed_factors = np.zeros((nr, nc, 6),
                                                dtype=np.asarray(factors).dtype)
        double[:] sums = np.zeros((5,), dtype=np.float64)
    with nogil:
        for r in range(nr):
            firstr = _int_max(0, r - radius)
            lastr = _int_min(nr - 1, r + radius)
            for c in range(nc):
                firstc = _int_max(0, c - radius)
                lastc = _int_min(nc - 1, c + radius)
                sums[:] = 0
                for i in range(firstr, 1 + lastr):
                    for j in range(firstc, 1 + lastc):
                        mu = factors[i, j, 0]
                        nu = factors[i, j, 1]
                        sfm = factors[i, j, 2]
                        sff = factors[i, j, 3]
                        smm = factors[i, j, 4]
                        
                        if sff == 0.0 or smm == 0.0:
                            continue
                        sums[0] += sfm / (sff * smm)
                        sums[1] += (sfm * sfm) / (sff * sff * smm)
                        sums[2] += (sfm * sfm) / (sff * smm * smm)
                        sums[3] += (nu * sfm) / (sff * smm) - (mu * sfm * sfm) / (sff * sff * smm)
                        sums[4] += (mu * sfm) / (sff * smm) - (nu * sfm * sfm) / (sff * smm * smm)

                sfm = factors[r, c, 2]
                sff = factors[r, c, 3]
                smm = factors[r, c, 4]

                for i in range(5):
                    fixed_factors[r, c, i] = 2 * sums[i]

                if sff * smm > 1e-5 :
                    fixed_factors[r, c, 5] = sfm * sfm / (sff * smm)
                    if fixed_factors[r, c, 5] > 1:
                        fixed_factors[r, c, 5] = 0
    return fixed_factors


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def fix_cc_factors_2d(floating[:, :, :] factors, cnp.npy_intp radius):
    r"""Correction to ANTS CC bug
    """
    ftype = np.asarray(factors).dtype
    cdef:
        cnp.npy_intp side = 2 * radius + 1
        cnp.npy_intp nr = factors.shape[0]
        cnp.npy_intp nc = factors.shape[1]
        cnp.npy_intp r, c, i, j, t, q, qq, firstc, lastc
        double mu, nu, sfm, sff, smm
        floating[:, :, :] fixed_factors = np.zeros((nr, nc, 6), dtype=ftype)
        double[:, :] lines = np.zeros((5, side), dtype=np.float64)
        double[:] sums = np.zeros((5,), dtype=np.float64)

    with nogil:

        for c in range(nc):
            firstc = _int_max(0, c - radius)
            lastc = _int_min(nc - 1, c + radius)
            # compute factors for row [:,c]
            for t in range(5):
                for q in range(side):
                    lines[t,q] = 0
            # Compute all rows and set the sums on the fly
            # compute row [i, j = {c-radius, c + radius}]
            for i in range(nr):
                q = i % side
                for t in range(5):
                    lines[t, q] = 0
                for j in range(firstc, lastc + 1):
                    mu = factors[i, j, 0]
                    nu = factors[i, j, 1]
                    sfm = factors[i, j, 2]
                    sff = factors[i, j, 3]
                    smm = factors[i, j, 4]
                    
                    if sff == 0.0 or smm == 0.0:
                        continue
                    lines[0, q] += sfm / (sff * smm)
                    lines[1, q] += (sfm * sfm) / (sff * sff * smm)
                    lines[2, q] += (sfm * sfm) / (sff * smm * smm)
                    lines[3, q] += (nu * sfm) / (sff * smm) - (mu * sfm * sfm) / (sff * sff * smm)
                    lines[4, q] += (mu * sfm) / (sff * smm) - (nu * sfm * sfm) / (sff * smm * smm)

                for t in range(5):
                    sums[t] = 0
                    for qq in range(side):
                        sums[t] += lines[t, qq]
                if(i >= radius):
                    # r is the pixel that is affected by the cube with slices
                    # [r - radius.. r + radius, :]
                    r = i - radius
                    sfm = factors[r, c, 2]
                    sff = factors[r, c, 3]
                    smm = factors[r, c, 4]
                    for t in range(5):
                        fixed_factors[r, c, t] = sums[t]
                    if sff * smm > 1e-5 :
                        fixed_factors[r, c, 5] = sfm * sfm / (sff * smm)
                        if fixed_factors[r, c, 5] > 1:
                            fixed_factors[r, c, 5] = 0
            # Finally set the values at the end of the line
            for r in range(nr - radius, nr):
                # this would be the last slice to be processed for pixel
                # [r, c], if it existed
                i = r + radius
                q = i % side
                for t in range(5):
                    sums[t] -= lines[t, q]
                sfm = factors[r, c, 2]
                sff = factors[r, c, 3]
                smm = factors[r, c, 4]
                for t in range(5):
                    fixed_factors[r, c, t] = sums[t]
                if sff * smm > 1e-5:
                    fixed_factors[r, c, 5] = sfm * sfm / (sff * smm)
                    if fixed_factors[r, c, 5] > 1:
                        fixed_factors[r, c, 5] = 0
    return fixed_factors


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_cc_forward_step_2d(floating[:, :] I,
                               floating[:, :] J,
                               floating[:, :, :] grad_static,
                               floating[:, :, :] factors,
                               cnp.npy_intp radius):
    r"""Fixed forward step
    """
    cdef:
        cnp.npy_intp nr = grad_static.shape[0]
        cnp.npy_intp nc = grad_static.shape[1]
        double energy = 0
        double sa, sb, sd
        cnp.npy_intp s,r,c
        floating[:, :, :] out = np.zeros((nr, nc, 2),
                                            dtype=np.asarray(grad_static).dtype)
    with nogil:
        for r in range(radius, nr-radius):
            for c in range(radius, nc-radius):
                sa = factors[r, c, 0]
                sb = factors[r, c, 1]
                sd = factors[r, c, 3]
                out[r, c, 0] -= (sa * J[r, c] - sb * I[r, c] - sd) * grad_static[r, c, 0]
                out[r, c, 1] -= (sa * J[r, c] - sb * I[r, c] - sd) * grad_static[r, c, 1]
                energy -= factors[r, c, 5]
    return out, energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_cc_backward_step_2d(floating[:, :] I,
                                floating[:, :] J,
                                floating[:, :, :] grad_moving,
                                floating[:, :, :] factors,
                                cnp.npy_intp radius):
    r"""Fixed backward step
    """
    ftype = np.asarray(grad_moving).dtype
    cdef:
        cnp.npy_intp nr = grad_moving.shape[0]
        cnp.npy_intp nc = grad_moving.shape[1]
        cnp.npy_intp s,r,c
        double energy = 0
        double sa, sc, se
        floating[:, :, :] out = np.zeros((nr, nc, 2), dtype=ftype)

    with nogil:
        for r in range(radius, nr-radius):
            for c in range(radius, nc-radius):
                sa = factors[r, c, 0]
                sc = factors[r, c, 2]
                se = factors[r, c, 4]
                out[r, c, 0] -= (sa * I[r, c] - sc * J[r, c] - se) * grad_moving[r, c, 0]
                out[r, c, 1] -= (sa * I[r, c] - sc * J[r, c] - se) * grad_moving[r, c, 1]
                energy -= factors[r, c, 5]
    return out, energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def precompute_cc_factors_2d(floating[:, :] static, floating[:, :] moving,
                             cnp.npy_intp radius):
    factors = _precompute_cc_factors_2d(static, moving, radius)
    fixed_factors = fix_cc_factors_2d(factors, radius-1)
    return fixed_factors


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def precompute_cc_factors_3d(floating[:, :, :] static, floating[:, :, :] moving,
                             cnp.npy_intp radius):
    factors = _precompute_cc_factors_3d(static, moving, radius)
    fixed_factors = fix_cc_factors_3d(factors, radius-1)
    return fixed_factors