#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
import numpy as np
cimport cython
cimport numpy as cnp
from dipy.align.transforms cimport (Transform)
from dipy.align.fused_types cimport floating, number
from dipy.correct.splines import CubicSplineField
from dipy.align.crosscorr import precompute_cc_factors_3d

cdef extern from "math.h":
    double sqrt(double x) nogil
    double floor(double x) nogil

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

cdef inline int _mod(int x, int m)nogil:
    if x<0:
        return x + m
    return x

cdef inline void _increment_factors(double[:,:,:,:] factors, double[:,:,:,:] temp,
                                    int ss, int rr, int cc, int s, int r, int c, int weight)nogil:
    cdef:
        double fa, fb, fb2, fc
    if s>=temp.shape[0] or r>=temp.shape[1] or c>=temp.shape[2]:
        if weight == 0:
            factors[ss, rr, cc, 0] = 0
            factors[ss, rr, cc, 1] = 0
    else:
        fa = temp[s, r, c, 2]  # dot product: A
        fb = temp[s, r, c, 4]  # moving norm: B
        fb2 = fb * fb  # B^2
        fc = temp[s, r, c, 3]  # static norm: C
        if (fb2*fc > 1e-5):
            if weight == 0:
                factors[ss, rr, cc, 0] = fa/(fb*fc)
                factors[ss, rr, cc, 1] = (fa*fa)/(fb2*fc)
            elif weight == -1:
                factors[ss, rr, cc, 0] -= fa/(fb*fc)
                factors[ss, rr, cc, 1] -= (fa*fa)/(fb2*fc)
            elif weight == 1:
                factors[ss, rr, cc, 0] += fa/(fb*fc)
                factors[ss, rr, cc, 1] += (fa*fa)/(fb2*fc)
        elif weight == 0:
            factors[ss, rr, cc, 0] = 0
            factors[ss, rr, cc, 1] = 0

def cc_splines_gradient(double[:,:,:] g, double[:,:,:] f,
                        double[:,:,:] dg, double[:,:,:] df,
                        double pedir_factor,
                        int[:,:,:] mf, int[:,:,:] mg,
                        double[:,:,:] kernel, double[:,:,:] dkernel,
                        double[:,:,:] dfield, int[:] kspacing,
                        int[:] kshape, cnp.npy_intp radius,
                        double[:,:,:] kcoef):
    cdef:
        int ns = f.shape[0]
        int nr = f.shape[1]
        int nc = f.shape[2]
        int* ksize = [kernel.shape[0], kernel.shape[1], kernel.shape[2]]
        cnp.npy_intp side = 2 * radius + 1
        int* kcenter = [ksize[0]//2, ksize[1]//2, ksize[2]//2]
        double[:,:,:] F = np.empty((ns,nr,nc), dtype=np.float64)
        double[:, :, :, :] temp = np.zeros((ns, nr, nc, 5), dtype=np.float64)
        double[:, :, :, :] factors = np.zeros((2, nr, nc, 2), dtype=np.float64)
        cnp.npy_intp firstc, lastc, firstr, lastr, firsts, lasts
        cnp.npy_intp sss, ss, rr, cc, prev_sss, prev_rr, prev_cc
        int s, r, c, i, j, k, i0, i1, j0, j1, k0, k1, sc_s, sc_r, sc_c, sides, sider, sidec, nwindows
        double J, Fbar, gbar, alpha, beta, sp_contrib, dF, A, B, C, energy
        double deltay = kspacing[1]
    kcoef[...] = 0
    with nogil:
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):
                    J = 1 + pedir_factor * dfield[s,r,c]  # local Jacobian
                    F[s,r,c] = f[s,r,c] * J

    # Precompute window sums
    temp = precompute_cc_factors_3d(g, F, radius)
    with nogil:
        nwindows = 0
        energy = 0
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
                    _increment_factors(factors, temp, sss, rr, cc, s, r, c, 0)
                    # Add signed sub-volumes
                    if s>0:
                        prev_sss = 1 - sss
                        for idx in range(2):
                            factors[sss, rr, cc, idx] += factors[prev_sss, rr, cc, idx]
                        if r>0:
                            prev_rr = _mod(rr-1, nr)
                            for idx in range(2):
                                factors[sss, rr, cc, idx] -= factors[prev_sss, prev_rr, cc, idx]
                            if c>0:
                                prev_cc = _mod(cc-1, nc)
                                for idx in range(2):
                                    factors[sss, rr, cc, idx] += factors[prev_sss, prev_rr, prev_cc, idx]
                        if c>0:
                            prev_cc = _mod(cc-1, nc)
                            for idx in range(2):
                                factors[sss, rr, cc, idx] -= factors[prev_sss, rr, prev_cc, idx]
                    if(r>0):
                        prev_rr = _mod(rr-1, nr)
                        for idx in range(2):
                            factors[sss, rr, cc, idx] += factors[sss, prev_rr, cc, idx]
                        if(c>0):
                            prev_cc = _mod(cc-1, nc)
                            for idx in range(2):
                                factors[sss, rr, cc, idx] -= factors[sss, prev_rr, prev_cc, idx]
                    if(c>0):
                        prev_cc = _mod(cc-1, nc)
                        for idx in range(2):
                            factors[sss, rr, cc, idx] += factors[sss, rr, prev_cc, idx]
                    # Add signed corners
                    if s>=side:
                        _increment_factors(factors, temp, sss, rr, cc, s-side, r, c, -1)
                        if r>=side:
                            _increment_factors(factors, temp, sss, rr, cc, s-side, r-side, c, 1)
                            if c>=side:
                                _increment_factors(factors, temp, sss, rr, cc, s-side, r-side, c-side, -1)
                        if c>=side:
                            _increment_factors(factors, temp, sss, rr, cc, s-side, r, c-side, 1)
                    if r>=side:
                        _increment_factors(factors, temp, sss, rr, cc, s, r-side, c, -1)
                        if c>=side:
                            _increment_factors(factors, temp, sss, rr, cc, s, r-side, c-side, 1)

                    if c>=side:
                        _increment_factors(factors, temp, sss, rr, cc, s, r, c-side, -1)
                    # Compute final factors
                    if ss>=radius and rr>=radius and cc>=radius:
                        nwindows += 1
                        firstc = _int_max(0, cc - radius)
                        lastc = _int_min(nc - 1, cc + radius)
                        sidec = (lastc - firstc + 1)
                        cnt = sides*sider*sidec

                        # Iterate over all splines affected by (ss, rr, cc)
                        # The first-order rectangle integrals are at temp[ss, rr, cc, :]
                        # The second-order rectangle integrals are at factors[sss, rr, cc, :]
                        gbar = temp[ss, rr, cc, 0]
                        Fbar = temp[ss, rr, cc, 1]

                        A = temp[ss, rr, cc, 2]  # dot product: A
                        C = temp[ss, rr, cc, 3]  # static sq norm: C
                        B = temp[ss, rr, cc, 4]  # moving sq norm: B
                        if B*C > 1e-5:
                            energy += (A * A) / (B * C)

                            alpha = factors[sss, rr, cc, 0]
                            beta = factors[sss, rr, cc, 1]

                            # First and last knot indices being affected by (ss, rr, cc)
                            k0 = (ss - kcenter[0] + kspacing[0] - 1) // kspacing[0]
                            k1 = (ss + kcenter[0]) // kspacing[0]
                            i0 = (rr - kcenter[1] + kspacing[1] - 1) // kspacing[1]
                            i1 = (rr + kcenter[1]) // kspacing[1]
                            j0 = (cc - kcenter[2] + kspacing[2] - 1) // kspacing[2]
                            j1 = (cc + kcenter[2]) // kspacing[2]
                            for k in range(k0, 1 + k1):
                                sc_s = k * kspacing[0] - kcenter[0]  # First grid cell affected by this spline
                                for i in range(i0, 1 + i1):
                                    sc_r = i * kspacing[1] - kcenter[1]
                                    for j in range(j0, 1 + j1):
                                        sc_c = j * kspacing[0] - kcenter[2]
                                        # The *first* grid cell affected by this
                                        # spline (sc_s, sc_r, sc_c) may be out of
                                        # bounds, but it doesn't matter, because
                                        # we are evaluating the spline at
                                        # [ss - sc_s, rr - sc_r, cc - sc_c]
                                        # which we know if a valid kernel index
                                        J = 1 + pedir_factor * dfield[ss,rr,cc]  # local Jacobian
                                        dF = pedir_factor * (df[ss, rr, cc] * kernel[ss - sc_s, rr - sc_r, cc - sc_c] * J +
                                             Fbar * dkernel[ss - sc_s, rr - sc_r, cc - sc_c] / deltay)
                                        sp_contrib = (alpha * gbar - beta * Fbar) * dF

                                        # The actual index of this knot in the knot coefficient array is (k+1, i+1, j+1)
                                        kcoef[k+1, i+1, j+1] += sp_contrib
        for k in range(kcoef.shape[0]):
            for i in range(kcoef.shape[1]):
                for j in range(kcoef.shape[2]):
                    kcoef[k, i, j] = -2.0 * kcoef[k, i, j] / nwindows
        energy = 1.0 - energy/nwindows
    return energy




cdef inline void _increment_factors_epicor(double[:,:,:,:] factors, double[:,:,:,:] temp,
                                    int ss, int rr, int cc, int s, int r, int c, int weight)nogil:
    cdef:
        double fa, fb, fb2, fc, fc2
    if s>=temp.shape[0] or r>=temp.shape[1] or c>=temp.shape[2]:
        if weight == 0:
            factors[ss, rr, cc, 0] = 0
            factors[ss, rr, cc, 1] = 0
            factors[ss, rr, cc, 2] = 0
    else:
        fa = temp[s, r, c, 2]  # dot product: A
        fb = temp[s, r, c, 3]  # moving norm: B
        fb2 = fb * fb  # B^2
        fc = temp[s, r, c, 4]  # static norm: C
        fc2 = fc * fc  # C^2
        if ((fb2*fc > 1e-7) and (fc2*fb > 1e-7)):
            if weight == 0:
                factors[ss, rr, cc, 0] = fa/(fb*fc)
                factors[ss, rr, cc, 1] = (fa*fa)/(fb2*fc)
                factors[ss, rr, cc, 2] = (fa*fa)/(fb*fc2)
            elif weight == -1:
                factors[ss, rr, cc, 0] -= fa/(fb*fc)
                factors[ss, rr, cc, 1] -= (fa*fa)/(fb2*fc)
                factors[ss, rr, cc, 2] -= (fa*fa)/(fb*fc2)
            elif weight == 1:
                factors[ss, rr, cc, 0] += fa/(fb*fc)
                factors[ss, rr, cc, 1] += (fa*fa)/(fb2*fc)
                factors[ss, rr, cc, 2] += (fa*fa)/(fb*fc2)
        elif weight == 0:
            factors[ss, rr, cc, 0] = 0
            factors[ss, rr, cc, 1] = 0
            factors[ss, rr, cc, 2] = 0


def cc_splines_gradient_epicor(double[:,:,:] f1, double[:,:,:] f2,
                               double[:,:,:] df1, double[:,:,:] df2,
                               double pedir_factor,
                               int[:,:,:] mf, int[:,:,:] mg,
                               double[:,:,:] kernel, double[:,:,:] dkernel,
                               double[:,:,:] dfield, int[:] kspacing,
                               int[:] kshape, cnp.npy_intp radius,
                               double[:,:,:] kcoef):
    cdef:
        int ns = f1.shape[0]
        int nr = f1.shape[1]
        int nc = f1.shape[2]
        int* ksize = [kernel.shape[0], kernel.shape[1], kernel.shape[2]]
        cnp.npy_intp side = 2 * radius + 1
        int* kcenter = [ksize[0]//2, ksize[1]//2, ksize[2]//2]
        double[:,:,:] F2 = np.empty((ns,nr,nc), dtype=np.float64)
        double[:,:,:] F1 = np.empty((ns,nr,nc), dtype=np.float64)
        double[:, :, :, :] temp = np.zeros((ns, nr, nc, 5), dtype=np.float64)
        int nfactors = 3
        double[:, :, :, :] factors = np.zeros((2, nr, nc, nfactors), dtype=np.float64)
        cnp.npy_intp firstc, lastc, firstr, lastr, firsts, lasts
        cnp.npy_intp sss, ss, rr, cc, prev_sss, prev_rr, prev_cc
        int s, r, c, i, j, k, i0, i1, j0, j1, k0, k1, sc_s, sc_r, sc_c, sides, sider, sidec, nwindows
        double Jf2, Jf1, F2bar, F1bar, alpha, beta, gamma, sp_contrib, dF2, dF1, A, B, C, energy
        double deltay = kspacing[1]
        double minJ1=1e100, maxJ1=-1e100, minJ2=1e100, maxJ2=-1e100
    kcoef[...] = 0
    with nogil:
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):
                    Jf1 = 1.0 + pedir_factor * dfield[s,r,c]  # local Jacobian
                    Jf2 = 1.0 - pedir_factor * dfield[s,r,c]  # local Jacobian
                    minJ1 = Jf1 if Jf1<minJ1 else minJ1
                    minJ2 = Jf2 if Jf2<minJ2 else minJ2
                    maxJ1 = Jf1 if Jf1>maxJ1 else maxJ1
                    maxJ2 = Jf2 if Jf2>maxJ2 else maxJ2
                    F1[s,r,c] = f1[s,r,c] * Jf1
                    F2[s,r,c] = f2[s,r,c] * Jf2
    #print("norot: J1[%f, %f], J2[%f, %f]"%(minJ1, maxJ1, minJ2, maxJ2))

    # Precompute window sums
    temp = precompute_cc_factors_3d(F1, F2, radius)
    with nogil:
        nwindows = 0
        energy = 0
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
                    _increment_factors_epicor(factors, temp, sss, rr, cc, s, r, c, 0)
                    # Add signed sub-volumes
                    if s>0:
                        prev_sss = 1 - sss
                        for idx in range(nfactors):
                            factors[sss, rr, cc, idx] += factors[prev_sss, rr, cc, idx]
                        if r>0:
                            prev_rr = _mod(rr-1, nr)
                            for idx in range(nfactors):
                                factors[sss, rr, cc, idx] -= factors[prev_sss, prev_rr, cc, idx]
                            if c>0:
                                prev_cc = _mod(cc-1, nc)
                                for idx in range(nfactors):
                                    factors[sss, rr, cc, idx] += factors[prev_sss, prev_rr, prev_cc, idx]
                        if c>0:
                            prev_cc = _mod(cc-1, nc)
                            for idx in range(nfactors):
                                factors[sss, rr, cc, idx] -= factors[prev_sss, rr, prev_cc, idx]
                    if(r>0):
                        prev_rr = _mod(rr-1, nr)
                        for idx in range(nfactors):
                            factors[sss, rr, cc, idx] += factors[sss, prev_rr, cc, idx]
                        if(c>0):
                            prev_cc = _mod(cc-1, nc)
                            for idx in range(nfactors):
                                factors[sss, rr, cc, idx] -= factors[sss, prev_rr, prev_cc, idx]
                    if(c>0):
                        prev_cc = _mod(cc-1, nc)
                        for idx in range(nfactors):
                            factors[sss, rr, cc, idx] += factors[sss, rr, prev_cc, idx]
                    # Add signed corners
                    if s>=side:
                        _increment_factors_epicor(factors, temp, sss, rr, cc, s-side, r, c, -1)
                        if r>=side:
                            _increment_factors_epicor(factors, temp, sss, rr, cc, s-side, r-side, c, 1)
                            if c>=side:
                                _increment_factors_epicor(factors, temp, sss, rr, cc, s-side, r-side, c-side, -1)
                        if c>=side:
                            _increment_factors_epicor(factors, temp, sss, rr, cc, s-side, r, c-side, 1)
                    if r>=side:
                        _increment_factors_epicor(factors, temp, sss, rr, cc, s, r-side, c, -1)
                        if c>=side:
                            _increment_factors_epicor(factors, temp, sss, rr, cc, s, r-side, c-side, 1)

                    if c>=side:
                        _increment_factors_epicor(factors, temp, sss, rr, cc, s, r, c-side, -1)
                    # Compute final factors
                    if ss>=radius and rr>=radius and cc>=radius:
                        nwindows += 1
                        firstc = _int_max(0, cc - radius)
                        lastc = _int_min(nc - 1, cc + radius)
                        sidec = (lastc - firstc + 1)
                        cnt = sides*sider*sidec

                        # Iterate over all splines affected by (ss, rr, cc)
                        # The first-order rectangle integrals are at temp[ss, rr, cc, :]
                        # The second-order rectangle integrals are at factors[sss, rr, cc, :]
                        F1bar = temp[ss, rr, cc, 0]
                        F2bar = temp[ss, rr, cc, 1]

                        A = temp[ss, rr, cc, 2]  # dot product: A
                        B = temp[ss, rr, cc, 3]  # moving sq norm: B
                        C = temp[ss, rr, cc, 4]  # static sq norm: C

                        #if (B*B*C > 1e-5) and (C*C*B > 1e-5):
                        if (B*C > 1e-7):
                            energy += (A * A) / (B * C)

                            alpha = factors[sss, rr, cc, 0]
                            beta = factors[sss, rr, cc, 1]
                            gamma = factors[sss, rr, cc, 2]

                            # First and last knot indices being affected by (ss, rr, cc)
                            k0 = (ss - kcenter[0] + kspacing[0] - 1) // kspacing[0]
                            k1 = (ss + kcenter[0]) // kspacing[0]
                            i0 = (rr - kcenter[1] + kspacing[1] - 1) // kspacing[1]
                            i1 = (rr + kcenter[1]) // kspacing[1]
                            j0 = (cc - kcenter[2] + kspacing[2] - 1) // kspacing[2]
                            j1 = (cc + kcenter[2]) // kspacing[2]
                            for k in range(k0, 1 + k1):
                                sc_s = k * kspacing[0] - kcenter[0]  # First grid cell affected by this spline
                                for i in range(i0, 1 + i1):
                                    sc_r = i * kspacing[1] - kcenter[1]
                                    for j in range(j0, 1 + j1):
                                        sc_c = j * kspacing[0] - kcenter[2]
                                        # The *first* grid cell affected by this
                                        # spline (sc_s, sc_r, sc_c) may be out of
                                        # bounds, but it doesn't matter, because
                                        # we are evaluating the spline at
                                        # [ss - sc_s, rr - sc_r, cc - sc_c]
                                        # which we know is a valid kernel index
                                        Jf1 = 1 + pedir_factor * dfield[ss,rr,cc]  # local Jacobian (for F1)
                                        Jf2 = 1 - pedir_factor * dfield[ss,rr,cc]  # local Jacobian (for F2)

                                        dF1 = pedir_factor * (df1[ss, rr, cc] * kernel[ss - sc_s, rr - sc_r, cc - sc_c] * Jf1 +
                                             F1bar * dkernel[ss - sc_s, rr - sc_r, cc - sc_c] / deltay)
                                        dF2 = pedir_factor * (-1.0*df2[ss, rr, cc] * kernel[ss - sc_s, rr - sc_r, cc - sc_c] * Jf2 +
                                             F2bar * dkernel[ss - sc_s, rr - sc_r, cc - sc_c] / deltay)

                                        sp_contrib = ((alpha * F2bar - beta * F1bar) * dF1 +
                                                     (alpha * F1bar - gamma * F2bar) * dF2)

                                        # The actual index of this knot in the knot coefficient array is (k+1, i+1, j+1)
                                        kcoef[k+1, i+1, j+1] += sp_contrib
        for k in range(kcoef.shape[0]):
            for i in range(kcoef.shape[1]):
                for j in range(kcoef.shape[2]):
                    kcoef[k, i, j] = -2.0 * kcoef[k, i, j] / nwindows
        energy = 1-energy/nwindows
    return energy


def cc_splines_grad_epicor_motion(double[:,:,:] f1, double[:,:,:] f2,
                                 double[:,:,:,:] df1, double[:,:,:,:] df2,
                                 double pedir_factor,
                                 int[:,:,:] mf, int[:,:,:] mg,
                                 double[:,:,:] kernel, double[:,:,:] dkernel,
                                 double[:,:,:] dfield, double[:,:] fieldg2w,
                                 int[:] kspacing,
                                 int[:] kshape, cnp.npy_intp radius,
                                 Transform transform, double[:] theta,
                                 double[:,:,:] kcoef, double[:] dtheta):
    cdef:
        int ns = f1.shape[0]
        int nr = f1.shape[1]
        int nc = f1.shape[2]
        int* ksize = [kernel.shape[0], kernel.shape[1], kernel.shape[2]]
        cnp.npy_intp side = 2 * radius + 1
        int* kcenter = [ksize[0]//2, ksize[1]//2, ksize[2]//2]
        double[:,:,:] F2 = np.empty((ns,nr,nc), dtype=np.float64)
        double[:,:,:] F1 = np.empty((ns,nr,nc), dtype=np.float64)
        double[:, :, :, :] temp = np.zeros((ns, nr, nc, 5), dtype=np.float64)
        double[:,:] J = None
        double[:] h = None
        double[:] x = np.zeros((3,), dtype=np.float64)
        double[:] center = np.zeros((3,), dtype=np.float64)
        int nfactors = 3
        double[:, :, :, :] factors = np.zeros((2, nr, nc, nfactors), dtype=np.float64)
        cnp.npy_intp firstc, lastc, firstr, lastr, firsts, lasts, constant_jacobian=0
        cnp.npy_intp sss, ss, rr, cc, prev_sss, prev_rr, prev_cc
        int s, r, c, i, j, k, ii, jj, kk, i0, i1, j0, j1, k0, k1, sc_s, sc_r, sc_c, sides, sider, sidec, nwindows
        double Jf2, Jf1, F2bar, F1bar, alpha, beta, gamma, sp_contrib, dF2, dF1, A, B, C, energy
        double deltay = kspacing[1]
        double factor
        double minJ1=1e100, maxJ1=-1e100, minJ2=1e100, maxJ2=-1e100

    n = transform.number_of_parameters
    J = np.zeros((3, n), dtype=np.float64)
    h = np.zeros((n,), dtype=np.float64)
    center[0] = ns * 0.5
    center[1] = nr * 0.5
    center[2] = nc * 0.5
    dtheta[:] = 0
    kcoef[...] = 0
    with nogil:
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):
                    Jf1 = 1.0 + pedir_factor * dfield[s,r,c]  # local Jacobian
                    Jf2 = 1.0 - pedir_factor * dfield[s,r,c]  # local Jacobian
                    minJ1 = Jf1 if Jf1<minJ1 else minJ1
                    minJ2 = Jf2 if Jf2<minJ2 else minJ2
                    maxJ1 = Jf1 if Jf1>maxJ1 else maxJ1
                    maxJ2 = Jf2 if Jf2>maxJ2 else maxJ2
                    F1[s,r,c] = f1[s,r,c] * Jf1
                    F2[s,r,c] = f2[s,r,c] * Jf2
    #print("general: J1[%f, %f], J2[%f, %f]"%(minJ1, maxJ1, minJ2, maxJ2))
    # Precompute window sums
    temp = precompute_cc_factors_3d(F1, F2, radius)
    with nogil:
        nwindows = 0
        energy = 0
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
                    _increment_factors_epicor(factors, temp, sss, rr, cc, s, r, c, 0)
                    # Add signed sub-volumes
                    if s>0:
                        prev_sss = 1 - sss
                        for idx in range(nfactors):
                            factors[sss, rr, cc, idx] += factors[prev_sss, rr, cc, idx]
                        if r>0:
                            prev_rr = _mod(rr-1, nr)
                            for idx in range(nfactors):
                                factors[sss, rr, cc, idx] -= factors[prev_sss, prev_rr, cc, idx]
                            if c>0:
                                prev_cc = _mod(cc-1, nc)
                                for idx in range(nfactors):
                                    factors[sss, rr, cc, idx] += factors[prev_sss, prev_rr, prev_cc, idx]
                        if c>0:
                            prev_cc = _mod(cc-1, nc)
                            for idx in range(nfactors):
                                factors[sss, rr, cc, idx] -= factors[prev_sss, rr, prev_cc, idx]
                    if(r>0):
                        prev_rr = _mod(rr-1, nr)
                        for idx in range(nfactors):
                            factors[sss, rr, cc, idx] += factors[sss, prev_rr, cc, idx]
                        if(c>0):
                            prev_cc = _mod(cc-1, nc)
                            for idx in range(nfactors):
                                factors[sss, rr, cc, idx] -= factors[sss, prev_rr, prev_cc, idx]
                    if(c>0):
                        prev_cc = _mod(cc-1, nc)
                        for idx in range(nfactors):
                            factors[sss, rr, cc, idx] += factors[sss, rr, prev_cc, idx]
                    # Add signed corners
                    if s>=side:
                        _increment_factors_epicor(factors, temp, sss, rr, cc, s-side, r, c, -1)
                        if r>=side:
                            _increment_factors_epicor(factors, temp, sss, rr, cc, s-side, r-side, c, 1)
                            if c>=side:
                                _increment_factors_epicor(factors, temp, sss, rr, cc, s-side, r-side, c-side, -1)
                        if c>=side:
                            _increment_factors_epicor(factors, temp, sss, rr, cc, s-side, r, c-side, 1)
                    if r>=side:
                        _increment_factors_epicor(factors, temp, sss, rr, cc, s, r-side, c, -1)
                        if c>=side:
                            _increment_factors_epicor(factors, temp, sss, rr, cc, s, r-side, c-side, 1)

                    if c>=side:
                        _increment_factors_epicor(factors, temp, sss, rr, cc, s, r, c-side, -1)
                    # Compute final factors
                    if ss>=radius and rr>=radius and cc>=radius:
                        nwindows += 1
                        firstc = _int_max(0, cc - radius)
                        lastc = _int_min(nc - 1, cc + radius)
                        sidec = (lastc - firstc + 1)
                        cnt = sides*sider*sidec

                        # Iterate over all splines affected by (ss, rr, cc)
                        # The first-order rectangle integrals are at temp[ss, rr, cc, :]
                        # The second-order rectangle integrals are at factors[sss, rr, cc, :]
                        F1bar = temp[ss, rr, cc, 0]
                        F2bar = temp[ss, rr, cc, 1]

                        A = temp[ss, rr, cc, 2]  # dot product: A
                        B = temp[ss, rr, cc, 3]  # moving sq norm: B
                        C = temp[ss, rr, cc, 4]  # static sq norm: C

                        #if (B*B*C > 1e-5) and (C*C*B > 1e-5):
                        if (B*C > 1e-7):
                            energy += (A * A) / (B * C)

                            alpha = factors[sss, rr, cc, 0]
                            beta = factors[sss, rr, cc, 1]
                            gamma = factors[sss, rr, cc, 2]

                            # First and last knot indices being affected by (ss, rr, cc)
                            k0 = (ss - kcenter[0] + kspacing[0] - 1) // kspacing[0]
                            k1 = (ss + kcenter[0]) // kspacing[0]
                            i0 = (rr - kcenter[1] + kspacing[1] - 1) // kspacing[1]
                            i1 = (rr + kcenter[1]) // kspacing[1]
                            j0 = (cc - kcenter[2] + kspacing[2] - 1) // kspacing[2]
                            j1 = (cc + kcenter[2]) // kspacing[2]
                            for k in range(k0, 1 + k1):
                                sc_s = k * kspacing[0] - kcenter[0]  # First grid cell affected by this spline
                                for i in range(i0, 1 + i1):
                                    sc_r = i * kspacing[1] - kcenter[1]
                                    for j in range(j0, 1 + j1):
                                        sc_c = j * kspacing[0] - kcenter[2]
                                        # The *first* grid cell affected by this
                                        # spline (sc_s, sc_r, sc_c) may be out of
                                        # bounds, but it doesn't matter, because
                                        # we are evaluating the spline at
                                        # [ss - sc_s, rr - sc_r, cc - sc_c]
                                        # which we know is a valid kernel index
                                        Jf1 = 1 + pedir_factor * dfield[ss,rr,cc]  # local Jacobian (for F1)
                                        Jf2 = 1 - pedir_factor * dfield[ss,rr,cc]  # local Jacobian (for F2)

                                        dF1 = pedir_factor * (df1[ss, rr, cc, 1] * kernel[ss - sc_s, rr - sc_r, cc - sc_c] * Jf1 +
                                             F1bar * dkernel[ss - sc_s, rr - sc_r, cc - sc_c] / deltay)
                                        dF2 = pedir_factor * (-1.0*df2[ss, rr, cc, 1] * kernel[ss - sc_s, rr - sc_r, cc - sc_c] * Jf2 +
                                             F2bar * dkernel[ss - sc_s, rr - sc_r, cc - sc_c] / deltay)

                                        sp_contrib = ((alpha * F2bar - beta * F1bar) * dF1 +
                                                     (alpha * F1bar - gamma * F2bar) * dF2)

                                        # The actual index of this knot in the knot coefficient array is (k+1, i+1, j+1)
                                        kcoef[k+1, i+1, j+1] += sp_contrib

                                        # Accumulate contribution to motion gradient
                                        factor = (alpha * F1bar - gamma * F2bar)
                                        # Must multiply the grid point by field's grid2world
                                        x[0] = _apply_affine_3d_x0(ss, rr, cc, 1, fieldg2w)
                                        x[1] = _apply_affine_3d_x1(ss, rr, cc, 1, fieldg2w)
                                        x[2] = _apply_affine_3d_x2(ss, rr, cc, 1, fieldg2w)
                                        if constant_jacobian == 0:
                                            constant_jacobian = transform._jacobian(theta, x, J)

                                        for jj in range(n):
                                            h[jj] = 0
                                            for ii in range(3):
                                                h[jj] += df2[ss, rr, cc, ii] * J[ii,jj]
                                            h[jj] *= Jf2
                                            dtheta[jj] += factor * h[jj]

        for k in range(n):
            dtheta[k] = -2.0 * dtheta[k] / nwindows

        for k in range(kcoef.shape[0]):
            for i in range(kcoef.shape[1]):
                for j in range(kcoef.shape[2]):
                    kcoef[k, i, j] = -2.0 * kcoef[k, i, j] / nwindows
        energy = 1-energy/nwindows
    return energy





def cc_splines_grad_epicor_general(double[:,:,:] f1, double[:,:,:] f2,
                                  double[:,:] f1_g2w, double[:,:] f2_g2w,
                                  double[:] f1_pe, double[:] f2_pe,
                                  double[:,:,:,:] df1, double[:,:,:,:] df2,
                                  int[:,:,:] mf1, int[:,:,:] mf2,
                                  double[:,:,:] kernel, double[:,:,:,:] gkernel,
                                  double[:,:,:,:] gfield, double[:,:] fieldg2w,
                                  int[:] kspacing, int[:] kshape,
                                  cnp.npy_intp radius,
                                  Transform transform, double[:] theta,
                                  double[:,:,:] kcoef, double[:] dtheta):
    cdef:
        cnp.npy_intp ns = f1.shape[0]
        cnp.npy_intp nr = f1.shape[1]
        cnp.npy_intp nc = f1.shape[2]
        int* ksize = [kernel.shape[0], kernel.shape[1], kernel.shape[2]]
        cnp.npy_intp side = 2 * radius + 1
        int* kcenter = [ksize[0]//2, ksize[1]//2, ksize[2]//2]
        double[:,:,:] F1 = np.empty((ns,nr,nc), dtype=np.float64)
        double[:,:,:] F2 = np.empty((ns,nr,nc), dtype=np.float64)
        double[:,:,:] J1 = np.empty((ns,nr,nc), dtype=np.float64)
        double[:,:,:] J2 = np.empty((ns,nr,nc), dtype=np.float64)
        double[:, :, :, :] temp = np.zeros((ns, nr, nc, 5), dtype=np.float64)
        double[:,:] J = None
        double[:] h = None
        double[:] x = np.zeros((3,), dtype=np.float64)
        double[:] rot_center = np.zeros((3,), dtype=np.float64)
        cnp.npy_intp nfactors = 3
        double[:, :, :, :] factors = np.zeros((2, nr, nc, nfactors), dtype=np.float64)
        cnp.npy_intp constant_jacobian=0
        cnp.npy_intp sss, ss, rr, cc, prev_sss, prev_rr, prev_cc
        cnp.npy_intp s, r, c, i, j, k, ii, jj, i0, i1, j0, j1, k0, k1, sc_s, sc_r, sc_c
        double F2bar, F1bar, alpha, beta, gamma, sp_contrib, dF2, dF1, A, B, C, energy
        double factor

    n = transform.number_of_parameters
    J = np.zeros((3, n), dtype=np.float64)
    h = np.zeros((n,), dtype=np.float64)
    dtheta[:] = 0
    kcoef[...] = 0
    rot_center[0] = 0.5 * ns
    rot_center[1] = 0.5 * nr
    rot_center[2] = 0.5 * nc
    with nogil:
        for s in range(ns):
            for r in range(nr):
                for c in range(nc):
                    J1[s,r,c] = 1.0 + gfield[s,r,c,0]*f1_pe[0]+gfield[s,r,c,1]*f1_pe[1]+gfield[s,r,c,2]*f1_pe[2]
                    J2[s,r,c] = 1.0 + gfield[s,r,c,0]*f2_pe[0]+gfield[s,r,c,1]*f2_pe[1]+gfield[s,r,c,2]*f2_pe[2]
                    F1[s,r,c] = f1[s,r,c] * J1[s, r, c]
                    F2[s,r,c] = f2[s,r,c] * J2[s, r, c]

    # Precompute window sums
    temp = precompute_cc_factors_3d(F1, F2, radius)
    with nogil:
        energy = 0
        sss = 1
        for s in range(ns):
            ss = _mod(s - radius, ns)
            sss = 1 - sss
            for r in range(nr):
                rr = _mod(r - radius, nr)
                for c in range(nc):
                    cc = _mod(c - radius, nc)
                    # New corner
                    _increment_factors_epicor(factors, temp, sss, rr, cc, s, r, c, 0)
                    # Add signed sub-volumes
                    if s>0:
                        prev_sss = 1 - sss
                        for idx in range(nfactors):
                            factors[sss, rr, cc, idx] += factors[prev_sss, rr, cc, idx]
                        if r>0:
                            prev_rr = _mod(rr-1, nr)
                            for idx in range(nfactors):
                                factors[sss, rr, cc, idx] -= factors[prev_sss, prev_rr, cc, idx]
                            if c>0:
                                prev_cc = _mod(cc-1, nc)
                                for idx in range(nfactors):
                                    factors[sss, rr, cc, idx] += factors[prev_sss, prev_rr, prev_cc, idx]
                        if c>0:
                            prev_cc = _mod(cc-1, nc)
                            for idx in range(nfactors):
                                factors[sss, rr, cc, idx] -= factors[prev_sss, rr, prev_cc, idx]
                    if(r>0):
                        prev_rr = _mod(rr-1, nr)
                        for idx in range(nfactors):
                            factors[sss, rr, cc, idx] += factors[sss, prev_rr, cc, idx]
                        if(c>0):
                            prev_cc = _mod(cc-1, nc)
                            for idx in range(nfactors):
                                factors[sss, rr, cc, idx] -= factors[sss, prev_rr, prev_cc, idx]
                    if(c>0):
                        prev_cc = _mod(cc-1, nc)
                        for idx in range(nfactors):
                            factors[sss, rr, cc, idx] += factors[sss, rr, prev_cc, idx]
                    # Add signed corners
                    if s>=side:
                        _increment_factors_epicor(factors, temp, sss, rr, cc, s-side, r, c, -1)
                        if r>=side:
                            _increment_factors_epicor(factors, temp, sss, rr, cc, s-side, r-side, c, 1)
                            if c>=side:
                                _increment_factors_epicor(factors, temp, sss, rr, cc, s-side, r-side, c-side, -1)
                        if c>=side:
                            _increment_factors_epicor(factors, temp, sss, rr, cc, s-side, r, c-side, 1)
                    if r>=side:
                        _increment_factors_epicor(factors, temp, sss, rr, cc, s, r-side, c, -1)
                        if c>=side:
                            _increment_factors_epicor(factors, temp, sss, rr, cc, s, r-side, c-side, 1)

                    if c>=side:
                        _increment_factors_epicor(factors, temp, sss, rr, cc, s, r, c-side, -1)
                    # Compute final factors
                    if ss>=radius and rr>=radius and cc>=radius:
                        # Iterate over all splines affected by (ss, rr, cc)
                        # The first-order rectangle integrals are at temp[ss, rr, cc, :]
                        # The second-order rectangle integrals are at factors[sss, rr, cc, :]
                        F1bar = temp[ss, rr, cc, 0]
                        F2bar = temp[ss, rr, cc, 1]

                        A = temp[ss, rr, cc, 2]  # dot product: A
                        B = temp[ss, rr, cc, 3]  # moving sq norm: B
                        C = temp[ss, rr, cc, 4]  # static sq norm: C

                        if (B*C > 1e-7):
                            energy += (A * A) / (B * C)

                            alpha = factors[sss, rr, cc, 0]
                            beta = factors[sss, rr, cc, 1]
                            gamma = factors[sss, rr, cc, 2]

                            # First and last knot indices being affected by (ss, rr, cc)
                            k0 = (ss - kcenter[0] + kspacing[0] - 1) // kspacing[0]
                            k1 = (ss + kcenter[0]) // kspacing[0]
                            i0 = (rr - kcenter[1] + kspacing[1] - 1) // kspacing[1]
                            i1 = (rr + kcenter[1]) // kspacing[1]
                            j0 = (cc - kcenter[2] + kspacing[2] - 1) // kspacing[2]
                            j1 = (cc + kcenter[2]) // kspacing[2]
                            for k in range(k0, 1 + k1):
                                sc_s = k * kspacing[0] - kcenter[0]  # First grid cell affected by this spline
                                for i in range(i0, 1 + i1):
                                    sc_r = i * kspacing[1] - kcenter[1]
                                    for j in range(j0, 1 + j1):
                                        sc_c = j * kspacing[0] - kcenter[2]
                                        # The *first* grid cell affected by this
                                        # spline (sc_s, sc_r, sc_c) may be out of
                                        # bounds, but it doesn't matter, because
                                        # we are evaluating the spline at
                                        # [ss - sc_s, rr - sc_r, cc - sc_c]
                                        # which we know is a valid kernel index

                                        # Add first term of the derivative of F_i w.r.t. c_k
                                        dF1 = df1[ss, rr, cc, 0]*f1_pe[0] +\
                                              df1[ss, rr, cc, 1]*f1_pe[1] +\
                                              df1[ss, rr, cc, 2]*f1_pe[2]
                                        dF2 = df2[ss, rr, cc, 0]*f2_pe[0] +\
                                              df2[ss, rr, cc, 1]*f2_pe[1] +\
                                              df2[ss, rr, cc, 2]*f2_pe[2]
                                        dF1 *= kernel[ss - sc_s, rr - sc_r, cc - sc_c] * J1[ss,rr,cc]
                                        dF2 *= kernel[ss - sc_s, rr - sc_r, cc - sc_c] * J2[ss,rr,cc]

                                        dF1 += F1bar*(gkernel[ss - sc_s, rr - sc_r, cc - sc_c, 0]*f1_pe[0]/kspacing[0] +\
                                               gkernel[ss - sc_s, rr - sc_r, cc - sc_c, 1]*f1_pe[1]/kspacing[1] +\
                                               gkernel[ss - sc_s, rr - sc_r, cc - sc_c, 2]*f1_pe[2]/kspacing[2])
                                        dF2 += F2bar*(gkernel[ss - sc_s, rr - sc_r, cc - sc_c, 0]*f2_pe[0]/kspacing[0] +\
                                               gkernel[ss - sc_s, rr - sc_r, cc - sc_c, 1]*f2_pe[1]/kspacing[1] +\
                                               gkernel[ss - sc_s, rr - sc_r, cc - sc_c, 2]*f2_pe[2]/kspacing[2])

                                        sp_contrib = ((alpha * F2bar - beta * F1bar) * dF1 +
                                                     (alpha * F1bar - gamma * F2bar) * dF2)

                                        # The actual index of this knot in the knot coefficient array is (k+1, i+1, j+1)
                                        kcoef[k+1, i+1, j+1] += sp_contrib

                                        # Accumulate contribution to motion gradient
                                        factor = (alpha * F1bar - gamma * F2bar)

                                        if theta is not None and constant_jacobian == 0:
                                            x[0] = ss - rot_center[0]
                                            x[1] = rr - rot_center[1]
                                            x[2] = cc - rot_center[2]
                                            constant_jacobian = transform._jacobian(theta, x, J)

                                        for jj in range(n):
                                            h[jj] = 0
                                            for ii in range(3):
                                                h[jj] += df2[ss, rr, cc, ii] * J[ii,jj]
                                            h[jj] *= J2[ss,rr,cc]
                                            dtheta[jj] += factor * h[jj]
        for k in range(n):
            dtheta[k] = -2.0 * dtheta[k]

        for k in range(kcoef.shape[0]):
            for i in range(kcoef.shape[1]):
                for j in range(kcoef.shape[2]):
                    kcoef[k, i, j] = -2.0 * kcoef[k, i, j]
        energy = -energy
    return energy










