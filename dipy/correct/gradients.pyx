#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
import numpy as np
cimport numpy as cnp
cimport cython
from dipy.align.fused_types cimport floating, number
from dipy.correct.splines import CubicSplineField

from dipy.align.vector_fields cimport(_apply_affine_3d_x0,
                                      _apply_affine_3d_x1,
                                      _apply_affine_3d_x2,
                                      _apply_affine_2d_x0,
                                      _apply_affine_2d_x1)

cdef extern from "math.h":
    double sqrt(double x) nogil
    double floor(double x) nogil


cdef inline int _interpolate_scalar_3d(floating[:, :, :] volume,
                                       double dkk, double dii, double djj,
                                       floating *out) nogil:
    r"""Trilinear interpolation of a 3D scalar image

    Interpolates the 3D image at (dkk, dii, djj) and stores the
    result in out. If (dkk, dii, djj) is outside the image's domain,
    zero is written to out instead.

    Parameters
    ----------
    image : array, shape (R, C)
        the input 2D image
    dkk : floating
        the first coordinate of the interpolating position
    dii : floating
        the second coordinate of the interpolating position
    djj : floating
        the third coordinate of the interpolating position
    out : array, shape (2,)
        the array which the interpolation result will be written to

    Returns
    -------
    inside : int
        if (dkk, dii, djj) is inside the domain of the image,
        inside == 1, otherwise inside == 0
    """
    cdef:
        cnp.npy_intp ns = volume.shape[0]
        cnp.npy_intp nr = volume.shape[1]
        cnp.npy_intp nc = volume.shape[2]
        cnp.npy_intp kk, ii, jj
        int inside
        double alpha, beta, calpha, cbeta, gamma, cgamma
    if not (-1 < dkk < ns and -1 < dii < nr and -1 < djj < nc):
        out[0] = 0
        return 0
    # find the top left index and the interpolation coefficients
    kk = <int>floor(dkk)
    ii = <int>floor(dii)
    jj = <int>floor(djj)
    # no one is affected

    cgamma = dkk - kk
    calpha = dii - ii
    cbeta = djj - jj
    alpha = 1 - calpha
    beta = 1 - cbeta
    gamma = 1 - cgamma

    inside = 0
    # ---top-left
    if (ii >= 0) and (jj >= 0) and (kk >= 0):
        out[0] = alpha * beta * gamma * volume[kk, ii, jj]
        inside += 1
    else:
        out[0] = 0
    # ---top-right
    jj += 1
    if (ii >= 0) and (jj < nc) and (kk >= 0):
        out[0] += alpha * cbeta * gamma * volume[kk, ii, jj]
        inside += 1
    # ---bottom-right
    ii += 1
    if (ii < nr) and (jj < nc) and (kk >= 0):
        out[0] += calpha * cbeta * gamma * volume[kk, ii, jj]
        inside += 1
    # ---bottom-left
    jj -= 1
    if (ii < nr) and (jj >= 0) and (kk >= 0):
        out[0] += calpha * beta * gamma * volume[kk, ii, jj]
        inside += 1
    kk += 1
    if(kk < ns):
        ii -= 1
        if (ii >= 0) and (jj >= 0):
            out[0] += alpha * beta * cgamma * volume[kk, ii, jj]
            inside += 1
        jj += 1
        if (ii >= 0) and (jj < nc):
            out[0] += alpha * cbeta * cgamma * volume[kk, ii, jj]
            inside += 1
        # ---bottom-right
        ii += 1
        if (ii < nr) and (jj < nc):
            out[0] += calpha * cbeta * cgamma * volume[kk, ii, jj]
            inside += 1
        # ---bottom-left
        jj -= 1
        if (ii < nr) and (jj >= 0):
            out[0] += calpha * beta * cgamma * volume[kk, ii, jj]
            inside += 1
    return 1 if inside == 8 else 0


cdef inline int _interpolate_vector_3d(floating[:, :, :, :] field, double dkk,
                                       double dii, double djj,
                                       floating[:] out) nogil:
    r"""Trilinear interpolation of a 3D vector field

    Interpolates the 3D displacement field at (dkk, dii, djj) and stores the
    result in out. If (dkk, dii, djj) is outside the vector field's domain, a
    zero vector is written to out instead.

    Parameters
    ----------
    field : array, shape (S, R, C)
        the input 3D displacement field
    dkk : floating
        the first coordinate of the interpolating position
    dii : floating
        the second coordinate of the interpolating position
    djj : floating
        the third coordinate of the interpolating position
    out : array, shape (3,)
        the array which the interpolation result will be written to

    Returns
    -------
    inside : int
        if (dkk, dii, djj) is inside the domain of the displacement field,
        inside == 1, otherwise inside == 0
    """
    cdef:
        cnp.npy_intp ns = field.shape[0]
        cnp.npy_intp nr = field.shape[1]
        cnp.npy_intp nc = field.shape[2]
        cnp.npy_intp kk, ii, jj
        int inside
        double alpha, beta, gamma, calpha, cbeta, cgamma
    if not (-1 < dkk < ns and -1 < dii < nr and -1 < djj < nc):
        out[0] = 0
        out[1] = 0
        out[2] = 0
        return 0
    #---top-left
    kk = <int>floor(dkk)
    ii = <int>floor(dii)
    jj = <int>floor(djj)

    cgamma = dkk - kk
    calpha = dii - ii
    cbeta = djj - jj
    alpha = 1 - calpha
    beta = 1 - cbeta
    gamma = 1 - cgamma

    inside = 0
    if (ii >= 0) and (jj >= 0) and (kk >= 0):
        out[0] = alpha * beta * gamma * field[kk, ii, jj, 0]
        out[1] = alpha * beta * gamma * field[kk, ii, jj, 1]
        out[2] = alpha * beta * gamma * field[kk, ii, jj, 2]
        inside += 1
    else:
        out[0] = 0
        out[1] = 0
        out[2] = 0
    # ---top-right
    jj += 1
    if (jj < nc) and (ii >= 0) and (kk >= 0):
        out[0] += alpha * cbeta * gamma * field[kk, ii, jj, 0]
        out[1] += alpha * cbeta * gamma * field[kk, ii, jj, 1]
        out[2] += alpha * cbeta * gamma * field[kk, ii, jj, 2]
        inside += 1
    # ---bottom-right
    ii += 1
    if (jj < nc) and (ii < nr) and (kk >= 0):
        out[0] += calpha * cbeta * gamma * field[kk, ii, jj, 0]
        out[1] += calpha * cbeta * gamma * field[kk, ii, jj, 1]
        out[2] += calpha * cbeta * gamma * field[kk, ii, jj, 2]
        inside += 1
    # ---bottom-left
    jj -= 1
    if (jj >= 0) and (ii < nr) and (kk >= 0):
        out[0] += calpha * beta * gamma * field[kk, ii, jj, 0]
        out[1] += calpha * beta * gamma * field[kk, ii, jj, 1]
        out[2] += calpha * beta * gamma * field[kk, ii, jj, 2]
        inside += 1
    kk += 1
    if (kk < ns):
        ii -= 1
        if (jj >= 0) and (ii >= 0):
            out[0] += alpha * beta * cgamma * field[kk, ii, jj, 0]
            out[1] += alpha * beta * cgamma * field[kk, ii, jj, 1]
            out[2] += alpha * beta * cgamma * field[kk, ii, jj, 2]
            inside += 1
        jj += 1
        if (jj < nc) and (ii >= 0):
            out[0] += alpha * cbeta * cgamma * field[kk, ii, jj, 0]
            out[1] += alpha * cbeta * cgamma * field[kk, ii, jj, 1]
            out[2] += alpha * cbeta * cgamma * field[kk, ii, jj, 2]
            inside += 1
        # ---bottom-right
        ii += 1
        if (jj < nc) and (ii < nr):
            out[0] += calpha * cbeta * cgamma * field[kk, ii, jj, 0]
            out[1] += calpha * cbeta * cgamma * field[kk, ii, jj, 1]
            out[2] += calpha * cbeta * cgamma * field[kk, ii, jj, 2]
            inside += 1
        # ---bottom-left
        jj -= 1
        if (jj >= 0) and (ii < nr):
            out[0] += calpha * beta * cgamma * field[kk, ii, jj, 0]
            out[1] += calpha * beta * cgamma * field[kk, ii, jj, 1]
            out[2] += calpha * beta * cgamma * field[kk, ii, jj, 2]
            inside += 1
    return 1 if inside == 8 else 0


cdef inline int _interpolate_scalar_nn_3d(number[:, :, :] volume, double dkk,
                                         double dii, double djj,
                                         number *out) nogil:
    r"""Nearest-neighbor interpolation of a 3D scalar image

    Interpolates the 3D image at (dkk, dii, djj) using nearest neighbor
    interpolation and stores the result in out. If (dkk, dii, djj) is outside
    the image's domain, zero is written to out instead.

    Parameters
    ----------
    image : array, shape (S, R, C)
        the input 2D image
    dkk : float
        the first coordinate of the interpolating position
    dii : float
        the second coordinate of the interpolating position
    djj : float
        the third coordinate of the interpolating position
    out : array, shape (1,)
        the variable which the interpolation result will be written to

    Returns
    -------
    inside : int
        if (dkk, dii, djj) is inside the domain of the image,
        inside == 1, otherwise inside == 0
    """
    cdef:
        cnp.npy_intp ns = volume.shape[0]
        cnp.npy_intp nr = volume.shape[1]
        cnp.npy_intp nc = volume.shape[2]
        cnp.npy_intp kk, ii, jj
        double alpha, beta, calpha, cbeta, gamma, cgamma
    if not (0 <= dkk <= ns - 1 and 0 <= dii <= nr - 1 and 0 <= djj <= nc - 1):
        out[0] = 0
        return 0
    # find the top left index and the interpolation coefficients
    kk = <int>floor(dkk)
    ii = <int>floor(dii)
    jj = <int>floor(djj)
    # no one is affected
    if not ((0 <= kk < ns) and (0 <= ii < nr) and (0 <= jj < nc)):
        out[0] = 0
        return 0
    cgamma = dkk - kk
    calpha = dii - ii
    cbeta = djj - jj
    alpha = 1 - calpha
    beta = 1 - cbeta
    gamma = 1 - cgamma
    if(gamma < cgamma):
        kk += 1
    if(alpha < calpha):
        ii += 1
    if(beta < cbeta):
        jj += 1
    # no one is affected
    if not ((0 <= kk < ns) and (0 <= ii < nr) and (0 <= jj < nc)):
        out[0] = 0
        return 0
    out[0] = volume[kk, ii, jj]
    return 1


def warp_with_orfield(floating[:,:,:] f, floating[:,:,:] b, double[:] dir,
                        double[:, :] affine_idx_in=None,
                        double[:, :] affine_idx_out=None,
                        double[:, :] affine_disp=None,
                        int[:] sampling_shape=None):
    ftype=np.asarray(f).dtype
    cdef:
        int nslices = f.shape[0]
        int nrows = f.shape[1]
        int ncols = f.shape[2]
        int nsVol = f.shape[0]
        int nrVol = f.shape[1]
        int ncVol = f.shape[2]
        int i, j, k, inside
        double dkk, dii, djj, dk, di, dj
        floating tmp
    if sampling_shape is not None:
        nslices = sampling_shape[0]
        nrows = sampling_shape[1]
        ncols = sampling_shape[2]
    elif b is not None:
        nslices = b.shape[0]
        nrows = b.shape[1]
        ncols = b.shape[2]

    cdef floating[:, :, :] warped = np.zeros(shape=(nslices, nrows, ncols), dtype=ftype)
    cdef int[:, :, :] mask = np.zeros(shape=(nslices, nrows, ncols), dtype=np.int32)

    with nogil:

        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if affine_idx_in is None:
                        dkk = dir[0] * b[k,i,j]
                        dii = dir[1] * b[k,i,j]
                        djj = dir[2] * b[k,i,j]
                    else:
                        dk = _apply_affine_3d_x0(
                            k, i, j, 1, affine_idx_in)
                        di = _apply_affine_3d_x1(
                            k, i, j, 1, affine_idx_in)
                        dj = _apply_affine_3d_x2(
                            k, i, j, 1, affine_idx_in)
                        inside = _interpolate_scalar_3d[floating](b, dk, di, dj, &tmp)
                        dkk = dir[0] * tmp
                        dii = dir[1] * tmp
                        djj = dir[2] * tmp

                    if affine_disp is not None:
                        dk = _apply_affine_3d_x0(
                            dkk, dii, djj, 0, affine_disp)
                        di = _apply_affine_3d_x1(
                            dkk, dii, djj, 0, affine_disp)
                        dj = _apply_affine_3d_x2(
                            dkk, dii, djj, 0, affine_disp)
                    else:
                        dk = dkk
                        di = dii
                        dj = djj

                    if affine_idx_out is not None:
                        dkk = dk + _apply_affine_3d_x0(k, i, j, 1,
                                                       affine_idx_out)
                        dii = di + _apply_affine_3d_x1(k, i, j, 1,
                                                       affine_idx_out)
                        djj = dj + _apply_affine_3d_x2(k, i, j, 1,
                                                       affine_idx_out)
                    else:
                        dkk = dk + k
                        dii = di + i
                        djj = dj + j

                    mask[k,i,j] = _interpolate_scalar_3d[floating](f, dkk, dii, djj,
                                                          &warped[k,i,j])
    return warped, mask


def warp_with_orfield_nn(number[:,:,:] f, floating[:,:,:] b, double[:] dir,
                         double[:, :] affine_idx_in=None,
                         double[:, :] affine_idx_out=None,
                         double[:, :] affine_disp=None,
                         int[:] sampling_shape=None):
    ftype=np.asarray(f).dtype
    cdef:
        int nslices = f.shape[0]
        int nrows = f.shape[1]
        int ncols = f.shape[2]
        int nsVol = f.shape[0]
        int nrVol = f.shape[1]
        int ncVol = f.shape[2]
        int i, j, k, inside
        double dkk, dii, djj, dk, di, dj
        floating tmp
    if sampling_shape is not None:
        nslices = sampling_shape[0]
        nrows = sampling_shape[1]
        ncols = sampling_shape[2]
    elif b is not None:
        nslices = b.shape[0]
        nrows = b.shape[1]
        ncols = b.shape[2]

    cdef number[:, :, :] warped = np.zeros(shape=(nslices, nrows, ncols), dtype=ftype)
    cdef int[:, :, :] mask = np.zeros(shape=(nslices, nrows, ncols), dtype=np.int32)

    with nogil:

        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if affine_idx_in is None:
                        dkk = dir[0] * b[k,i,j]
                        dii = dir[1] * b[k,i,j]
                        djj = dir[2] * b[k,i,j]
                    else:
                        dk = _apply_affine_3d_x0(
                            k, i, j, 1, affine_idx_in)
                        di = _apply_affine_3d_x1(
                            k, i, j, 1, affine_idx_in)
                        dj = _apply_affine_3d_x2(
                            k, i, j, 1, affine_idx_in)
                        inside = _interpolate_scalar_3d(b, dk, di, dj, &tmp)
                        dkk = dir[0] * tmp
                        dii = dir[1] * tmp
                        djj = dir[2] * tmp

                    if affine_disp is not None:
                        dk = _apply_affine_3d_x0(
                            dkk, dii, djj, 0, affine_disp)
                        di = _apply_affine_3d_x1(
                            dkk, dii, djj, 0, affine_disp)
                        dj = _apply_affine_3d_x2(
                            dkk, dii, djj, 0, affine_disp)
                    else:
                        dk = dkk
                        di = dii
                        dj = djj

                    if affine_idx_out is not None:
                        dkk = dk + _apply_affine_3d_x0(k, i, j, 1,
                                                       affine_idx_out)
                        dii = di + _apply_affine_3d_x1(k, i, j, 1,
                                                       affine_idx_out)
                        djj = dj + _apply_affine_3d_x2(k, i, j, 1,
                                                       affine_idx_out)
                    else:
                        dkk = dk + k
                        dii = di + i
                        djj = dj + j

                    mask[k,i,j] = _interpolate_scalar_nn_3d(f, dkk, dii, djj,
                                                          &warped[k,i,j])
    return warped, mask




cdef void _der_y(floating[:,:,:] f, floating[:,:,:] df) nogil:
    cdef:
        int ns = f.shape[0]
        int nr = f.shape[1]
        int nc = f.shape[2]
        int k, i, j
        double h
    for k in range(ns):
        for i in range(nr):
            for j in range(nc):
                h = 0.5
                if i<nr-1:
                    df[k,i,j] = f[k,i+1,j]
                else:
                    h = 1.0
                    df[k,i,j] = f[k,i,j]

                if i>0:
                    df[k,i,j] = h * (df[k,i,j] - f[k,i-1,j])
                else:
                    h=1.0
                    df[k,i,j] = h * (df[k,i,j] - f[k,i,j])


def der_y(floating[:,:,:] f):
    ftype = np.asarray(f).dtype
    cdef:
        floating[:,:,:] df = np.empty_like(f, dtype=ftype)
    _der_y(f, df)
    return df


def resample_vector_field(floating[:,:,:,:] f, floating[:,:,:] b, double[:] dir,
                        double[:, :] affine_idx_in,
                        double[:, :] affine_idx_out,
                        double[:, :] affine_disp,
                        int[:] sampling_shape,
                        floating[:,:,:,:] out):
    ftype=np.asarray(f).dtype
    cdef:
        int nslices = f.shape[0]
        int nrows = f.shape[1]
        int ncols = f.shape[2]
        int nsVol = f.shape[0]
        int nrVol = f.shape[1]
        int ncVol = f.shape[2]
        int i, j, k, inside
        double dkk, dii, djj, dk, di, dj
        floating tmp
    if sampling_shape is not None:
        nslices = sampling_shape[0]
        nrows = sampling_shape[1]
        ncols = sampling_shape[2]
    elif b is not None:
        nslices = b.shape[0]
        nrows = b.shape[1]
        ncols = b.shape[2]

    #cdef floating[:, :, :, :] warped = np.zeros(shape=(nslices, nrows, ncols,3), dtype=ftype)
    #cdef int[:, :, :] mask = np.zeros(shape=(nslices, nrows, ncols), dtype=np.int32)

    with nogil:

        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if affine_idx_in is None:
                        dkk = dir[0] * b[k,i,j]
                        dii = dir[1] * b[k,i,j]
                        djj = dir[2] * b[k,i,j]
                    else:
                        dk = _apply_affine_3d_x0(
                            k, i, j, 1, affine_idx_in)
                        di = _apply_affine_3d_x1(
                            k, i, j, 1, affine_idx_in)
                        dj = _apply_affine_3d_x2(
                            k, i, j, 1, affine_idx_in)
                        inside = _interpolate_scalar_3d[floating](b, dk, di, dj, &tmp)
                        dkk = dir[0] * tmp
                        dii = dir[1] * tmp
                        djj = dir[2] * tmp

                    if affine_disp is not None:
                        dk = _apply_affine_3d_x0(
                            dkk, dii, djj, 0, affine_disp)
                        di = _apply_affine_3d_x1(
                            dkk, dii, djj, 0, affine_disp)
                        dj = _apply_affine_3d_x2(
                            dkk, dii, djj, 0, affine_disp)
                    else:
                        dk = dkk
                        di = dii
                        dj = djj

                    if affine_idx_out is not None:
                        dkk = dk + _apply_affine_3d_x0(k, i, j, 1,
                                                       affine_idx_out)
                        dii = di + _apply_affine_3d_x1(k, i, j, 1,
                                                       affine_idx_out)
                        djj = dj + _apply_affine_3d_x2(k, i, j, 1,
                                                       affine_idx_out)
                    else:
                        dkk = dk + k
                        dii = di + i
                        djj = dj + j

                    _interpolate_vector_3d[floating](f, dkk, dii, djj,
                                                          out[k,i,j])





cdef void _grad_holland(floating[:,:,:] fp, floating[:,:,:] fm,
                        floating[:,:,:] dfp, floating[:,:,:] dfm,
                        floating[:,:,:] b, floating[:,:,:] db,
                        double l1, double l2, floating[:,:,:] out_grad):
    cdef:
        int NUM_NEIGHBORS = 6
        int[:] dSlice = np.array([-1,  0, 0, 0,  0, 1], dtype=np.int32)
        int[:] dRow = np.array([0, -1, 0, 1,  0, 0], dtype=np.int32)
        int[:] dCol = np.array([0,  0, 1, 0, -1, 0], dtype=np.int32)
        int ns = fp.shape[0]
        int nr = fp.shape[1]
        int nc = fp.shape[2]
        int i, j, k, kk, ii, jj, idx
        double delta_c, delta_p, delta_m
        double diff_c, diff_p, diff_m
        double Jc, Jp, Jm
    with nogil:
        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    Jc = db[k, i, j]
                    delta_c = fp[k,i,j]*(1.0-Jc) - fm[k,i,j]*(1.0+Jc)

                    if(i < nr - 1):
                        Jp = db[k, i+1, j]
                        delta_p = fp[k,i+1,j]*(1.0-Jp) - fm[k,i+1,j]*(1.0+Jp)

                    if(i > 0):
                        Jm = db[k, i-1, j]
                        delta_m = fp[k,i-1,j]*(1.0-Jm) - fm[k,i-1,j]*(1.0+Jm)

                    if(i == nr-1):
                        # Derivative of [fp*(1-db) - fm(1+db)] at [k,i,j]
                        # w.r.t b[k,i,j]
                        # Here the derivative of db[k,i,j] w.r.t. b[k,i,j] is not
                        # zero because db[k,i,j] = b[k,i,j] - b[k,i-1,j]
                        diff_c = -1.0*(dfp[k,i,j]*(1.0 - Jc) + fp[k,i,j]) + \
                                 -1.0*(dfm[k,i,j]*(1.0 + Jc) + fm[k,i,j])

                        # Derivative of [fp*(1-db) - fm(1+db)] at [k,i-1,j]
                        # w.r.t b[k,i,j]
                        diff_m = -1.0*(fm[k,i-1,j] + fp[k,i-1,j])
                        out_grad[k,i,j] = delta_c*diff_c + delta_m*diff_m
                    elif(i == 0):
                        # Derivative of [fp*(1-db) - fm(1+db)] at [k,i,j]
                        # w.r.t b[k,i,j]
                        # Here the derivative of db[k,i,j] w.r.t. b[k,i,j] is not
                        # zero because db[k,i,j] = b[k,i+1,j] - b[k,i,j]
                        diff_c = -1.0*dfp[k,i,j]*(1.0 - Jc) + fp[k,i,j] + \
                                 -1.0*dfm[k,i,j]*(1.0 + Jc) + fm[k,i,j]

                        # Derivative of [fp*(1-db) - fm(1+db)] at [k,i+1,j]
                        # w.r.t b[k,i,j]
                        diff_p = fm[k,i+1,j] + fp[k,i+1,j]
                        out_grad[k,i,j] = delta_c*diff_c + delta_p*diff_p
                    else:
                        # Derivative of [fp*(1-db) - fm(1+db)] at [k,i,j]
                        # w.r.t b[k,i,j]
                        # Here the derivative of db[k,i,j] w.r.t. b[k,i,j] is zero
                        diff_c = -1.0*dfp[k,i,j]*(1.0 - Jc) + \
                                 -1.0*dfm[k,i,j]*(1.0 + Jc)
                        # Derivative of [fp*(1-db) - fm(1+db)] at [k,i+1,j]
                        # w.r.t b[k,i,j]
                        diff_p = 0.5 * (fp[k,i+1,j] + fm[k,i+1,j])
                        # Derivative of [fp*(1-db) - fm(1+db)] at [k,i-1,j]
                        # w.r.t b[k,i,j]
                        diff_m = -0.5 * (fp[k,i-1,j] + fm[k,i-1,j])

                        out_grad[k,i,j] = delta_c*diff_c + delta_p*diff_p + delta_m*diff_m

                    # Spatial regularization
                    for idx in range(NUM_NEIGHBORS):
                        kk = k + dSlice[idx]
                        if((kk < 0) or (kk >= ns)):
                            continue
                        ii = i + dRow[idx]
                        if((ii < 0) or (ii >= nr)):
                            continue
                        jj = j + dCol[idx]
                        if((jj < 0) or (jj >= nc)):
                            continue

                        out_grad[k,i,j] += l2 * (b[k,i,j] - b[kk,ii,jj])

                    # Tikonov regularization
                    out_grad[k,i,j] += l1 * b[k,i,j]


def grad_holland(floating[:,:,:] fp, floating[:,:,:] fm,
                 floating[:,:,:] dfp, floating[:,:,:] dfm,
                 floating[:,:,:] b, floating[:,:,:] db,
                 double l1, double l2):
    ftype=np.asarray(b).dtype
    cdef:
        int ns = b.shape[0]
        int nr = b.shape[1]
        int nc = b.shape[2]
        floating[:,:,:] out_grad = np.zeros(shape=(ns, nr, nc), dtype=ftype)

    _grad_holland(fp, fm, dfp, dfm, b, db, l1, l2, out_grad)

    return out_grad

def test_gauss_newton_holland(floating[:,:,:] fp, floating[:,:,:] fm,
                              floating[:,:,:] dfp, floating[:,:,:] dfm,
                              floating[:,:,:] db, double l1, double l2):
    cdef:
        int NUM_NEIGHBORS = 6
        int[:] dSlice = np.array([-1,  0, 0, 0,  0, 1], dtype=np.int32)
        int[:] dRow = np.array([0, -1, 0, 1,  0, 0], dtype=np.int32)
        int[:] dCol = np.array([0,  0, 1, 0, -1, 0], dtype=np.int32)
        int ns = fp.shape[0]
        int nr = fp.shape[1]
        int nc = fp.shape[2]
        int i, j, k, row, col, cell_count
        double d_prev, d_center, d_next
        double[:,:] Jt = np.zeros((ns*nr*nc, ns*nr*nc), dtype=np.float64)
        double[:] residual = np.zeros((ns*nr*nc), dtype=np.float64)
    with nogil:
        # voxel indices are in lexicographical order:
        # (k,i,j) -> k*(nr*nc) + i*(nc) + j
        # Build Jt
        row = 0
        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    residual[row] = fp[k,i,j] * (1 + db[k,i,j]) -\
                                    fm[k,i,j] * (1 - db[k,i,j])
                    d_prev = 0
                    d_center = 0
                    d_next = 0

                    # Derivative of h(k, i, j) w.r.t. b(k, i, j)
                    if i == 0:
                        d_center -= fp[k,i,j] + fm[k,i,j]
                    if i == nr -1:
                        d_center += fp[k,i,j] + fm[k,i,j]
                    d_center += dfp[k,i,j] * (1 + db[k,i,j]) +\
                                dfm[k,i,j] * (1 - db[k,i,j])
                    Jt[row, row] = d_center

                    # Derivative of h(k, i-1, j) w.r.t. b(k, i, j)
                    if i > 0:
                        if i > 1:
                            factor = 0.5
                        else:
                            factor = 1.0
                        d_prev = factor * (fp[k,i-1,j] + fm[k,i-1,j])
                        Jt[row, row - nc] = d_prev

                    #derivative of h(k, i+1, j) w.r.t. b(k, i, j)
                    if i < nr - 1:
                        if i < nr-2:
                            factor = -0.5
                        else:
                            factor = -1.0
                        d_next = factor * (fp[k,i+1,j] + fm[k,i+1,j])
                        Jt[row, row + nc] = d_next
                    row += 1
    Jth = np.dot(Jt, residual)
    JtJ = np.dot(Jt, np.transpose(Jt))
    JtJ[np.diag_indices(ns*nr*nc, 2)] += l1

    return Jth, JtJ


def gauss_newton_system_holland(floating[:,:,:] fp, floating[:,:,:] fm,
                                floating[:,:,:] dfp, floating[:,:,:] dfm,
                                floating[:,:,:] db, double l1, double l2):
    cdef:
        int NUM_NEIGHBORS = 6
        int[:] dSlice = np.array([-1,  0, 0, 0,  0, 1], dtype=np.int32)
        int[:] dRow = np.array([0, -1, 0, 1,  0, 0], dtype=np.int32)
        int[:] dCol = np.array([0,  0, 1, 0, -1, 0], dtype=np.int32)
        int ns = fp.shape[0]
        int nr = fp.shape[1]
        int nc = fp.shape[2]
        int i, j, k, row, col, cell_count
        double d_prev, d_center, d_next, residual
        double[:,:] Jt = np.ndarray((ns*nr*nc, 3), dtype=np.float64)
        double[:] JtJ = np.zeros((ns*nr*nc*5), dtype=np.float64)
        double[:] Jth = np.zeros((ns*nr*nc), dtype=np.float64)
        double energy
        int[:] indices = np.ndarray((ns*nr*nc*5), dtype=np.int32)
        int[:] indptr = np.ndarray((1 + ns*nr*nc), dtype=np.int32)

    with nogil:
        # voxel indices are in lexicographical order:
        # (k,i,j) -> k*(nr*nc) + i*(nc) + j
        # Build Jt
        energy = 0
        row = 0
        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    d_prev = 0
                    d_center = 0
                    d_next = 0

                    # Derivative of h(k, i, j) w.r.t. b(k, i, j)
                    if i == 0:
                        d_center -= fp[k,i,j] + fm[k,i,j]
                    if i == nr -1:
                        d_center += fp[k,i,j] + fm[k,i,j]
                    d_center += dfp[k,i,j] * (1 + db[k,i,j]) +\
                                dfm[k,i,j] * (1 - db[k,i,j])
                    residual = fp[k,i,j] * (1 + db[k,i,j]) -\
                               fm[k,i,j] * (1 - db[k,i,j])
                    Jth[row] += d_center * residual
                    energy += residual*residual

                    # Derivative of h(k, i-1, j) w.r.t. b(k, i, j)
                    if i > 0:
                        if i > 1:
                            factor = 0.5
                        else:
                            factor = 1.0
                        residual = fp[k,i-1,j] * (1 + db[k,i-1,j]) -\
                                   fm[k,i-1,j] * (1 - db[k,i-1,j])
                        d_prev = factor * (fp[k,i-1,j] + fm[k,i-1,j])
                        Jth[row] += d_prev * residual

                    #derivative of h(k, i+1, j) w.r.t. b(k, i, j)
                    if i < nr - 1:
                        if i < nr-2:
                            factor = -0.5
                        else:
                            factor = -1.0
                        residual = fp[k,i+1,j] * (1 + db[k,i+1,j]) -\
                                   fm[k,i+1,j] * (1 - db[k,i+1,j])
                        d_next = factor * (fp[k,i+1,j] + fm[k,i+1,j])
                        Jth[row] += d_next * residual
                    Jt[row, 0] = d_prev
                    Jt[row, 1] = d_center
                    Jt[row, 2] = d_next
                    row += 1

        row = 0
        cell_count = 0
        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    indptr[row] = cell_count
                    if i < nr - 2:
                        col = row + 2*nc
                        JtJ[cell_count] = Jt[row, 2] * Jt[col, 0]
                        indices[cell_count] = col
                        cell_count +=1
                    if i < nr -1:
                        col = row + nc
                        JtJ[cell_count] = Jt[row, 1] * Jt[col, 0]+\
                                          Jt[row, 2] * Jt[col, 1]
                        indices[cell_count] = col
                        cell_count +=1

                    col = row
                    JtJ[cell_count] = Jt[row, 0] * Jt[col, 0] +\
                                      Jt[row, 1] * Jt[col, 1] +\
                                      Jt[row, 2] * Jt[col, 2] + l1
                    indices[cell_count] = col
                    cell_count +=1

                    if i > 0:
                        col = row - nc
                        JtJ[cell_count] = Jt[row, 0] * Jt[col, 1] +\
                                          Jt[row, 1] * Jt[col, 2]
                        indices[cell_count] = col
                        cell_count +=1
                    if i > 1:
                        col = row - 2*nc
                        JtJ[cell_count] = Jt[row, 0] * Jt[col, 2]
                        indices[cell_count] = col
                        cell_count +=1
                    row += 1
        indptr[row] = cell_count
    print(">>%f\n"%(energy,))
    return Jth, JtJ, indices, indptr


cdef inline void _overlap_offsets(int idx0, int idx1, int kspacing, int sp_len,
                           int *begin0, int *begin1, int *overlap_len) nogil:
    if idx0 < idx1: # then idx0-th spline requires offset
        begin0[0] = (idx1 - idx0) * kspacing
        begin1[0] = 0
        overlap_len[0] = sp_len - begin0[0]
    else:
        begin0[0] = 0
        begin1[0] = (idx0 - idx1) * kspacing
        overlap_len[0] = sp_len - begin1[0]


cdef double _mult_overlapping_splines(double[:,:] Jt, int[:] kspacing,
                                      int[:] kshape, Py_ssize_t[:] kernel_shape,
                                      int k0, int i0, int j0,
                                      int k1, int i1, int j1):
    cdef:
        int* begin0 = [0, 0, 0]
        int* begin1 = [0, 0, 0]
        int* overlap_len = [0, 0, 0]
        # Rows in sparse matrix corresponding to (k0, i0, j0) and (k1, i1, j1)
        int row0 = (k0 * kshape[1] + i0 ) * kshape[2] +  j0
        int row1 = (k1 * kshape[1] + i1 ) * kshape[2] +  j1
        # Iterators over corresponding indices in both spline kernels
        int index0, index1
        double prod
        int k, i, j

    with nogil:
        _overlap_offsets(k0, k1, kspacing[0], kernel_shape[0], &begin0[0], &begin1[0], &overlap_len[0])
        _overlap_offsets(i0, i1, kspacing[1], kernel_shape[1], &begin0[1], &begin1[1], &overlap_len[1])
        _overlap_offsets(j0, j1, kspacing[2], kernel_shape[2], &begin0[2], &begin1[2], &overlap_len[2])

        #if overlap_len[0]<0 or overlap_len[1]<0 or overlap_len[2]<0:
        #    with gil:
        #        print("Negative length: %d, %d, %d", overlap_len[0], overlap_len[1], overlap_len[2])

        # We no longer require k0, k1, i0, i1, j0, j1, we will re-use them
        prod = 0
        for k in range(overlap_len[0]):
            k0 = begin0[0] + k
            k1 = begin1[0] + k
            for i in range(overlap_len[1]):
                i0 = begin0[1] + i
                i1 = begin1[1] + i

                index0 = (k0 * kernel_shape[1] + i0) * kernel_shape[2] + begin0[2]
                index1 = (k1 * kernel_shape[1] + i1) * kernel_shape[2] + begin1[2]
                for j in range(overlap_len[2]):
                    prod += Jt[row0, index0] * Jt[row1, index1]
                    index0 += 1
                    index1 += 1
    return prod


cdef void _JtJ(double[:,:] Jt, int[:] kspacing, int[:] kshape,
               Py_ssize_t[:] kernel_shape, double[:] JtJ,
               int[:] indices, int[:] indptr, double tau=0):
    r""" Returns JtJ + tau*Id
    """
    cdef:
        # Spline grid size
        int nks = kshape[0]
        int nkr = kshape[1]
        int nkc = kshape[2]
        # Iterators over spline kernels
        int k, i, j, kk, ii, jj
        # Index bounds of spline kernels interacting with that at (k,i,j)
        int k0, k1, i0, i1, j0, j1
        # Iterators over the sparse JtJ
        int kindex, cell_count
    with nogil:
        kindex = 0
        cell_count = 0
        for k in range(nks):
            k0 = k - 3 if k >= 3 else 0
            k1 = k + 4 if k + 4 <= nks else nks
            for i in range(nkr):
                i0 = i - 3 if i >= 3 else 0
                i1 = i + 4 if i + 4 <= nkr else nkr
                for j in range(nkc):
                    j0 = j - 3 if j >= 3 else 0
                    j1 = j + 4 if j + 4 <= nkc else nkc

                    # Mark start of entries corresponding to this row
                    indptr[kindex] = cell_count
                    for kk in range(k0, k1):
                        for ii in range(i0, i1):
                            for jj in range(j0, j1):
                                with gil:
                                    prod = _mult_overlapping_splines(Jt, kspacing,
                                            kshape, kernel_shape, k, i, j, kk, ii, jj)
                                JtJ[cell_count] = prod
                                indices[cell_count] = (kk * nkr + ii) * nkc + jj
                                if indices[cell_count] == kindex:
                                    JtJ[cell_count] += tau
                                cell_count += 1
                    kindex += 1
        indptr[kindex] = cell_count



def _append_right(double[:] Hdata, int[:] Hindices, int[:] Hindptr,
                  double[:] Rdata, int[:] Rindices, int[:] Rindptr,
                  int n, int k):
    r""" Append sparse matrix R to the right of H and R^T to the bottom
    H is nxn
    L is nxk
    O will be (n+k)x(n+k)
    The resulting sparse matrix will be of the form:

                |  H   R |
                | R^T  0 |

    """
    cdef:
        int nnz = Hindices[n] + Rindices[n]
        double[:] Odata = np.ndarray(nnz, dtype=np.float64)
        int[:] Oindices = np.ndarray(nnz, dtype=np.int32)
        int[:] Oindptr = np.ndarray(1+n+k, dtype=np.int32)
        int i, j, p, q, r
    with nogil:
        p = 0
        q = 0
        r = 0
        for i in range(n):
            Oindptr[i] = r
            # Copy H[i, ...]
            for j in range(Hindptr[i], Hindptr[i+1]):
                Odata[r] = Hdata[p]
                Oindices[r] = Hindices[p]
                p += 1
                r += 1
            # Copy R[i, ...]
            for j in range(Rindptr[i], Rindptr[i+1]):
                Odata[r] = Rdata[q]
                Oindices[r] = Rindices[q]
                q += 1
                r += 1
        Oindptr[n] = r
    return Odata, Oindices, Oindptr


def gauss_newton_system_andersson(floating[:,:,:] fp, floating[:,:,:] fm,
                                  floating[:,:,:] dfp, floating[:,:,:] dfm,
                                  int[:,:,:] mp, int[:,:,:] mm,
                                  double[:,:,:] kernel, double[:,:,:] dkernel,
                                  floating[:,:,:] db, int[:] kspacing, int[:] kshape,
                                  double l1, double l2):
    cdef:
        # Image grid size
        int ns = fp.shape[0]
        int nr = fp.shape[1]
        int nc = fp.shape[2]
        int nvox = ns*nr*nc

        # Spline grid size
        int nks = kshape[0]
        int nkr = kshape[1]
        int nkc = kshape[2]
        int ncoeff = nks*nkr*nkc

        # Spline kernel size
        int len_s = kernel.shape[0]
        int len_r = kernel.shape[1]
        int len_c = kernel.shape[2]
        int kernel_size = len_s * len_r * len_c

        # Iterators over spline knots in the spline grid
        int i, j, k
        # Iterators over voxels in the image grid affected by spline [i,j,k]
        int ii, jj, kk
        # Iterators over kernel cells, multiplying voxel [ii,jj,kk]
        int si, sj, sk
        # Delimiters of the image region affected by spline [i,j,k]
        int k0, k1, i0, i1, j0, j1
        # Kernel cell corresponding to the first voxel affected by spline [i,j,k]
        int koff, joff, ioff
        # Iterators over the sparse Jacobian
        int row, col, cell_count, vindex
        # Containers for energy computation
        double energy, residual, derivative

        double[:,:] Jt = np.zeros((ncoeff, kernel_size), dtype=np.float64)
        double[:] JtJ = np.zeros(ncoeff * 343, dtype=np.float64)
        double[:] Jth = np.zeros(ncoeff, dtype=np.float64)
        int[:] indices = np.ndarray(ncoeff * 343, dtype=np.int32)
        int[:] indptr = np.ndarray(1 + ncoeff, dtype=np.int32)

    with nogil:
        # voxel- and coefficient- indices  are in lexicographical order:
        # (k,i,j) -> k*(nkr*nkc) + i*(nkc) + j
        # (kk,ii,jj) -> kk*(nr*nc) + ii*nc + j

        # Build Jt
        energy = 0
        row = 0
        for k in range(nks):
            _volume_range_affected(k, kspacing[0], ns, len_s, &k0, &k1, &koff)
            for i in range(nkr):
                _volume_range_affected(i, kspacing[1], nr, len_r, &i0, &i1, &ioff)
                for j in range(nkc):
                    _volume_range_affected(j, kspacing[2], nc, len_c, &j0, &j1, &joff)

                    # Compute the contribution of this coefficient to volume
                    # region [k0,k1)x[i0,i1)x[j0,j1)

                    for kk in range(k0, k1):
                        for ii in range(i0, i1):
                            for jj in range(j0, j1):
                                sk = koff + kk - k0
                                si = ioff + ii - i0
                                sj = joff + jj - j0
                                residual = fp[kk,ii,jj] * (1 + db[kk,ii,jj]) -\
                                           fm[kk,ii,jj] * (1 - db[kk,ii,jj])
                                # The index of voxel [kk,ii,jj]
                                vindex = (kk*nr + ii)*nc + jj
                                if (mp[kk,ii,jj] == 0) or (mm[kk,ii,jj] == 0):
                                    continue

                                # The index of the spline component [sk,si,sj]
                                col = (sk * len_r + si) * len_c + sj

                                # Add contribution to voxel [kk,ii,jj]
                                # The derivative of the residual at [kk,ii,jj]
                                # with respect to coefficient [i,j,k]
                                derivative = \
                                    dfp[kk,ii,jj] * kernel[sk,si,sj] * (1+db[kk,ii,jj]) +\
                                    fp[kk,ii,jj] * dkernel[sk,si,sj] +\
                                    dfm[kk,ii,jj] * kernel[sk,si,sj] * (1-db[kk,ii,jj]) +\
                                    fm[kk,ii,jj] * dkernel[sk,si,sj]
                                Jt[row, col] = 2.0 * derivative
                                Jth[row] += 2.0 * derivative * residual
                                cell_count += 1
                    row += 1
        energy = 0
        nvox = 0
        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    if (mp[k, i, j] == 0) or (mm[k, i, j] == 0):
                        continue
                    residual = fp[k,i,j] * (1 + db[k,i,j]) -\
                               fm[k,i,j] * (1 - db[k,i,j])
                    energy += residual * residual
                    nvox += 1
    _JtJ(Jt, kspacing, kshape, kernel.shape, JtJ, indices, indptr, l1)
    energy /= nvox
    return Jth, JtJ, indices, indptr, energy


def test_gauss_newton_andersson(floating[:,:,:] fp, floating[:,:,:] fm,
                                floating[:,:,:] dfp, floating[:,:,:] dfm,
                                double[:,:,:] kernel, double[:,:,:] dkernel,
                                floating[:,:,:] db, int[:] kspacing, int[:] kshape,
                                double l1, double l2):
    cdef:
        # Image grid size
        int ns = fp.shape[0]
        int nr = fp.shape[1]
        int nc = fp.shape[2]
        int nvox = ns*nr*nc

        # Spline grid size
        int nks = kshape[0]
        int nkr = kshape[1]
        int nkc = kshape[2]
        int ncoeff = nks*nkr*nkc

        # Spline kernel size
        int len_s = kernel.shape[0]
        int len_r = kernel.shape[1]
        int len_c = kernel.shape[2]
        int kernel_size = len_s * len_r * len_c

        # Iterators over spline knots in the spline grid
        int i, j, k
        # Iterators over voxels in the image grid affected by spline [i,j,k]
        int ii, jj, kk
        # Iterators over kernel cells, multiplying voxel [ii,jj,kk]
        int si, sj, sk
        # Delimiters of the image region affected by spline [i,j,k]
        int k0, k1, i0, i1, j0, j1
        # Kernel cell corresponding to the first voxel affected by spline [i,j,k]
        int koff, joff, ioff
        # Iterators over the sparse Jacobian
        int row, col, cell_count
        # Containers for energy computation
        double energy, derivative

        double[:,:] Jt = np.zeros((ncoeff, nvox), dtype=np.float64)
        double[:] residual = np.zeros(nvox, dtype=np.float64)
    print("Test Gauss newton Andersson")
    print("Image grid shape: (%d, %d, %d)"%(ns, nr, nc))
    print("Spline grid shape: (%d, %d, %d)"%(nks, nkr, nkc))
    print("Spline kernel shape: (%d, %d, %d)"%(len_s, len_r, len_c))
    with nogil:
        # voxel- and coefficient- indices  are in lexicographical order:
        # (k,i,j) -> k*(nkr*nkc) + i*(nkc) + j
        # (kk,ii,jj) -> kk*(nr*nc) + ii*nc + j

        # Build Jt
        energy = 0
        row = 0
        # For each spline knot
        for k in range(nks):
            _volume_range_affected(k, kspacing[0], ns, len_s, &k0, &k1, &koff)
            for i in range(nkr):
                _volume_range_affected(i, kspacing[1], nr, len_r, &i0, &i1, &ioff)
                for j in range(nkc):
                    _volume_range_affected(j, kspacing[2], nc, len_c, &j0, &j1, &joff)
                    # This knot affects region [k0, k1)x[i0, i1)x[j0, j1)

                    for kk in range(k0, k1):
                        for ii in range(i0, i1):
                            for jj in range(j0, j1):
                                # Position of kernel cell multiplying voxel [kk,ii,jj]
                                sk = koff + kk - k0
                                si = ioff + ii - i0
                                sj = joff + jj - j0

                                # Index of voxel [kk,ii,jj]
                                col = (kk * nr + ii) * nc + jj

                                residual[col] = fp[kk,ii,jj] * (1 + db[kk,ii,jj]) -\
                                                fm[kk,ii,jj] * (1 - db[kk,ii,jj])

                                derivative = \
                                    dfp[kk,ii,jj] * kernel[sk,si,sj] * (1+db[kk,ii,jj]) +\
                                    fp[kk,ii,jj] * dkernel[sk,si,sj] +\
                                    dfm[kk,ii,jj] * kernel[sk,si,sj] * (1-db[kk,ii,jj]) +\
                                    fm[kk,ii,jj] * dkernel[sk,si,sj]
                                with gil:
                                    if sk<0 or si<0 or sj<0 or sk>=len_s or si>=len_r or sj>= len_c:
                                        print("Out: %d, %d, %d"%(sk,si,sj))

                                Jt[row, col] = derivative

                    row += 1
    Jth = np.dot(Jt, residual)
    JtJ = np.dot(Jt, np.transpose(Jt))
    JtJ[np.diag_indices(nks*nkr*nkc, 2)] += l1
    return Jth, JtJ, Jt


cpdef wrap_scalar_field(double[:] v, int[:] sh):
    cdef:
        double[:,:,:] vol = np.ndarray((sh[0], sh[1], sh[2]), dtype=np.float64)
        int i, j, k, idx
    if sh[0]*sh[1]*sh[2] != len(v):
        raise ValueError("Wrong number of coefficients")
    with nogil:
        idx = 0
        for k in range(sh[0]):
            for i in range(sh[1]):
                for j in range(sh[2]):
                    vol[k,i,j] = v[idx]
                    idx += 1
    return vol


cpdef unwrap_scalar_field(double[:,:,:] v):
    cdef:
        int n = v.shape[0] * v.shape[1] * v.shape[2]
        double[:] out = np.ndarray(n, dtype=np.float64)
        int i, j, k, idx
    with nogil:
        idx = 0
        for k in range(v.shape[0]):
            for i in range(v.shape[1]):
                for j in range(v.shape[2]):
                    out[idx] = v[k,i,j]
                    idx += 1
    return out


def resample_orfield(floating[:, :, :] b, double[:] factors, int[:] target_shape):
    ftype = np.asarray(b).dtype
    cdef:
        int tslices = target_shape[0]
        int trows = target_shape[1]
        int tcols = target_shape[2]
        int inside, k, i, j
        double dkk, dii, djj
        floating[:, :, :] expanded = np.zeros((tslices, trows, tcols), dtype=ftype)

    for k in range(tslices):
        for i in range(trows):
            for j in range(tcols):
                dkk = <double>k*factors[0]
                dii = <double>i*factors[1]
                djj = <double>j*factors[2]
                _interpolate_scalar_3d(b, dkk, dii, djj, &expanded[k, i, j])
    return expanded

######################################################
#################### Spline field ####################
######################################################


#################### Spline kernels ##################

def cubic_spline(double[:] x, double[:] sx):
    r''' Evaluates the cubic spline at a set of values
    Parameters
    ----------
    x : array, shape (n)
        input values
    sx : array, shape (n)
        buffer in which to write the evaluated spline
    '''
    cdef:
        int i
        int n = x.shape[0]
    with nogil:
        for i in range(n):
            sx[i] =  _cubic_spline(x[i])


cdef inline double _cubic_spline(double x) nogil:
    r''' Cubic B-Spline evaluated at x
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


def cubic_spline_derivative(double[:] x, double[:] sx):
    r''' Evaluates the cubic spline derivative at a set of values
    Parameters
    ----------
    x : array, shape (n)
        input values
    sx : array, shape (n)
        buffer in which to write the evaluated spline derivative
    '''
    cdef:
        int i
        int n = x.shape[0]
    with nogil:
        for i in range(n):
            sx[i] =  _cubic_spline_derivative(x[i])


cdef inline double _cubic_spline_derivative(double x) nogil:
    r''' Derivative of cubic B-Spline evaluated at x
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


def cubic_spline_second_derivative(double[:] x, double[:] sx):
    r''' Evaluates the cubic spline's second derivative at a set of values
    Parameters
    ----------
    x : array, shape (n)
        input values
    sx : array, shape (n)
        buffer in which to write the evaluated spline derivative
    '''
    cdef:
        int i
        int n = x.shape[0]
    with nogil:
        for i in range(n):
            sx[i] =  _cubic_spline_second_derivative(x[i])


cdef inline double _cubic_spline_second_derivative(double x) nogil:
    r''' Second derivative of cubic B-Spline evaluated at x
    See eq. (3) of [1].
    [1] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K., & Eubank, W.
        PET-CT image registration in the chest using free-form deformations.
        IEEE Transactions on Medical Imaging, 22(1), 120–8, 2003.
    '''
    cdef:
        double absx = -x if x < 0.0 else x
        double sqrx = x * x
    if absx < 1.0:
        return 3.0 * absx - 2.0
    elif absx < 2:
        return 2 - absx
    return 0

####################### Spline fields #####################
ctypedef double (*spline_kernel)(double) nogil


cdef spline_kernel _get_spline_kernel(int derivative_order) nogil:
    if derivative_order == 0:
        return _cubic_spline
    elif derivative_order == 1:
        return _cubic_spline_derivative
    elif derivative_order == 2:
        return _cubic_spline_second_derivative
    return NULL


cdef _compute_spline_1d(int knot_sp, spline_kernel kernel, double[:] out):
    cdef:
        int n = out.shape[0]
        int center = n // 2
        int i
        double x, dx

    dx = 1.0 / knot_sp
    x = -1.0 * center * dx
    for i in range(n):
        out[i] = kernel(x)
        x += dx


cdef _compute_spline_3d(int[:] knot_sps, int[:] der_orders, double[:,:,:] out):
    cdef:
        int ns = out.shape[0]
        int nr = out.shape[1]
        int nc = out.shape[2]
        int s, r, c
        double[:] spline_s = np.ndarray(ns, dtype=np.float64)
        double[:] spline_r = np.ndarray(nr, dtype=np.float64)
        double[:] spline_c = np.ndarray(nc, dtype=np.float64)
        double cs = ns // 2
        double cr = nr // 2
        double cc = nc // 2

    _compute_spline_1d(knot_sps[0], _get_spline_kernel(der_orders[0]), spline_s)
    _compute_spline_1d(knot_sps[1], _get_spline_kernel(der_orders[1]), spline_r)
    _compute_spline_1d(knot_sps[2], _get_spline_kernel(der_orders[2]), spline_c)
    for s in range(ns):
        for r in range(nr):
            for c in range(nc):
                out[s, r, c] = spline_s[s] * spline_r[r] * spline_c[c]


class Spline3D:
    r""" Precomputed Spline in 3D over a regular grid
    """
    def __init__(self, knot_spacings, derivative_orders = None):
        self.knot_spacings = knot_spacings

        # Compute the grid shape based on the knot spacing
        self.grid_shape = 4 * knot_spacings - 1

        # Precompute the centered spline
        self.spline = np.ndarray(shape=tuple(self.grid_shape), dtype=np.float64)
        _compute_spline_3d(knot_spacings, derivative_orders, self.spline)


def create_spline_3d(resolution, vox_size, derivative_orders):
    knot_spacing = np.round(vox_size * 1.0 / resolution).astype(np.int32)
    knot_spacing[knot_spacing < 1] = 1
    spline = Spline3D(knot_spacing, derivative_orders)
    return spline


cdef inline void _volume_range_affected(int index, int kspacing, int vol_side,
                                        int spline_side, int *first, int *last,
                                        int *spline_offset) nogil:
    r"""
    Spline knots are centered at cells {-1, 0, 1, 2, ..., m-1, m, m+1}*kspacing.
    If the spline knot is centered at c = j*kspacing, then this spline affects
    cells c+{-s, -s+1, ..., -1, 0, 1, ..., s-1, s}, where the length of the
    spline is 2*s+1 (it is always odd). Note that the 0-th spline is centered
    outside the volume grid: at -kspacing.

    If the first affected cell is inside the volume, then the spline cell
    corresponding to the first affected volume cell is the first one. Else,
    we need to shift the spline.
    """
    first[0] = (index - 1) * kspacing - spline_side // 2
    spline_offset[0] = -first[0]
    last[0] = (index - 1) * kspacing + spline_side // 2 + 1
    if first[0] < 0:
        first[0] = 0;
    if last[0] > vol_side:
        last[0] = vol_side
    if spline_offset[0] < 0:
        spline_offset[0] = 0


cdef void _eval_spline_field(double[:,:,:] coefs, int[:] kspacing,
                        double[:,:,:] spline, double[:,:,:] out) nogil:
    cdef:
        int ns = coefs.shape[0]
        int nr = coefs.shape[1]
        int nc = coefs.shape[2]
        int i,j,k # Iterators over coefs
        int ii, jj, kk # Iterators for voxels affected by current coefficient
        int isp, jsp, ksp # Corresponding grid locations on precomputed spline
        # Range of voxels affected by current coefficient
        int i_first, i_last, j_first, j_last, k_first, k_last
        int isp_offset, jsp_offset, ksp_offset
        double coef
    # Initialize output volume to zeros
    for k in range(out.shape[0]):
        for i in range(out.shape[1]):
            for j in range(out.shape[2]):
                out[k,i,j] = 0

    # Iterate over all spline coefficients
    for k in range(ns):
        _volume_range_affected(k, kspacing[0], out.shape[0],
                               spline.shape[0], &k_first, &k_last, &ksp_offset)
        for i in range(nr):
            _volume_range_affected(i, kspacing[1], out.shape[1],
                                   spline.shape[1], &i_first, &i_last, &isp_offset)
            for j in range(nc):
                _volume_range_affected(j, kspacing[2], out.shape[2],
                                       spline.shape[2], &j_first, &j_last, &jsp_offset)
                coef = coefs[k,i,j]
                # Add contribution of this coefficient to all neighboring voxels
                for kk in range(k_first, k_last):
                    ksp = ksp_offset + kk - k_first
                    for ii in range(i_first, i_last):
                        isp = isp_offset + ii - i_first
                        for jj in range(j_first, j_last):
                            jsp = jsp_offset + jj - j_first
                            out[kk, ii, jj] += coef * spline[ksp, isp, jsp]


class SplineField:
    r"""
    This spline field is intended to be evaluated at a fixed regular grid. The
    spline knots are also distributed regularly along the grid. Each knot is
    centered on a cell of the volume grid. The distance (in voxels) between
    neighboring knots is integer.
    """
    def __init__(self, vol_shape, kspacing):
        self.grid_shape = vol_shape / kspacing + 3
        self.vol_shape = vol_shape
        self.kspacing = kspacing
        self.splines = {}
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    self.splines[(i,j,k)] = Spline3D(kspacing, np.array([i,j,k]))

    def num_coefficients(self):
        return self.grid_shape[0] * self.grid_shape[1] * self.grid_shape[2]

    def set_coefficients(self, coef):
        if len(coef.shape) != len(self.grid_shape):
            raise ValueError("Coefficient field dimension mismatch")
        if not np.all(coef.shape == self.grid_shape):
            raise ValueError("Coefficient field dimension mismatch")
        self.coef = coef

    def copy_coefficients(self, coef):
        if len(coef.shape) == 1: # Vectorized form
            if coef.size != self.num_coefficients():
                raise ValueError("Coefficient field dimension mismatch")
            self.coef = wrap_scalar_field(coef, self.grid_shape)
            return

        if len(coef.shape) != len(self.grid_shape):
            raise ValueError("Coefficient field dimension mismatch")
        if not np.all(coef.shape == self.grid_shape):
            raise ValueError("Coefficient field dimension mismatch")
        self.coef = coef.copy()

    def get_volume(self, der_orders = (0,0,0)):
        volume = np.zeros(tuple(self.vol_shape), dtype=np.float64)
        _eval_spline_field(self.coef, self.kspacing, self.splines[der_orders].spline, volume)
        return np.array(volume)


def regrid(floating[:,:,:]vol, double[:] factors, int[:] new_shape):
    ftype=np.asarray(vol).dtype
    cdef:
        int ns = new_shape[0]
        int nr = new_shape[1]
        int nc = new_shape[2]
        int k,i,j
        double kk, ii, jj
        floating[:,:,:] out = np.ndarray((ns, nr, nc), dtype=ftype)
    with nogil:
        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    kk = k * factors[0]
                    ii = i * factors[1]
                    jj = j * factors[2]

                    _interpolate_scalar_3d(vol, kk, ii, jj, &out[k,i,j])
    return out
