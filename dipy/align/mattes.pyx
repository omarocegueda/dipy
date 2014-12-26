#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as cnp
cimport cython

from dipy.align.vector_fields cimport(_apply_affine_3d_x0,
                                      _apply_affine_3d_x1,
                                      _apply_affine_3d_x2,
                                      _apply_affine_2d_x0,
                                      _apply_affine_2d_x1)

from dipy.align.transforms cimport (jacobian_function,
                                    param_to_matrix_function,
                                    get_jacobian_function)

cdef extern from "dpy_math.h" nogil:
    double cos(double)
    double sin(double)
    double log(double)

class MattesBase(object):
    def __init__(self, nbins):
        r""" MattesBase
        Base class for the Mattes' Mutual Information metric

        Notes: we need this class in cython to allow
        _joint_pdf_gradient_dense_2d and _joint_pdf_gradient_dense_3d to receive
        a pointer to a nogil function that computes the Jacobian of a transform,
        which allows us to compute Jacobians inside a nogil loop.

        The reason we need a class is to encapsulate all the parameters related
        to the joint and marginal distributions.

        Parameters
        ----------
        nbins : int
            the number of bins of the joint and margianl probability density
            functions (the actual number of bins of the joint PDF is nbins^2)
        """
        self.nbins = nbins
        # Since the kernel used to compute the Parzen histogram covers more than
        # one bin, we need to add extra bins to both sides of the histogram
        # to account for the contributions of the minimum and maximum
        # intensities. Padding is the number of extra bins used at each side
        # of the histogram (a total of 2 * padding extra bins). Since the
        # support of the cubic spline is 5 bins (the center plus 2 bins at each
        # side) we need a padding of 2, in this case.
        self.padding = 2

    def setup(self, static, moving, smask=None, mmask=None):
        r""" Compute histogram settings to store PDF of input images

        Parameters
        ----------
        static : array
            static image
        moving : array
            moving image
        smask : array
            mask of static object being registered (a binary array with 1's
            inside the object of interest and 0's along the background).
            If None, ones_like(static) is used as mask
        mmask : array
            mask of moving object being registered (a binary array with 1's
            inside the object of interest and 0's along the background).
            If None, ones_like(moving) is used as mask
        """

        if smask is None:
            smask = np.ones_like(static)
        if mmask is None:
            mmask = np.ones_like(moving)

        self.smin = np.min(static[smask!=0])
        self.smax = np.max(static[smask!=0])
        self.mmin = np.min(moving[mmask!=0])
        self.mmax = np.max(moving[mmask!=0])

        self.sdelta = (self.smax - self.smin)/(self.nbins - 2 * self.padding)
        self.mdelta = (self.mmax - self.mmin)/(self.nbins - 2 * self.padding)
        self.smin = self.smin/self.sdelta - self.padding
        self.mmin = self.mmin/self.sdelta - self.padding

        self.joint_grad = None
        self.metric_grad = None
        self.metric_val = 0
        self.joint = np.ndarray(shape = (self.nbins, self.nbins))
        self.smarginal = np.ndarray(shape = (self.nbins,), dtype = np.float64)
        self.mmarginal = np.ndarray(shape = (self.nbins,), dtype = np.float64)


    def bin_normalize_static(self, x):
        r""" Normalizes intensity x to the range covered by the static histogram

        If the input intensity is in [self.smin, self.smax] then the normalized
        intensity will be in [self.padding, self.nbins - 1 - self.padding]

        Parameters
        ----------
        x : float
            the intensity to be normalized

        Returns
        -------
        xnorm : float
            normalized intensity to the range covered by the static histogram
        """
        return _bin_normalize(x, self.smin, self.sdelta)


    def bin_normalize_moving(self, x):
        r""" Normalizes intensity x to the range covered by the moving histogram

        If the input intensity is in [self.mmin, self.mmax] then the normalized
        intensity will be in [self.padding, self.nbins - 1 - self.padding]

        Parameters
        ----------
        x : float
            the intensity to be normalized

        Returns
        -------
        xnorm : float
            normalized intensity to the range covered by the moving histogram
        """
        return _bin_normalize(x, self.mmin, self.mdelta)


    def bin_index(self, xnorm):
        r""" Bin index associated to the given normalized intensity

        The return value is an integer in [padding, nbins - 1 - padding]

        Parameters
        ----------
        xnorm : float
            intensity value normalized to the range covered by the histogram

        Returns
        -------
        bin : int
            the bin index associated to the given normalized intensity
        """
        return _bin_index(xnorm, self.nbins, self.padding)


    def update_pdfs_dense(self, static, moving, smask=None, mmask=None):
        r''' Computes the Joint Probability Density Function of of two images

        Parameters
        ----------
        static: array, shape (S, R, C)
            static image
        moving: array, shape (S, R, C)
            moving image
        smask: array, shape (S, R, C)
            mask of static object being registered (a binary array with 1's
            inside the object of interest and 0's along the background).
            If None, ones_like(static) is used as mask.
        mmask: array, shape (S, R, C)
            mask of moving object being registered (a binary array with 1's
            inside the object of interest and 0's along the background).
            If None, ones_like(moving) is used as mask.
        '''
        dim = len(static.shape)
        if dim == 2:
            _compute_pdfs_dense_2d(static, moving, smask, mmask, self.smin,
                                   self.sdelta, self.mmin, self.mdelta,
                                   self.nbins, self.padding, self.joint,
                                   self.smarginal, self.mmarginal)
        elif dim == 3:
            _compute_pdfs_dense_3d(static, moving, smask, mmask, self.smin,
                                   self.sdelta, self.mmin, self.mdelta,
                                   self.nbins, self.padding, self.joint,
                                   self.smarginal, self.mmarginal)
        else:
            msg = 'Only dimensions 2 and 3 are supported. '+str(dim)+' received'
            raise ValueError(msg)


    def update_pdfs_sparse(self, sval, mval):
        r''' Computes the Probability Density Functions of paired intensities

        Parameters
        ----------
        sval: array, shape (n,)
            sampled intensities from the static image at sampled_points
        mval: array, shape (n,)
            sampled intensities from the moving image at sampled_points
        '''
        energy = _compute_pdfs_sparse(sval, mval, self.smin, self.sdelta,
                                      self.mmin, self.mdelta, self.nbins,
                                      self.padding, self.joint, self.smarginal,
                                      self.mmarginal)


    def update_gradient_dense(self, theta, transform, static, moving,
                              grid_to_space, mgradient, smask, mmask):
        r''' Computes the Gradient of the joint PDF w.r.t. transform parameters

        Computes the vector of partial derivatives of the joint histogram w.r.t.
        each transformation parameter.

        Parameters
        ----------
        theta: array, shape (n,)
            parameters of the transformation to compute the gradient from
        transform: int
            the type of the transformation (use transform_type dictionary from
            the transforms module to map transformation name to int)
        static: array, shape (S, R, C)
            static image
        moving: array, shape (S, R, C)
            moving image
        grid_to_space: array, shape (4, 4)
            the grid-to-space transform associated to images static and moving
            (we assume that both images have already been sampled at a common
            grid)
        mgradient: array, shape (S, R, C, 3)
            the gradient of the moving image
        smask: array, shape (S, R, C)
            mask of static object being registered (a binary array with 1's
            inside the object of interest and 0's along the background)
        mmask: array, shape (S, R, C)
            mask of moving object being registered (a binary array with 1's
            inside the object of interest and 0's along the background)
        '''
        cdef:
            jacobian_function jacobian = NULL

        dim = len(static.shape)
        jacobian = get_jacobian_function(transform, dim)

        if jacobian == NULL:
            raise(ValueError('Unknown transform type: "'+transform+'"'))

        n = theta.shape[0]
        nbins = self.nbins

        if (self.joint_grad is None) or (self.joint_grad.shape[2] != n):
                self.joint_grad = np.ndarray((nbins, nbins, n))
        if dim == 2:
            _joint_pdf_gradient_dense_2d(theta, jacobian, static, moving,
                                         grid_to_space, mgradient, smask, mmask,
                                         self.smin, self.sdelta, self.mmin,
                                         self.mdelta, self.nbins, self.padding,
                                         self.joint_grad)
        elif dim ==3:
            _joint_pdf_gradient_dense_3d(theta, jacobian, static, moving,
                                         grid_to_space, mgradient, smask, mmask,
                                         self.smin, self.sdelta, self.mmin,
                                         self.mdelta, self.nbins, self.padding,
                                         self.joint_grad)
        else:
            msg = 'Only dimensions 2 and 3 are supported. '+str(dim)+' received'
            raise ValueError(msg)


    def update_gradient_sparse(self, theta, transform, sval, mval,
                               sample_points, mgradient):
        r''' Computes the Gradient of the joint PDF w.r.t. transform parameters

        Computes the vector of partial derivatives of the joint histogram w.r.t.
        each transformation parameter.

        Parameters
        ----------
        theta: array, shape (n,)
            parameters to compute the gradient at
        transform: int
            the type of the transformation (use transform_type dictionary from
            the transforms module to map transformation name to int)
        sval: array, shape (m,)
            sampled intensities from the static image at sampled_points
        mval: array, shape (m,)
            sampled intensities from the moving image at sampled_points
        sample_points: array, shape (m, 3)
            coordinates (in physical space) of the points the images were
            sampled at
        mgradient: array, shape (m, 3)
            the gradient of the moving image at the sample points
        '''
        cdef:
            jacobian_function jacobian = NULL

        dim = len(sample_points.shape[1])
        jacobian = get_jacobian_function(transform, dim)

        if jacobian == NULL:
            raise(ValueError('Unknown transform type: "'+transform+'"'))
        n = theta.shape[0]
        nbins = self.nbins

        if (self.joint_grad is None) or (self.joint_grad.shape[2] != n):
            self.joint_grad = np.ndarray(shape = (nbins, nbins, n))

        if dim == 2:
            _joint_pdf_gradient_sparse_2d(theta, jacobian, sval, mval,
                                          sample_points, mgradient, self.smin,
                                          self.sdelta, self.mmin, self.mdelta,
                                          self.nbins, self.padding,
                                          self.joint_grad)
        elif dim ==3:
            _joint_pdf_gradient_sparse_3d(theta, jacobian, sval, mval,
                                          sample_points, mgradient, self.smin,
                                          self.sdelta, self.mmin, self.mdelta,
                                          self.nbins, self.padding,
                                          self.joint_grad)
        else:
            msg = 'Only dimensions 2 and 3 are supported. '+str(dim)+' received'
            raise ValueError(msg)


    def update_mi_metric(self, update_gradient=True):
        r""" Computes current value and gradient (if requested) of the MI metric

        Parameters
        ----------
        update_gradient : Boolean
            boolean indicating if the gradient must be computed (if False,
            only the value is computed). Default is True.
        """
        grad = None
        if update_gradient:
            sh = self.joint_grad.shape[2]
            if (self.metric_grad is None) or (self.metric_grad.shape[0] != sh):
                self.metric_grad = np.empty(sh)
            grad = self.metric_grad

        self.metric_val = _compute_mattes_mi(self.joint, self.joint_grad,
                                             self.smarginal, self.mmarginal,
                                             grad)


cdef inline double _bin_normalize(double x, double mval, double delta) nogil:
    r''' Normalizes intensity x to the range covered by the Parzen histogram
    We assume that mval was computed as:

    (1) mval = xmin / delta - padding

    where xmin is the minimum observed image intensity and delta is the
    bin size, computed as:

    (2) delta = (xmax - xmin)/(nbins - 2 * padding)

    If the minimum and maximum intensities were assigned to the first and last
    bins (with no padding), it could be possible that samples at the first and
    last bins contribute to "non-existing" bins beyond the boundary (because
    the support of the Parzen window may be larger than one bin). The padding
    bins are used to collect such contributions (i.e. the probability of
    observing a value beyond the minimum and maximum observed intensities may
    be assigned a positive value).

    The normalized intensity is (from eq(1) ):

    (3) nx = (x - xmin) / delta + padding = x / delta - mval

    This means that the observed intensity x must be between
    bins padding and nbins-1-padding, although it may affect bins 0 to nbins-1.
    '''
    return x / delta - mval


cdef inline cnp.npy_intp _bin_index(double normalized, int nbins,
                                    int padding) nogil:
    r''' Index of the bin in which the normalized intensity 'normalized' lies
    The intensity is assumed to have been normalized to the range of intensities
    covered by the histogram: the bin index is the integer part of the argument,
    which must be within the interval [padding, nbins - 1 - padding].

    Parameters
    ----------
    normalized : float
        normalized intensity
    nbins : int
        number of histogram bins
    padding : int
        number of bins used as padding (the total bins used for padding at
        both sides of the histogram is actually 2*padding)

    Returns
    -------
    bin : int
        index of the bin in which the normalized intensity 'normalized' lies
    '''
    cdef:
        cnp.npy_intp bin

    bin = <cnp.npy_intp>(normalized)
    if bin < padding:
        return padding
    if bin > nbins - 1 - padding:
        return nbins - 1 - padding
    return bin


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


cdef _compute_pdfs_dense_2d(double[:,:] static, double[:,:] moving,
                            int[:,:] smask, int[:,:] mmask,
                            double smin, double sdelta,
                            double mmin, double mdelta,
                            int nbins, int padding, double[:,:] joint,
                            double[:] smarginal, double[:] mmarginal):
    r''' Joint Probability Density Function of intensities of two 2D images

    Parameters
    ----------
    static: array, shape (R, C)
        static image
    moving: array, shape (R, C)
        moving image
    smask: array, shape (R, C)
        mask of static object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    mmask: array, shape (R, C)
        mask of moving object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    smin: float
        the minimum observed intensity associated to the static image, which
        was used to define the joint PDF
    sdelta: float
        bin size associated to the intensities of the static image
    mmin: float
        the minimum observed intensity associated to the moving image, which
        was used to define the joint PDF
    mdelta: float
        bin size associated to the intensities of the moving image
    nbins: int
        number of histogram bins
    padding: int
        number of bins used as padding (the total bins used for padding at both
        sides of the histogram is actually 2*padding)
    joint: array, shape (nbins, nbins)
        the array to write the joint PDF
    smarginal: array, shape (nbins,)
        the array to write the marginal PDF associated to the static image
    mmarginal: array, shape (nbins,)
        the array to write the marginal PDF associated to the moving image
    '''
    cdef:
        int nrows = static.shape[0]
        int ncols = static.shape[1]
        int offset, valid_points
        cnp.npy_intp i, j, r, c
        double rn, cn
        double val, spline_arg, sum

    joint[...] = 0
    sum = 0
    valid_points = 0
    with nogil:
        smarginal[:] = 0
        for i in range(nrows):
            for j in range(ncols):
                if smask is not None and smask[i, j] == 0:
                    continue
                if mmask is not None and mmask[i, j] == 0:
                    continue
                valid_points += 1
                rn = _bin_normalize(static[i, j], smin, sdelta)
                r = _bin_index(rn, nbins, padding)
                cn = _bin_normalize(moving[i, j], mmin, mdelta)
                c = _bin_index(cn, nbins, padding)
                spline_arg = (c-2) - cn

                smarginal[r] += 1
                for offset in range(-2, 3):
                    val = _cubic_spline(spline_arg)
                    joint[r, c + offset] += val
                    sum += val
                    spline_arg += 1.0

        if sum > 0:
            for i in range(nbins):
                for j in range(nbins):
                    joint[i, j] /= sum

            for i in range(nbins):
                smarginal[i] /= valid_points

            for j in range(nbins):
                mmarginal[j] = 0
                for i in range(nbins):
                    mmarginal[j] += joint[i, j]



cdef _compute_pdfs_dense_3d(double[:,:,:] static, double[:,:,:] moving,
                            int[:,:,:] smask, int[:,:,:] mmask,
                            double smin, double sdelta,
                            double mmin, double mdelta,
                            int nbins, int padding, double[:,:] joint,
                            double[:] smarginal, double[:] mmarginal):
    r''' Joint Probability Density Function of intensities of two 3D images

    Parameters
    ----------
    static: array, shape (S, R, C)
        static image
    moving: array, shape (S, R, C)
        moving image
    smask: array, shape (S, R, C)
        mask of static object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    mmask: array, shape (S, R, C)
        mask of moving object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    smin: float
        the minimum observed intensity associated to the static image, which
        was used to define the joint PDF
    sdelta: float
        bin size associated to the intensities of the static image
    mmin: float
        the minimum observed intensity associated to the moving image, which
        was used to define the joint PDF
    mdelta: float
        bin size associated to the intensities of the moving image
    nbins: int
        number of histogram bins
    padding: int
        number of bins used as padding (the total bins used for padding at both
        sides of the histogram is actually 2*padding)
    joint: array, shape(nbins, nbins)
        the array to write the joint PDF to
    smarginal: array, shape (nbins,)
        the array to write the marginal PDF associated to the static image
    mmarginal: array, shape (nbins,)
        the array to write the marginal PDF associated to the moving image
    '''
    cdef:
        int nslices = static.shape[0]
        int nrows = static.shape[1]
        int ncols = static.shape[2]
        int offset, valid_points
        cnp.npy_intp k, i, j, r, c
        double rn, cn
        double val, spline_arg, sum

    joint[...] = 0
    sum = 0
    with nogil:
        smarginal[:] = 0
        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if smask is not None and smask[k, i, j] == 0:
                        continue
                    if mmask is not None and mmask[k, i, j] == 0:
                        continue
                    valid_points += 1
                    rn = _bin_normalize(static[k, i, j], smin, sdelta)
                    r = _bin_index(rn, nbins, padding)
                    cn = _bin_normalize(moving[k, i, j], mmin, mdelta)
                    c = _bin_index(cn, nbins, padding)
                    spline_arg = (c-2) - cn

                    smarginal[r] += 1
                    for offset in range(-2, 3):
                        val = _cubic_spline(spline_arg)
                        joint[r, c + offset] += val
                        sum += val
                        spline_arg += 1.0

        if sum > 0:
            for i in range(nbins):
                for j in range(nbins):
                    joint[i, j] /= sum

            for i in range(nbins):
                smarginal[i] /= valid_points

            for j in range(nbins):
                mmarginal[j] = 0
                for i in range(nbins):
                    mmarginal[j] += joint[i, j]


cdef _compute_pdfs_sparse(double[:] sval, double[:] mval, double smin,
                          double sdelta, double mmin, double mdelta,
                          int nbins, int padding, double[:,:] joint,
                          double[:] smarginal, double[:] mmarginal):
    r''' Probability Density Functions of paired intensities

    Parameters
    ----------
    sval: array, shape (n,)
        sampled intensities from the static image at sampled_points
    mval: array, shape (n,)
        sampled intensities from the moving image at sampled_points
    smin: float
        the minimum observed intensity associated to the static image, which
        was used to define the joint PDF
    sdelta: float
        bin size associated to the intensities of the static image
    mmin: float
        the minimum observed intensity associated to the moving image, which
        was used to define the joint PDF
    mdelta: float
        bin size associated to the intensities of the moving image
    nbins: int
        number of histogram bins
    padding: int
        number of bins used as padding (the total bins used for padding at both
        sides of the histogram is actually 2*padding)
    joint: array, shape(nbins, nbins)
        the array to write the joint PDF to
    smarginal: array, shape (nbins,)
        the array to write the marginal PDF associated to the static image
    mmarginal: array, shape (nbins,)
        the array to write the marginal PDF associated to the moving image
    '''
    cdef:
        int n = sval.shape[0]
        int offset, valid_points
        cnp.npy_intp i, r, c
        double rn, cn
        double val, spline_arg, sum

    joint[...] = 0
    sum = 0
    valid_points = 0
    with nogil:
        smarginal[:] = 0
        for i in range(n):
            valid_points += 1
            rn = _bin_normalize(sval[i], smin, sdelta)
            r = _bin_index(rn, nbins, padding)
            cn = _bin_normalize(mval[i], mmin, mdelta)
            c = _bin_index(cn, nbins, padding)
            spline_arg = (c-2) - cn

            smarginal[r] += 1
            for offset in range(-2, 3):
                val = _cubic_spline(spline_arg)
                joint[r, c + offset] += val
                sum += val
                spline_arg += 1.0

        if sum > 0:
            for i in range(nbins):
                for j in range(nbins):
                    joint[i, j] /= sum

            for i in range(nbins):
                smarginal[i] /= valid_points

            for j in range(nbins):
                mmarginal[j] = 0
                for i in range(nbins):
                    mmarginal[j] += joint[i, j]


cdef _joint_pdf_gradient_dense_2d(double[:] theta, jacobian_function jacobian,
                                  double[:,:] static, double[:,:] moving,
                                  double[:,:] grid_to_space,
                                  double[:,:,:] mgradient, int[:,:] smask,
                                  int[:,:] mmask, double smin, double sdelta,
                                  double mmin, double mdelta, int nbins,
                                  int padding, double[:,:,:] grad_pdf):
    r''' Gradient of the joint PDF w.r.t. transform parameters theta

    Computes the vector of partial derivatives of the joint histogram w.r.t.
    each transformation parameter. The transformation itself is not necessary
    to compute the gradient, but only its Jacobian.

    Parameters
    ----------
    theta: array, shape (n,)
        parameters of the transformation to compute the gradient from
    jacobian: jacobian_function
        function that computes the Jacobian matrix of a transformation
    static: array, shape (R, C)
        static image
    moving: array, shape (R, C)
        moving image
    grid_to_space: array, shape (3, 3)
        the grid-to-space transform associated to images static and moving
        (we assume that both images have already been sampled at a common grid)
    mgradient: array, shape (R, C, 2)
        the gradient of the moving image
    smask: array, shape (R, C)
        mask of static object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    mmask: array, shape (R, C)
        mask of moving object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    smin: float
        the minimum observed intensity associated to the static image, which
        was used to define the joint PDF
    sdelta: float
        bin size associated to the intensities of the static image
    mmin: float
        the minimum observed intensity associated to the moving image, which
        was used to define the joint PDF
    mdelta: float
        bin size associated to the intensities of the moving image
    nbins: int
        number of histogram bins
    padding: int
        number of bins used as padding (the total bins used for padding at both
        sides of the histogram is actually 2*padding)
    grad_pdf: array, shape (nbins, nbins, len(theta))
        the array to write the gradient to
    '''
    cdef:
        int nrows = static.shape[0]
        int ncols = static.shape[1]
        int n = theta.shape[0]
        int offset, valid_points, constant_jacobian=0
        cnp.npy_intp k, i, j, r, c
        double rn, cn
        double val, spline_arg
        double[:,:] J = np.ndarray(shape=(2, n), dtype=np.float64)
        double[:] prod = np.ndarray(shape=(n,), dtype=np.float64)
        double[:] x = np.ndarray(shape=(2,), dtype=np.float64)

    grad_pdf[...] = 0
    with nogil:
        valid_points = 0
        for i in range(nrows):
            for j in range(ncols):
                if smask is not None and smask[i, j] == 0:
                    continue
                if mmask is not None and mmask[i, j] == 0:
                    continue

                valid_points += 1
                x[0] = _apply_affine_2d_x0(i, j, 1, grid_to_space)
                x[1] = _apply_affine_2d_x1(i, j, 1, grid_to_space)

                if constant_jacobian == 0:
                    constant_jacobian = jacobian(theta, x, J)

                for k in range(n):
                    prod[k] = J[0,k] * mgradient[i,j,0] +\
                              J[1,k] * mgradient[i,j,1]

                rn = _bin_normalize(static[i, j], smin, sdelta)
                r = _bin_index(rn, nbins, padding)
                cn = _bin_normalize(moving[i, j], mmin, mdelta)
                c = _bin_index(cn, nbins, padding)
                spline_arg = (c-2) - cn

                for offset in range(-2, 3):
                    val = _cubic_spline_derivative(spline_arg)
                    for k in range(n):
                        grad_pdf[r, c + offset,k] -= val * prod[k]
                    spline_arg += 1.0

        if valid_points * mdelta > 0:
            for i in range(nbins):
                for j in range(nbins):
                    for k in range(n):
                        grad_pdf[i, j, k] /= (valid_points * mdelta)


cdef _joint_pdf_gradient_dense_3d(double[:] theta, jacobian_function jacobian,
                                  double[:,:,:] static, double[:,:,:] moving,
                                  double[:,:] grid_to_space,
                                  double[:,:,:,:] mgradient, int[:,:,:] smask,
                                  int[:,:,:] mmask, double smin, double sdelta,
                                  double mmin, double mdelta, int nbins,
                                  int padding, double[:,:,:] grad_pdf):
    r''' Gradient of the joint PDF w.r.t. transform parameters theta

    Computes the vector of partial derivatives of the joint histogram w.r.t.
    each transformation parameter. The transformation itself is not necessary
    to compute the gradient, but only its Jacobian.

    Parameters
    ----------
    theta: array, shape (n,)
        parameters of the transformation to compute the gradient from
    jacobian: jacobian_function
        function that computes the Jacobian matrix of a transformation
    static: array, shape (S, R, C)
        static image
    moving: array, shape (S, R, C)
        moving image
    grid_to_space: array, shape (4, 4)
        the grid-to-space transform associated to images static and moving
        (we assume that both images have already been sampled at a common grid)
    mgradient: array, shape (S, R, C, 3)
        the gradient of the moving image
    smask: array, shape (S, R, C)
        mask of static object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    mmask: array, shape (S, R, C)
        mask of moving object being registered (a binary array with 1's inside
        the object of interest and 0's along the background)
    smin: float
        the minimum observed intensity associated to the static image, which
        was used to define the joint PDF
    sdelta: float
        bin size associated to the intensities of the static image
    mmin: float
        the minimum observed intensity associated to the moving image, which
        was used to define the joint PDF
    mdelta: float
        bin size associated to the intensities of the moving image
    nbins: int
        number of histogram bins
    padding: int
        number of bins used as padding (the total bins used for padding at both
        sides of the histogram is actually 2*padding)
    grad_pdf: array, shape (nbins, nbins, len(theta))
        the array to write the gradient to
    '''
    cdef:
        int nslices = static.shape[0]
        int nrows = static.shape[1]
        int ncols = static.shape[2]
        int n = theta.shape[0]
        int offset, constant_jacobian=0
        cnp.npy_intp l, k, i, j, r, c
        double rn, cn
        double val, spline_arg
        double[:,:] J = np.ndarray(shape=(3, n), dtype=np.float64)
        double[:] prod = np.ndarray(shape=(n,), dtype=np.float64)
        double[:] x = np.ndarray(shape=(3,), dtype=np.float64)

    grad_pdf[...] = 0
    with nogil:

        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if smask is not None and smask[k, i, j] == 0:
                        continue
                    if mmask is not None and mmask[k, i, j] == 0:
                        continue
                    x[0] = _apply_affine_3d_x0(k, i, j, 1, grid_to_space)
                    x[1] = _apply_affine_3d_x1(k, i, j, 1, grid_to_space)
                    x[2] = _apply_affine_3d_x2(k, i, j, 1, grid_to_space)

                    if constant_jacobian == 0:
                        constant_jacobian = jacobian(theta, x, J)

                    for l in range(n):
                        prod[l] = J[0,l] * mgradient[k,i,j,0] +\
                                  J[1,l] * mgradient[k,i,j,1] +\
                                  J[2,l] * mgradient[k,i,j,2]

                    rn = _bin_normalize(static[k, i, j], smin, sdelta)
                    r = _bin_index(rn, nbins, padding)
                    cn = _bin_normalize(moving[k, i, j], mmin, mdelta)
                    c = _bin_index(cn, nbins, padding)
                    spline_arg = (c-2) - cn

                    for offset in range(-2, 3):
                        val = _cubic_spline_derivative(spline_arg)
                        for l in range(n):
                            grad_pdf[r, c + offset,l] -= val * prod[l]
                        spline_arg += 1.0


cdef _joint_pdf_gradient_sparse_2d(double[:] theta, jacobian_function jacobian,
                                   double[:] sval, double[:] mval,
                                   double[:,:] sample_points,
                                   double[:,:] mgradient, double smin,
                                   double sdelta, double mmin, double mdelta,
                                   int nbins, int padding,
                                   double[:,:,:] grad_pdf):
    r''' Gradient of the joint PDF w.r.t. transform parameters theta

    Computes the vector of partial derivatives of the joint histogram w.r.t.
    each transformation parameter. The transformation itself is not necessary
    to compute the gradient, but only its Jacobian.

    Parameters
    ----------
    theta: array, shape (n,)
        parameters to compute the gradient at
    jacobian: jacobian_function
        function that computes the Jacobian matrix of a transformation
    sval: array, shape (m,)
        sampled intensities from the static image at sampled_points
    mval: array, shape (m,)
        sampled intensities from the moving image at sampled_points
    sample_points: array, shape (m, 2)
        coordinates (in physical space) of the points the images were sampled at
    mgradient: array, shape (m, 2)
        the gradient of the moving image at the sample points
    smin: float
        the minimum observed intensity associated to the static image, which
        was used to define the joint PDF
    sdelta: float
        bin size associated to the intensities of the static image
    mmin: float
        the minimum observed intensity associated to the moving image, which
        was used to define the joint PDF
    mdelta: float
        bin size associated to the intensities of the moving image
    nbins: int
        number of histogram bins
    padding: int
        number of bins used as padding (the total bins used for padding at both
        sides of the histogram is actually 2*padding)
    grad_pdf: array, shape (nbins, nbins, len(theta))
        the array to write the gradient to
    '''
    cdef:
        int n = theta.shape[0]
        int m = sval.shape[0]
        int offset, constant_jacobian=0
        cnp.npy_intp i, j, r, c
        double rn, cn
        double val, spline_arg
        double[:,:] J = np.ndarray(shape=(2, n), dtype=np.float64)
        double[:] prod = np.ndarray(shape=(n,), dtype=np.float64)

    grad_pdf[...] = 0
    with nogil:

        for i in range(m):
            if constant_jacobian == 0:
                constant_jacobian = jacobian(theta, sample_points[i], J)

            for j in range(n):
                prod[j] = J[0,j] * mgradient[i,0] +\
                          J[1,j] * mgradient[i,1]

            rn = _bin_normalize(sval[i], smin, sdelta)
            r = _bin_index(rn, nbins, padding)
            cn = _bin_normalize(mval[i], mmin, mdelta)
            c = _bin_index(cn, nbins, padding)
            spline_arg = (c-2) - cn

            for offset in range(-2, 3):
                val = _cubic_spline_derivative(spline_arg)
                for j in range(n):
                    grad_pdf[r, c + offset,j] -= val * prod[j]
                spline_arg += 1.0


cdef _joint_pdf_gradient_sparse_3d(double[:] theta, jacobian_function jacobian,
                                   double[:] sval, double[:] mval,
                                   double[:,:] sample_points,
                                   double[:,:] mgradient, double smin,
                                   double sdelta, double mmin, double mdelta,
                                   int nbins, int padding,
                                   double[:,:,:] grad_pdf):
    r''' Gradient of the joint PDF w.r.t. transform parameters theta

    Computes the vector of partial derivatives of the joint histogram w.r.t.
    each transformation parameter. The transformation itself is not necessary
    to compute the gradient, but only its Jacobian.

    Parameters
    ----------
    theta: array, shape (n,)
        parameters to compute the gradient at
    jacobian: jacobian_function
        function that computes the Jacobian matrix of a transformation
    sval: array, shape (m,)
        sampled intensities from the static image at sampled_points
    mval: array, shape (m,)
        sampled intensities from the moving image at sampled_points
    sample_points: array, shape (m, 3)
        coordinates (in physical space) of the points the images were sampled at
    mgradient: array, shape (m, 3)
        the gradient of the moving image at the sample points
    smin: float
        the minimum observed intensity associated to the static image, which
        was used to define the joint PDF
    sdelta: float
        bin size associated to the intensities of the static image
    mmin: float
        the minimum observed intensity associated to the moving image, which
        was used to define the joint PDF
    mdelta: float
        bin size associated to the intensities of the moving image
    nbins: int
        number of histogram bins
    padding: int
        number of bins used as padding (the total bins used for padding at both
        sides of the histogram is actually 2*padding)
    grad_pdf: array, shape (nbins, nbins, len(theta))
        the array to write the gradient to
    '''
    cdef:
        int n = theta.shape[0]
        int m = sval.shape[0]
        int offset, constant_jacobian=0
        cnp.npy_intp i, j, r, c
        double rn, cn
        double val, spline_arg
        double[:,:] J = np.ndarray(shape=(3, n), dtype=np.float64)
        double[:] prod = np.ndarray(shape=(n,), dtype=np.float64)

    grad_pdf[...] = 0
    with nogil:

        for i in range(m):
            if constant_jacobian == 0:
                constant_jacobian = jacobian(theta, sample_points[i], J)

            for j in range(n):
                prod[j] = J[0,j] * mgradient[i,0] +\
                          J[1,j] * mgradient[i,1] +\
                          J[2,j] * mgradient[i,2]

            rn = _bin_normalize(sval[i], smin, sdelta)
            r = _bin_index(rn, nbins, padding)
            cn = _bin_normalize(mval[i], mmin, mdelta)
            c = _bin_index(cn, nbins, padding)
            spline_arg = (c-2) - cn

            for offset in range(-2, 3):
                val = _cubic_spline_derivative(spline_arg)
                for j in range(n):
                    grad_pdf[r, c + offset,j] -= val * prod[j]
                spline_arg += 1.0


cdef double _compute_mattes_mi(double[:,:] joint, double[:,:,:] joint_gradient,
                               double[:] smarginal, double[:] mmarginal,
                               double[:] mi_gradient) nogil:
    r""" Computes the mutual information and its gradient (if requested)

    Parameters
    ----------
    joint : array, shape (nbins, nbins)
        the joint intensity distribution
    joint_gradient : array, shape(nbins, nbins, n)
        the gradient of the joint distribution w.r.t. the transformation
        parameters
    smarginal : array, shape (nbins,)
        the marginal intensity distribution of the static image
    mmarginal : array, shape (nbins,)
        the marginal intensity distribution of the moving image
    mi_gradient : array, shape (n,)
        the buffer in which to write the gradient of the mutual information.
        If None, the gradient is not computed
    """
    cdef:
        double epsilon = 2.2204460492503131e-016
        double metric_value
        int nrows = joint_gradient.shape[0]
        int ncols = joint_gradient.shape[1]
        int n = joint_gradient.shape[2]

    mi_gradient[:] = 0
    metric_value = 0
    for i in range(nrows):
        for j in range(ncols):
            if joint[i,j] < epsilon or mmarginal[j] < epsilon:
                continue

            factor = log(joint[i,j] / mmarginal[j])

            if mi_gradient is not None:
                for k in range(n):
                    mi_gradient[k] -= joint_gradient[i, j, k] * factor

            if smarginal[i] > epsilon:
                metric_value -= joint[i,j] * (factor - log(smarginal[i]))

    return metric_value


def sample_domain_2d(int[:] shape, int n, double[:,:] samples,
                     int[:,:] mask=None):
    r""" Take n samples from a grid of the given shape where mask is not zero

    The sampling is made without replacement. If the number of grid cells is
    less than n, then all cells are selected (in random order).
    Returns the number of samples actually taken

    Parameters
    ----------
    shape : array, shape (2,)
        the shape of the grid to be sampled
    n : int
        the number of samples to be acquired
    samples : array, shape (n, 2)
        buffer in which to write the acquired samples
    mask : array, shape (shape[0], shape[1])
        the mask indicating the valid samples. Only grid cells x with
        mask[x] != 0 will be selected

    Returns
    -------
    m : int
        the number of samples actually acquired

    Example
    -------
    >>> from dipy.align.mattes import sample_domain_2d
    >>> import dipy.align.vector_fields as vf
    >>> mask = np.array(vf.create_circle(10, 10, 3), dtype=np.int32)
    >>> n = 5
    >>> dim = 2
    >>> samples = np.empty((n, dim))
    >>> sample_domain_2d(np.array(mask.shape, dtype=np.int32), n, samples, mask)
    5
    >>> [mask[tuple(x)] for x in samples]
    [1, 1, 1, 1, 1]
    """
    cdef:
        int tmp, m, r, i, j
        double p, q
        int[:] index = np.empty(shape=(shape[0]*shape[1], ), dtype=np.int32)
    with nogil:
        # make an array of all avalable indices
        m = 0
        for i in range(shape[0]):
            for j in range(shape[1]):
                if mask is None or mask[i, j] != 0:
                    index[m] = i * shape[1] + j
                    m += 1
        if n > m:
            n = m
    selected = np.random.choice(index[:m], n)
    for i in range(n):
        samples[i,0] = selected[i] // shape[1]
        samples[i,1] = selected[i] % shape[1]
    return n


def sample_domain_3d(int[:] shape, int n, double[:,:] samples,
                     int[:,:,:] mask=None):
    r""" Take n samples from a grid of the given shape where mask is not zero

    The sampling is made without replacement. If the number of grid cells is
    less than n, then all cells are selected (in random order).
    Returns the number of samples actually taken

    Parameters
    ----------
    shape : array, shape (3,)
        the shape of the grid to be sampled
    n : int
        the number of samples to be acquired
    samples : array, shape (n, 3)
        buffer in which to write the acquired samples
    mask : array, shape (shape[0], shape[1], shape[2])
        the mask indicating the valid samples. Only grid cells x with
        mask[x] != 0 will be selected

    Returns
    -------
    m : int
        the number of samples actually acquired

    Example
    -------
    >>> from dipy.align.mattes import sample_domain_3d
    >>> import dipy.align.vector_fields as vf
    >>> mask = np.array(vf.create_sphere(10, 10, 10, 3), dtype=np.int32)
    >>> n = 10
    >>> dim = 3
    >>> samples = np.empty((n, dim))
    >>> sample_domain_3d(np.array(mask.shape, dtype=np.int32), n, samples, mask)
    10
    >>> [mask[tuple(x)] for x in samples]
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    """
    cdef:
        int tmp, m, r, i, j, k, ss
        double p, q
        int[:] index = np.empty((shape[0]*shape[1]*shape[2], ), dtype=np.int32)
    with nogil:
        # make an array of all avalable indices
        m = 0
        ss = shape[1] * shape[2]
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    if mask is None or mask[i, j, k] != 0:
                        index[m] = i * ss + j * shape[2] + k
                        m += 1
        if n > m:
            n = m
    selected = np.random.choice(index[:m], n)
    for i in range(n):
        samples[i,2] = selected[i] % shape[2]
        samples[i,1] = (selected[i] % ss) // shape[2]
        samples[i,0] = selected[i] // ss
    return n