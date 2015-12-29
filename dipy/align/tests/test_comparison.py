import numpy as np
import scipy as sp
import scipy.stats
import nibabel as nib
import matplotlib.pyplot as plt
import dipy.viz.regtools as rt
import experiments.registration.dataset_info as info
from dipy.align.comparison import count_masked_values
from dipy.data import get_data
from dipy.align import vector_fields as vfu
import dipy.align.imwarp as imwarp
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align import VerbosityLevels
from numpy.testing import (assert_equal)
from dipy.align import floating
from dipy.align.polynomial import PolynomialTransfer


def get_warped_stacked_image(image, nslices, b, m):
    r""" Creates a volume by stacking copies of a deformed image

    The image is deformed under an invertible field, and a 3D volume is
    generated as follows:
    the first and last `nslices`//3 slices are filled with zeros
    to simulate background. The remaining middle slices are filled with
    copies of the deformed `image` under the action of the invertible
    field.

    Parameters
    ----------
    image : 2d array shape(r, c)
        the image to be deformed
    nslices : int
        the number of slices in the final volume
    b, m : float
        parameters of the harmonic field (as in [1]).

    Returns
    -------
    vol : array shape(r, c) if `nslices`==1 else (r, c, `nslices`)
        the volumed generated using the undeformed image
    wvol : array shape(r, c) if `nslices`==1 else (r, c, `nslices`)
        the volumed generated using the warped image

    References
    ----------
    [1] Chen, M., Lu, W., Chen, Q., Ruchala, K. J., & Olivera, G. H. (2008).
        A simple fixed-point approach to invert a deformation field.
        Medical Physics, 35(1), 81. doi:10.1118/1.2816107
    """
    shape = image.shape
    #create a synthetic invertible map and warp the circle
    d, dinv = vfu.create_harmonic_fields_2d(shape[0], shape[1], b, m)
    d = np.asarray(d, dtype=floating)
    dinv = np.asarray(dinv, dtype=floating)
    mapping = DiffeomorphicMap(2, shape)
    mapping.forward, mapping.backward = d, dinv
    wimage = mapping.transform(image)

    if(nslices == 1):
        return image, wimage

    #normalize and form the 3d by piling slices
    image = image.astype(floating)
    image = (image-image.min())/(image.max() - image.min())
    zero_slices = nslices // 3
    vol = np.zeros(shape=image.shape + (nslices,))
    vol[..., zero_slices:(2 * zero_slices)] = image[..., None]
    wvol = np.zeros(shape=image.shape + (nslices,))
    wvol[..., zero_slices:(2 * zero_slices)] = wimage[..., None]

    return vol, wvol

def estimate_polynomial_transfer(x, y, c=None, p=9, theta=None, tolerance=1e-9):
    n = len(x)
    if c is None:
        c = int(0.8 * n)

    if n < 10 * p:
        return None, None

    # Initialize parameters with random sample
    if theta is None:
        sel = np.random.choice(range(n), c, replace=False)
        xsel = x[sel]
        ysel = y[sel]
        theta = np.polyfit(xsel, ysel, deg=p)

    # Start LTS
    print('Starting LTS using %d/%d samples (%f percent)'%(c, n, (100.0*c)/n))
    prev_residual = np.inf
    ysel_hat = np.polyval(theta, xsel)
    residual = np.sum((ysel-ysel_hat)**2)
    print("delta: %f"%(prev_residual - residual,))
    while tolerance < (prev_residual - residual):
        y_hat = np.polyval(theta, x)
        residual_vector = (y - y_hat)**2

        # Select the c smallest residuals
        sel = np.argsort(residual_vector)[:c]
        xsel = x[sel]
        ysel = y[sel]
        theta = np.polyfit(xsel, ysel, deg=p)

        ysel_hat = np.polyval(theta, xsel)
        prev_residual = residual
        residual = np.sum((ysel - ysel_hat)**2) # this is the sum of the rho's, eq. (9) in Guimod 2001
        print("delta: %f"%(prev_residual - residual,))

    # Start RLS
    # get the 0.5 + c/2N quantile of the standard Gaussian distribution
    q = 0.5 + c/(2.0*n)
    alpha = sp.stats.norm.ppf(q, 0, 1)

    # Numerically integrate the second non-central moment of the Gaussian distribution within [-alpha, alpha]
    integral = sp.integrate.quad(lambda x: (x**2)*scipy.stats.norm.pdf(x, loc=0.0, scale=1), -alpha, alpha)
    K = (integral[0]*n) / c
    # Estimate standard deviation
    sigma_hat = np.sqrt(residual/(K*n))

    print("RLS with sigma_hat=%f"%(sigma_hat,))

    # Select all samples whose residual is within 3*sigma_hat
    y_hat = np.polyval(theta, x)
    rho = (y - y_hat)**2
    sel = rho <= 3*sigma_hat
    xsel = x[sel]
    ysel = y[sel]
    print('RLS using %d/%d samples (%f percent)'%(sel.sum(), n, (100.0*sel.sum())/n))
    theta = np.polyfit(xsel, ysel, deg=p)

    # Plot the resulting transfer
    xlin = np.linspace(x.min(), x.max(), 300)
    ylin = np.polyval(theta, xlin)
    plt.plot(xlin, ylin)
    return theta, sigma_hat, sel



def test_polynomial_fit():
    # Load data
    t1_name = info.get_brainweb('t1','strip')
    t2_name = info.get_brainweb('t2','strip')
    t1_nib = nib.load(t1_name)
    t2_nib = nib.load(t2_name)
    t1 = t1_nib.get_data()
    t2 = t2_nib.get_data()

    # Vectorize images
    x = t1.reshape(-1).astype(np.int32)
    y = t2.reshape(-1).astype(np.int32)

    drop_zeros = False
    if drop_zeros:
        non_zero = (x>0) * (y>0)
        x = x[non_zero]
        y = y[non_zero]

    # Allocate count vectors
    n0 = np.empty(1 + np.max(x), dtype=np.int32)
    n1 = np.empty(1 + np.max(x), dtype=np.int32)

    # Estimate first transfer
    theta0, sigma0, sel0 = estimate_polynomial_transfer(x, y)
    # Estimate second transfer
    x1 = x[~sel0]
    y1 = y[~sel0]
    theta1, sigma1, sel1 = estimate_polynomial_transfer(x1, y1)

    # Estimate marginal probabilities
    count_masked_values(x, sel0.astype(np.int32), n0)
    count_masked_values(x1, sel1.astype(np.int32), n1)
    total = n0 + n1
    pi0 = n0.astype(np.float64)/(total)
    pi1 = n1.astype(np.float64)/(total)
    pi0[total==0] = 0.5
    pi1[total==0] = 0.5

    # Estimate sigma
    n0 = sel0.sum()
    n1 = sel1.sum()
    p0 = (n0)/float(n0+n1)
    p1 = (n1)/float(n0+n1)
    sigma = p0*sigma0 + p1*sigma1

    #Gaussian distribution evaluated at all residuals w.r.t. each transfer
    yhat0 = np.polyval(theta0, x)
    yhat1 = np.polyval(theta1, x)
    d0 = yhat0 - y
    d1 = yhat1 - y
    G0 = sp.stats.norm.pdf(d0, loc=0, scale=sigma)
    G1 = sp.stats.norm.pdf(d1, loc=0, scale=sigma)

    #Final weights
    den = pi0[x] * G0 + pi1[x] * G1
    P0 = (pi0[x] * G0) / den
    P1 = (pi1[x] * G1) / den

    P0[den==0] = 0.5
    P1[den==0] = 0.5

    rt.overlay_slices(yhat0.reshape(t1.shape), yhat1.reshape(t1.shape))

    yhat = yhat0*P0 + yhat1*P1
    rt.overlay_slices(y.reshape(t1.shape), yhat0.reshape(t1.shape))
    rt.overlay_slices(y.reshape(t1.shape), yhat.reshape(t1.shape))


def test_polynomial_transfer_2d():
    r''' Test 2D SyN with PolynomialTransfer metric

    Register a coronal slice from a T1w brain MRI before and after warping
    it under a synthetic invertible map. We verify that the final
    registration is of good quality.
    '''

    fname = get_data('t1_coronal_slice')
    nslices = 1
    b = 0.1
    m = 4

    image = np.load(fname)
    moving, static = get_warped_stacked_image(image, nslices, b, m)

    #Configure the metric
    degree = 9
    smooth = 5.0
    q_levels = 256
    cprop = 0.95
    drop_zeros = False
    metric = PolynomialTransfer(dim=2, degree=degree, smooth=smooth,
                                q_levels=q_levels, cprop=cprop,
                                drop_zeros=drop_zeros)

    #Configure and run the Optimizer
    level_iters = [40, 20, 10]
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, level_iters)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)

    warped = mapping.transform(moving)
    starting_energy = np.sum((static - moving)**2)
    final_energy = np.sum((static - warped)**2)
    reduced = 1.0 - final_energy/starting_energy

    assert(reduced > 0.9)


def test_polynomial_transfer_3d():
    r''' Test 3D SyN with PolynomialTransfer metric

    Register a volume created by stacking copies of a coronal slice from
    a T1w brain MRI before and after warping it under a synthetic
    invertible map. We verify that the final registration is of good
    quality.
    '''
    fname = get_data('t1_coronal_slice')
    nslices = 21
    b = 0.1
    m = 4

    image = np.load(fname)
    moving, static = get_warped_stacked_image(image, nslices, b, m)

    #Configure the metric
    degree = 9
    smooth = 5.0
    q_levels = 256
    cprop = 0.95
    drop_zeros = True
    metric = PolynomialTransfer(dim=3, degree=degree, smooth=smooth,
                                q_levels=q_levels, cprop=cprop,
                                drop_zeros=drop_zeros)

    #Create the optimizer
    level_iters = [20, 5]
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    step_length=0.25
    ss_sigma_factor = 1.0
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric,
        level_iters, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)

    warped = mapping.transform(moving)
    starting_energy = np.sum((static - moving)**2)
    final_energy = np.sum((static - warped)**2)
    reduced = 1.0 - final_energy/starting_energy

    assert(reduced > 0.9)
