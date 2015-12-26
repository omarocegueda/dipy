import numpy as np
import scipy as sp
import scipy.stats
import nibabel as nib
import matplotlib.pyplot as plt
import dipy.viz.regtools as rt
import experiments.registration.dataset_info as info
from dipy.align.comparison import count_masked_values

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















np.random.choice()