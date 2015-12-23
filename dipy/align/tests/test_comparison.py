import numpy as np
import scipy as sp
import scipy.stats
import nibabel as nib
import matplotlib.pyplot as plt
import dipy.viz.regtools as rt
import experiments.registration.dataset_info as info

def test_polynomial_fit():
    # Load data
    t1_name = info.get_brainweb('t1','strip')
    t2_name = info.get_brainweb('t2','strip')
    t1_nib = nib.load(t1_name)
    t2_nib = nib.load(t2_name)
    t1 = t1_nib.get_data()
    t2 = t2_nib.get_data()


    x = t1.reshape(-1)
    y = t2.reshape(-1)

    drop_zeros = False
    if drop_zeros:
        non_zero = (x>0) * (y>0)
        x = x[non_zero]
        y = y[non_zero]
    n = len(x)

    # Set parameters
    p = 9
    tolerance = 1e-9
    #c = (n + p + 2 + 1) // 2
    c = int(0.8 * n)

    # Initialize parameters with random sample
    sel = np.random.choice(range(n), c, replace=False)
    xsel = x[sel]
    ysel = y[sel]
    theta = np.polyfit(xsel, ysel, deg=p)

    # Start LTS
    prev_residual = np.inf
    ysel_hat = np.polyval(theta, xsel)
    residual = np.sum((ysel-ysel_hat)**2)
    print("delta: ", prev_residual - residual)
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
        print("delta: ", prev_residual - residual)

    # Start RLS
    # get the 0.5 + c/2N quantile of the standard Gaussian distribution
    q = 0.5 + c/(2.0*n)
    alpha = sp.stats.norm.ppf(q, 0, 1)

    # Numerically integrate the second non-central moment of the Gaussian distribution within [-alpha, alpha]
    integral = sp.integrate.quad(lambda x: (x**2)*scipy.stats.norm.pdf(x, loc=0.0, scale=1), -alpha, alpha)
    K = (integral[0]*n) / c

    # Estimate standard deviation
    sigma_hat = np.sqrt(residual/(K*n))

    # Select all samples whose residual is within 3*sigma_hat
    y_hat = np.polyval(theta, x)
    rho = (y - y_hat)**2
    sel = rho <= 3*sigma_hat
    xsel = x[sel]
    ysel = y[sel]
    theta = np.polyfit(xsel, ysel, deg=p)


    # Plot the resulting transfer
    xlin = linspace(x.min(), x.max(), 300)
    ylin = np.polyval(theta, xlin)
    plot(xlin, ylin)



    yhat= np.polyval(theta, x)

    yhat = yhat.reshape(t1.shape)
    rt.plot_slices(yhat)
    rt.plot_slices(x.reshape(t1.shape))

    rt.overlay_slices(yhat, y.reshape(t1.shape))




np.random.choice()