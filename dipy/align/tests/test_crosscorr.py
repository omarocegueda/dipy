import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from numpy.testing import assert_array_almost_equal
from dipy.align import floating
from dipy.align import crosscorr as cc
import nibabel as nib
from dipy.viz import regtools as rt
import time


def test_precision_loss():
    ftype = np.float32
    exponent = 2
    a_fname = r'D:\opt\registration\data\IBSR_nifti_stripped\IBSR_01\IBSR_01_ana_strip.nii.gz'
    a_nib = nib.load(a_fname)
    a = a_nib.get_data().squeeze().astype(ftype)

    #for radius in range(1, 5):
    for radius in [3]:
        rs_proposed = np.array(cc.window_sums(a, radius, exponent=exponent))
        rs_integral = np.array(cc.window_sums_integral(a, radius, exponent=exponent))
        rs_direct = np.array(cc.window_sums_direct(a, radius, exponent=exponent))

        error_proposed = np.abs(rs_proposed - rs_direct)
        error_integral = np.abs(rs_integral - rs_direct)

        sad_proposed_2 = error_proposed.mean(2)
        sad_integral_2 = error_integral.mean(2)

        fig = plt.figure(facecolor='white')
        axes = fig.add_subplot(1,1,1)
        axes.set_axis_off()
        mapable = axes.imshow(sad_integral_2)
        colorbar = fig.colorbar(mapable, label='Mean Absolute Difference')
        colorbar.ax.tick_params(labelsize=32)
        cb_axes = colorbar.ax
        cb_axes.yaxis.label.set_fontsize(48)

        fig = plt.figure(facecolor='white')
        axes = fig.add_subplot(1,1,1)
        axes.set_axis_off()
        mapable = axes.imshow(sad_proposed_2)
        colorbar = fig.colorbar(mapable, label='Mean Absolute Differences')
        colorbar.ax.tick_params(labelsize=32)
        cb_axes = colorbar.ax
        cb_axes.yaxis.label.set_fontsize(48)






def plot_times(times, names):
    fig = plt.figure(facecolor='white')
    axes = fig.add_subplot(1,1,1)
    k = len(times)
    styles = ['-', '--', '-.', ':']
    for i in range(k):
        t = times[i]
        n = t.shape[0]
        xticks = range(1, n+1)
        label = names[i]
        line, = axes.plot(xticks, t, linestyle=styles[i], linewidth=2)
        line.set_label(label)
    axes.legend(loc=0, fontsize=48)
    axes.grid()
    axes.set_xticklabels(xticks, fontsize=32)

    maxy = np.max([I.max() for I in times])
    miny = -50 if maxy>1000 else 0
    axes.set_ylim((miny, maxy))
    deltay = 200 if maxy>1000 else 10
    yticks = range(int(miny), int(maxy)+deltay-1, deltay)
    axes.set_yticks(yticks)
    axes.set_yticklabels(yticks, fontsize=32)

    axes.set_xlabel('Window radius', fontsize=40)
    axes.set_ylabel('Time (s)', fontsize=40)



def test_cc_steps_no_factors():
    r"""
    Compares the output of the optimized function to compute the cross-
    correlation factors against a direct (not optimized, but less error prone)
    implementation.
    """
    proposed = np.array([2.783000e+00,3.053000e+00,3.120000e+00,3.205000e+00,3.276000e+00,3.323000e+00,3.329000e+00,3.320000e+00,3.288000e+00,3.236000e+00])
    integral = np.array([4.232000e+00,4.392000e+00,4.220000e+00,4.218000e+00,3.750000e+00,3.744000e+00,3.597000e+00,3.527000e+00,3.408000e+00,3.401000e+00])
    avants = np.array([1.241000e+00,2.143000e+00,3.350000e+00,5.394000e+00,8.064000e+00,1.068300e+01,1.488700e+01,2.225700e+01,4.003100e+01,5.840800e+01])
    direct = np.array([2.983000e+00,1.194600e+01,2.546700e+01,4.165200e+01,7.699900e+01,1.221640e+02,1.889960e+02,3.339210e+02,7.338560e+02,1.396136e+03])
    n = direct.shape[0]
    k = 0
    times = [proposed[0:(n-k)], avants[0:(n-k)], direct[0:(n-k)]]
    names = ['Proposed: $\Theta(N^{3})$', 'ANTs: $\Theta(m^{2}N^{3})$', 'Direct: $\Theta(m^{3}N^{3})$']

    times = [proposed[0:(n-k)], avants[0:(n-k)]]
    names = ['Proposed: $\Theta(N^{3})$', 'ANTs: $\Theta(m^{2}N^{3})$']

    a_fname = r'D:\opt\registration\data\IBSR_nifti_stripped\IBSR_01\IBSR_01_ana_strip.nii.gz'
    b_fname = r'D:\opt\registration\data\IBSR_nifti_stripped\IBSR_02\IBSR_02_ana_strip.nii.gz'
    a_nib = nib.load(a_fname)
    b_nib = nib.load(b_fname)
    a = a_nib.get_data().squeeze().astype(floating)
    b = b_nib.get_data().squeeze().astype(floating)

    #sz = 200
    #a = np.array(range(sz*sz*sz), dtype=floating).reshape(sz,sz,sz)
    #b = np.array(range(sz*sz*sz)[::-1], dtype=floating).reshape(sz,sz,sz)
    #a = np.random.randn(sz*sz*sz).reshape(sz,sz,sz).astype(floating)
    #b = np.random.randn(sz*sz*sz).reshape(sz,sz,sz).astype(floating)
    #a /= np.abs(a).max()
    #b /= np.abs(b).max()
    #a*=100
    #b*=100
    #a = a**2
    #b = b**3
    ga = np.empty(shape=(a.shape)+(3,), dtype=floating)
    gb = np.empty(shape=(a.shape)+(3,), dtype=floating)
    for i, grad in enumerate(sp.gradient(a)):
        ga[..., i] = grad
    for i, grad in enumerate(sp.gradient(b)):
        gb[..., i] = grad

    for radius in range(1,11):
        print("-------------- Radius: %d --------------"%(radius,))
        # Compute using factors
        start = time.time()
        factors = np.asarray(cc.precompute_cc_factors_3d_old(a,b,radius))
        factors_fwd, factors_fwd_energy = cc.compute_cc_forward_step_3d(ga, factors, radius)
        factors_fwd = np.array(factors_fwd)
        factors_bwd, factors_bwd_energy = cc.compute_cc_backward_step_3d(gb, factors, radius)
        factors_bwd = np.array(factors_bwd)
        end = time.time()
        elapsed = end - start
        print("Avants: %e"%(elapsed,))

    for radius in range(1,11):
        print("-------------- Radius: %d --------------"%(radius,))
        # Compute using integral images
        start = time.time()
        integral_fwd, integral_bwd, integral_fwd_energy, integral_bwd_energy = cc.compute_cc_steps_3d_integral(a, b, ga, gb, radius)
        integral_fwd = np.array(integral_fwd)
        integral_bwd = np.array(integral_bwd)
        end = time.time()
        elapsed = end - start
        print("Integral images: %e"%(elapsed,))

        # Compute using DP
    for radius in range(1,11):
        print("-------------- Radius: %d --------------"%(radius,))
        start = time.time()
        dp_fwd, dp_bwd, dp_fwd_energy, dp_bwd_energy = cc.compute_cc_steps_3d_nofactors(a, b, ga, gb, radius)
        dp_fwd = np.array(dp_fwd)
        dp_bwd = np.array(dp_bwd)
        end = time.time()
        elapsed = end - start
        print("Proposed: %e"%(elapsed,))

    for radius in range(1,11):
        print("-------------- Radius: %d --------------"%(radius,))
        start = time.time()
        brute_fwd, brute_bwd, brute_fwd_energy, brute_bwd_energy = cc.compute_cc_steps_3d_nofactors_test(a, b, ga, gb, radius)
        brute_fwd = np.array(brute_fwd)
        brute_bwd = np.array(brute_bwd)
        end = time.time()
        elapsed = end - start
        print("Direct: %e"%(elapsed,))

        if False:
            dif_brute_dp_fwd = np.abs(brute_fwd - dp_fwd).max()
            dif_brute_dp_bwd = np.abs(brute_bwd - dp_bwd).max()
            print("---Difference between brute and DP ---")
            print("Max diff. fwd: %e. Max diff bwd: %e"%(dif_brute_dp_fwd, dif_brute_dp_bwd))
            print("Diff energy fwd: %e. Diff energy bwd: %e"%(np.abs(brute_fwd_energy-dp_fwd_energy), np.abs(brute_bwd_energy-dp_bwd_energy)))
            print("------------------------------------------------")
            print("Difference between brute and using factors")
            dif_brute_factors_fwd = np.abs(brute_fwd - factors_fwd).max()
            dif_brute_factors_bwd = np.abs(brute_bwd - factors_bwd).max()
            print("Max diff. fwd: %e. Max diff bwd: %e"%(dif_brute_factors_fwd, dif_brute_factors_bwd))
            print("Diff energy fwd: %e. Diff energy bwd: %e"%(np.abs(brute_fwd_energy-factors_fwd_energy), np.abs(brute_bwd_energy-factors_bwd_energy)))
            print("------------------------------------------------")
            print("Difference between brute and using integral images")
            dif_brute_integral_fwd = np.abs(brute_fwd - integral_fwd).max()
            dif_brute_integral_bwd = np.abs(brute_bwd - integral_bwd).max()
            print("Max diff. fwd: %e. Max diff bwd: %e"%(dif_brute_integral_fwd, dif_brute_integral_bwd))
            print("Diff energy fwd: %e. Diff energy bwd: %e"%(np.abs(brute_fwd_energy-integral_fwd_energy), np.abs(brute_bwd_energy-integral_bwd_energy)))
            print("-------------------------------------"%(radius,))
            #assert_array_almost_equal(factors, expected, decimal=5)


def test_cc_factors_2d():
    r"""
    Compares the output of the optimized function to compute the cross-
    correlation factors against a direct (not optimized, but less error prone)
    implementation.
    """
    a = np.array(range(20*20), dtype=floating).reshape(20,20)
    b = np.array(range(20*20)[::-1], dtype=floating).reshape(20,20)
    a /= a.max()
    b /= b.max()
    for radius in [0, 1, 3, 6]:
        factors = np.asarray(cc.precompute_cc_factors_2d(a,b,radius))
        expected = np.asarray(cc.precompute_cc_factors_2d_test(a,b,radius))
        assert_array_almost_equal(factors, expected)


def test_cc_factors_3d():
    r"""
    Compares the output of the optimized function to compute the cross-
    correlation factors against a direct (not optimized, but less error prone)
    implementation.
    """
    a = np.array(range(20*20*20), dtype=floating).reshape(20,20,20)
    b = np.array(range(20*20*20)[::-1], dtype=floating).reshape(20,20,20)
    a /= a.max()
    b /= b.max()
    for radius in [0, 1, 3, 6]:
        factors = np.asarray(cc.precompute_cc_factors_3d(a,b,radius))
        expected = np.asarray(cc.precompute_cc_factors_3d_test(a,b,radius))
        assert_array_almost_equal(factors, expected, decimal=5)


def test_compute_cc_steps_2d():
    #Select arbitrary images' shape (same shape for both images)
    sh = (32, 32)
    radius = 2

    #Select arbitrary centers
    c_f = (np.asarray(sh)/2) + 1.25
    c_g = c_f + 2.5

    #Compute the identity vector field I(x) = x in R^2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    X = np.ndarray(sh + (2,), dtype = np.float64)
    O = np.ones(sh)
    X[...,0]= x_0[:, None] * O
    X[...,1]= x_1[None, :] * O

    #Compute the gradient fields of F and G
    np.random.seed(1147572)

    grad_F = np.array(X - c_f, dtype = floating)
    grad_G = np.array(X - c_g, dtype = floating)

    Fnoise = np.random.ranf(np.size(grad_F)).reshape(grad_F.shape) * grad_F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    grad_F += Fnoise

    Gnoise = np.random.ranf(np.size(grad_G)).reshape(grad_G.shape) * grad_G.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    grad_G += Gnoise

    sq_norm_grad_G = np.sum(grad_G**2,-1)

    F = np.array(0.5*np.sum(grad_F**2,-1), dtype = floating)
    G = np.array(0.5*sq_norm_grad_G, dtype = floating)

    Fnoise = np.random.ranf(np.size(F)).reshape(F.shape) * F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    F += Fnoise

    Gnoise = np.random.ranf(np.size(G)).reshape(G.shape) * G.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    G += Gnoise

    #precompute the cross correlation factors
    factors = cc.precompute_cc_factors_2d_test(F, G, radius)
    factors = np.array(factors, dtype = floating)

    #test the forward step against the exact expression
    I = factors[..., 0]
    J = factors[..., 1]
    sfm = factors[..., 2]
    sff = factors[..., 3]
    smm = factors[..., 4]
    expected = np.ndarray(shape = sh + (2,), dtype = floating)
    expected[...,0] = (-2.0 * sfm / (sff * smm)) * (J - (sfm / sff) * I) * grad_F[..., 0]
    expected[...,1] = (-2.0 * sfm / (sff * smm)) * (J - (sfm / sff) * I) * grad_F[..., 1]
    actual, energy = cc.compute_cc_forward_step_2d(grad_F, factors, 0)
    assert_array_almost_equal(actual, expected)
    for radius in range(1,5):
        expected[:radius, ...] = 0
        expected[:, :radius, ...] = 0
        expected[-radius::, ...] = 0
        expected[:, -radius::, ...] = 0
        actual, energy = cc.compute_cc_forward_step_2d(grad_F, factors, radius)
        assert_array_almost_equal(actual, expected)

    #test the backward step against the exact expression
    expected[...,0] = (-2.0 * sfm / (sff * smm)) * (I - (sfm / smm) * J) * grad_G[..., 0]
    expected[...,1] = (-2.0 * sfm / (sff * smm)) * (I - (sfm / smm) * J) * grad_G[..., 1]
    actual, energy = cc.compute_cc_backward_step_2d(grad_G, factors, 0)
    assert_array_almost_equal(actual, expected)
    for radius in range(1,5):
        expected[:radius, ...] = 0
        expected[:, :radius, ...] = 0
        expected[-radius::, ...] = 0
        expected[:, -radius::, ...] = 0
        actual, energy = cc.compute_cc_backward_step_2d(grad_G, factors, radius)
        assert_array_almost_equal(actual, expected)


def test_compute_cc_steps_3d():
    sh = (32, 32, 32)
    radius = 2

    #Select arbitrary centers
    c_f = (np.asarray(sh)/2) + 1.25
    c_g = c_f + 2.5

    #Compute the identity vector field I(x) = x in R^2
    x_0 = np.asarray(range(sh[0]))
    x_1 = np.asarray(range(sh[1]))
    x_2 = np.asarray(range(sh[2]))
    X = np.ndarray(sh + (3,), dtype = np.float64)
    O = np.ones(sh)
    X[...,0]= x_0[:, None, None] * O
    X[...,1]= x_1[None, :, None] * O
    X[...,2]= x_2[None, None, :] * O

    #Compute the gradient fields of F and G
    np.random.seed(12465825)
    grad_F = np.array(X - c_f, dtype = floating)
    grad_G = np.array(X - c_g, dtype = floating)

    Fnoise = np.random.ranf(np.size(grad_F)).reshape(grad_F.shape) * grad_F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    grad_F += Fnoise

    Gnoise = np.random.ranf(np.size(grad_G)).reshape(grad_G.shape) * grad_G.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    grad_G += Gnoise

    sq_norm_grad_G = np.sum(grad_G**2,-1)

    F = np.array(0.5*np.sum(grad_F**2,-1), dtype = floating)
    G = np.array(0.5*sq_norm_grad_G, dtype = floating)

    Fnoise = np.random.ranf(np.size(F)).reshape(F.shape) * F.max() * 0.1
    Fnoise = Fnoise.astype(floating)
    F += Fnoise

    Gnoise = np.random.ranf(np.size(G)).reshape(G.shape) * G.max() * 0.1
    Gnoise = Gnoise.astype(floating)
    G += Gnoise
    expected = np.ndarray(shape = sh + (3,), dtype = floating)
    for radius in range(1,5):
        #precompute the cross correlation factors
        factors = cc.precompute_cc_factors_3d_test(F, G, radius)
        factors = np.array(factors, dtype = floating)

        #test the forward step against the exact expression
        I = factors[..., 0]
        J = factors[..., 1]
        sfm = factors[..., 2]
        sff = factors[..., 3]
        smm = factors[..., 4]

        expected[...,0] = (-2.0 * sfm / (sff * smm)) * (J - (sfm / sff) * I) * grad_F[..., 0]
        expected[...,1] = (-2.0 * sfm / (sff * smm)) * (J - (sfm / sff) * I) * grad_F[..., 1]
        expected[...,2] = (-2.0 * sfm / (sff * smm)) * (J - (sfm / sff) * I) * grad_F[..., 2]
        expected[:radius, ...] = 0
        expected[:, :radius, ...] = 0
        expected[:, :, :radius, :] = 0
        expected[-radius::, ...] = 0
        expected[:, -radius::, ...] = 0
        expected[:, :, -radius::, ...] = 0
        actual, energy = cc.compute_cc_forward_step_3d(grad_F, factors, radius)
        assert_array_almost_equal(actual, expected)
        actual, energy = cc.compute_cc_forward_step_3d_nofactors(F, G, grad_F, radius)
        assert_array_almost_equal(actual, expected)

    for radius in range(1,5):
        #precompute the cross correlation factors
        factors = cc.precompute_cc_factors_3d_test(F, G, radius)
        factors = np.array(factors, dtype = floating)

        #test the forward step against the exact expression
        I = factors[..., 0]
        J = factors[..., 1]
        sfm = factors[..., 2]
        sff = factors[..., 3]
        smm = factors[..., 4]
        expected[...,0] = (-2.0 * sfm / (sff * smm)) * (I - (sfm / smm) * J) * grad_G[..., 0]
        expected[...,1] = (-2.0 * sfm / (sff * smm)) * (I - (sfm / smm) * J) * grad_G[..., 1]
        expected[...,2] = (-2.0 * sfm / (sff * smm)) * (I - (sfm / smm) * J) * grad_G[..., 2]
        expected[:radius, ...] = 0
        expected[:, :radius, ...] = 0
        expected[:, :, :radius, :] = 0
        expected[-radius::, ...] = 0
        expected[:, -radius::, ...] = 0
        expected[:, :, -radius::, ...] = 0
        actual, energy = cc.compute_cc_backward_step_3d(grad_G, factors, radius)
        assert_array_almost_equal(actual, expected)
        actual, energy = cc.compute_cc_backward_step_3d_nofactors(F, G, grad_G, radius)
        assert_array_almost_equal(actual, expected)


if __name__=='__main__':
    test_cc_factors_2d()
    test_cc_factors_3d()
    test_compute_cc_steps_2d()
    test_compute_cc_steps_3d()
