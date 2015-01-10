import numpy as np
import scipy as sp
from dipy.align.mattes import *
from dipy.core.ndindex import ndindex
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_almost_equal,
                           assert_equal)
from dipy.align.transforms import (transform_type,
                                   number_of_parameters,
                                   param_to_matrix,
                                   get_identity_parameters)

import dipy.align.imaffine as imaffine
from dipy.align.imaffine import aff_warp
import dipy.viz.regtools as rt
from dipy.data import get_data

def create_random_image_pair_2d(nr, nc, nvals):
    r""" Create a pair of images with an arbitrary, non-uniform joint
    distribution
    """
    sh = (nr, nc)
    static = np.random.randint(0, nvals, nr*nc).reshape(sh)

    # This is just a simple way of making  the distribution non-uniform
    moving = static.copy()
    moving += np.random.randint(0, nvals//2, nr*nc).reshape(sh) - nvals//4

    # This is just a simple way of making  the distribution non-uniform
    static = moving.copy()
    static += np.random.randint(0, nvals//2, nr*nc).reshape(sh) - nvals//4

    return static.astype(np.float64), moving.astype(np.float64)


def create_random_image_pair_3d(ns, nr, nc, nvals):
    r""" Create a pair of images with an arbitrary, non-uniform joint
    distribution
    """
    sh = (ns, nr, nc)
    static = np.random.randint(0, nvals, ns*nr*nc).reshape(sh)

    # This is just a simple way of making  the distribution non-uniform
    moving = static.copy()
    moving += np.random.randint(0, nvals//2, ns*nr*nc).reshape(sh) - nvals//4

    # This is just a simple way of making  the distribution non-uniform
    static = moving.copy()
    static += np.random.randint(0, nvals//2, ns*nr*nc).reshape(sh) - nvals//4

    return static.astype(np.float64), moving.astype(np.float64)


def test_mattes_densities_dense():
    # Test pdf dense
    np.random.seed(1246592)
    nbins = 32
    nr = 30
    nc = 35
    ns = 20
    nvals = 50

    for dim in [2, 3]:
        if dim == 2:
            shape = (nr, nc)
            static, moving = create_random_image_pair_2d(nr, nc, nvals)
        else:
            shape = (ns, nr, nc)
            static, moving = create_random_image_pair_3d(ns, nr, nc, nvals)

        # Initialize
        mbase = MattesBase(nbins)
        mbase.setup(static, moving)
        mbase.update_pdfs_dense(static, moving)
        actual_joint = mbase.joint
        actual_mmarginal = mbase.mmarginal
        actual_smarginal = mbase.smarginal

        # Compute the expected joint distribution
        expected_joint = np.zeros(shape=(nbins, nbins))
        for index in ndindex(shape):
            sval = mbase.bin_normalize_static(static[index])
            mval = mbase.bin_normalize_moving(moving[index])
            sbin = mbase.bin_index(sval)
            mbin = mbase.bin_index(mval)
            # The spline is centered at mval, will evaluate for all row
            spline_arg = np.array([i - mval for i in range(nbins)])
            contribution = np.empty(nbins)
            cubic_spline(spline_arg, contribution)
            expected_joint[sbin, :] += contribution

        # Verify joint distribution
        expected_joint /= expected_joint.sum()
        assert_array_almost_equal(actual_joint, expected_joint)

        # Verify moving marginal
        expected_mmarginal = expected_joint.sum(0)
        expected_mmarginal /= expected_mmarginal.sum()
        assert_array_almost_equal(actual_mmarginal, expected_mmarginal)

        # Verivy static marginal
        expected_smarginal = expected_joint.sum(1)
        expected_smarginal /= expected_smarginal.sum()
        assert_array_almost_equal(actual_smarginal, expected_smarginal)


def setup_random_transform_2d(ttype, rfactor):
    np.random.seed(3147702)
    dim = 2
    fname = get_data('t1_coronal_slice')
    moving = np.load(fname)
    moving = moving[40:180, 50:210]
    moving_aff = np.eye(dim + 1)
    mmask = np.ones_like(moving, dtype=np.int32)

    n = number_of_parameters(ttype, dim)
    theta = np.empty(n)
    get_identity_parameters(ttype, dim, theta)
    theta += np.random.rand(n) * rfactor

    T = np.empty((dim + 1, dim + 1))
    param_to_matrix(ttype, dim, theta, T)

    static = aff_warp(moving, moving_aff, moving, moving_aff, T)
    static = static.astype(np.float64)
    static_aff = moving_aff
    smask = np.ones_like(static, dtype=np.int32)

    return static, moving, static_aff, moving_aff, smask, mmask, T





def test_mattes_gradient_2d():
    h = 1e-5
    dim = 2
    transforms = transform_type.keys()
    factors = {'TRANSLATION':2.0,
               'ROTATION':0.1,
               'RIGID':0.1,
               'SCALING':0.1,
               'AFFINE':0.1}
    for transform in transforms:
        factor = factors[transform]
        t = transform_type[transform]
        n = number_of_parameters(t, dim)
        theta = np.ndarray(shape=(n,))
        get_identity_parameters(t, dim, theta)
        theta += h

        static, moving, static_aff, moving_aff, smask, mmask, T = \
                                            setup_random_transform_2d(t, factor)

        metric = imaffine.MattesMIMetric(32)
        spacing = np.ones(2)
        metric.setup(transform, static, moving, spacing, static_aff, moving_aff,
                     smask, mmask, prealign=None, precondition=False)

        # Compute the gradient with the implementation under test
        val0, actual = metric.value_and_gradient(theta)
        # Compute the gradient using finite-diferences
        expected = np.empty_like(actual)
        for i in range(n):
            dtheta = theta.copy()
            dtheta[i] += h
            val1 = metric.distance(dtheta)
            expected[i] = (val1 - val0) / h

        nprod = expected.dot(actual) / (linalg.norm(expected) * linalg.norm(actual))
        print("Test: ", transform, nprod)
        assert_equal(nprod >= 0.999, True)



def test_mattes_densities_sparse():
    # Test pdf dense
    np.random.seed(3147702)
    nbins = 32
    nr = 30
    nc = 35
    nvals = 50

    shape = (nr, nc)
    static, moving = create_random_image_pair_2d(nr, nc, nvals)
    sval = static.reshape(-1)
    mval = moving.reshape(-1)

    # Initialize
    mbase = MattesBase(nbins)
    mbase.setup(static, moving)
    mbase.update_pdfs_sparse(sval, mval)
    actual_joint = mbase.joint
    actual_mmarginal = mbase.mmarginal
    actual_smarginal = mbase.smarginal

    # Compute the expected joint distribution
    expected_joint = np.zeros(shape=(nbins, nbins))
    for index in range(sval.shape[0]):
        sv = mbase.bin_normalize_static(sval[index])
        mv = mbase.bin_normalize_moving(mval[index])
        sbin = mbase.bin_index(sv)
        mbin = mbase.bin_index(mv)
        # The spline is centered at mval, will evaluate for all row
        spline_arg = np.array([i - mv for i in range(nbins)])
        contribution = np.empty(nbins)
        cubic_spline(spline_arg, contribution)
        expected_joint[sbin, :] += contribution

    # Verify joint distribution
    expected_joint /= expected_joint.sum()
    assert_array_almost_equal(actual_joint, expected_joint)

    # Verify moving marginal
    expected_mmarginal = expected_joint.sum(0)
    expected_mmarginal /= expected_mmarginal.sum()
    assert_array_almost_equal(actual_mmarginal, expected_mmarginal)

    # Verivy static marginal
    expected_smarginal = expected_joint.sum(1)
    expected_smarginal /= expected_smarginal.sum()
    assert_array_almost_equal(actual_smarginal, expected_smarginal)

if __name__ == '__main__':
    test_mattes_densities_dense()
    test_mattes_densities_sparse()
