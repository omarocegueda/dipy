import numpy as np
from dipy.align import floating
import dipy.align.vector_fields as vf
import dipy.align.imaffine as imaffine
import dipy.core.geometry as geometry
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_almost_equal,
                           assert_equal)
import nibabel as nib
import dipy.align.mattes as mattes
import scipy as sp
from dipy.align.transforms import (Transform,
                                   regtransforms)
from dipy.align.imaffine import *
import dipy.viz.regtools as rt
import dipy.align.imaffine as imaffine
from dipy.data import get_data


def test_aff_centers_of_mass_3d():
    np.random.seed(1246592)
    shape = (64, 64, 64)
    rm = 8
    sp = vf.create_sphere(shape[0]//2, shape[1]//2, shape[2]//2, rm)
    moving = np.zeros(shape)
    # The center of mass will be (16, 16, 16), in image coordinates
    moving[:shape[0]//2, :shape[1]//2, :shape[2]//2] = sp[...]

    rs = 16
    # The center of mass will be (32, 32, 32), in image coordinates
    static = vf.create_sphere(shape[0], shape[1], shape[2], rs)

    # Create arbitrary image-to-space transforms
    axis = np.array([.5, 2.0, 1.5])
    t = 0.15 #translation factor
    trans = np.array([[1, 0, 0, -t*shape[0]],
                      [0, 1, 0, -t*shape[1]],
                      [0, 0, 1, -t*shape[2]],
                      [0, 0, 0, 1]])
    trans_inv = np.linalg.inv(trans)

    for theta in [-1 * np.pi/6.0, 0.0, np.pi/5.0]: #rotation angle
        for s in [0.83,  1.3, 2.07]: #scale
            rot = np.zeros(shape=(4,4))
            rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
            rot[3,3] = 1.0
            scale = np.array([[1*s, 0, 0, 0],
                              [0, 1*s, 0, 0],
                              [0, 0, 1*s, 0],
                              [0, 0, 0, 1]])

            static_affine = trans_inv.dot(scale.dot(rot.dot(trans)))
            moving_affine = np.linalg.inv(static_affine)

            # Expected translation
            c_static = static_affine.dot((32, 32, 32, 1))[:3]
            c_moving = moving_affine.dot((16, 16, 16, 1))[:3]
            expected = np.eye(4);
            expected[:3, 3] = c_moving - c_static

            # Implementation under test
            actual = imaffine.aff_centers_of_mass(static, static_affine, moving, moving_affine)
            assert_array_almost_equal(actual, expected)


def test_aff_geometric_centers_3d():
    # Create arbitrary image-to-space transforms
    axis = np.array([.5, 2.0, 1.5])
    t = 0.15 #translation factor

    for theta in [-1 * np.pi/6.0, 0.0, np.pi/5.0]: #rotation angle
        for s in [0.83,  1.3, 2.07]: #scale
            for shape_moving in [(256, 256, 128), (255, 255, 127), (64, 127, 142)]:
                for shape_static in [(256, 256, 128), (255, 255, 127), (64, 127, 142)]:
                    moving = np.ndarray(shape=shape_moving)
                    static = np.ndarray(shape=shape_static)
                    trans = np.array([[1, 0, 0, -t*shape_static[0]],
                                      [0, 1, 0, -t*shape_static[1]],
                                      [0, 0, 1, -t*shape_static[2]],
                                      [0, 0, 0, 1]])
                    trans_inv = np.linalg.inv(trans)
                    rot = np.zeros(shape=(4,4))
                    rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
                    rot[3,3] = 1.0
                    scale = np.array([[1*s, 0, 0, 0],
                                      [0, 1*s, 0, 0],
                                      [0, 0, 1*s, 0],
                                      [0, 0, 0, 1]])

                    static_affine = trans_inv.dot(scale.dot(rot.dot(trans)))
                    moving_affine = np.linalg.inv(static_affine)

                    # Expected translation
                    c_static = tuple(np.array(shape_static, dtype = np.float64)*0.5)
                    c_static = static_affine.dot(c_static+(1,))[:3]
                    c_moving = tuple(np.array(shape_moving, dtype = np.float64)*0.5)
                    c_moving = moving_affine.dot(c_moving+(1,))[:3]
                    expected = np.eye(4);
                    expected[:3, 3] = c_moving - c_static

                    # Implementation under test
                    actual = imaffine.aff_geometric_centers(static, static_affine, moving, moving_affine)
                    assert_array_almost_equal(actual, expected)


def test_aff_origins_3d():
    # Create arbitrary image-to-space transforms
    axis = np.array([.5, 2.0, 1.5])
    t = 0.15 #translation factor

    for theta in [-1 * np.pi/6.0, 0.0, np.pi/5.0]: #rotation angle
        for s in [0.83,  1.3, 2.07]: #scale
            for shape_moving in [(256, 256, 128), (255, 255, 127), (64, 127, 142)]:
                for shape_static in [(256, 256, 128), (255, 255, 127), (64, 127, 142)]:
                    moving = np.ndarray(shape=shape_moving)
                    static = np.ndarray(shape=shape_static)
                    trans = np.array([[1, 0, 0, -t*shape_static[0]],
                                      [0, 1, 0, -t*shape_static[1]],
                                      [0, 0, 1, -t*shape_static[2]],
                                      [0, 0, 0, 1]])
                    trans_inv = np.linalg.inv(trans)
                    rot = np.zeros(shape=(4,4))
                    rot[:3, :3] = geometry.rodrigues_axis_rotation(axis, theta)
                    rot[3,3] = 1.0
                    scale = np.array([[1*s, 0, 0, 0],
                                      [0, 1*s, 0, 0],
                                      [0, 0, 1*s, 0],
                                      [0, 0, 0, 1]])

                    static_affine = trans_inv.dot(scale.dot(rot.dot(trans)))
                    moving_affine = np.linalg.inv(static_affine)

                    # Expected translation
                    c_static = static_affine[:3, 3]
                    c_moving = moving_affine[:3, 3]
                    expected = np.eye(4);
                    expected[:3, 3] = c_moving - c_static

                    # Implementation under test
                    actual = imaffine.aff_origins(static, static_affine, moving, moving_affine)
                    assert_array_almost_equal(actual, expected)


def setup_random_transform_2d(transform, rfactor):
    np.random.seed(3147702)
    dim = 2
    fname = get_data('t1_coronal_slice')
    moving = np.load(fname)
    moving_aff = np.eye(dim + 1)
    mmask = np.ones_like(moving)

    n = transform.get_number_of_parameters()
    theta = transform.get_identity_parameters()
    theta += np.random.rand(n) * rfactor

    T = transform.param_to_matrix(theta)

    static = aff_warp(moving, moving_aff, moving, moving_aff, T)
    static = static.astype(np.float64)
    static_aff = moving_aff
    smask = np.ones_like(static)

    return static, moving, static_aff, moving_aff, smask, mmask, T


def test_mi_gradient():
    # Test the gradient of mutual information using MattesMIMetric,
    # which extends MattesBase
    h = 1e-5
    factors = {('TRANSLATION', 2):2.0,
               ('ROTATION', 2):0.1,
               ('RIGID', 2):0.1,
               ('SCALING', 2):0.01,
               ('AFFINE', 2):0.1,
               ('TRANSLATION', 3):2.0,
               ('ROTATION', 3):0.2,
               ('RIGID', 3):0.1,
               ('SCALING', 3):0.1,
               ('AFFINE', 3):0.1}
    for ttype in factors.keys():
        dim = ttype[1]
        if dim == 2:
            nslices = 1
        else:
            nslices = 45

        transform = regtransforms[ttype]
        factor = factors[ttype]
        static, moving, static_aff, moving_aff, smask, mmask, T = \
                        setup_random_transform(transform, factor, nslices, 5.0)
        smask=None
        mmask=None
        theta = transform.get_identity_parameters().copy()
        metric = MattesMIMetric(32)
        spacing = np.ones(dim)

        metric.setup(transform, static, moving, spacing, static_aff,moving_aff,
                     smask, mmask, prealign=None, precondition=False)

        # Compute the gradient with the implementation under test
        val0, actual = metric.value_and_gradient(theta)
        # Compute the gradient using finite-diferences
        n = transform.get_number_of_parameters()
        expected = np.empty_like(actual)
        for i in range(n):
            dtheta = theta.copy()
            dtheta[i] += h
            val1 = metric.distance(dtheta)
            expected[i] = (val1 - val0) / h

        dp = expected.dot(actual)
        enorm = np.linalg.norm(expected)
        anorm = np.linalg.norm(actual)
        nprod = dp / (enorm * anorm)
        assert_equal(nprod >= 0.999, True)


def test_mattes_mi_registration_2d():
    factors = {('TRANSLATION', 2):25.0,
               ('ROTATION', 2):0.35,
               ('RIGID', 2):0.15,
               ('SCALING', 2):0.3,
               ('AFFINE', 2):0.2}
    for ttype in factors.keys():
        factor = factors[ttype]
        transform = regtransforms[ttype]
        static, moving, static_aff, moving_aff, smask, mmask, T = \
                        setup_random_transform_2d(transform, factor)
        start_sad = np.abs(static - moving).sum().sum()

        # In case of failure, it is useful to see the overlaid input images
        #rt.overlay_images(static, moving)

        metric = imaffine.MattesMIMetric(32)
        affreg = imaffine.AffineRegistration(metric, 'BFGS',
                                             [10000, 111110, 11110], 1e-5, 1.0,
                                             [4, 2, 1],[3, 1, 0],
                                             options=None)
        x0 = None
        sol = affreg.optimize(static, moving, transform, x0, static_aff,
                              moving_aff, smask, mmask)
        warped = aff_warp(static, static_aff, moving, moving_aff, sol)
        end_sad = np.abs(static - warped).sum().sum()

        # In case of failure, it is useful to see the overlaid resulting images
        #rt.overlay_images(static, warped)

        reduction = 1 - end_sad / start_sad
        print("%s>>%f"%(transform, reduction))
        assert_equal(reduction > 0.99, True)


if __name__ == "__main__":
    test_aff_centers_of_mass_3d()
    test_aff_geometric_centers_3d()
    test_aff_origins_3d()
    test_mattes_mi_registration_2d()
