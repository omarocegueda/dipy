from __future__ import print_function
import numpy as np
from numpy.testing import (assert_equal,
                           assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises)
import matplotlib.pyplot as plt
import dipy.align.imwarp as imwarp
import dipy.align.metrics as metrics
import dipy.align.vector_fields as vfu
from dipy.data import get_data
from dipy.align import floating
import nibabel.eulerangles as eulerangles
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align import VerbosityLevels

def test_mult_aff():
    r"""mult_aff from imwarp returns the matrix product A.dot(B) considering None
    as the identity
    """
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[2.0, 0.0], [0.0, 2.0]])

    C = imwarp.mult_aff(A, B)
    expected_mult = np.array([[2.0, 4.0], [6.0, 8.0]])
    assert_array_almost_equal(C, expected_mult)

    C = imwarp.mult_aff(A, None)
    assert_array_almost_equal(C, A)

    C = imwarp.mult_aff(None, B)
    assert_array_almost_equal(C, B)

    C = imwarp.mult_aff(None, None)
    assert_equal(C, None)


def test_diffeomorphic_map_2d():
    r"""
    Creates a random displacement field that exactly maps pixels from an input
    image to an output image. First a discrete random assignment between the
    images is generated, then each pair of mapped points are transformed to
    the physical space by assigning a pair of arbitrary, fixed affine matrices
    to input and output images, and finaly the difference between their positions
    is taken as the displacement vector. The resulting displacement, although
    operating in physical space, maps the points exactly (up to numerical
    precision).
    """
    np.random.seed(2022966)
    domain_shape = (10, 10)
    codomain_shape = (10, 10)
    #create a simple affine transformation
    nr = domain_shape[0]
    nc = domain_shape[1]
    s = 1.1
    t = 0.25
    trans = np.array([[1, 0, -t*nr],
                      [0, 1, -t*nc],
                      [0, 0, 1]])
    trans_inv = np.linalg.inv(trans)
    scale = np.array([[1*s, 0, 0],
                      [0, 1*s, 0],
                      [0, 0, 1]])
    gt_affine = trans_inv.dot(scale.dot(trans))

    #create the random displacement field
    domain_affine = gt_affine
    codomain_affine = gt_affine
    disp, assign = vfu.create_random_displacement_2d(np.array(domain_shape, dtype=np.int32),
                                                     domain_affine,
                                                     np.array(codomain_shape, dtype=np.int32),
                                                     codomain_affine)
    disp = np.array(disp, dtype=floating)
    assign = np.array(assign)
    #create a random image (with decimal digits) to warp
    moving_image = np.ndarray(codomain_shape, dtype=floating)
    moving_image[...] = np.random.randint(0, 10, np.size(moving_image)).reshape(tuple(codomain_shape))
    #set boundary values to zero so we don't test wrong interpolation due to floating point precision
    moving_image[0,:] = 0
    moving_image[-1,:] = 0
    moving_image[:,0] = 0
    moving_image[:,-1] = 0

    #warp the moving image using the (exact) assignments
    expected = moving_image[(assign[...,0], assign[...,1])]

    #warp using a DiffeomorphicMap instance
    diff_map = imwarp.DiffeomorphicMap(2, domain_shape, domain_affine,
                                          domain_shape, domain_affine,
                                          codomain_shape, codomain_affine,
                                          None)
    diff_map.forward = disp

    #Verify that the transform method accepts different image types (note that
    #the actual image contained integer values, we don't want to test rounding)
    for type in [floating, np.float64, np.int64, np.int32]:
        moving_image = moving_image.astype(type)

        #warp using linear interpolation
        warped = diff_map.transform(moving_image, 'linear')
        #compare the images (the linear interpolation may introduce slight precision errors)
        assert_array_almost_equal(warped, expected, decimal=5)

        #Now test the nearest neighbor interpolation
        warped = diff_map.transform(moving_image, 'nearest')
        #compare the images (now we dont have to worry about precision, it is n.n.)
        assert_array_almost_equal(warped, expected)

        #verify the is_inverse flag
        inv = diff_map.inverse()
        warped = inv.transform_inverse(moving_image, 'linear')
        assert_array_almost_equal(warped, expected, decimal=5)

        warped = inv.transform_inverse(moving_image, 'nearest')
        assert_array_almost_equal(warped, expected)

    #Now test the inverse functionality
    diff_map = imwarp.DiffeomorphicMap(2, codomain_shape, codomain_affine,
                                          codomain_shape, codomain_affine,
                                          domain_shape, domain_affine, None)
    diff_map.backward = disp
    for type in [floating, np.float64, np.int64, np.int32]:
        moving_image = moving_image.astype(type)

        #warp using linear interpolation
        warped = diff_map.transform_inverse(moving_image, 'linear')
        #compare the images (the linear interpolation may introduce slight precision errors)
        assert_array_almost_equal(warped, expected, decimal=5)

        #Now test the nearest neighbor interpolation
        warped = diff_map.transform_inverse(moving_image, 'nearest')
        #compare the images (now we dont have to worry about precision, it is n.n.)
        assert_array_almost_equal(warped, expected)

    #Verify that DiffeomorphicMap raises the appropriate exceptions when
    #the sampling information is undefined
    diff_map = imwarp.DiffeomorphicMap(2, domain_shape, domain_affine,
                                          domain_shape, domain_affine,
                                          codomain_shape, codomain_affine,
                                          None)
    diff_map.forward = disp
    diff_map.domain_shape = None
    #If we don't provide the sampling info, it should try to use the map's info, but it's None...
    assert_raises(ValueError, diff_map.transform, moving_image, 'linear')

    #Same test for diff_map.transform_inverse
    diff_map = imwarp.DiffeomorphicMap(2, domain_shape, domain_affine,
                                          domain_shape, domain_affine,
                                          codomain_shape, codomain_affine,
                                          None)
    diff_map.forward = disp
    diff_map.codomain_shape = None
    #If we don't provide the sampling info, it should try to use the map's info, but it's None...
    assert_raises(ValueError, diff_map.transform_inverse, moving_image, 'linear')

    #We must provide, at least, the reference grid shape
    assert_raises(ValueError, imwarp.DiffeomorphicMap, 2, None)


def test_diffeomorphic_map_simplification_2d():
    r"""
    Create an invertible deformation field, and define a DiffeomorphicMap
    using different voxel-to-space transforms for domain, codomain, and
    reference discretizations, also use a non-identity pre-aligning matrix.
    Warp a circle using the diffeomorphic map to obtain the expected warped
    circle. Now simplify the DiffeomorphicMap and warp the same circle using
    this simplified map. Verify that the two warped circles are equal up to
    numerical precision.
    """
    #create a simple affine transformation
    domain_shape = (64, 64)
    codomain_shape = (80, 80)
    nr = domain_shape[0]
    nc = domain_shape[1]
    s = 1.1
    t = 0.25
    trans = np.array([[1, 0, -t*nr],
                      [0, 1, -t*nc],
                      [0, 0, 1]])
    trans_inv = np.linalg.inv(trans)
    scale = np.array([[1*s, 0, 0],
                      [0, 1*s, 0],
                      [0, 0, 1]])
    gt_affine = trans_inv.dot(scale.dot(trans))
    # Create the invertible displacement fields and the circle
    radius = 16
    circle = vfu.create_circle(codomain_shape[0], codomain_shape[1], radius)
    d, dinv = vfu.create_harmonic_fields_2d(domain_shape[0], domain_shape[1], 0.3, 6)
    #Define different voxel-to-space transforms for domain, codomain and reference grid,
    #also, use a non-identity pre-align transform
    D = gt_affine
    C = imwarp.mult_aff(gt_affine, gt_affine)
    R = np.eye(3)
    P = gt_affine

    #Create the original diffeomorphic map
    diff_map = imwarp.DiffeomorphicMap(2, domain_shape, R,
                                          domain_shape, D,
                                          codomain_shape, C,
                                          P)
    diff_map.forward = np.array(d, dtype = floating)
    diff_map.backward = np.array(dinv, dtype = floating)
    #Warp the circle to obtain the expected image
    expected = diff_map.transform(circle, 'linear')

    #Simplify
    simplified = diff_map.get_simplified_transform()
    #warp the circle
    warped = simplified.transform(circle, 'linear')
    #verify that the simplified map is equivalent to the
    #original one
    assert_array_almost_equal(warped, expected)
    #And of course, it must be simpler...
    assert_equal(simplified.domain_affine, None)
    assert_equal(simplified.codomain_affine, None)
    assert_equal(simplified.disp_affine, None)
    assert_equal(simplified.domain_affine_inv, None)
    assert_equal(simplified.codomain_affine_inv, None)
    assert_equal(simplified.disp_affine_inv, None)


def test_diffeomorphic_map_simplification_3d():
    r"""
    Create an invertible deformation field, and define a DiffeomorphicMap
    using different voxel-to-space transforms for domain, codomain, and
    reference discretizations, also use a non-identity pre-aligning matrix.
    Warp a sphere using the diffeomorphic map to obtain the expected warped
    sphere. Now simplify the DiffeomorphicMap and warp the same sphere using
    this simplified map. Verify that the two warped spheres are equal up to
    numerical precision.
    """
    #create a simple affine transformation
    domain_shape = (64, 64, 64)
    codomain_shape = (80, 80, 80)
    nr = domain_shape[0]
    nc = domain_shape[1]
    ns = domain_shape[2]
    s = 1.1
    t = 0.25
    trans = np.array([[1, 0, 0, -t*ns],
                      [0, 1, 0, -t*nr],
                      [0, 0, 1, -t*nc],
                      [0, 0, 0, 1]])
    trans_inv = np.linalg.inv(trans)
    scale = np.array([[1*s, 0, 0, 0],
                      [0, 1*s, 0, 0],
                      [0, 0, 1*s, 0],
                      [0, 0, 0, 1]])
    gt_affine = trans_inv.dot(scale.dot(trans))
    # Create the invertible displacement fields and the sphere
    radius = 16
    sphere = vfu.create_sphere(codomain_shape[0], codomain_shape[1], codomain_shape[2], radius)
    d, dinv = vfu.create_harmonic_fields_3d(domain_shape[0], domain_shape[1], domain_shape[2], 0.3, 6)
    #Define different voxel-to-space transforms for domain, codomain and reference grid,
    #also, use a non-identity pre-align transform
    D = gt_affine
    C = imwarp.mult_aff(gt_affine, gt_affine)
    R = np.eye(4)
    P = gt_affine

    #Create the original diffeomorphic map
    diff_map = imwarp.DiffeomorphicMap(3, domain_shape, R,
                                          domain_shape, D,
                                          codomain_shape, C,
                                          P)
    diff_map.forward = np.array(d, dtype = floating)
    diff_map.backward = np.array(dinv, dtype = floating)
    #Warp the sphere to obtain the expected image
    expected = diff_map.transform(sphere, 'linear')

    #Simplify
    simplified = diff_map.get_simplified_transform()
    #warp the sphere
    warped = simplified.transform(sphere, 'linear')
    #verify that the simplified map is equivalent to the
    #original one
    assert_array_almost_equal(warped, expected)
    #And of course, it must be simpler...
    assert_equal(simplified.domain_affine, None)
    assert_equal(simplified.codomain_affine, None)
    assert_equal(simplified.disp_affine, None)
    assert_equal(simplified.domain_affine_inv, None)
    assert_equal(simplified.codomain_affine_inv, None)
    assert_equal(simplified.disp_affine_inv, None)

def test_optimizer_exceptions():
    #An arbitrary valid metric
    metric = metrics.SSDMetric(2)
    # The metric must not be None
    assert_raises(ValueError, imwarp.SymmetricDiffeomorphicRegistration, None)
    # The iterations list must not be empty
    assert_raises(ValueError, imwarp.SymmetricDiffeomorphicRegistration, metric, [])

    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, None)
    #Verify the default iterations list
    assert_array_equal(optimizer.level_iters, [100,100,25])

    #Verify exception thrown when attepting to fit the energy profile without enough data
    assert_raises(ValueError, optimizer._get_energy_derivative)


def test_scale_space_exceptions():
    np.random.seed(2022966)

    target_shape = (32, 32)
    #create a random image
    image = np.ndarray(target_shape, dtype=floating)
    image[...] = np.random.randint(0, 10, np.size(image)).reshape(tuple(target_shape))
    zeros = (image == 0).astype(np.int32)

    ss = imwarp.ScaleSpace(image,3)

    for invalid_level in [-1, 3, 4]:
        assert_raises(ValueError, ss.get_image, invalid_level)

    # Verify that the mask is correctly applied, when requested
    ss = imwarp.ScaleSpace(image,3, mask0=True)
    for level in range(3):
        img = ss.get_image(level)
        z = (img == 0).astype(np.int32)
        assert_array_equal(zeros, z)


def test_get_direction_and_spacings():
    xrot = 0.5
    yrot = 0.75
    zrot = 1.0
    direction_gt = eulerangles.euler2mat(zrot, yrot, xrot)
    spacings_gt = np.array([1.1, 1.2, 1.3])
    scaling_gt = np.diag(spacings_gt)
    translation_gt = np.array([1,2,3])

    affine = np.eye(4)
    affine[:3, :3] = direction_gt.dot(scaling_gt)
    affine[:3, 3] = translation_gt

    direction, spacings = imwarp.get_direction_and_spacings(affine, 3)
    assert_array_almost_equal(direction, direction_gt)
    assert_array_almost_equal(spacings, spacings_gt)

def simple_callback(sdr, status):
    if status == imwarp.RegistrationStages.INIT_START:
        sdr.INIT_START_CALLED = 1
    if status == imwarp.RegistrationStages.INIT_END:
        sdr.INIT_END_CALLED = 1
    if status == imwarp.RegistrationStages.OPT_START:
        sdr.OPT_START_CALLED = 1
    if status == imwarp.RegistrationStages.OPT_END:
        sdr.OPT_END_CALLED = 1
    if status == imwarp.RegistrationStages.SCALE_START:
        sdr.SCALE_START_CALLED = 1
    if status == imwarp.RegistrationStages.SCALE_END:
        sdr.SCALE_END_CALLED = 1
    if status == imwarp.RegistrationStages.ITER_START:
        sdr.ITER_START_CALLED = 1
    if status == imwarp.RegistrationStages.ITER_END:
        sdr.ITER_END_CALLED = 1


def test_ssd_2d_demons():
    r'''
    Classical Circle-To-C experiment for 2D Monomodal registration. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of SSD in
    2D using the Demons step, and this test checks that the current energy
    profile matches the saved one.
    '''
    fname_moving = get_data('reg_o')
    fname_static = get_data('reg_c')

    moving = plt.imread(fname_moving)
    static = plt.imread(fname_static)
    moving = np.array(moving, dtype=floating)
    static = np.array(static, dtype=floating)
    moving = (moving-moving.min())/(moving.max() - moving.min())
    static = (static-static.min())/(static.max() - static.min())
    #Create the SSD metric
    smooth = 4
    step_type = 'demons'
    similarity_metric = metrics.SSDMetric(2, smooth=smooth, step_type=step_type)

    #Configure and run the Optimizer
    level_iters = [200, 100, 50, 25]
    step_length = 0.25
    opt_tol = 1e-4
    inv_iter = 40
    inv_tol = 1e-3
    ss_sigma_factor = 0.2
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric,
        level_iters, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)

    #test callback being called
    optimizer.INIT_START_CALLED = 0
    optimizer.INIT_END_CALLED = 0
    optimizer.OPT_START_CALLED = 0
    optimizer.OPT_END_CALLED = 0
    optimizer.SCALE_START_CALLED = 0
    optimizer.SCALE_END_CALLED = 0
    optimizer.ITER_START_CALLED = 0
    optimizer.ITER_END_CALLED = 0

    optimizer.callback_counter_test = 0
    optimizer.callback = simple_callback

    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)

    subsampled_energy_profile = np.array(optimizer.full_energy_profile[::10])
    if floating is np.float32:
        expected_profile = \
            np.array([312.6813333, 162.36237287, 99.3092602, 77.58768946,
                      61.79626624, 55.39742359, 46.38704372, 41.67221537,
                      36.28463836, 32.27768916, 31.73040702, 66.06495496,
                      24.62779771, 16.13855133, 13.62532387, 110.96794829,
                      37.60848963, 35.84714255, 108.62943537, 88.9151439 ])
    else:
        expected_profile = \
            np.array([312.68133361, 162.36227997, 99.30927424, 77.58751953,
                      61.79614527, 55.39739106, 46.38685359, 41.6719058,
                      36.29200056, 32.64887131, 31.07222716, 78.05180223,
                      28.80263071, 16.53375467, 13.51399519, 12.12131417,
                      54.46296899, 37.05664608, 172.55394639, 91.88413331,
                      82.77240577])

    assert_array_almost_equal(subsampled_energy_profile, expected_profile)
    assert_equal(optimizer.OPT_START_CALLED, 1)
    assert_equal(optimizer.OPT_END_CALLED, 1)
    assert_equal(optimizer.SCALE_START_CALLED, 1)
    assert_equal(optimizer.SCALE_END_CALLED, 1)
    assert_equal(optimizer.ITER_START_CALLED, 1)
    assert_equal(optimizer.ITER_END_CALLED, 1)



def test_ssd_2d_gauss_newton():
    r'''
    Classical Circle-To-C experiment for 2D Monomodal registration. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of SSD in
    2D using the Gauss Newton step, and this test checks that the current energy
    profile matches the saved one.
    '''
    fname_moving = get_data('reg_o')
    fname_static = get_data('reg_c')

    moving = plt.imread(fname_moving)
    static = plt.imread(fname_static)
    moving = np.array(moving, dtype=floating)
    static = np.array(static, dtype=floating)
    moving = (moving-moving.min())/(moving.max() - moving.min())
    static = (static-static.min())/(static.max() - static.min())
    #Create the SSD metric
    smooth = 4
    inner_iter = 5
    step_type = 'gauss_newton'
    similarity_metric = metrics.SSDMetric(2, smooth, inner_iter, step_type)

    #Configure and run the Optimizer
    level_iters = [200, 100, 50, 25]
    step_length = 0.5
    opt_tol = 1e-4
    inv_iter = 40
    inv_tol = 1e-3
    ss_sigma_factor = 0.2
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric,
        level_iters, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)

    #test callback not being called
    optimizer.INIT_START_CALLED = 0
    optimizer.INIT_END_CALLED = 0
    optimizer.OPT_START_CALLED = 0
    optimizer.OPT_END_CALLED = 0
    optimizer.SCALE_START_CALLED = 0
    optimizer.SCALE_END_CALLED = 0
    optimizer.ITER_START_CALLED = 0
    optimizer.ITER_END_CALLED = 0

    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, np.eye(3), np.eye(3), np.eye(3))
    m = optimizer.get_map()
    assert_equal(mapping, m)
    subsampled_energy_profile = np.array(optimizer.full_energy_profile[::10])
    if floating is np.float32:
        expected_profile = \
            np.array([312.68133316, 79.37098045, 23.21411977, 124.02404618,
                      60.13340398, 32.6870562, 25.25477472, 82.33576097,
                      97.95014357, 116.23146452, 106.72164233])
    else:
        expected_profile = \
            np.array([312.68133361, 79.370983, 23.26112647, 22.36798589,
                      53.75336336, 40.90032523, 38.83952825, 72.15134797,
                      131.28842158, 144.41662653, 129.45806628])
    assert_array_almost_equal(subsampled_energy_profile, expected_profile)
    assert_equal(optimizer.OPT_START_CALLED, 0)
    assert_equal(optimizer.OPT_END_CALLED, 0)
    assert_equal(optimizer.SCALE_START_CALLED, 0)
    assert_equal(optimizer.SCALE_END_CALLED, 0)
    assert_equal(optimizer.ITER_START_CALLED, 0)
    assert_equal(optimizer.ITER_END_CALLED, 0)


def get_synthetic_warped_circle(nslices):
    #get a subsampled circle
    fname_cicle = get_data('reg_o')
    circle = plt.imread(fname_cicle)[::4,::4].astype(floating)

    #create a synthetic invertible map and warp the circle
    d, dinv = vfu.create_harmonic_fields_2d(64, 64, 0.1, 4)
    d = np.asarray(d, dtype=floating)
    dinv = np.asarray(dinv, dtype=floating)
    mapping = DiffeomorphicMap(2, (64, 64))
    mapping.forward, mapping.backward = d, dinv
    wcircle = mapping.transform(circle)

    if(nslices == 1):
        return circle, wcircle

    #normalize and form the 3d by piling slices
    circle = (circle-circle.min())/(circle.max() - circle.min())
    circle_3d = np.ndarray(circle.shape + (nslices,), dtype=floating)
    circle_3d[...] = circle[...,None]
    circle_3d[...,0] = 0
    circle_3d[...,-1] = 0

    #do the same with the warped circle
    wcircle = (wcircle-wcircle.min())/(wcircle.max() - wcircle.min())
    wcircle_3d = np.ndarray(wcircle.shape + (nslices,), dtype=floating)
    wcircle_3d[...] = wcircle[...,None]
    wcircle_3d[...,0] = 0
    wcircle_3d[...,-1] = 0

    return circle_3d, wcircle_3d


def test_ssd_3d_demons():
    r'''
    Register a stack of circles ('cylinder') before and after warping them with
    a synthetic diffeomorphism. This test is intended to detect regressions
    only: we saved the energy profile (the sequence of energy values at each
    iteration) of a working version of SSD in 3D using the Demons step, and this
    test checks that the current energy profile matches the saved one. The
    validation of the "working version" was done by registering the 18 manually
    annotated T1 brain MRI database IBSR with each other and computing the
    jaccard index for all 31 common anatomical regions.
    '''
    moving, static = get_synthetic_warped_circle(30)
    moving[...,:8] = 0
    moving[...,-1:-9:-1] = 0
    static[...,:8] = 0
    static[...,-1:-9:-1] = 0

    #Create the SSD metric
    smooth = 4
    step_type = 'demons'
    similarity_metric = metrics.SSDMetric(3, smooth=smooth, step_type=step_type)

    #Create the optimizer
    level_iters = [10, 5]
    step_length = 0.25
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric,
        level_iters, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)
    energy_profile = np.array(optimizer.full_energy_profile)
    if floating is np.float32:
        expected_profile = \
            np.array([312.22706987, 154.65556885, 53.88455398, 9.11770682,
                      36.48642824, 13.21706748, 48.67710635, 14.91782047,
                      49.84142899, 14.92531294, 543.47869573, 172.30622181,
                      164.23837284, 147.15846991, 157.57208267])
    else:
        expected_profile = \
            np.array([312.22709468, 154.65706498, 53.88424337, 8.79304783,
                      34.90097908, 12.38605031, 48.62619406, 14.38621352,
                      50.72048699, 14.2310842, 561.39521267, 169.39665598,
                      163.28538697, 146.03958517, 157.36024629])
    assert_array_almost_equal(energy_profile, expected_profile, decimal=6)


def test_ssd_3d_gauss_newton():
    r'''
    Register a stack of circles ('cylinder') before and after warping them with
    a synthetic diffeomorphism. This test is intended to detect regressions
    only: we saved the energy profile (the sequence of energy values at each
    iteration) of a working version of SSD in 3D using the Gauss-Newton step,
    and this test checks that the current energy profile matches the saved
    one. The validation of the "working version" was
    done by registering the 18 manually annotated T1 brain MRI database IBSR
    with each other and computing the jaccard index for all 31 common anatomical
    regions.
    '''
    moving, static = get_synthetic_warped_circle(35)
    moving[...,:10] = 0
    moving[...,-1:-11:-1] = 0
    static[...,:10] = 0
    static[...,-1:-11:-1] = 0

    #Create the SSD metric
    smooth = 4
    inner_iter = 5
    step_type = 'gauss_newton'
    similarity_metric = metrics.SSDMetric(3, smooth, inner_iter, step_type)

    #Create the optimizer
    level_iters = [10, 5]
    step_length = 0.25
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric,
        level_iters, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)
    energy_profile = np.array(optimizer.full_energy_profile)
    if floating is np.float32:
        expected_profile = \
            np.array([348.3204721, 143.48075646, 44.30003413, 8.73624841,
                      3.13227181, 14.70806845, 6.48360884, 23.52499421,
                      17.25667176, 48.997691, 261.31606232, 82.68180383,
                      207.9123664, 65.94775368, 217.83317551])
    else:
        expected_profile = \
            np.array([348.32049917, 143.48075242, 44.30002695, 8.73624595,
                      3.13227079, 14.70800999, 6.4835393, 23.52449876,
                      17.25637125, 48.99735377, 261.3155413, 82.68163179,
                      207.91199923, 65.94684288, 217.83366983])
    assert_array_almost_equal(energy_profile, expected_profile, decimal=6)


def test_cc_2d():
    r'''
    Register a circle to itself after warping it under a synthetic invertible
    map. This test is intended to detect regressions only: we saved the energy
    profile (the sequence of energy values at each iteration) of a working
    version of CC in 2D, and this test checks that the current energy profile
    matches the saved one.
    '''

    moving, static = get_synthetic_warped_circle(1)
    #Configure the metric
    sigma_diff = 3.0
    radius = 4
    metric = metrics.CCMetric(2, sigma_diff, radius)

    #Configure and run the Optimizer
    level_iters = [10, 5]
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, level_iters)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)
    energy_profile = np.array(optimizer.full_energy_profile)
    if floating is np.float32:
        expected_profile = \
            [-681.02276236, -920.57714783, -1008.82241171, -1021.91021701,
             -994.86961164, -1026.52978164, -1015.83587405, -1020.02780802,
             -993.8576053, -1026.4369566, -2708.69637013, -2818.69689899,
             -2819.0057186, -2828.96458529, -2838.98236872]
    else:
        expected_profile = \
            [-685.02275452, -928.57719958, -1020.82238769, -1029.40493009,
             -1007.20253961, -1007.54244118, -1017.23536561, -997.79933896,
             -1021.66992244, -993.12855571, -2782.15634529, -2818.14101957,
             -2792.39799167, -2820.35851663, -2805.37854206]
    expected_profile = np.asarray(expected_profile)
    assert_array_almost_equal(energy_profile, expected_profile)


def test_cc_3d():
    r'''
    Register a stack of circles ('cylinder') before and after warping them with
    a synthetic diffeomorphism. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of CC in
    3D, and this test checks that the current energy profile matches the saved
    one. The validation of the "working version" was
    done by registering the 18 manually annotated T1 brain MRI database IBSR
    with each other and computing the jaccard index for all 31 common anatomical
    regions. The "working version" of CC in 3D obtains very similar results as
    those reported for ANTS on the same database with the same number of
    iterations. Any modification that produces a change in the energy profile
    should be carefully validated to ensure no accuracy loss.
    '''
    moving, static = moving, static = get_synthetic_warped_circle(20)

    #Create the CC metric
    sigma_diff = 2.0
    radius = 4
    similarity_metric = metrics.CCMetric(3, sigma_diff, radius)

    #Create the optimizer
    level_iters = [20, 10]
    step_length = 0.25
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric,
        level_iters, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    optimizer.verbosity = VerbosityLevels.DEBUG

    mapping = optimizer.optimize(static, moving, None, None, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)
    energy_profile = np.array(optimizer.full_energy_profile)*1e-4
    if floating is np.float32:
        expected_profile = \
            [-0.17136006, -0.19243038, -0.20632291, -0.20896195, -0.2038927,
             -0.20904329, -0.20688352, -0.20913435, -0.20821154, -0.20830725,
             -0.20909298, -0.20826442, -0.20872891, -0.20834017, -0.20933514,
             -2.98555799, -3.06861497, -3.08087159, -3.07851062, -3.08394814,
             -3.09589079, -3.10001981, -3.10085246, -3.1014803, -3.10175915]
    else:
        expected_profile = \
            [-0.17416006, -0.19343038, -0.20672292, -0.20976197, -0.20709272,
             -0.2118433, -0.21008353, -0.21213436, -0.21121155, -0.21190727,
             -0.21149299, -0.21186443, -0.21152893, -0.21194018, -0.21193516,
             -0.21254031, -3.04824214, -3.20586669, -3.31143326, -3.38055358,
             -3.42858846, -3.43867684, -3.43396941, -3.43370172, -3.45179141,
             -3.43580858]
    expected_profile = np.asarray(expected_profile)
    assert_array_almost_equal(energy_profile, expected_profile, decimal=6)


def test_em_3d_gauss_newton():
    r'''
    Register a stack of circles ('cylinder') before and after warping them with
    a synthetic diffeomorphism. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of EM in
    3D, and this test checks that the current energy profile matches the saved
    one. The validation of the "working version" was
    done by registering the 18 manually annotated T1 brain MRI database IBSR
    with each other and computing the jaccard index for all 31 common anatomical
    regions. The "working version" of EM in 3D obtains very similar results as
    those reported for ANTS on the same database. Any modification that produces
    a change in the energy profile should be carefully validated to ensure no
    accuracy loss.
    '''
    moving, static = get_synthetic_warped_circle(30)
    moving[...,:8] = 0
    moving[...,-1:-9:-1] = 0
    static[...,:8] = 0
    static[...,-1:-9:-1] = 0

    #Create the EM metric
    smooth=25.0
    inner_iter=20
    step_length=0.25
    q_levels=256
    double_gradient=True
    iter_type='gauss_newton'
    similarity_metric = metrics.EMMetric(
        3, smooth, inner_iter, q_levels, double_gradient, iter_type)

    #Create the optimizer
    level_iters = [10, 5]
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric,
        level_iters, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)
    energy_profile = np.array(optimizer.full_energy_profile)
    if floating is np.float32:
        expected_profile = \
            np.array([144.03694724, 63.06874148, 51.84694881, 39.63740417,
                      31.84981481, 44.37788414, 37.84961844, 38.00509881,
                      38.67423954, 38.47003339, 1645.65999126, 1440.27446829,
                      1199.90976637, 1065.18430878, 980.32387957])
    else:
        expected_profile = \
            np.array([144.03695787, 63.06869345, 51.84692311, 39.63740853,
                      31.849781, 44.36773439, 38.05328459, 36.77452848,
                      38.61107635, 39.90232685, 1673.14727089, 1455.03440222,
                      1249.30254166, 1107.48675055, 997.67646155])
    assert_array_almost_equal(energy_profile, expected_profile, decimal=6)


def test_em_2d_gauss_newton():
    r'''
    Register a circle to itself after warping it under a synthetic invertible
    map. This test is intended to detect regressions only: we saved the energy
    profile (the sequence of energy values at each iteration) of a working
    version of EM in 2D, and this test checks that the current energy profile
    matches the saved one.
    '''

    moving, static = get_synthetic_warped_circle(1)

    #Configure the metric
    smooth=25.0
    inner_iter=20
    q_levels=256
    double_gradient=False
    iter_type='gauss_newton'
    metric = metrics.EMMetric(
        2, smooth, inner_iter, q_levels, double_gradient, iter_type)

    #Configure and run the Optimizer
    level_iters = [40, 20, 10]
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, level_iters)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)
    energy_profile = np.array(optimizer.full_energy_profile)[::4]
    if floating is np.float32:
        expected_profile = \
            [2.50773392, 1.21385918, 0.26439402, 0.3577335, 0.09560039,
             0.11573476, 3.4162687, 1.47890964, 1.15214868, 0.93098839,
             49.37186785, 42.18746867, 44.56830102]
    else:
        expected_profile = \
            [2.50773436, 1.21386361, 0.26444436, 0.40518006, 0.25217539,
             0.68813006, 2.65249834, 2.65081422, 1.64096636, 1.93552956,
             60.13197926, 57.73984356, 56.56663992]
    assert_array_almost_equal(energy_profile, np.array(expected_profile))


def test_em_3d_demons():
    r'''
    Register a stack of circles ('cylinder') before and after warping them with
    a synthetic diffeomorphism. This test
    is intended to detect regressions only: we saved the energy profile (the
    sequence of energy values at each iteration) of a working version of EM in
    3D, and this test checks that the current energy profile matches the saved
    one. The validation of the "working version" was
    done by registering the 18 manually annotated T1 brain MRI database IBSR
    with each other and computing the jaccard index for all 31 common anatomical
    regions. The "working version" of EM in 3D obtains very similar results as
    those reported for ANTS on the same database. Any modification that produces
    a change in the energy profile should be carefully validated to ensure no
    accuracy loss.
    '''
    moving, static = get_synthetic_warped_circle(30)
    moving[...,:8] = 0
    moving[...,-1:-9:-1] = 0
    static[...,:8] = 0
    static[...,-1:-9:-1] = 0

    #Create the EM metric
    smooth=25.0
    inner_iter=20
    step_length=0.25
    q_levels=256
    double_gradient=True
    iter_type='demons'
    similarity_metric = metrics.EMMetric(
        3, smooth, inner_iter, q_levels, double_gradient, iter_type)

    #Create the optimizer
    level_iters = [10, 5]
    opt_tol = 1e-4
    inv_iter = 20
    inv_tol = 1e-3
    ss_sigma_factor = 0.5
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(similarity_metric,
        level_iters, step_length, ss_sigma_factor, opt_tol, inv_iter, inv_tol)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)
    energy_profile = np.array(optimizer.full_energy_profile)
    if floating is np.float32:
        expected_profile = \
            np.array([144.03694708, 122.39512227, 111.31924572, 90.91010482,
                      93.93707059, 104.22996918, 110.57822649, 140.45298465,
                      133.87831302, 119.20826433, 3873.76524215, 4003.54252593,
                      3874.82104759, 4166.40172048, 3454.62144793])
    else:
        expected_profile = \
            np.array([144.03695787, 121.73905118, 108.00194514, 86.8358979,
                      104.96530524, 109.84548772, 108.01483915, 111.65458765,
                      142.7822616, 113.71345843, 3292.21066768, 3658.30948342,
                      3473.65873292, 3587.70530204, 3496.2063634])
    assert_array_almost_equal(energy_profile, expected_profile, decimal=6)


def test_em_2d_demons():
    r'''
    Register a circle to itself after warping it under a synthetic invertible
    map. This test is intended to detect regressions only: we saved the energy
    profile (the sequence of energy values at each iteration) of a working
    version of EM in 2D, and this test checks that the current energy profile
    matches the saved one.
    '''

    moving, static = get_synthetic_warped_circle(1)

    #Configure the metric
    smooth=25.0
    inner_iter=20
    q_levels=256
    double_gradient=False
    iter_type='demons'
    metric = metrics.EMMetric(
        2, smooth, inner_iter, q_levels, double_gradient, iter_type)

    #Configure and run the Optimizer
    level_iters = [40, 20, 10]
    optimizer = imwarp.SymmetricDiffeomorphicRegistration(metric, level_iters)
    optimizer.verbosity = VerbosityLevels.DEBUG
    mapping = optimizer.optimize(static, moving, None)
    m = optimizer.get_map()
    assert_equal(mapping, m)
    energy_profile = np.array(optimizer.full_energy_profile)[::2]
    if floating is np.float32:
        expected_profile = \
            [2.50773393, 5.65549561, 3.26942352, 3.26011675, 1.8168445,
             2.93135032, 5.44879264, 4.95081391, 40.01956373, 28.96091049,
             31.65616398, 26.28163996, 32.43115903, 35.32112344, 35.24130742,
             214.78815439, 192.89072697, 205.73092183, 195.456909, 211.96622016]
    else:
        expected_profile = \
            [2.50773436, 5.65549788, 3.26942361, 3.2601172, 1.81684552,
             2.9313531, 5.44879215, 4.95081721, 40.0195569, 28.96088091,
             31.8697309, 25.87929996, 28.00015449, 30.25857833, 31.59863726,
             209.46413311, 200.50693617, 207.44586051, 207.8159257,
             206.17505953]
    assert_array_almost_equal(energy_profile, np.array(expected_profile))

if __name__=='__main__':
    test_scale_space_exceptions()
    test_optimizer_exceptions()
    test_mult_aff()
    test_diffeomorphic_map_2d()
    test_diffeomorphic_map_simplification_2d()
    test_diffeomorphic_map_simplification_3d()
    test_get_direction_and_spacings()
    test_ssd_2d_demons()
    test_ssd_2d_gauss_newton()
    test_ssd_3d_demons()
    test_ssd_3d_gauss_newton()
    test_cc_2d()
    test_cc_3d()
    test_em_2d_gauss_newton()
    test_em_3d_gauss_newton()
    test_em_3d_demons()
    test_em_2d_demons()
