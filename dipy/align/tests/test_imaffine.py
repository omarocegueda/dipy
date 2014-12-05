import numpy as np
from dipy.align import floating
import dipy.align.vector_fields as vf
import imaffine
import dipy.core.geometry as geometry
from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           assert_almost_equal,
                           assert_equal)

def test_aff_centers_of_mass_3d():
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
            actual = imaffine.aff_centers_of_mass_3d(static, static_affine, moving, moving_affine)
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
                    actual = imaffine.aff_geometric_centers_3d(static, static_affine, moving, moving_affine)
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
                    actual = imaffine.aff_origins_3d(static, static_affine, moving, moving_affine)
                    assert_array_almost_equal(actual, expected)


def test_mattes():
    import nibabel as nib
    import mattes
    import matplotlib.pyplot as plt
    import imaffine
    i10_nib = nib.load("D:/opt/registration/runs/reference/IBSR_10_ana_strip.nii.gz")
    i15_nib = nib.load("D:/opt/registration/runs/target/IBSR_15_ana_strip.nii.gz")
    i10 = i10_nib.get_data().squeeze()
    i10 = np.array(i10, dtype=np.float64)
    i10_aff = i10_nib.get_affine()
    i15 = i15_nib.get_data().squeeze()
    i15 = np.array(i15, dtype=np.float64)
    i15_aff = i15_nib.get_affine()

    pdf = mattes.joint_pdf_dense_3d(i10, i15, 32)
    plt.figure()
    plt.imshow(pdf)

    # Register the centers of mass
    T = imaffine.aff_centers_of_mass_3d(i10, i10_aff, i15, i15_aff)
    w15 = imaffine.aff_warp(i10, i10_aff, i15, i15_aff, T).astype(np.float64)
    pdf_reg = mattes.joint_pdf_dense_3d(i10, w15, 32)
    plt.figure()
    plt.imshow(pdf_reg)

def test_cc_residuals():
    import dipy.align.mattes as mattes
    import experiments.registration.dataset_info as info
    import experiments.registration.semi_synthetic as ss
    import nibabel as nib
    import dipy.viz.regtools as rt

    I = np.array(range(10*15*20), dtype=np.float64).reshape(10, 15, 20)
    J = 2*I+5
    r = mattes.compute_cc_residuals(I, J, 3)
    r = np.array(r)
    
    t1_name = info.get_brainweb('t1','strip')
    t1_nib = nib.load(t1_name)
    t1 = t1_nib.get_data().squeeze()
    t1_n = t1.astype(np.float64)
    t1_n = (t1_n - t1_n.min())/(t1_n.max() - t1_n.min())
    
    t2_name = info.get_brainweb('t2','strip')
    t2_nib = nib.load(t2_name)
    t2 = t2_nib.get_data().squeeze()
    t2_n = t2.astype(np.float64)
    t2_n = (t2_n - t2_n.min())/(t2_n.max() - t2_n.min())
    
    residuals = mattes.compute_cc_residuals_noboundary(t1_n, t2_n, 4)
    residuals = np.array(residuals)
    
    means, vars = ss.get_mean_transfer(t1, t2)
    sst2 = means[t1]
    sst2 = (sst2 - sst2.min())/(sst2.max() - sst2.min())
    
    residuals_ss = mattes.compute_cc_residuals_noboundary(sst2, t2_n, 4)
    residuals_ss = np.array(residuals_ss)
    
    
    
    
    rr = mattes.compute_cc_residuals_noboundary(t1_n, t1_n, 4)
    rr = np.array(rr)
    
    
if __name__ == "__main__":
    test_aff_centers_of_mass_3d()
    test_aff_geometric_centers_3d()
    test_aff_origins_3d()

