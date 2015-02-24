from __future__ import print_function
import numpy as np
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises)
import matplotlib.pyplot as plt
import dipy.align.imwarp as imwarp
import dipy.align.metrics as metrics
import dipy.align.vector_fields as vfu
from dipy.data import get_data
from dipy.align import floating
import nibabel as nib
import nibabel.eulerangles as eulerangles
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align import VerbosityLevels
import dipy.viz.regtools as rt
import scipy as sp
import scipy
import scipy.sparse
import scipy.sparse.linalg
import dipy.correct.gradients as gr

def read_topup_field(fname):
    fname = 'results_fieldcoef.nii'
    coef_nib = nib.load(fname)
    coef = coef_nib.get_data()
    coef_shape = np.array(coef.shape)
    h = coef_nib.get_header()
    knot_spacings = np.round(np.array(h.get_zooms()) + 0.5).astype(np.int32)
    vox_size = np.array([h._structarr['intent_p'+str(i + 1)] for i in range(3)])
    vox_size = vox_size.astype(np.float64)
    vol_shape = coef_nib.get_affine()[:3,3].astype(np.int32)

    sfield = gr.SplineField(vol_shape, knot_spacings)
    sfield.copy_coefficients(coef.astype(np.float64))
    vol = sfield.get_volume()



def extend_volume(vol, margin):
    dims = np.array(vol.shape)
    dims += 2*margin
    new_vol = np.zeros(tuple(dims))
    new_vol[margin:-margin, margin:-margin, margin:-margin] = vol[...]
    return new_vol


up_nib = nib.load("b0_blipup.nii")
down_nib = nib.load("b0_blipdown.nii")
up = up_nib.get_data().squeeze().astype(np.float64)
down = down_nib.get_data().squeeze().astype(np.float64)


use_extend_volume = True
if use_extend_volume:
    up = extend_volume(up, 16)
    down = extend_volume(down, 16)

#up_dir, up_spacings = imwarp.get_direction_and_spacings(up_nib.get_affine(), 3)
#down_dir, down_spacings = imwarp.get_direction_and_spacings(down_nib.get_affine(), 3)

# ====Simplify sampling transforms====
up_dir, up_spacings = np.eye(4), np.ones(3)*1.8
down_dir, down_spacings = np.eye(4), np.ones(3)*1.8
up_affine = up_dir
up_affine[:3, :3] *= 1.8
up_affine_inv = np.linalg.inv(up_affine)
down_affine = down_dir
down_affine[:3, :3] *= 1.8
down_affine_inv = np.linalg.inv(down_affine)
# ====================================

levels = 4
ss_sigma_factor = 0.2

up_ss = imwarp.ScaleSpace(up, levels, up_affine, up_spacings, ss_sigma_factor, False)
down_ss = imwarp.ScaleSpace(down, levels, down_affine, down_spacings, ss_sigma_factor, False)

ref_shape = up_ss.get_domain_shape(levels-1)
ref_affine = up_ss.get_affine(levels-1)
ref_affine_inv = up_ss.get_affine_inv(levels-1)
b = np.zeros(shape=tuple(ref_shape), dtype=floating)
d_up = np.array([0, -1, 0], dtype=np.float64)
d_down = np.array([0, 1, 0], dtype=np.float64)





#one level
#for level in range(levels-1, levels-4, -1):
if True:
    level = levels-1
    current_up = up_ss.get_image(level)
    current_down = down_ss.get_image(level)
    dcurrent_up = gr.der_y(current_up)
    dcurrent_down = gr.der_y(current_down)
    print("Range up: %f %f"%(current_up.min(), current_up.max()))
    print("Range down: %f %f"%(current_down.min(), current_down.max()))
    print("Means:",current_up.mean(), current_down.mean())
    #current_up *= 10.0/current_up.mean()
    #current_down *= 10.0/current_down.mean()
    print("New range up: %f %f"%(current_up.min(), current_up.max()))
    print("New range down: %f %f"%(current_down.min(), current_down.max()))
    #current_up/=10000.0
    #current_down/=10000.0

    if level<levels-1:
        rt.plot_slices(b)
        new_shape = up_ss.get_domain_shape(level)
        factors = up_ss.get_expand_factors(level + 1, level)
        new_b = gr.resample_orfield(b, factors, new_shape)

        b = np.array(new_b)
        ref_shape = new_shape
        ref_affine = up_ss.get_affine(level)
        ref_affine_inv = up_ss.get_affine_inv(level)
        rt.plot_slices(b)

    # warp up
    S = ref_affine
    Rinv = ref_affine_inv
    Tinv = up_affine_inv

    affine_idx_in = Rinv.dot(S)
    affine_idx_out = Tinv.dot(S)
    affine_disp = Tinv

    # Warp down
    S = ref_affine
    Rinv = ref_affine_inv
    Tinv = down_affine_inv

    affine_idx_in = Rinv.dot(S)
    affine_idx_out = Tinv.dot(S)
    affine_disp = Tinv

    l1 = 0.01
    l2 = 0.01
    max_iter = 2000
    it = 0
    epsilon = 0.000001
    nrm = 1 + epsilon
    energy = None
    prev_energy=None

    test_holland_hessian = True
    if test_holland_hessian:
        w_up = gr.warp_with_orfield(current_up, b, d_up, affine_idx_in, affine_idx_out,
                                    affine_disp, ref_shape)
        w_down = gr.warp_with_orfield(current_down, b, d_down, affine_idx_in, affine_idx_out,
                                      affine_disp, ref_shape)
        w_up = np.array(w_up)
        w_down = np.array(w_down)

        dw_up = gr.warp_with_orfield(dcurrent_up, b, d_up, affine_idx_in, affine_idx_out,
                                     affine_disp, ref_shape)
        dw_down = gr.warp_with_orfield(dcurrent_down, b, d_down, affine_idx_in, affine_idx_out,
                                       affine_disp, ref_shape)
        db = np.array(gr.der_y(b))

        Jth, data, indices, indptr = gr.gauss_newton_system_holland(w_up, w_down, dw_up, dw_down, db, l1, l2)
        Jth = np.array(Jth)
        data = np.array(data)
        indices = np.array(indices)
        indptr = np.array(indptr)

        JtJ = sp.sparse.csr_matrix((data, indices, indptr), shape=(w_up.size, w_up.size))
        x = sp.sparse.linalg.spsolve(JtJ, -1.0*Jth)
        step = gr.wrap_scalar_field(x, np.array(w_up.shape, dtype=np.int32))
        step = np.array(step)
        b += step

        rt.overlay_slices(w_up*(1+db), w_down*(1-db), slice_type=2)

        #Jth_test, JtJ_test = gr.test_gauss_newton_holland(w_up, w_down, dw_up, dw_down, db, l1, l2)
        #dd = np.abs(JtJ - JtJ_test)
        #dd.max()


    while it<max_iter and nrm > epsilon:# and (prev_energy is None or prev_energy>=energy):
        w_up = gr.warp_with_orfield(current_up, b, d_up, affine_idx_in, affine_idx_out,
                                    affine_disp, ref_shape)
        w_down = gr.warp_with_orfield(current_down, b, d_down, affine_idx_in, affine_idx_out,
                                      affine_disp, ref_shape)
        w_up = np.array(w_up)
        w_down = np.array(w_down)

        dw_up = gr.warp_with_orfield(dcurrent_up, b, d_up, affine_idx_in, affine_idx_out,
                                     affine_disp, ref_shape)
        dw_down = gr.warp_with_orfield(dcurrent_down, b, d_down, affine_idx_in, affine_idx_out,
                                       affine_disp, ref_shape)
        db = np.array(gr.der_y(b))

        gh = np.array(gr.grad_holland(w_up, w_down, dw_up, dw_down, b, db, l1, l2))


        #w_up *= ( 1.0 - db)
        #w_down *= ( 1.0 + db)

        prev_energy = energy
        energy = np.sum((w_up - w_down)**2)
        #if prev_energy is not None:
        #    print("Decreased energy:%f", prev_energy - energy)

        nrm = np.abs(gh).max()
        print("Level: %d. Iter: %d/%d. Grad norm: %f. Energy: %f"%(level, it+1, max_iter, nrm,energy))

        if it % 200 == 0:
            rt.overlay_slices(w_up, w_down, slice_type=2)

        tau = 0.1
        if nrm > tau:
            gh = gh * (tau / nrm)
        b -= gh

        it +=1
    rt.overlay_slices(w_up, w_down, slice_type=2)
