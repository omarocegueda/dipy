from __future__ import print_function
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import abc
from scipy import gradient, ndimage
import dipy.align.vector_fields as vfu
from dipy.align import sumsqdiff as ssd
from dipy.align import crosscorr as cc
from dipy.align import expectmax as em
from dipy.align import floating
from dipy.align.metrics import SimilarityMetric

class ECCMetric(SimilarityMetric):

    def __init__(self, dim, sigma_diff=2.0, radius=4, q_levels=256):
        r"""
        Normalized Cross-Correlation Similarity metric.

        Parameters
        ----------
        dim : int (either 2 or 3)
            the dimension of the image domain
        sigma_diff : the standard deviation of the Gaussian smoothing kernel to
            be applied to the update field at each iteration
        radius : int
            the radius of the squared (cubic) neighborhood at each voxel to be
            considered to compute the cross correlation
        """
        super(ECCMetric, self).__init__(dim)
        self.sigma_diff = sigma_diff
        self.radius = radius
        self.q_levels = q_levels

        self.static_image_mask = None
        self.moving_image_mask = None
        self.staticq_means_field = None
        self.movingq_means_field = None
        self.movingq_levels = None
        self.staticq_levels = None

        self._connect_functions()

    def _connect_functions(self):
        if self.dim == 2:
            self.precompute_factors = cc.precompute_cc_factors_2d
            self.compute_forward_step = cc.compute_cc_forward_step_2d
            self.compute_backward_step = cc.compute_cc_backward_step_2d
            self.reorient_vector_field = vfu.reorient_vector_field_2d

            self.quantize = em.quantize_positive_image
            self.compute_stats = em.compute_masked_image_class_stats
        elif self.dim == 3:
            self.precompute_factors = cc.precompute_cc_factors_3d
            self.compute_forward_step = cc.compute_cc_forward_step_3d
            self.compute_backward_step = cc.compute_cc_backward_step_3d
            self.reorient_vector_field = vfu.reorient_vector_field_3d

            self.quantize = em.quantize_positive_volume
            self.compute_stats = em.compute_masked_volume_class_stats
        else:
            print('CC Metric not defined for dimension %d'%(self.dim))


    def initialize_iteration(self):
        r"""
        Pre-computes the cross-correlation factors
        """
        ##################################################
        #Compute the CC factors (CC-initialization)
        ##################################################
        self.staticq_factors = self.precompute_factors(self.staticq_means_field,
                                                       self.moving_image,
                                                       self.radius)
        self.movingq_factors = self.precompute_factors(self.static_image,
                                                       self.movingq_means_field,
                                                       self.radius)
        self.staticq_factors = np.array(self.staticq_factors)
        self.movingq_factors = np.array(self.movingq_factors)
        
        ##################################################
        #Compute the gradients (common initialization)
        ##################################################
        self.gradient_moving = np.empty(
            shape=(self.moving_image.shape)+(self.dim,), dtype=floating)
        for i, grad in enumerate(sp.gradient(self.moving_image)):
            self.gradient_moving[..., i] = grad

        self.gradient_static = np.empty(
            shape=(self.static_image.shape)+(self.dim,), dtype=floating)
        for i, grad in enumerate(sp.gradient(self.static_image)):
            self.gradient_static[..., i] = grad

        #Convert the moving image's gradient field from voxel to physical space
        if self.moving_spacing is not None:
            self.gradient_moving /= self.moving_spacing
        if self.moving_direction is not None:
            self.reorient_vector_field(self.gradient_moving, 
                                       self.moving_direction)

        #Convert the moving image's gradient field from voxel to physical space
        if self.static_spacing is not None:
            self.gradient_static /= self.static_spacing
        if self.static_direction is not None:
            self.reorient_vector_field(self.gradient_static, 
                                       self.static_direction)

    def free_iteration(self):
        r"""
        Frees the resources allocated during initialization
        """
        del self.staticq_factors
        del self.movingq_factors
        del self.gradient_moving
        del self.gradient_static
    
    def compute_forward(self):
        r"""
        Computes the update displacement field to be used for registration of
        the moving image towards the static image
        """
        #This step uses the gradient of the static image
        displacement, self.energy = self.compute_forward_step(
            self.gradient_static, self.gradient_moving, self.movingq_factors)
        displacement=np.array(displacement)
        for i in range(self.dim):
            displacement[..., i] = ndimage.filters.gaussian_filter(
                                        displacement[..., i], self.sigma_diff)
        return displacement

    def compute_backward(self):
        r"""
        Computes the update displacement field to be used for registration of
        the static image towards the moving image
        """
        #This step uses the gradient of the moving image
        displacement, energy=self.compute_backward_step(
            self.gradient_static, self.gradient_moving, self.staticq_factors)
        displacement=np.array(displacement)
        for i in range(self.dim):
            displacement[..., i] = ndimage.filters.gaussian_filter(
                                        displacement[..., i], self.sigma_diff)
        return displacement

    def use_static_image_dynamics(self, original_static_image, transformation):
        r"""
        ECCMetric takes advantage of the image dynamics by computing the
        current static image mask from the originalstaticImage mask (warped
        by nearest neighbor interpolation)
        
        Parameters
        ----------
        original_static_image : array, shape (R, C) or (S, R, C)
            the original static image from which the current static image was
            generated, the current static image is the one that was provided 
            via 'set_static_image(...)', which may not be the same as the
            original static image but a warped version of it (even the static 
            image changes during Symmetric Normalization, not only the moving
            one).
        transformation : DiffeomorphicMap object
            the transformation that was applied to the original_static_image 
            to generate the current static image
        """
        #self.static_image_mask = (original_static_image>0).astype(np.int32)
        #if transformation == None:
        #    return
        #self.static_image_mask = \
        #    transformation.transform(self.static_image_mask,'nn')
        pass

    def use_moving_image_dynamics(self, original_moving_image, transformation):
        r"""
        ECCMetric takes advantage of the image dynamics by computing the
        current moving image mask from the original_moving_image mask (warped
        by nearest neighbor interpolation)

        Parameters
        ----------
        original_moving_image : array, shape (R, C) or (S, R, C)
            the original moving image from which the current moving image was
            generated, the current moving image is the one that was provided 
            via 'set_moving_image(...)', which may not be the same as the
            original moving image but a warped version of it.
        transformation : DiffeomorphicMap object
            the transformation that was applied to the original_moving_image 
            to generate the current moving image
        """
        #self.moving_image_mask = (original_moving_image>0).astype(np.int32)
        #if transformation == None:
        #    return
        #self.moving_image_mask = \
        #    transformation.transform(self.moving_image_mask,'nn')
        pass

    def use_image_dynamics(self, original_static, static_affine, static_transform, 
                           original_moving, moving_affine, moving_transform):
        r"""
        ECCMetric takes advantage of the image dynamics by computing the
        current moving/static image masks from the original moving/static 
        image masks(warped with nearest neighbor interpolation)

        Parameters
        ----------
        original_static : array, shape (R, C) or (S, R, C)
            the original moving image from which the current moving image was
            generated, the current moving image is the one that was provided 
            via 'set_moving_image(...)', which may not be the same as the
            original moving image but a warped version of it.
        static_affine: array, shape 3x3 (for 2D images) or 4x4 (for 3D images)
            the voxel-to-space transformation associated to the static image
        static_transform : DiffeomorphicMap object
            the transformation that was applied to the original_moving_image 
            to generate the current moving image
        original_moving_image : array, shape (R, C) or (S, R, C)
            the original moving image from which the current moving image was
            generated, the current moving image is the one that was provided 
            via 'set_moving_image(...)', which may not be the same as the
            original moving image but a warped version of it.
        moving_affine: array, shape 3x3 (for 2D images) or 4x4 (for 3D images)
            the voxel-to-space transformation associated to the moving image
        static_transform : DiffeomorphicMap object
            the transformation that was applied to the original_moving_image 
            to generate the current moving image
        moving_transform : DiffeomorphicMap object
            the transformation that was applied to the original_moving_image 
            to generate the current moving image
        """
        self.static_image_mask = (original_static>0).astype(np.int32)
        self.moving_image_mask = (original_moving>0).astype(np.int32)
        #We assume that either both transforms are None or none of them is
        if (static_transform is not None) and (moving_transform is not None):
            #compute the full warping
            composition = moving_transform.warp_endomorphism(static_transform.inverse()).inverse()

            static_shape = np.asarray(original_static.shape, dtype=np.int32)
            moving_shape = np.asarray(original_moving.shape, dtype=np.int32)
            #The cost of the following inversions is negligible compared to the
            #overall complexity of the registration algorithm
            static_affine_inv = np.linalg.inv(static_affine)
            moving_affine_inv = np.linalg.inv(moving_affine)

            #Warp the original moving image/mask at its original resolution
            self.wmoving = composition.transform(original_moving, 'lin', moving_affine_inv,
                                                 static_shape, static_affine)
            self.wmoving_mask = composition.transform(self.moving_image_mask, 'nn', moving_affine_inv,
                                                      static_shape, static_affine)
        
            #Warp the original static image/mask at its original resolution
            self.wstatic = composition.transform_inverse(original_static, 'lin', static_affine_inv,
                                                 moving_shape, moving_affine)
            self.wstatic_mask = composition.transform_inverse(self.static_image_mask, 'nn', static_affine_inv,
                                                      moving_shape, moving_affine)
        else:#We assume the transforms are identities
            self.wmoving = original_moving
            self.wstatic = original_static
            self.wstatic_mask = self.static_image_mask
            self.wmoving_mask = self.moving_image_mask

        #Compute the sampling masks and quantizations
        #Note: since now we iterate at the fool resolution, the quantization and
        #native masks don't change over time
        self.sampling_static = self.static_image_mask * self.wmoving_mask
        self.sampling_moving = self.moving_image_mask * self.wstatic_mask
        
        #Quantize the original static (fine resolution) and compute the 
        #mean/sigma_sq fields from it
        self.staticq, self.staticq_levels, hist = \
            self.quantize(original_static,self.q_levels)
        self.staticq = np.asarray(self.staticq)
        staticq_means, staticq_variances = self.compute_stats(self.sampling_static,
                                                              self.wmoving,
                                                              self.q_levels,
                                                              self.staticq)
        staticq_means[0] = 0
        self.staticq_means = np.array(staticq_means)
        self.staticq_variances = np.array(staticq_variances)
        self.staticq_variances[np.isinf(self.staticq_variances)] = -1
        self.staticq_variances[self.staticq_variances < 0 ] = self.staticq_variances.max()
        self.staticq_sigma_sq_field_fine = self.staticq_variances[self.staticq]
        self.staticq_means_field_fine = self.staticq_means[self.staticq]

        #Quantize the original moving (fine resolution) and compute the 
        #mean/sigma_sq fields from it
        self.movingq, self.movingq_levels, hist = \
            self.quantize(original_moving,self.q_levels)
        self.movingq = np.asarray(self.movingq)
        movingq_means, movingq_variances = self.compute_stats(self.sampling_moving,
                                                              self.wstatic,
                                                              self.q_levels,
                                                              self.movingq)
        movingq_means[0] = 0
        self.movingq_means = np.array(movingq_means)
        self.movingq_variances = np.array(movingq_variances)
        self.movingq_variances[np.isinf(self.movingq_variances)] = -1
        self.movingq_variances[self.movingq_variances < 0] = self.movingq_variances.max()
        self.movingq_sigma_sq_field_fine = self.movingq_variances[self.movingq]
        self.movingq_means_field_fine = self.movingq_means[self.movingq]

        #Now this is the point of executing EM at the full resolution:
        #Instead of first warping the images to a coarser resolution and
        #computing statistics from its quantizations, compute the stats
        #at the fine resolution and then warp them to a coarser resolution
        self.staticq_means_field = static_transform.transform(self.staticq_means_field_fine)
        self.movingq_means_field = moving_transform.transform(self.movingq_means_field_fine)
        self.staticq_sigma_sq_field = static_transform.transform(self.staticq_sigma_sq_field_fine)
        self.movingq_sigma_sq_field = moving_transform.transform(self.movingq_sigma_sq_field_fine)
        

    def get_energy(self):
        r"""
        Returns the Cross Correlation (data term) energy computed at the largest
        iteration
        """
        return self.energy
