from __future__ import division, print_function, absolute_import
import numpy as np


def total_energy(masked_image, masked_segmentation,
                 mu, var, index, label, beta):
    r""" Computes the total energy for the ICM model

    Parameters
    -----------
    masked_image :
    masked_segmentation :
    mu : 
    var : 
    index : 
    label : 
    beta : 

    Returns
    --------

    energytotal : float number corresponding to the total energy


    """

    energytotal = neg_log_likelihood(masked_image, mu, var, index, label)
    energytotal += gibbs_energy(masked_segmentation, index, label, beta)

    return energytotal


def neg_log_likelihood(img, mu, var, index, label):
    r""" Computes the negative log-likelihood

    Parameters
    -----------

    img : The grey scale T1 image. It should be masked already
    mu : This is a 1X3 ndarray with mean of each each tissue type
        [CSF, Grey Matter, White Matter]
    var : Also a 1x3 ndarray with the variance of each tissue type
    index : The voxel index
    label : The index of the label that is looping through, 0, 1 or 2

    Returns
    --------

    loglike : the negative log-likelihood

    """

    loglike = ((img[index] - mu[label]) ** 2) / (2 * var[label])

    loglike += np.log(np.sqrt(var[label]))

    return loglike


def gibbs_energy(seg, index, label, beta):

    label = label + 1
    energy = 0

    if label == seg[index[0] + 1 - 1, index[1] + 1, index[2] + 1]:
        energy = energy - beta
    else:
        energy = energy + beta

    if label == seg[index[0] + 1 + 1, index[1] + 1, index[2] + 1]:
        energy = energy - beta
    else:
        energy = energy + beta

    if label == seg[index[0] + 1, index[1] + 1 - 1, index[2] + 1]:
        energy = energy - beta
    else:
        energy = energy + beta

    if label == seg[index[0] + 1, index[1] + 1 + 1, index[2] + 1]:
        energy = energy - beta
    else:
        energy = energy + beta

    if label == seg[index[0] + 1, index[1] + 1, index[2] + 1 - 1]:
        energy = energy - beta
    else:
        energy = energy + beta

    if label == seg[index[0] + 1, index[1] + 1, index[2] + 1 + 1]:
        energy = energy - beta
    else:
        energy = energy + beta

    return energy


def ising(l, voxel, beta):
    r""" Ising model

    Parameters
    -----------
    l : The value of the label that is being tested in each voxel (1, 2 or 3)
    voxel : The value of the voxel
    beta : the value of the weight given to the neighborhood

    Returns
    --------
    beta : Negative or positive float number

    """

    if l == voxel:
        return - beta
    else:
        return beta