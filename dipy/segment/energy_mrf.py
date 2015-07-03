from __future__ import division, print_function, absolute_import
import numpy as np


def total_energy(masked_image, masked_segmentation,
                 mu, var, index, label, beta):

    energytotal = neg_log_likelihood(masked_image, mu, var, index, label)
    energytotal += gibbs_energy(masked_segmentation, index, label, beta)

    return energytotal


def neg_log_likelihood(img, mu, var, index, label):

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
    """ Ising model

    Parameters
    -----------
    l :
    voxel :

    Returns
    --------
    beta : float
    """

    if l == voxel:
        return - beta
    else:
        return beta
