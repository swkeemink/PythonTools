"""General functions.

Author: s w keemink
"""
from __future__ import division
import numpy as np
from skimage.feature import register_translation as rt
from scipy.ndimage.fourier import fourier_shift


def find_max_correlation_shifted(x, y):
    """Finding the maximum correlation between two arrays after shifting.

    It finds the best shift with the scikit-image register_translation function
    for finding the best way to shift an image in the fourrier space.

    Parameters
    ----------
    x, y: arrays
        Equal length arrays for which the correlation should be found

    Returns
    -------
    float
        The maximum correlation
    int
        The shift at maximum correlation
    array
        The shifted y-array
    """
    # find shift
    shift = rt(x, y)[0]

    # apply shift
    y_f_shift = fourier_shift(np.fft.fftn(y), shift)
    y_shifted = np.fft.ifftn(y_f_shift).real

    # return shift and correlation
    return np.corrcoef(x, y_shifted)[0, 1], shift, y_shifted


def popvec(X, ang):
    """Population vector for the set of responses X.

    Each value in the vector X corresponds to an angle in ang

    Parameters
    ----------
    X : array
        1D vector of length len(ang)
    ang : array
        preferred orientations

    Returns
    -------
    double
        The angle of the population vector.
    """
    # define vector coordinates
    v = np.zeros((2, len(ang)))
    v[0, :] = np.cos(2*ang)
    v[1, :] = np.sin(2*ang)

    # find population vector
    vest0 = np.sum(X*v, 1)

    # return the angle of the population vector
    return 0.5*np.arctan2(vest0[1], vest0[0])
