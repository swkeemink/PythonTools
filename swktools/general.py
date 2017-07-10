"""General functions.

Author: s w keemink
"""
from __future__ import division
import numpy as np
from skimage.feature import register_translation as rt
from scipy.ndimage.fourier import fourier_shift


def spike_match(real, est, maxdelay):
    """Match real spiketimes with estimated spiketimes.

    Parameters
    ----------
    real : array
      real spike times
    estimated : array
      estimated spike times
    maxdelay : float
      maximum delay for matching spiketimes

    Returns
    -------
    float
      ER score (1 - F1 = 1- true_p/(true_p+false_p+false_n) )
    int
      True positives
    int
      False positives
    int
      False negatives
    array
      Labels on real times (true_p: 1, false_n: 2)
    array
      Labels on estimated times (true_p: 1, false_p: 3)
    """
    # start counters
    true_p = 0
    false_n = 0

    # start labels
    real_labels = np.zeros(len(real))
    est_labels = np.zeros(len(est))

    # loop over all real spikes
    for i in range(len(real)):
        # get current time
        cur_time = real[i]

        # get differences with current estimated times
        diffs = abs(cur_time - est)

        # find all match ids
        ids = np.nonzero(np.logical_and(diffs <= maxdelay, est_labels == 0))
        ids = np.array(ids)[0]

        # if any match, remember the first as true positive
        if ids.size != 0:
            est_labels[ids[0]] = 1
            real_labels[i] = 1
            true_p = true_p + 1
        else:  # otherwise, this is a false negative
            real_labels[i] = 2
            false_n = false_n + 1

    # set leftover estimated spikes to false positives
    est_labels[est_labels == 0] = 3
    false_p = len(est) - true_p

    # get ER
    F1 = 2*true_p/(2*true_p + false_n + false_p)
    ER = 1 - F1

    return ER, true_p, false_p, false_n, real_labels, est_labels



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
