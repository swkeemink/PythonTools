"""General functions.

Author: s w keemink
"""
from __future__ import division
import numpy as np
from skimage.feature import register_translation as rt
from scipy.ndimage.fourier import fourier_shift
from scipy import signal


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



def lowPassFilter(F, fs=40, nfilt=40, fw_base=10, axis=0):
    '''
    Low pass filters a fluorescence imaging trace line.

    Parameters
    ----------
    F : array_like
        Fluorescence signal.
    fs : float, optional
        Sampling frequency of F, in Hz. Default 40.
    nfilt : int, optional
        Number of taps to use in FIR filter, default 40
    fw_base : float, optional
        Cut-off frequency for lowpass filter, default 1
    axis : int, optional
        Along which axis to apply low pass filtering, default 0

    Returns
    -------
    array
        Low pass filtered signal of len(F)

    Based on implementation by S. Lowe.
    '''
    # The Nyquist rate of the signal is half the sampling frequency
    nyq_rate = fs / 2.0

    # Make a set of weights to use with our taps.
    # We use an FIR filter with a Hamming window.
    b = signal.firwin(nfilt, cutoff=fw_base/nyq_rate, window='hamming')

    # Use lfilter to filter with the FIR filter.
    # We filter along the second dimension because that represents time
    filtered_f = signal.filtfilt(b, [1.0], F, axis=axis)

    return filtered_f

def highPassFilter(F, fs=40, nfilt=40, fw_base=10, axis=0):
    '''
    High pass filters a fluorescence imaging trace line.

    Parameters
    ----------
    F : array_like
        Fluorescence signal.
    fs : float, optional
        Sampling frequency of F, in Hz. Default 40.
    nfilt : int, optional
        Number of taps to use in FIR filter, default 40
    fw_base : float, optional
        Cut-off frequency for lowpass filter, default 1
    axis : int, optional
        Along which axis to apply low pass filtering, default 0

    Returns
    -------
    array
        Low pass filtered signal of len(F)

    Based on implementation by S. Lowe.
    '''
    # The Nyquist rate of the signal is half the sampling frequency
    nyq_rate = fs / 2.0

    # Make a set of weights to use with our taps.
    # We use an FIR filter with a Hamming window.
    b = signal.firwin(nfilt, cutoff=fw_base/nyq_rate,
                      pass_zero=False)

    # Use lfilter to filter with the FIR filter.
    # We filter along the second dimension because that represents time
    filtered_f = signal.filtfilt(b, [1.0], F, axis=axis)

    return filtered_f


def bandPassFilter(F, fs=40, nfilt=40, fw_low=10, fw_up=40, axis=0):
    '''
    Band pass filters a fluorescence imaging trace line.

    Parameters
    ----------
    F : array_like
        Fluorescence signal.
    fs : float, optional
        Sampling frequency of F, in Hz. Default 40.
    nfilt : int, optional
        Number of taps to use in FIR filter, default 40
    fw_low : float, optional
        Lower cut-off frequency for bandpass filter, default 10
    fw_up : float, optional
        Upper cut-off frequency for bandpass filter, default 10
    axis : int, optional
        Along which axis to apply low pass filtering, default 0

    Returns
    -------
    array
        Low pass filtered signal of len(F)

    Based on implementation by S. Lowe.
    '''
    # The Nyquist rate of the signal is half the sampling frequency
    nyq_rate = fs / 2.0

    # Make a set of weights to use with our taps.
    # We use an FIR filter with a Hamming window.
    b = signal.firwin(nfilt, [fw_low/nyq_rate, fw_up/nyq_rate],
                      pass_zero=False)
    # taps = firwin(ntaps, [lowcut, highcut], nyq=nyq, pass_zero=False,
    #               window=window, scale=False)
    # Use lfilter to filter with the FIR filter.
    # We filter along the second dimension because that represents time
    filtered_f = signal.filtfilt(b, [1.0], F, axis=axis)

    return filtered_f
