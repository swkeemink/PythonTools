""" Code for generating filters for neural layers (like gabor filters for
V1).

Author: S W keemink
"""

from __future__ import division
import numpy as np
from scipy.stats import multivariate_normal

def RandomFilter(settings, l):
    """Generates a completely random filter.
    """
    return np.random.rand(l, l)

def DoG(settings, l):
    ''' For a given image size, make Difference of Gaussians filter.

    Parameters
    ----------
    settings : dictionary
        contains:
        'scales' : arry
            scales of each Gaussian
        'means' : list of arrays
            list of 2D means
        'sigmas' : array
            variances
    l : int
        width and height of image

    Returns
    -------
    filt
        The spatial filter (2d array)
    '''
    scales, means, sigmas = settings['scales'], settings['means'], settings['sigmas']

    # get x,y values for gaussians
    x, y = np.mgrid[0:l, 0:l]
    pos = np.dstack((x, y))

    # generate positive filter
    filt = np.zeros((l, l))
    for i in range(2):
        cov = np.array([[sigmas[i], 0], [0, sigmas[i]]])
        rv = multivariate_normal(means[i], cov)
        filt += scales[i]*rv.pdf(pos) # spatial filter

    return filt

def CreateFilterBank(filtfunc, settings_list, l):
    """Creates a filter bank for a given filter function

    Parameters
    ----------
    filtfunc : function
        Fuction of form filtfunc(settings, l)
        where settings is some input, and l is the same as the input to the current function
    settings_list : list
        a list of settings inputs for each filter
    l : int
        Image size (full image will be l by l)

    Returns
    -------
    list
        A list of arrays, of the same length as settings_list
    """
    out = ['']*len(settings_list)
    for i, settings in enumerate(settings_list):
        filt = filtfunc(settings, l)
        out[i] = filt.reshape(l*l)
    return out

def CreateSettingsList(N, filter_type, l):
    """Creates a settings list for CreateFilterBank function.

    Parameters
    ----------
    N : int
        How many filters settings should be created for
    filter_type : string
        'DoG': Difference of Gaussians
        'Gabor' : Gabor filters (Not yet implemented)
    l : int
        Image size

    Returns
    -------
    list
        A list of settings
    """
    out = ['']*N

    if filter_type == 'DoG':
        scales = np.sign(np.random.rand(N)-0.1)
        sigmas = np.random.rand(N)*l/2
        means = np.random.rand(N, 2)*l
        for i in range(N):
            settings = {}
            settings['scales'] = [scales[i], -scales[i]]
            settings['means'] = [means[i], means[i]]
            settings['sigmas'] = [sigmas[i], sigmas[i]*2]
            out[i] = settings

    elif filter_type == 'Gabor':
        raise NotImplementedError()
    else:
        raise ValueError('Use either DoG or Gabor for filter_type.')

    return out
