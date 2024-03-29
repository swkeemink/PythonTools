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
    return np.random.randn(l, l)

def DoG(settings, l):
    """ For a given image size, make Difference of Gaussians filter.

    Parameters
    ----------
    settings : dictionary
        contains:
        'scales' : array
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
    """
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

def gabor(settings, l):
    """ For a given image size, make Gabor filter.

    Parameters
    ----------
    settings : dictionary
        contains following keys:
        'frequency', 'theta', 'sigma',
        'offset', 'xoffset', 'yoffset'
        See swktools.filters.gabor_kernel() help for more information on
        what these should be.
    l : int
        width and height of image

    Returns
    -------
    array
        The spatial filter (2d array)
    """
    # get settings
    frequency = settings['frequency']
    theta = settings['theta']
    sigma = settings['sigma']
    offset = settings['offset']
    xoffset, yoffset = settings['xoffset'], settings['yoffset']

    # return filter
    filt =  gabor_kernel(frequency, l, theta, sigma, offset,
                                xoffset, yoffset)
    return filt

def gabor_kernel(frequency, imsize=10, theta=0, sigma=1, offset=0, xoffset=0,
                 yoffset=0):
    """Return complex 2D Gabor filter kernel.

    Adopted from https://github.com/scikit-image/scikit-image/blob/master/skimage/filters/_gabor.py#L16

    Gabor kernel is a Gaussian kernel modulated by a complex harmonic function.
    Harmonic function consists of an imaginary sine function and a real
    cosine function. Spatial frequency is inversely proportional to the
    wavelength of the harmonic and to the standard deviation of a Gaussian
    kernel. The bandwidth is also inversely proportional to the standard
    deviation.

    Parameters
    ----------
    frequency : float
        Spatial frequency of the harmonic function. Specified in pixels.
    imsize : int
        Size of image
    theta : float, optional
        Orientation in radians. If 0, the harmonic is in the x-direction.
    sigma : float, optional
        Standard deviation in x- and y-directions.
    offset : float, optional
        Phase offset of harmonic function in radians.
    xoffset, yoffset : floats
        Coordinate offsets from middle

    Returns
    -------
    g : complex array
        Complex filter kernel.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gabor_filter
    .. [2] https://web.archive.org/web/20180127125930/http://mplab.ucsd.edu/tutorials/gabor.pdf
    """
    # find limits
    x0 = (imsize-1)/2.
    y0 = (imsize-1)/2.
    y, x = np.mgrid[-y0:y0 + 1, -x0:x0 + 1]

    # offset values
    x -= xoffset
    y -= yoffset

    # calculate rotation
    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)

    # calculate filter
    g = np.exp(-0.5 * (rotx ** 2 / sigma ** 2 + roty ** 2 / sigma ** 2))
    # g /= 2 * np.pi * sigma**2
    g *= np.cos(2 * np.pi * frequency * rotx + offset)

    # normalize filter
    # g /= g.sum()

    return g

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

def CreateSettingsList(N, filter_type, l, distribution='Random'):
    """Creates a settings list for CreateFilterBank function.

    Parameters
    ----------
    N : int or list o fints
        How many filters settings should be created
        If the distribution is even, it should be a list as follows:
        N = [Nxy, Nfreq, Nthet, Nsig, Noff]
    filter_type : string
        'DoG': Difference of Gaussians
        'Gabor' : Gabor filters (Not yet implemented)
    l : int
        Image size
    distribution : string
        'Random' - filters are just completely random within the image bounds,
                   and for filter settings
        'Even' - filters are distributed evenly across the different feature
                 dimensions.
                 In this case it can not be guaranteed you get exactly N
                 neurons, N is used as a reference

    Returns
    -------
    list
        A list of settings
    """
    if distribution == 'Random':
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
            frequencies = np.random.rand(N)*5/l
            thetas = np.random.rand(N)*np.pi
            sigma = np.random.rand(N)*l/5+1
            offset = np.random.rand(N)
            xyoffsets = (np.random.rand(N, 2)-0.5)*l
            for i in range(N):
                settings = {}
                settings['frequency'] = frequencies[i]
                settings['theta'] = thetas[i]
                settings['sigma'] = sigma[i]
                settings['offset'] = offset[i]
                settings['xoffset'] = xyoffsets[i, 0]
                settings['yoffset'] = xyoffsets[i, 1]
                out[i] = settings
        else:
            raise ValueError('Use either DoG or Gabor for filter_type.')
    elif distribution == 'Even':
        if filter_type == 'DoG':
            raise NotImplementedError()
        elif filter_type == 'Gabor':
            Nxy, Nfreq, Nthet, Nsig, Noff = N
            xyoffsets = np.linspace(-l/2, l/2, Nxy+2)[1:-1]
            frequencies = np.linspace(3, 8, Nfreq)/l
            thetas = np.linspace(0, np.pi, Nthet+1)[:-1]
            sigmas = np.linspace(l/10., l/20., Nsig)
            offsets = np.linspace(0, 1, Noff+1)[:-1]
            count = 0
            out = ['']*len(xyoffsets)**2*len(frequencies)*len(thetas)*len(sigmas)*len(offsets)
            for x in xyoffsets:
                for y in xyoffsets:
                    for freq in frequencies:
                        for theta in thetas:
                            for sigmai, sigma in enumerate(sigmas):
                                for offset in offsets:
                                    settings = {}
                                    settings['frequency'] = freq
                                    settings['theta'] = theta
                                    settings['sigma'] = sigma
                                    settings['offset'] = offset
                                    settings['xoffset'] = x
                                    settings['yoffset'] = y
                                    out[count] = settings
                                    count += 1
    else:
        raise ValueError('Use correct distribution type.')

    return out
