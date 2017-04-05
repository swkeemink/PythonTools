"""This code motion corrects tifffiles.

Author : sander keemink
"""
import tifffile
import glob
import os
import numpy as np
from skimage.feature import register_translation as rt
from scipy.ndimage.fourier import fourier_shift
from multiprocessing import Pool


def multi_fun(inputs):
    '''Function to apply motion correction across files in parallel.

    Parameters
    ----------
    inputs : list
        [0], array, lower bounds
        [1], array, upper bounds
        [2], array, means
        [3], array, covariance matrix

    Returns
    -------
    array
        probabilities for each stimulus

    '''
    # get inputs
    img = inputs[1]
    refframes = inputs[2]
    tiff_mc = inputs[3]
    cutoff = inputs[4]

    # correct data
    out = correct_array(img, refframes, cutoff)[0]
    tifffile.imsave(tiff_mc, out)

    print 'finished tiff ' + str(inputs[0]) + '...'

def multi_fun_across(inputs):
    '''Function to apply motion correction across files in parallel.

    Parameters
    ----------
    inputs : list
        [0], array, lower bounds
        [1], array, upper bounds
        [2], array, means
        [3], array, covariance matrix

    Returns
    -------
    array
        probabilities for each stimulus

    '''
    # get inputs
    img = inputs[1]
    shift = inputs[2]
    tiff_mc = inputs[3]
    out = np.copy(img)
    numf = out.shape[0]

    # correct data
    for f in range(numf):
        curimg = fourier_shift(np.fft.fftn(img[f, :]), shift)
        out[f, :] = np.fft.ifftn(curimg).real

    tifffile.imsave(tiff_mc, out)

    print 'finished tiff ' + str(inputs[0]) + '...'


def correct_array(X, refframes=100, cutoff=True):
    '''Motion correct an array.

    Parameters
    ----------
    X : array
        framex*x*y array of a video
    refframes : int
        number of frames to average over for reference image (default 100)
    cutoff : bool
        Whether to cutoff based on maximum displacements (default True)

    Returns
    -------
    array
        Motion corrected array, will have slightly smaller dimensions because
        of cutoff.
    array
        The shifts per frame to match to the average of the
        middle refframes frames.
    '''
    # get number of frames
    numf = X.shape[0]

    # find shifts relative to reference mean
    meanimg = X[int(numf/2)-refframes:int(numf/2)+refframes].mean(axis=0)
    shifts = np.array([rt(meanimg, X[f, :, :])[0] for f in range(numf)])

    # do shifts
    out = np.copy(X)
    for f in range(numf):
        curimg = fourier_shift(np.fft.fftn(X[f, :]), shifts[f])
        out[f, :] = np.fft.ifftn(curimg).real

    # cutoff based on max shifts
    if cutoff:
        minx = int(min([0, shifts[:, 0].min()]))
        maxx = int(max([0, shifts[:, 0].max()]))
        miny = int(min([0, shifts[:, 1].min()]))
        maxy = int(max([0, shifts[:, 1].max()]))
        if minx == 0:
            minx = out.shape[0]
        if miny == 0:
            miny = out.shape[1]
        out = out[:, maxx:minx, maxy:miny]

    return out, shifts


def correct_tiffs_sep(folder, folder_mc, refframes=100, cutoff=False):
    '''For each tiff in folder, motion corrects to the middle refframes frames
    using the register_translation function from scikit-image.

    Parameters
    ----------
    folder : string
        folder where tiffs are located
    folder_mc : string
        folder where motion corrected tiffs should be stored
    refframes : int
        number of frames to average over for reference image (default 100)
    cutoff : bool
        Whether to cutoff based on maximum displacements (default True)
    '''
    if folder == folder_mc:
        raise ValueError('input and output folder should not be the same')

    # Find tiffs in folder
    tiffs = sorted(glob.glob(folder+'/*.tif'))
    tiffs_mc = [cycle.replace(folder, folder_mc) for cycle in tiffs]

    # check for folder existence and create if not
    if not os.path.exists(folder_mc):
        os.makedirs(folder_mc)

    # motion correct and save each
    print 'loading in tiffs...'
    inputs = [[i, tifffile.imread(tiffs[i]),
               refframes, tiffs_mc[i], cutoff] for i in range(len(tiffs))]
    print 'starting correction...'
    pool = Pool(None)  # to use less than max processes, change None to number
    pool.map(multi_fun, inputs)
    pool.close()

def correct_tiffs_across(folder, folder_mc, refframes=100, cutoff=False):
    '''For each tiff in folder, motion corrects according to the means
    using the register_translation function from scikit-image.

    Parameters
    ----------
    folder : string
        folder where tiffs are located
    refframes : int
        number of frames to average over for reference image (default 100)
    cutoff : bool
        Whether to cutoff based on maximum displacements (default True)
    '''

    # check for folder existence and create if not
    if not os.path.exists(folder_mc):
        os.makedirs(folder_mc)

    # Find tiffs in folder
    tiffs = sorted(glob.glob(folder+'/*.tif'))
    tiffs_mc = [cycle.replace(folder, folder_mc) for cycle in tiffs]

    # load tiffs and find means
    imgs = [tifffile.imread(tiff) for tiff in tiffs]
    means = np.array([img.mean(axis=0) for img in imgs])

    # get corrections
    out, shifts = correct_array(means, refframes=1, cutoff=False)

    # do shifts
    inputs = [[i, imgs[i], shifts[i], tiffs_mc[i]] for i in range(len(imgs))]

    pool = Pool(None)  # to use less than max processes, change None to number
    pool.map(multi_fun_across, inputs)
    pool.close()
