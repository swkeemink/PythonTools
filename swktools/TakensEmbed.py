"""An implementation of the Taken's embedding theorem.

Based on Sugihara et al (2012) - Detecting Causality in Complex Ecosystems.

Author: Sander Keemink.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def get_delayed_manifold(data, tau=10, ndelay=3):
    """Get the delayed manifolds of the variables in data.

    Parameters
    ----------
    data : array
        ntimepoints*nvariable data array
    tau : int, optional
        how many timepoints per delay step
    ndelay : int
        how many delay steps, optional

    Returns
    -------
    array
        ndim*ndata*ndelay array

    """
    N = data.shape[0]  # number of data points
    ndim = data.shape[1]  # number of input dimensions
    delayed_manifolds = np.zeros((ndim, N-ndelay*tau, ndelay))
    for dim in range(ndim):
        for n in range(ndelay):
            delayed_manifolds[dim, :, n] = np.roll(data[:, dim],
                                                   -n*tau)[:-ndelay*tau]
    return delayed_manifolds


def findknearest(data, k):
    """Find k nearest neighbours.

    Parameters
    ----------
    data : array
        Data to apply NearestNeighbors to
        (see sklearn.neighbors.NearestNeighbors)
    k : int
        How many neighbours to find.

    """
    neigh = NearestNeighbors(n_neighbors=k+1)
    neigh.fit(data)
    dists, ids = neigh.kneighbors(data, n_neighbors=k+1)
    return dists[:, 1:], ids[:, 1:]


def do_embedding(delayed_manifolds, rnge=range(20, 5000, 20)):
    """Do embedding at different time-point-lengths.

    Parameters
    ----------
    delayed_manifolds : array
        ndim*ndata*ndelay array with delayed single time courses
    rnge : array, optional
        at which time points to calculate the predictability of the variables

    Returns
    -------
    array
        The correlations

    """
    # get some information about data size
    ndelay = delayed_manifolds.shape[2]
    ndims = delayed_manifolds.shape[0]
    # N = delayed_manifolds.shape[1]

    # start analysis
    data = delayed_manifolds
    k = ndelay+3  # how many neighbours to find
    cors = np.zeros((ndims, ndims, len(rnge)))
    # loop over time lengths
    for i, l in enumerate(rnge):
        dists, ids, weights, preds = {}, {}, {}, {}
        # loop over actual dimensions
        for dim in range(ndims):
            # get nearest neighbours
            dists[dim], ids[dim] = findknearest(data[dim, :l, :], k)

            # get weights as per pop paper
            minim = dists[dim].min(axis=1)
            weights[dim] = np.exp(-dists[dim]/minim[:, None])
            weights[dim] /= weights[dim].sum(axis=1)[:, None]
        # get predictions from cross embeddings for all dimension combinations
        for dim1 in range(ndims):  # dimension to use to predict
            for dim2 in range(ndims):  # dimension to predict
                points_to_use = delayed_manifolds[dim2, ids[dim1], 1]
                preds[dim1, dim2] = np.sum(weights[dim1][:, :]*points_to_use,
                                           axis=1)
                cors[dim1, dim2, i] = np.corrcoef(preds[dim1, dim2],
                                                  delayed_manifolds[dim2,
                                                                    :l,
                                                                    1])[0, 1]

    return cors
