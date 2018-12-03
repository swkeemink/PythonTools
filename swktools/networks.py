'''Network functions.

In this script will be functions related to running networks.
'''
import numpy as np
import warnings

def run_scn(x, D, beta, tau, dt, alpha=None):
    ''' Runs a simple spike-coding network (scn), given stimulus x, decoding weights D, sparsity Beta, and decoder timescale tau.

    Should have N neurons and M stimuli, with N >= M, and nT data points.

    If the decoder needs to be adjusted (for example for suboptimal resets, or other changes), an optional
    function 'alpha' can be given, which will be a function of x and x_ (the estimated x). at the moment
    alpha is not applied to the last stimulus.

    Parameters
    ----------
    x : 2D array
        The variables being tracked. Should be M by nT in size.
    D : 2D array
        Decoding weights. Should be M by N in size.
    beta : float
        Firing rate cost.
    tau : float
        Timescale of decoder and thus membrane potential.
    dt : float
        Time step.
    alpha : function
        scaling function of decoder, of form alpha(x, x_)

    Returns
    -------
    array (N by nT in size)
        An array with voltages
    array (N by nT in size)
        An array with 0's and 1's, for the spikes
    array (M by nT in size)
        An array of the decoded variables
    '''
    # get the derivative of x
    dx = np.diff(x, axis=1)/dt

    # get array sizes
    nT = dx.shape[1]
    M, N = D.shape

    # predefine arrays
#     V = np.random.rand(N, nT)/10
    V = np.zeros((N, nT))
    o = np.zeros((N, nT))
    x_ = np.zeros((M, nT))
    r = np.zeros((N, nT))
    x_[:, 0] = x[:, 0]
    Omeg = np.dot(D.T, D) + np.identity(N)*beta

    # find threshold
    T = np.diag(Omeg)/2

    # run neuron
    for i in range(1, nT):
        # dynamics
        dV = -V[:, i-1]/tau + np.dot(D.T, dx[:, i-1]+x[:, i-1]/tau) - np.dot(Omeg, o[:, i-1]/dt)
        V[:, i] = V[:, i-1] + dt*dV
        r[:, i] = r[:, i-1] + dt*(-r[:, i-1]/tau + o[:, i-1]/dt)
#         x_[:, i] = x_[:, i-1] + dt*(-x_[:, i-1]/tau + np.dot(D, o[:, i-1]/dt))

        # find neurons that should spikes, if any
        to_spike = np.where(V[:, i] > T)[0]

        if len(to_spike)>0:
            # if more than one, pick only one to spike (and give a warning that dt is too big)
            to_pick = np.argmax(V[to_spike,i] - T[to_spike])
            neuron_id = to_spike[to_pick]
            o[neuron_id, i] = 1

        if len(to_spike)>1:
            warnings.warn('More than one neuron wants to spike, consider lowering dt.')

    # get estimate
    x_ = np.dot(D, r)
    if alpha is not None: x_[:-1, :] *= alpha(x, x_)

    # return
    return V, o, x_
