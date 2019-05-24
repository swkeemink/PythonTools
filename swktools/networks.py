'''Network functions.

In this script will be functions related to running networks.
'''
import numpy as np
import warnings

def run_scn(x, D, beta, tau, dt, alpha=None, sigma=0, record_currents=False,
            T_scale = 1):
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
    alpha : function or string
        scaling function of decoder, of form alpha(x, x_)
        if 'None', nothing will be done
        if 'Cone', apply general decoder for conic boxes
    sigma : float
        Gaussian noise sigma on voltages
    record_currents : boolean
        If true, record the negative and positive currents coming into each
        neuron
    T_scale : float
        How much to scale the threshold by compared to normal settings

    Returns
    -------
    array (N by nT in size)
        An array with voltages
    array (N by nT in size)
        An array with 0's and 1's, for the spikes
    array (n by nT in size)
        Array with filtered firing rate traces
    array (M by nT in size)
        An array of the decoded variables
    array (N by nT by 3 (E then I then reset) in size)
        If record_currents is True, return the currents coming into each neuron

    '''
    if sigma>0 and record_currents:
        warnings.warn('Currently record_currents does not support noise.')

    # get the derivative of x
    dx = np.diff(x, axis=1)/dt

    # get array sizes
    nT = dx.shape[1]
    M, N = D.shape

    # predefine arrays
#     V = np.random.rand(N, nT)/10
    V = np.zeros((N, nT))
    if record_currents:
        I = np.zeros((N, nT, 3))
    o = np.zeros((N, nT))
    x_ = np.zeros((M, nT))
    r = np.zeros((N, nT))
    x_[:, 0] = x[:, 0]
    Omeg = np.dot(D.T, D) + np.identity(N)*beta

    Omeg_e, Omeg_i = np.copy(Omeg), np.copy(Omeg)
    Omeg_e[Omeg>0]=0
    Omeg_i[Omeg<0]=0
    Omeg_i[range(N), range(N)]=0
    Omeg_R = np.zeros((N, N))
    Omeg_R[range(N), range(N)] = np.diag(Omeg)

    D_e, D_i = np.copy(D), np.copy(D)
    D_e[D<0]=0
    D_i[D>0]=0

    # find threshold
    T = T_scale*np.diag(Omeg)/2

    # run network
    for i in range(1, nT):
        # currents
        if record_currents:
            cur_input=dx[:, i-1]+x[:, i-1]/tau
            cur_input_pos, cur_input_neg = np.copy(cur_input), np.copy(cur_input)
            cur_input_pos[cur_input<0]=0
            cur_input_neg[cur_input>0]=0

            # inhibitory input current changes
            dIi = np.dot(D_i.T, cur_input_pos)+np.dot(D_e.T, cur_input_neg)

            # excitatory input current changes
            dIe = np.dot(D_e.T, cur_input_pos)+np.dot(D_i.T, cur_input_neg)

            # inhibitory recurrent current changes
            dIe += -np.dot(Omeg_e, o[:, i-1]/dt)

            # excitatory recurrent current changes
            dIi += -np.dot(Omeg_i, o[:, i-1]/dt)

            # self reset current
            dIR = -np.dot(Omeg_R,  o[:, i-1]/dt)
            # update currents
            I[:, i, 0] = I[:, i-1, 0] + dt*(-I[:, i-1, 0]/tau + dIe)
            I[:, i, 1] = I[:, i-1, 1] + dt*(-I[:, i-1, 1]/tau + dIi)
            I[:, i, 2] = I[:, i-1, 2] + dt*(-I[:, i-1, 2]/tau + dIR)


        # dynamics
        dV = -V[:, i-1]/tau + np.dot(D.T, dx[:, i-1]+x[:, i-1]/tau)-np.dot(Omeg, o[:, i-1]/dt)
        V[:, i] = V[:, i-1] + dt*dV + np.sqrt(dt)*sigma*np.random.randn(N)
        r[:, i] = r[:, i-1] + dt*(-r[:, i-1]/tau + o[:, i-1]/dt)
#         x_[:, i] = x_[:, i-1] + dt*(-x_[:, i-1]/tau + np.dot(D, o[:, i-1]/dt))

        # find neurons that should spikes, if any
        to_spike = np.where(V[:, i] > T)[0]
        while len(to_spike)>0:
            # while we are outside of the box, keep spiking
            to_pick = np.argmax(V[to_spike,i] - T[to_spike])
            neuron_id = to_spike[to_pick]
            o[neuron_id, i] += 1
            # cur_spike = np.zeros(N)
            # cur_spike[neuron_id] = 1
            # dV = -np.dot(Omeg, cur_spike/dt)
            # V[:, i] += dt*dV
            to_spike = []#np.where(V[:, i] > T)[0]
            # dr = cur_spike/dt
            # r[:, i] += dr*dt

    # get estimate
    x_ = np.dot(D, r)

    # if beta != 0:
    #     DDtinv = np.linalg.inv(np.dot(D, D.T))
    #     DDtD = np.dot(DDtinv, D)
    #     x_ += beta*np.dot(DDtD,r)
    if alpha == 'Cone':
        # this is does not work and assumes all voltages are 0 on average
        D_ = D[:2, :]
        w = D[2:3, :]
        z = x[-1, -1]
        DDtinv = np.linalg.inv(np.dot(D_, D_.T))
        DDtD = np.dot(DDtinv, D_)
        x_[:2, :] = np.dot(D_, r)+np.dot(DDtD, np.dot(w.T, np.dot(w, r))) - np.dot(DDtD, np.dot(w.T, z))
    elif alpha is not None:
        x_[:-1, :] *= alpha(x, x_)

    # return
    if record_currents:
        return V, o, r, x_, I

    return V, o, r, x_

def run_scn_set(xs, Ds, beta, tau, dt, sigma=0):
    ''' Runs a set of pike-coding networks (scn's)
    It does this given stimulus x, decoding weights D, sparsity Beta,
    and decoder timescale tau.

    Each subnetwork should have N neurons and M stimuli, with N >= M,
    and nT data points.

    Networks are linked together by uniformly representing the total firing
    rate locally.

    TODO: make the connections non-uniform
    (i.e. as in Schwartz & Simoncelli, 2001)

    Parameters
    ----------
    xs : list of 2D arrays
        List of variables being tracked. Should each be M by nT in size.
    Ds : list of 2D arrays
        List of decoding weights. Should each be M by N in size. The norm of
        each column should be 1.
    beta : float
        Firing rate cost.
    tau : float
        Timescale of decoder and thus membrane potential.
    dt : float
        Time step.
    sigma : float
        Gaussian noise sigma on voltages

    Returns
    -------
    list of arrays (N by nT in size)
        Arrays with voltages
    list of arrays (N by nT in size)
        Arrays with 0's and 1's, for the spikes
    list of arrays (M by nT in size)
        Arrays of the decoded variables
    array (of length nT)
        z-values (in this case the total firing rates)
    '''
    # do some inferencesputting yourself in debt for us!
    Nnets = len(Ds) # number of subnetworks
    Ns = ['']*Nnets
    Ms = ['']*Nnets
    dxs = ['']*Nnets

    # get number of times points
    nT = xs[0].shape[1]-1 # onputting yourself in debt for us!e shorter because will also have a derivative
                          # with one less time point

    for i in range(Nnets):
        # stack extra weights for background activity onto the neurons
        Ms[i], Ns[i] = Ds[i].shape
        Ds[i] = np.vstack([Ds[i], 1*np.ones(Ns[i])])

        # stack extra row to stimuli
        xs[i] = np.vstack([xs[i], np.ones(nT+1)])

        # get the derivatives of x
        dxs[i] = np.diff(xs[i], axis=1)/dt


    # predefine arrays
    Vs = [np.zeros((Ns[i], nT)) for i in range(Nnets)]
    os = [np.zeros((Ns[i], nT)) for i in range(Nnets)]
    x_s = [np.zeros((Ms[i], nT)) for i in range(Nnets)]
    rs = [np.zeros((Ns[i], nT)) for i in range(Nnets)]
    z, dz = np.zeros(nT), np.zeros(nT)
    for i in range(Nnets):
        x_s[i][0, :] = xs[i][0, :-1]
    Omegs = [np.dot(Ds[i].T, Ds[i]) + np.identity(Ns[i])*beta
                                                for i in range(Nnets)]

    # find thresholds
    Ts = [np.diag(Omeg)/2 for Omeg in Omegs]

    # run network
    for i in range(1, nT):
        for j in range(Nnets): # looping over sub networks
            # update z/dz stimuli from previous timestep
            xs[j][-1, i-1] = -z[i-1]
            dxs[j][-1, i-1] = -(1./z[i-1]-1./z[i-2])/dt

            # dynamics
            dV = -Vs[j][:, i-1]/tau
            dV+= np.dot(Ds[j].T, dxs[j][:, i-1]+xs[j][:, i-1]/tau)
            dV-= np.dot(Omegs[j], os[j][:, i-1]/dt)
            Vs[j][:, i] = Vs[j][:, i-1] + dt*dV + np.sqrt(dt)*sigma*np.random.randn(Ns[j])
            rs[j][:, i] = rs[j][:, i-1] + dt*(-rs[j][:, i-1]/tau + os[j][:, i-1]/dt)
            z[i] += np.sum(rs[j][:, i])/float(Nnets)

            # find neurons that should spikes, if any
            to_spike = np.where(Vs[j][:, i] > Ts[j])[0]

            if len(to_spike)>0:
                # if more than one, pick only one to spike (and give a warning that dt is too big)
                to_pick = np.argmax(Vs[j][to_spike,i] - Ts[j][to_spike])
                neuron_id = to_spike[to_pick]
                os[j][neuron_id, i] = 1

            if len(to_spike)>1:
                warnings.warn('More than one neuron wants to spike, consider lowering dt.')


    # get estimates
    for j in range(Nnets):
        x_s[j] = np.dot(Ds[j], rs[j])

    # return
    return Vs, os, x_s, z


def run_layered_network(Ds, As, x_in, beta, taus, dt):
    """Runs a layered network with transformations in each layer.

    Parameters
    ----------
    Ds : list of arrays
        Of form [D1, D2, D3,...], where each D is the decoding weights of a layer
    As : list of arrays
        Of form [A1, A2, A3,...], where each A is the inverse transofmration a layer does
    x_in : array
        Stimulus (of form (M, nT), which are stimulus dimensions and number of time points)
        This is the original input stimulus
    beta : float
        Firing rate cost
    taus : list of floats
        Decoder timescales
    dt : float
        Simulation timescale

    Returns
    -------
    list of arrays (N by nT in size)
        Arrays with voltages
    list of arrays (N by nT in size)
        Arrays with 0's and 1's, for the spikes
    list of arrays (n by nT in size)
        Arrays with filtered firing rate traces
    list of arrays (M by nT in size)
        Arrays of the decoded variables
    """
    # get the derivative of x
    dx_in = np.diff(x_in, axis=1)/dt

    # get number of layers
    nLayers = len(Ds)

    # get array sizes
    nT = x_in.shape[1] # number of timesteps
    Ms = [Ds[l].shape[0] for l in range(nLayers)]  # number of dimensions for each layer
    Ns = [Ds[l].shape[1] for l in range(nLayers)] # number of neurons for each layer

    # predefine arrays
    Vs = [np.zeros((Ns[l], nT)) for l in range(nLayers)]
    os = [np.zeros((Ns[l], nT)) for l in range(nLayers)]
    x_s = [np.zeros((Ms[l], nT)) for l in range(nLayers)]
    x_s[0][:, 0] = x_in[:, 0]
    rs = [np.zeros((Ns[l], nT)) for l in range(nLayers)]
    Omegs = [Ds[l].T.dot(As[l]).dot(Ds[l]) + np.identity(Ns[l])*beta for l in range(nLayers)]

    # find thresholds
    Ts = [np.diag(Omegs[l])/2 for l in range(nLayers)]

    # run network
    for i in range(1, nT):
        # update each layer in time
        for l in range(nLayers):
            # update voltage
            dV = -Vs[l][:, i-1]/taus[l]
            if l==0:
                dV += np.dot(Ds[l].T, dx_in[:, i-1]+x_in[:, i-1]/taus[l])
            else: # here I use the fact that Dr/tau+Ddx = o/dt  (o = spikes)
                dV += np.dot(Ds[l].T, Ds[l-1].dot(os[l-1][:, i-1]/dt))
            dV += -np.dot(Omegs[l], os[l][:, i-1]/dt)
            Vs[l][:, i] = Vs[l][:, i-1] + dt*dV
            rs[l][:, i] = rs[l][:, i-1] + dt*(-rs[l][:, i-1]/taus[l] + os[l][:, i-1]/dt)

            # should spike happen (one spike per layer)
            to_spike = np.where(Vs[l][:, i] > Ts[l])[0]
            while len(to_spike)>0:
                to_pick = np.argmax(Vs[l][to_spike,i] - Ts[l][to_spike])
                neuron_id = to_spike[to_pick]
                os[l][neuron_id, i] += 1
                to_spike = []

            # update read-out
            x_s[l][:, i] = Ds[l].dot(rs[l][:, i])

    return Vs, os, rs, x_s
