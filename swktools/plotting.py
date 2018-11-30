"""Set of plotting aids for Holoviews."""
from __future__ import division
import holoviews as hv
import numpy as np
import scipy
import matplotlib.pyplot as plt


def plot3d(traj, ntraj=1, labels=['x', 'y', 'z'], ax=None):
    """Plot the 3d trajectory in traj in 4D.

    Parameters
    ----------
    traj : array
        3D array for plotting, should be datapoints*dimensions.
    ntraj : int, optional
        In how many trajectories to split up the data (can be useful to see
        the time dependence of the trajectory)
    labels : list of strings, optional
        What labels to put on the axes. By default is ['x', 'y', 'z']
    ax : matplotlib axis object, optional
        If None, will make a new axis.

    Example
    -------
    To plot three 3d subplots, do something like the following:
    fig = plt.figure()
    ax = fig.add_subplot(131, projection='3d')
    plot3d(delayed[0,::1,:], labels=['x(t)','x(t-tau)','x(t-2tau)'], ax=ax)
    ax = fig.add_subplot(132, projection='3d')
    plot3d(delayed[1,::1,:], labels=['y(t)','y(t-tau)','y(t-2tau)'], ax=ax)
    ax = fig.add_subplot(133, projection='3d')
    plot3d(delayed[2,::1,:], labels=['z(t)','z(t-tau)','z(t-2tau)'], ax=ax)

    To have interactable 3d plots in Jupyter, use %matplotlib nbagg

    """
    # define 3d plot
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # if necessary, split up data, then plot it
    data_split = np.array_split(traj, ntraj)
    for n in range(ntraj):
        xs = data_split[n][:, 0]
        ys = data_split[n][:, 1]
        zs = data_split[n][:, 2]
        ax.plot3D(xs, ys, zs, alpha=1)

    # set labels
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
#     ax.set_title('[x,y,z] = ' +str(var*180/pi))
    plt.show()


def whisk_compare(Points, labels=None, ref=0, vdims=['score'], group='Bars',
                  tests=None):
    """Show datapoints as a whisker plot for averages of all distributions.

    Show the significance between ref and all other datapoints.

    Parameters
    ----------
    Points : list
        list  of arrays of points to be compared. As [array1, array2, ...].
    labels : list of strings
        Labels of the distributions, list of form ['points1','points2']
    ref : int, optional
        Which distribution to use as a reference to test significance of
        other distributions against.
    vdims : Holoviews dimension object, optional
        Should be of form ['label']. Gives the y-axis label.
    group : string, optional
        The group that the Bars object should belong to
    tests : list of string, optional
        TO BE IMPLEMENTED
        Which tests to use, in order of distributions (skipping the reference
        distribution). Should be a list of strings, with the following options:
        'ttest_rel' : paired t-test
        'ttest_ind' : independent t-test
        The list should be of length len(Points)-1
        Default test is pairwise test.

    Returns
    -------
    Holoviews Overlay
        Whisker plot

    """
    ndists = len(Points)
    if labels is None:
        labels = ['points'+str(i) for i in range(ndists)]

    # make box plots
    groups = []
    for i in range(ndists):
        npoints = len(Points[i])
        groups += [labels[i] for j in range(npoints)]
    allpoints = np.concatenate(Points)
    print len(groups)
    whisk = hv.BoxWhisker((groups, allpoints),
                          kdims=[' '], vdims=vdims)

    # calculate and plot significance
    sig_plot = hv.Overlay()
    rnge = np.arange(ndists, dtype=int)
    rnge = rnge[rnge != ref]
    count = 0
    for i in rnge:
        points1 = Points[ref]
        points2 = Points[i]
        sign = scipy.stats.wilcoxon(points1, points2)[1]
        fontsize = 20
        offset = 0.02
        print sign
        if sign < 0.0005:
            sig_text = '***'
            fontsize = 20
            offset = 0.03
        elif sign < 0.005:
            sig_text = '**'
        elif sign < 0.05:
            sig_text = '*'
        else:
            fontsize = 15
            offset = 0.03
            sig_text = 'n.s.'

        # plot significance
        maxpoint = max(allpoints)
        xref = ref+1
        xcom = 1+i
        y = maxpoint+.1*(ndists-count-3)
        sig_plot *= hv.Curve(zip([xref, xcom], [y, y]),
                             extents=(0, -maxpoint/3, 5, maxpoint+0.2),
                             group='significance')(style={'color': 'k'})
        sig_plot *= hv.Curve(zip([xref, xref], [y, y-.02]),
                             group='significance')(style={'color': 'k'})
        sig_plot *= hv.Curve(zip([xcom, xcom], [y, y-.02]),
                             group='significance')(style={'color': 'k'})
        xloc = (ref+2+i)/2
        yloc = maxpoint+offset+.1*(ndists-count-3)
        text = hv.Text(xloc, yloc, sig_text, fontsize=fontsize)
        sig_plot *= text
        count += 1

    return whisk*sig_plot


def bar_compare(Points, labels=None, ref=0, vdims=['score'], group='Bars',
                tests=None):
    """Show datapoints and bars for averages of all distributions in Points.

    Show the significance between ref and all other datapoints.

    Parameters
    ----------
    Points : list
        list  of arrays of points to be compared. As [array1, array2, ...].
    labels : list of strings
        Labels of the distributions, list of form ['points1','points2']
    ref : int, optional
        Which distribution to use as a reference to test significance of
        other distributions against.
    vdims : Holoviews dimension object, optional
        Should be of form ['label']. Gives the y-axis label.
    group : string, optional
        The group that the Bars object should belong to
    tests : list of string, optional
        TO BE IMPLEMENTED
        Which tests to use, in order of distributions (skipping the reference
        distribution). Should be a list of strings, with the following options:
        'ttest_rel' : paired t-test
        'ttest_ind' : independent t-test
        The list should be of length len(Points)-1
        Default test is pairwise test.

    Returns
    -------
    Holoviews Overlay
        Bars with points overlay

    """
    ndists = len(Points)
    if labels is None:
        labels = ['points'+str(i) for i in range(ndists)]

    # make bars
    data = [(labels[i], np.mean(Points[i])) for i in range(ndists)]
    bars = hv.Bars(data, kdims=[hv.Dimension(' ')], vdims=vdims)

    # make points
    points_plot = hv.Overlay()
    for i in range(ndists):
        y = Points[i]
        x = np.ones(len(y))*i+.5
        points_plot *= hv.Scatter(zip(x, y))

    # calculate and plot significance
    sig_plot = hv.Overlay()
    rnge = np.arange(ndists, dtype=int)
    rnge = rnge[rnge != ref]
    for i in rnge:
        points1 = Points[ref]
        points2 = Points[i]
        sign = scipy.stats.ttest_rel(points1, points2)[1]
        fontsize = 20
        offset = 0.175
        if sign < 0.0005:
            sig_text = 'p<0.0005'
        elif sign < 0.005:
            sig_text = '**'
        elif sign < 0.05:
            sig_text = '*'
        else:
            fontsize = 15
            offset = 0.2
            sig_text = 'n.s.'

        # plot significance
        maxpoint = max([max(points1), max(points2)])
        sig_plot *= hv.Curve(zip([ref+0.5, .5+i],
                                 [maxpoint+0.15*(ref+0.5-i),
                                  maxpoint+0.15*(ref+0.5-i)]),
                             extents=(0, 0, 2, 1),
                             group='significance')(style={'color': 'k'})
        sig_plot *= hv.Curve(zip([ref+0.5, ref+0.5],
                                 [maxpoint+0.15*(ref+0.5-i),
                                  maxpoint+0.15*(ref+0.5-i)-.05]),
                             extents=(0, 0, 2, 1),
                             group='significance')(style={'color': 'k'})
        sig_plot *= hv.Curve(zip([.5+i, .5+i],
                                 [maxpoint+0.15*(ref+0.5-i),
                                  maxpoint+0.15*(ref+0.5-i)-.05]),
                             extents=(0, 0, 2, 1),
                             group='significance')(style={'color': 'k'})
        xloc = (ref+1.5+i)/2
        yloc = maxpoint+offset+.15*(ref-0.5-i)
        text = hv.Text(xloc, yloc, sig_text, fontsize=fontsize)
        sig_plot *= text

    return bars*points_plot*sig_plot


def plotspikes(spiketimes, yoffset, dt, text_offset=None):
    """Plot the spikes in spiketimes.

    Also indicates if there are more than one spike within distance dt.

    Parameters
    ----------
    spiketimes : array
        Array of spiketimes
    yoffset : array
        The y-offset for plotting spikes
    dt : float
        Distance dt at which spikes should be indicated as a group
    text_offset : array, optional
        If given (as [dx,dy]), this gives the offset of the text indicating
        the number of spikes, relative to the first spike in a series.
        If None, will set it to [-dt/2, 0]
    Returns
    -------
    holoviews Scatter object
        Shows the spikes. Can be interfaced with as any normal Scatter object.

    """
    # get groups of spikes
    small_diffs = np.diff(spiketimes) <= dt
    start_ends = np.diff(np.concatenate([[0], small_diffs, [0]]))
    starts = np.where(start_ends == 1)[0]
    ends = np.where(start_ends == -1)[0]
    numbers = ends-starts + 1  # number of spikes per start
    # plot spikes
    fig = hv.Scatter(zip(spiketimes, np.ones(len(spiketimes))*yoffset))

    # plot spike numbers
    if text_offset is None:
        dx, dy = [-dt/2, 0]
    else:
        dx, dy = text_offset
    for i, start in enumerate(starts):
        fig *= hv.Text(spiketimes[start]+dx, yoffset+dy, str(numbers[i]))
    return fig


def hist(x, bins=10, group='hist'):
    """Make a Holoviews histogram for the data provided.

    Parameters
    ----------
    x : array
        data
    bins : int or array, optional (default: 10)
        bins as normaly provided to numpy.histogram

    Returns
    -------
    Holoviews.histogram
        Holoviews histogram object

    """
    counts, edges = np.histogram(x, bins)
    return hv.Histogram(counts, edges, group=group)


def ScaleBars(x=0, y=0, scalex=1, scaley=1, labeldx=0.035,
              labeldy=2, labelx='x', labely='y', w=1, color='k'):
    """Make scalebars using HoloViews' Curve object, and puts them at (x,y).

    Parameters
    ----------
    x : float (0)
        The x start position
    y : float (0)
        The y start position
    labeldx,labeldy : floats
        The offsets of the x and y labels compared to the scale bars ends.
        The offset is away from the middle
    scalex : float (1)
        The scale of the horizontal scalebar
    scaley : float (1)
        The scale of the vertical scalebar
    labelx : string ('x')
        The label for the x scale
    labely : string ('y')
        The label for the y scale
    w : float (1)
        Width of the bars
    color : string ('k')
        Color of scale

    Returns
    -------
    A holoviews Curve object with both the horizontal and vertical
    scalebars

    """
    # define horizontal scalebar
    bar_x = hv.Curve(zip([x, x+scalex], [y, y]))(style={'color': color,
                                                        'linewidth': w})

    # define horizontal label
    label_x = hv.Text(x+scalex/2, y-labeldx, labelx)(style={'color': color})

    # define vertical scalebar
    bar_y = hv.Curve(zip([x, x], [y, y+scaley]))(style={'color': color,
                                                        'linewidth': w})

    # define vertical label
    label_y = hv.Text(x-labeldy, y+scaley/2, labely,
                      rotation=90)(style={'color': color})

    # return overlay
    return bar_x*label_x*bar_y*label_y

def plot_bounds_z(D, offset=(0,0,0), length=1, group='Curve'):
    ''' Plots the bounding box, for a given offset in z direction.

    For the offset in z it will be calculated what the net size of the bounding box is

    Parameters
    ----------
    D : array
        Decoding weights, 3D
    offset : list/tuple/array
        [x,y,z] offset of the bounding box
    length : float
        length of the bounding box vertices

    Returns
    -------
    HoloViews overlay
        The boundary box, offset by 'offset'
    HoloViews overlay
        The vectors determining the boundary box, offset by ''offset'
    '''
    # get offsets
    x, y, z = offset

    # get thresholds for 2D box at height 0
    D2 = D[:2, :]
    Omeg = np.dot(D2.T, D2) + np.identity(N)*beta
    T = np.diag(Omeg)
#     T = np.sqrt(2)*T

    # plot projection vectors and bounding box
    angle = np.pi/2
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    projectVs = hv.Overlay()
    bounds = hv.Overlay()
    for i in range(N):
        v = np.copy(D2[:, i])
        norm = np.linalg.norm(v)
        v/= norm
        scale = T[i]/norm**1-z
        if scale<0: scale=0
        prop = scale*norm**1/T[i]
        v*= scale
        projectVs *= hv.Curve(zip([x, x+v[0]], [y, y+v[1]]), group=group)
        v90 = np.dot(rotation, v)
        v90*= length*prop/np.linalg.norm(v90)
        bounds *= hv.Curve(zip([x+v[0]+v90[0], x+v[0]-v90[0]],
                               [y+v[1]+v90[1], y+v[1]-v90[1]]), group=group)
    return bounds, projectVs

def plot_bounds(D, T, beta=0, offset=(0,0), length=1, widths=None):
    ''' Plots spike coding network bounding box in 2D.

    Parameters
    ----------
    D : array
        Decoding weights
    T : array
        array of spiking thresholds
    beta : float
        network cost parameter
    offset : list/tuple/array
        [x,y] offset of the bounding box
    length : float
        length of the bounding box vertices
    widths : array
        Array of linewidths for each vertice

    Returns
    -------
    HoloViews overlay
        The boundary box, offset by 'offset'
    HoloViews overlay
        The vectors determining the boundary box, offset by ''offset'
    '''
    # infer some Parameters
    N = D.shape[1]
    if widths is None: widths = np.ones(N)*2

    # get offsets
    x, y = offset

    # get thresholds
    Omeg = np.dot(D.T, D) + np.identity(N)*beta
    T = np.diag(Omeg)/2

    # plot projection vectors and bounding box
    angle = np.pi/2
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    projectVs = hv.Overlay()
    bounds = hv.Overlay()
    for i in range(N):
        v = np.copy(D[:, i])
        norm = np.linalg.norm(v)
        v/= norm**2
        v*= T[i]
        projectVs *= hv.Curve(zip([x, x+v[0]], [y, y+v[1]]))
        v90 = np.dot(rotation, v)
        v90*= length/np.linalg.norm(v90)
        style = {'line_width':widths[i]}
        bounds *= hv.Curve(zip([x+v[0]+v90[0], x+v[0]-v90[0]],
                               [y+v[1]+v90[1], y+v[1]-v90[1]]))(style=style)

    return bounds, projectVs

def plot_bounds_z(D,  T, beta=0, offset=(0,0,0), length=1, group='Curve',
                  widths=None, alphas=None):
    ''' Plots the bounding box, for a given offset in z directionself.

    This function is specifically for bounding boxes which are cone-shaped.

    For the offset in z it will be calculated what the net size of the
    bounding box is.

    Parameters
    ----------
    D : array
        Decoding weights, 3D
    T : array
        array of spiking thresholds
    beta : float
        network cost parameter
    offset : list/tuple/array
        [x,y,z] offset of the bounding box
    length : float
        length of the bounding box vertices
    widths : array
        Array of linewidths for each vertice
    alphas : array
        Array of alphas for each vertice

    Returns
    -------
    HoloViews overlay
        The boundary box, offset by 'offset'
    HoloViews overlay
        The vectors determining the boundary box, offset by ''offset'
    '''
    # infer some parameters
    N = D.shape[1]
    if widths is None: widths = np.ones(N)*2
    if alphas is None: widths = np.ones(N)

    # get offsets
    x, y, z = offset

    # get thresholds for 2D box at height 0
    D2 = D[:2, :]
    Omeg = np.dot(D2.T, D2) + np.identity(N)*beta
    T = np.diag(Omeg)

    # plot projection vectors and bounding box
    angle = np.pi/2
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    projectVs = hv.Overlay()
    bounds = hv.Overlay()
    for i in range(N):
        v = np.copy(D2[:, i])
        norm = np.linalg.norm(v)
        v/= norm
        scale = T[i]/norm**1-z
        if scale<0: scale=0
        prop = scale*norm**1/T[i]
        v*= scale
        projectVs *= hv.Curve(zip([x, x+v[0]], [y, y+v[1]]), group=group)
        v90 = np.dot(rotation, v)
        v90*= length*prop/np.linalg.norm(v90)
        style = {'line_width':widths[i], 'alpha':alphas[i]}
        bounds *= hv.Curve(zip([x+v[0]+v90[0], x+v[0]-v90[0]],
                               [y+v[1]+v90[1], y+v[1]-v90[1]]),
                               group=group)(style=style)
    return bounds, projectVs

def animate_error_box(D, T, beta, E, x, o, Tstart=0, Tend=None,
                      boundlength=0.05, trail_length=40, step_size=10,
                      spike_tau=1., dt=0.01):
    """For spike coding networks (SCNs), animates the error inside bounding box.

    This function is specifically for cone-shaped bounding boxes.

    Parameters
    ----------
    D : array
        2D array which is the SCN decoding matrix
    T : array
        1D array with SCN spiking thresholds
    beta : float
        SCN cost parameter
    E : array
        2D array of the error
    x : array
        2D array of the actual stimulus
    o : array
        N by nT array of 0s and 1s indicating spikes
    Tstart : int
        Starting timestep
    Tend : int
        Final timestep (if None, will use final timestep)
    boundlength : float
        How long to make each bounding edge
    trail_length : int
        How many timesteps to use for the error trail
    step_size : int
        How many timesteps to skip for each frame
    spike_tau/dt : floats
        Determine time constant on increased line thickness with spikes

    Output
    ------
    Holoviews HoloMap
    """
    # get some parameters
    if Tend is None: Tend = E.shape[1]
    framenums = range(Tstart, Tstart+Tend,step_size)

    # turn spikes into line ticknesses
    s = np.zeros(o.shape)
    for i in range(o.shape[1]-1):
        s[:, i+1]=s[:, i]+dt*(-s[:, i]/spike_tau+o[:, i]/dt)

    # based on spiking, determine bound widths (so for a spike, a cell's
    # bound changes size)
    widths = {f: np.ones(D.shape[1])*2 for f in framenums}
    alphas = {f: np.ones(D.shape[1]) for f in framenums}
    for f in framenums:
        widths[f] += s[:, f]*2
        alphas[f] += s[:, f]
        alphas[f]/=alphas[f].max()

    # Define the animation frames
    frames = {f: hv.Scatter(zip([E[0, f]], [E[1, f]]))
                 for f in framenums}
    frames = {f: frames[f]*hv.Curve(E[:2, f+1-trail_length:f+1].T)
                 for f in framenums}
    frames = {f: frames[f]*hv.VLine(x[0, f])*hv.HLine(x[1, f])
                 for f in framenums}
    frames = {f: frames[f]*plot_bounds_z(D, T, beta, (0, 0, E[2, f]),
                                         length=boundlength,
                                         widths=widths[f],
                                         alphas=alphas[f])[0]
                 for f in framenums}
    # return animation
    return hv.HoloMap(frames)*hv.Scatter(zip([0], [0]), group='origin')
