import holoviews as hv
import numpy as np


def hist(x, bins=10):
    """"Makes a Holoviews histogram for the data provided.

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
    return hv.Histogram(counts, edges)

def ScaleBars(x=0,y=0,scalex=1,scaley=1,labeldx = 0.035,labeldy=2,labelx='x',labely='y',w=1,color='k'):
    ''' Makes scalebars using HoloViews' Curve object, and puts them at
    (x,y).

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

    '''
    # define horizontal scalebar
    bar_x = hv.Curve(zip([x,x+scalex],[y,y]))(style={'color':color,'linewidth':w})

    # define horizontal label
    label_x = hv.Text(x+scalex/2,y-labeldx,labelx)(style={'color':color})

    # define vertical scalebar
    bar_y = hv.Curve(zip([x,x],[y,y+scaley]))(style={'color':color,'linewidth':w})

    # define vertical label
    label_y = hv.Text(x-labeldy,y+scaley/2,labely,rotation=90)(style={'color':color})

    # return overlay
    return bar_x*label_x*bar_y*label_y
