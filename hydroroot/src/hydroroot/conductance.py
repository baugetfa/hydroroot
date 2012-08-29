from math import pi
from collections import defaultdict

from openalea.mtg import *
from openalea.mtg import algo

import numpy as np
from scipy.interpolate import UnivariateSpline
import pylab

def poiseuille(radius, length, viscosity=1e-3):
    """
    Compute a conductance of a xylem element based on their radius and length.
    
    Parameters
    ==========
    radius : float (m)
        radius of a xylem tube

    length: float (m) 
        length of a xylem element

    viscosity : float (Pa.s)
        dynamic viscosity of the liquid
    
    The poiseuille formula is:
        :math:` conductance = \frac{\pi r^4}{8 \mu L }` 
        with :math:`r` the radius of a pipe, 
        :math:`\mu` the viscosity of the liquid,
        :math:`L` the length of the pipe.
        
    .. seealso:: http://en.wikipedia.org/wiki/Poiseuille
    """
    return pi*(radius**4) / ( 8 * viscosity * length)


def compute_k(g, k0 = 0.1):
    """ Compute the radial conductances (k) of each segment of the MTG.

    Parameters
    ==========

        - `g` - the RSA
        - `k0` - the radial conductance for one element of surface
        - `length` - the length of a segment

    """
    radius = g.property('radius')
    length = g.property('length')
    k = dict( (vid,radius[vid]*2*pi*length[vid]*k0) for vid in g.vertices(scale=g.max_scale()))
    g.properties()['k'] = k
    return g


def compute_K(g, nb_xylem=5, radius_scale = 1/10.):
    """ Compute the axial conductances (K) in a MTG according to Poiseuille law. 

    The conductance depends on the radius of each xylem pipe, the number of xylem pipes,
    and on the length of a root segment.

    radius_scale allows to compute the radius of a xylem pipe from the radius of a root segment.
    """

    radius = g.property('radius_xylem')
    if not radius:
        full_radius = g.property('radius')
        radius = dict( (vid,r*radius_scale) for vid,r in full_radius.iteritems())
    nb_xylem = g.property('nb_xylem')
    length= g.property('length')
    if not nb_xylem:
        nb_xylem = defaultdict(lambda : 5)
    K = dict((vid, nb_xylem[vid]*poiseuille(radius[vid], length[vid])) 
                for vid in g.vertices(scale=g.max_scale()))
    g.properties()['K'] = K
    return g

def fit_property(g, x, y, prop_in, prop_out, s=3.): 
    """ Fit a 1D spline from x, y data.

    Retrieve the values from the prop_in of the MTG.
    And evaluate the spline to compute the property 'prop_out'
    """

    spline = UnivariateSpline(x, y, s=3)
    keys = g.property(prop_in).keys()
    x_values = np.array(g.property(prop_in).values())

    y_values = spline(x_values)

    g.properties()[prop_out] = dict(zip(keys,y_values))

    xx = np.linspace(0,1,1000)
    yy = spline(xx)

    pylab.clf()
    pylab.plot(x, y)
    pylab.plot(xx, yy)
    pylab.show()

    print 'Update figure ', yy.min(), yy.max()
    return g


def fit_K(g, s=0.):
    x = np.linspace(0.,1.,100)
    y = np.linspace(50, 500, 100)+100*np.random.random(100)-50

    if s == 0.:
        s = None
    fit_property(g,x,y,'relative_position', 'K', s=s)


    return g
