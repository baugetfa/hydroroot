from __future__ import absolute_import

import numpy as np
from hydroroot.length import fit_law
from hydroroot import radius, markov, flux, conductance


def hydroroot(
    n=1500,
    delta=2.e-3,
    beta=0.25,
    order_max=5,
    segment_length=1e-4,
    nude_length = 0.02,
    seed = 2,
    ref_radius = 1e-4,
    order_decrease_factor = 0.7,
    k0 = 300,
    Jv = 0.1,
    psi_e = 0.4,
    psi_base = 0.1,
    length_data=None,
    axial_conductivity_data=None,
    radial_conductivity_data=None ):

    """ Simulate a root system and compute global conductance and flux.

    Parameters
    ==========
    
    Returns
    =======
        - surface 
        - volume
        - Keq
        - Jv
        
    Example
    =======
    
    """
    xl, yl = length_data
    length_law = fit_law(xl, yl, scale=segment_length)
    
    xa, ya = axial_conductivity_data
    ya = list(np.array(ya) * (segment_length / 1e-4))
    axial_conductivity_law = fit_law(xa, ya)
    
    xr, yr = radial_conductivity_data
    radial_conductivity_law = fit_law(xr, yr)
    
    # compute the architecture
    nb_nude_vertices = int(nude_length / segment_length)
    branching_delay = int(delta / segment_length)

    g = markov.markov_binary_tree(
        nb_vertices=n,
        branching_variability=beta,
        branching_delay=branching_delay,
        length_law=length_law,
        nude_tip_length=nb_nude_vertices,
        order_max=order_max,
        seed=seed)
    
    # compute radius property on MTG
    g = radius.ordered_radius(g, ref_radius=ref_radius, order_decrease_factor=order_decrease_factor)

    # compute length property and parametrisation
    g = radius.compute_length(g, segment_length)
    g = radius.compute_relative_position(g)
    
    # Compute K using axial conductance data
    g = conductance.fit_property_from_spline(g, axial_conductivity_law, 'position', 'K')
    
    g, surface = radius.compute_surface(g)
    g, volume = radius.compute_volume(g)

    # Compute the flux
    # TODO: Use radial conducatnce law when available 
    
    g = conductance.fit_property_from_spline(g, radial_conductivity_law, 'position', 'k0')
    g = conductance.compute_k(g, k0='k0')
    
    # TODO: return Keq base and Jv
    g = flux.flux(g, Jv, psi_e, psi_base, invert_model=True)
    
    Keqs = g.property('Keq')
    v_base = g.component_roots_at_scale_iter(g.root, scale=1).next()
    
    Keq = Keqs[v_base]
    Jv_global = Keq * (psi_e - psi_base)

    return g, surface, volume, Keq, Jv_global