#!/usr/bin/env python
# coding: utf-8

# ## Example of direct simulation using an existing architecture
# 
# This is the notebook version of the example in read the doc.
# The following lines present a small example of simulation the sap flux from an Arabidopsis de-topped root plunged in a hydroponic solution at a hydrostatic pressure of 0.4 Mpa when its  base is at the atmospheric pressure.

# In[1]:


import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['../src'])


# In[2]:


from hydroroot.display import plot
from hydroroot.read_file import read_archi_data
from hydroroot.main import hydroroot_flow, root_builder


# # Read the architecture file and give the architecture properties:
# - radius of the primary in meter
# - a decrease factor between root orders

# In[3]:


df = read_archi_data('data/plant-01.txt')
r_pr = 7e-05 # radius of the primary in m
beta = 0.7 # decrease factor at each order


# # Building the MTG from the file

# In[16]:


g, primary_length, total_length, surface, seed = root_builder(df=df, segment_length=1.0e-4, order_decrease_factor = beta, ref_radius = r_pr)


# # Axial and radial conductance data versus distance to tip

# In[17]:


k_radial_data=([0, 0.2],[30.0,30.0])
K_axial_data=([0, 0.2],[3.0e-7,4.0e-4])


# # Calculation
# Flux and equivalent conductance calculation, for a root in an external hydroponic medium at 0.4 MPa, its base at 0.1 MPa, and with the conductances set above.

# In[18]:


g, keq, jv = hydroroot_flow(g, psi_e = 0.4, psi_base = 0.1, axial_conductivity_data = K_axial_data, radial_conductivity_data = k_radial_data)


# In[19]:


print('equivalent root conductance (microL/s/MPa): ',keq, 'sap flux (microL/s): ', jv)


# # Display the local water uptake heatmap in 3D

# In[20]:


get_ipython().run_line_magic('gui', 'qt')
plot(g, prop_cmap='j') # j is the radial flux in ul/s


# You may change the property to display to the hydrostatic pressure inside the xylem vessels for instance

# In[ ]:


plot(g, prop_cmap='psi_in')


# You may change the radial conductivity and see the impact on the water uptake

# In[ ]:


k_radial_data=([0, 0.2],[300.0,300.0])
g, keq, jv = hydroroot_flow(g, psi_e = 0.4, psi_base = 0.1, axial_conductivity_data = K_axial_data, radial_conductivity_data = k_radial_data)
print('sap flux (microL/s): ', jv)
plot(g, prop_cmap='j')


# Or the axial conductance

# In[12]:


k_radial_data=([0, 0.2],[30.0,30.0])
K_axial_data=([0, 0.2],[3.0e-7,1.0e-4])
g, keq, jv = hydroroot_flow(g, psi_e = 0.4, psi_base = 0.1, axial_conductivity_data = K_axial_data, radial_conductivity_data = k_radial_data)
print('sap flux (microL/s): ', jv)
plot(g, prop_cmap='j')


# In[ ]:




