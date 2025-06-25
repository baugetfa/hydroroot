Solutes and water transport model
---------------------------------

The following lines are taken or inspired from Bauget *et al.* 2023 [bauget2023]_.

In this solver, solute transport equations are added to the hydraulic model. This change leads to a
major difference in the resolution of the equation system on the whole
RSA. Thus, the hydraulic architecture can no longer be modeled by an
analogous electrical network (Boursiac *et al.*, 2022a) and the coupled
solute and water transport equations had to be solved in a matrix form.

The root is discretized in cylindrical elementary volumes (Figure 1), considered as representative elementary
volumes (REV) of diameter, :math:`d` and length :math:`l`. The local transport equations
described below are considered in each REV. Each REV can be seen as two
concentric media: the peripheral tissues (from epidermis to pericycle)
through which radial transports happen, and a central medium (stele with
xylem vessels) where the sap flows axially.

In the following model two type of solutes are considered:
 * the permeating solutes, that are able to be transported through the peripheral tissues
 * the non-permeating solutes, that are not able to pass through these tissues

In the following, the subscript S refer to the permeating solutes and np to the non-permeating ones.

Water transport
~~~~~~~~~~~~~~~

The axial flow is unchanged, but a term due to osmotic potential difference is added to the
radial water flow rate, that can be modeled as follows:

.. math:: j = k\left( \Delta\Psi_{H} + \Delta\Psi_{np} + \sigma \Delta\Psi_{S} \right)S

where :math:`j` is the local radial water flow rate, :math:`k` the radial hydraulic conductivity.
:math:`\Delta\Psi_{H}`, :math:`\Delta\Psi_{np}` and :math:`\Delta\Psi_{S}` are the hydrostatic water
potential difference between the bathing solution and the xylem sap, the osmotic water potential difference due to the
non-permeating solute and the osmotic water potential difference due to the permeating solutes, respectively. σ is the effective reflection coefficient. :math:`S`
(:math:`S = \pi\ d\ l`) is the external surface area of the REV. Expressing the water potentials, we obtain:

.. math:: j = k\left( P_{e} - P - \pi_{np}^{ext} + \pi_{np} - \sigma RT\left( C_{e} - C \right) \right)S

:math:`P_e` and :math:`P` are the hydrostatic pressure of the bathing solution and within the xylem vessels, respectively.
:math:`\pi_{np}^{ext}` and :math:`\pi_{np}` are the np contribution to the osmotic pressure of the bathing solution and
inside the xylem vessels, respectively. Note, :math:`\pi_{np}` is most of the time equal to 0 because these molecules can
not penetrate the root tissues. However, in the particular case of a cut and flow experiment [boursiac2022], they can enter
the xylem vessels through the cut tips of the root.
:math:`C_e` and :math:`C` are the permeating solute concentration in the bathing solution and in the xylem vessels,
respectively. :math:`R` is the gas constant, :math:`T` the temperature (set to 298°K here).

The axial sap flow rate was modeled with a Hagen-Poiseuille’s law type:

.. math:: J = K(\mu)\frac{\Delta P}{l}

where :math:`J` is the axial sap flow rate, :math:`K(μ)` the axial conductance that
depends on sap viscosity, :math:`ΔP` the local pressure difference between two REVs, :math:`l`
the length of the REV. The axial conductance is inversely proportional
to the fluid viscosity :math:`μ`, as illustrated for example by the
conductance in a cylindrical capillary of radius :math:`r`:
:math:`K(\mu) = \pi r^{4}/(8\mu)`. The sap is commonly considered having
water viscosity. However, when np penetrates the root vasculature in
cut-and-flow experiments, the viscosity may significantly increase with the
np concentration and may have to be taken into account.

Solute transport
~~~~~~~~~~~~~~~~

The solute transport model is inspired by the model of Fiscus (Fiscus,
1977). The radial solute flux is modeled as follows:

.. math:: j_{s} = \left\lbrack J_{s}^{*} - P_{s}\left( C - C_{e} \right) \right\rbrack S

where :math:`j_s` (mol.s\ :sup:`-1`) is the radial solute flux, :math:`J_s^*`
(mol.m\ :sup:`-2`.s\ :sup:`-1`) is the solute active uptake rate and
`P_s` (m.s\ :sup:`-1`) is the radial permeability of the root
peripheral tissues. As above, :math:`C` and :math:`C_e` correspond to the
solute concentration in the xylem vessels and in the bathing solution,
respectively. :math:`S` is the external surface area of the REV.

Since solutes are transported along xylem vessels by advection, axial
solute flux can be expressed as

 .. math:: J_{s} = JC

When np penetrates the root in cut-and-flow experiments, its axial flux
has the same form:

 .. math:: J_{np} = JC_{np}

where :math:`C_{np}` is the np concentration in the xylem vessels.


Notes on the numerical resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Boundary conditions:**

At the base, we consider a Dirichlet boundary condition for the pressure
and Neumann boundary condition for the concentration:

.. math:: P_{1} = P_{atm}

.. math:: \frac{\partial C}{\partial x} = 0

in other words, root base is at atmospheric pressure and solute and np
concentrations at the outlet are the same as in the first node.

**Discretization of the transport equations:**

The root system architecture (RSA) is represented by a Multiscale Tree
Graph (MTG) where the nodes are the discretized representation of
representative elementary volumes (REV). In the following, REV are
numbered from root base to tip.

In each REV, mass conservation is independently applied for water,
permeating solutes and non-permeating solutes. that gives for a REV numbered i:

.. math:: \left\{
    \begin{array}{l}
		J_{i} & = \sum_{j}^{}J_{j} + k_{i}\left\lbrack P_{e} - P_{i} - \pi_{np}^{ext} + {\pi_{np}^{}}_{i}- \sigma RT\left( C_{e} - C_{i} \right) \right\rbrack S_{i} \\
		J_{i}Χ_{i} & = \sum_{j}^{}{J_{j}Χ_{j}} + \left\lbrack J_{s}^{*} - P_{s}\left( C_{i} - C_{e} \right) \right\rbrack S_{i} \\
		J_{i}{Χ_{np}}_{i} &= \sum_{j}^{}{J_{j}{Χ_{np}}_{j}}
    \end{array}
	\right.

where :math:`P_e, \pi_{np}^{ext} \text{and}\ C_e` are the hydrostatic
pressure, the osmotic pressure due to the np and the solute
concentration of the external medium, respectively. The variables with
subscript *i* refer to the REV *i*: :math:`k_i` is the radial hydraulic
conductivity, :math:`P_i` the sap hydrostatic pressure,
:math:`{\pi_{np}}_{i}` the osmotic pressure corresponding to the
local np concentration in sap (:math:`{C_{np}}_{i})`, :math:`C_i` the
solute concentration in sap. :math:`S_i` is the surface area of the REV.
:math:`J_i` is the outgoing xylem sap flow and :math:`J_j` is the xylem sap flow
coming from node :math:`j`, which stands for one of the children of node :math:`i`.
This can be the next node on root axis, or any first node of a lateral
root branched on node :math:`i`. :math:`k_i` is the radial conductivity. :math:`σ` is the
effective reflection coefficient, :math:`R` the gas constant, and :math:`T` the
temperature. :math:`J_s^*` is the solute active uptake rate and :math:`P_s` is
the radial permeability of the root peripheral tissues. :math:`J_i\ \text{and}\ J_j` are proportional to the local pressure gradient as follows:

.. math:: J_{i} = K_{i}\frac{\left( P_{i + 1} - P_{i} \right)}{l_{i}}

.. math:: J_{j} = K_{j}\frac{\left( P_{j} - P_{i} \right)}{l_{j}}

with :math:`K` being the axial conductance, :math:`l` the REV length according to
the subscript.

:math:`Χ_i` is the solute concentration according to the sap flow direction,
with
:math:`Χ_{i} = \theta_{i}C_{i} + \left( 1 - \theta_{i} \right)C_{i + 1}`,
:math:`\theta_i` being a factor that depends on flow direction:
:math:`\theta_{i} = 1` if :math:`P_{i} > P_{i - 1}` and
:math:`\theta_{i} = 0` if :math:`P_{i} < P_{i - 1}`. Χ\ :sub:`j` is the
solute concentration flowing between node i and its child j:
:math:`Χ_{j} = \theta_{j}C_{j} + \left( 1 - \theta_{j} \right)C_{i}`,
with :math:`θ\ j` following the same rules as :math:`θ\ i` according to
:math:`\left( P_{j} - P_{i} \right)`. :math:`{Χ_{np}}_{i}` is the same
variable for the np concentration.

The system can be transformed as follows:

.. math:: \left\{
    \begin{array}{l}
		G_{w_{i}} = J_{i} - \sum_{j}^{}J_{j} - k_{i}\left\lbrack P_{e} - P_{i} - \pi_{np}^{ext} + {\pi_{np}^{}}_{i} - \sigma RT\left( C_{e} - C_{i} \right) \right\rbrack S_{i} = 0 \\
		G_{s_{i}} = J_{i}Χ_{i} - \sum_{j}^{}{J_{j}Χ_{j}} - \left\lbrack J_{s}^{*} - P_{s}\left( C_{i} - C_{e} \right) \right\rbrack S_{i} = 0 \\
		{G_{np}}_{i} = J_{i}{Χ_{np}}_{i} - \sum_{j}^{}{J_{j}{Χ_{np}}_{j}} = 0
    \end{array}
	\right.

The purpose is to solve the mass balance equation for the three components water
(w), permeating solutes (s) and non-permeating solute (np) i.e. to solve on each
grid block i:

.. math::
   G_{i} = 0\ \text{with}\ G_{i} = \begin{pmatrix}
   G_{{w}_{i}} \\
   G_{s_{i}} \\
   G_{np_{i}} \\
   \end{pmatrix}

or considering the whole grid system:

.. math::

   G = \left(G_{w_1}G_{s_1}G_{np_1}, \cdots\ G_{w_i}G_{s_i}G_{np_i}, \cdots\ G_{w_n}G_{s_n}G_{np_n} \right)

The dimension of G is 3N, N grid blocks for G\ :sub:`w`, G\ :sub:`s` and G\ :sub:`np`.

The system may be expressed according to the three unknowns: the hydrostatic pressure P,
the permeating solute C\ :sub:`s` and the non-permeating solute C\ :sub:`np`.
The unknowns are stored in a 3N vector Y, N elements for each :

.. math::

   Y = \left(P_{w_1}C_{s_1}C_{np_1}, \cdots\ P_{w_i}C_{s_i}C_{np_i}, \cdots\ P_{w_n}C_{s_n}C_{np_n} \right)

Now to solve the system a Newton-Raphson is used leading to:

.. math:: J\ dY = - G

dY is a 3N vector containing alternatively dP, dC\ :sub:`s` and dC\ :sub:`np`:

.. math::
	dY = (\cdots, dP_{i-1}, d{C_s}_{i-1}, d{C_{np}}_{i-1}, dP_{i}, d{C_s}_{i}, d{C_{np}}_{i}, dP_{i+1}, d{C_s}_{i+1}, d{C_{np}}_{i+1}, \cdots)

J is the Jacobian of G according to the three unknowns:

.. math::

	J = \begin{pmatrix}
		 & \vdots & \vdots & \vdots & & \vdots & \vdots & \vdots & & \vdots & \vdots & \vdots & \\
		\cdots  & \frac{\partial G_{w_i}}{\partial P_{i - 1}} & \frac{\partial G_{w_i}}{{\partial C_s}_{i - 1}} & \frac{\partial G_{w_i}}{{\partial C_{np}}_{i - 1}} & &
		\frac{\partial G_{w_i}}{\partial P_{i}} & \frac{\partial G_{w_i}}{{\partial C_s}_{i}} & \frac{\partial G_{w_i}}{{\partial C_{np}}_{i}} &  &
		\frac{\partial G_{w_i}}{\partial P_{i + 1}} & \frac{\partial G_{w_i}}{{\partial C_s}_{i + 1}} & \frac{\partial G_{w_i}}{{\partial C_{np}}_{i + 1}} & \cdots  \\
		\cdots  & \frac{\partial G_{s_i}}{\partial P_{i - 1}} & \frac{\partial G_{s_i}}{{\partial C_s}_{i - 1}} & \frac{\partial G_{s_i}}{{\partial C_{np}}_{i - 1}} & &
		\frac{\partial G_{s_i}}{\partial P_{i}} & \frac{\partial G_{s_i}}{{\partial C_s}_{i}} & \frac{\partial G_{s_i}}{{\partial C_{np}}_{i}} & &
		\frac{\partial G_{s_i}}{\partial P_{i + 1}} & \frac{\partial G_{s_i}}{{\partial C_s}_{i + 1}} & \frac{\partial G_{s_i}}{{\partial C_{np}}_{i + 1}} & \cdots  \\
		\cdots  & \frac{\partial G_{np_i}}{\partial P_{i - 1}} & \frac{\partial G_{np_i}}{{\partial C_s}_{i - 1}} & \frac{\partial G_{np_i}}{{\partial C_{np}}_{i - 1}} & &
		\frac{\partial G_{np_i}}{\partial P_{i}} & \frac{\partial G_{np_i}}{{\partial C_s}_{i}} & \frac{\partial G_{np_i}}{{\partial C_{np}}_{i}} & &
		\frac{\partial G_{np_i}}{\partial P_{i + 1}} & \frac{\partial G_{np_i}}{{\partial C_s}_{i + 1}} & \frac{\partial G_{np_i}}{{\partial C_{np}}_{i + 1}} & \cdots  \\
		 & \vdots & \vdots & \vdots & & \vdots & \vdots & \vdots & & \vdots & \vdots & \vdots &
   \end{pmatrix}

Most of the non diagonal terms of J are zero.

Finally, the linear system :math:`J\ dY = - G` is solved by a direct LU decomposition. This is not the most efficient in term of run time but this is the most robust.
