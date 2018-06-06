.. ref-model:

-------------------
Model and Theory 
-------------------


Let's take a look at the basic modeling concepts on which Pomato is build.



Power Flow
^^^^^^^^^^

Pomato uses the established DC load flow (DCLF) approximation for linear power flow equations in transmission systems. DCLF leverages the fact, that transmission systems usually have negligible active power losses and nodal voltage levels close unity. This allows to create a matrix of *Power Transfer Distribution Factors* (PTDFs) that map the net-injection at each bus to a flow on each line. 

.. math::
    PTDF = (B_d A)(A' B A)^{-1}

where :math:`A` is the :math:`l \times n` incidence matrix of the network with :math:`a_{(ij)} = 1` if line :math:`i` starts at bus :math:`j` and :math:`a_{(ij)} = -1` if line :math:`i` ends at bus :math:`j`, and :math:`B_d` is the :math:`l \times l`-diagonal matrix with the line susceptances on the diagonal. Because the physical interpretation of the power flow through the lines results from voltage angle *differences* between the nodes, the voltage angle at one bus has to be fixed (so called *slack bus*). Only then the matrix inversion to create the PTDF matrix is possible.

The vector of flows on each line based on the vector of net-injections at each matrix is then given by:

.. math::
    Flow = PTDF \cdot Inj


N-1 Outage Calculation
^^^^^^^^^^^^^^^^^^^^^^

Line contingencies can be calculated via a *Line Outage Distribution Factor* (LODF) matrix that provides insights on how the outage of one line (:math: `o`) affects the flow on all other lines (:math: `l`) in the network (cf.: [GAO2009]_).

.. math::
    LODF_{l,o} = \begin{cases} -1 &\text{if } l = o \\ \frac{(A_{(o,*)} \cdot PTDF_{(l,*)})}{(1 - A_{(o,*)} \cdot PTDF_{(o,*)})} &\text{else} \end{cases}


.. [GAO2009] Guo, Jiachun, et al. "Direct calculation of line outage distribution factors." IEEE Transactions on Power Systems 24.3 (2009): 1633-1634.

Based on the LODF matrix we create :math: `l` PTDF matrices for all possible line contingencies in the system.


Phase Shifting Transformers
^^^^^^^^^^^^^^^^^^^^^^^^^^^


A phase shifting transformer (PST) actively influences the power flow on a line by manipulating the voltage angle and thereby making a line more or less susceptible for power flow. PSTs change the voltage angle by :math: `α_l` for a line :math: `l` between nodes :math: `i` and :math: `j`.

We can derive the *Phase Shift Distribution Factor* PSDF matrix as:

.. math::
    PSDF = B_d - PTDF (B_d A)'

which computes the change of flows on each line per change of one unit voltage angle (radian) on each line. So when a line is systematically overloaded, a phase shift can be employed, to systematically make a line less susceptible to flows (if PST is available). The PTDF matrix can then be updated as:

.. math::
    PTDF + PSDF \cdot \alpha \cdot PTDF


Zonal PTDF and GSK
^^^^^^^^^^^^^^^^^^

While a nodal representation is useful, in reality usually zones/countries are usually the locational reference rather than individual grid nodes. It is assumed that the injection of a node is divided across all nodes with that zone in a specific ratio, the Generation Shift Key (GSK). Using the GSK, a complex network can be abstracted into a smaller, less complex grid. The line specific load is not lost, based on the overall injection in a zone/country, the nodal injection can be easily calculated with the GSK and individual line loadings and the corresponding contingency analysis is still possible. In other words, the zonal PTDF represents a zone to line sensitivity, where the zonal injection consists of all nodal injections.






