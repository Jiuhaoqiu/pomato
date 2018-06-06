.. ref-mato:

-----------------------
The Market Tool Object
-----------------------


This chapter provides an overview of the main functionality of the market tool object and its underlying classes and functions


Grid Model Object
^^^^^^^^^^^^^^^^^

**Utility:**

    * ``save()``: Stores the full object, e.g to be used by plotting functions.
    * ``check_grid_toplogy``: Checks grid topology for radial nodes and lines.
    * ``slack_zones()``: Returns number of nodes that are part of a synchronous area for each slack defined.
    * ``check_n_1_for_marketresult(injections, timeslice=None, threshold=3)``: Checks Market Result for N-1 Security. Optional timeslice as list of strings [t001, . . . , txxx]. Op- tional threshold for overloaded lines from which further check is canceled, injections dataframe from method gms.gams_symbol_to_df().
    * ``update_gsk()``: Updates the GSK based on the GSK values in the nodes table. Returns the GSK12.
    * ``update_ram(PTDF)``: Updates the RAM based on net injections and the resulting line-flows using the input PTDF. Returns RAM.
    * ``update_flows()``: Updates the flows in lines.

**Calculations:**

    * ``create_incedence_matrix()``: Creates incidence matrix
    * ``create_node_line_susceptance_matrix()``: Creates node-susceptance matrix Bn and line-susceptance matrix Bl
    * ``create_ptdf_matrix()``: Creates and returns PTDF matrix
    * ``create_lodf_matrix()``: Creates and returns LODF matrix
    * ``create_psdf_matrix()``: Creates and returns PSDF matrix (not yet implemented)
    * ``create_n_1_pdtf()``: Creates and returns N-1 PTDF matrices (number of lines plus N-0 PTDF matrix)
    * ``calc_flows()``: Calculates and returns flows with the N-0 PTDF
    * ``n_1_flows()``: Calculates and returns all N-1 Flows in a dictionary. For each outage (=key) it contains the flows on all lines.
    * ``check_n_0(Flow)``: Checks weather a flow (input) is viable in the N-0 grid
    * ``check_n_1()``: Checks N-1 load flows based on the return from N_1_Flows(). Returns overloaded_lines and model_status to the class.
    * ``reduce_ptdf(A, b)``: Based on a equations system in the form :math: `Ax = b` this methods find the equations which create the convex hull. This method is essential for the creation of the CBCOs and the Flow-Based Domain. 
    * ``create_zonal_ptdf(contingency)`` Creates a list of zonal PTDFs based on the input contingency, which usually is the list of N-1 PTDFs. This method returns a list with all equations for both the lower limit and upper limit of the corresponding line. Because of related methods, the right hand side of all equations is positive.

**Tools:**

    * ``return_cbco(contingency={}, option='zonal', lines = [], outages = [])``: Creates CB- COs for a specified contingency. It is possible to generate zonal oder nodal CBCOs, however the implementation incorporates some math-fuzzyness, which is why it is possible to manually add specific CBCOs.
    * ``create_eq_list_zonalptdf(list_zonal_PTDF, domain_x=[], domain_y=[], gsk_sink=[])`` Creates a list of two dimensional PTDFs to allow plotting the trade domain for chosen zones and specified zones that act as sinks for excess capacity.
    * ``get_xy_hull(contingency, domain_x, domain_y, gsk_sink={}, external_zonal_ptdf=True)``: Based on the reduced zonal PTDF equations, this methods constructs the resulting convex hull based on the intersection of the relevant equations, returns the x and y coordinates.
    * ``plot_domain(contingency, domain_x, domain_y, gsk_sink={}, reduce=True, external_zonal_ptdf=False)``: Creates x and y coordinates for all (or the reduced set) of zonal PTDF equations from create_eq_list_zonalPTDF.
    * ``plot_fbmc(domain_x, domain_y, gsk_sink={})``: Combines plot_domain and get_xy_hull to plot the resulting FBMC domain with everything anybody could want.


Plotting
^^^^^^^^^

Needs Bokeh_.

.. _Bokeh: https://bokeh.pydata.org/en/latest/ 



