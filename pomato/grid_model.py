"""
GRID Model
"""
import sys
import logging
import numpy as np
import pandas as pd
import tools as tools

from .cbco_module import CBCOModule

class GridModel(object):
    """GRID Model Class"""
    numpy_settings = np.seterr(divide="raise")
    def __init__(self):
        self.logger = logging.getLogger('Log.MarketModel.GridModel')
        self.logger.info("Initializing GridModel..")
        self.is_empty = True
            
    def build_grid_model(self, nodes, lines):
        try:
            # import logger
            self.nodes = nodes
            self.lines = lines
            self.mult_slack = bool(len(nodes.index[nodes.slack]) > 1)
            self.ptdf = self.create_ptdf_matrix()
            self.psdf = self.create_psdf_matrix()

            self.check_grid_topology()
            self.lodf = self.create_lodf_matrix()
            self.n_1_ptdf = self.create_n_1_ptdf()

            self.cbco_index = None
            self.add_cbco = None
            self._cbco_option = "convex_hull" ## or gams
            self.is_empty = False
            self.logger.info("GridModel initialized!")
        except:
            self.logger.exception("Error in GridModel!")
        
    def __getstate__(self):
        """
        Method to remove logger attribute from __dict__
        needed when pickeled
        """
        d = dict(self.__dict__)
        del d["logger"]
        return d

    def __setstate__(self, d):
        """
        Method updates self with modified __dict__ without logger
        needed when pickeled
        """
        self.__dict__.update(d) # I *think* this is a safe way to do it

    def check_grid_topology(self):
        """Checking grid topology for radial nodes and lines"""
        self.logger.info("Checking Grid Topology...")

        radial_nodes = []
        for node in self.nodes.index:
            if len(self.lines[(self.lines.node_i == node)|(self.lines.node_j == node)]) < 2:
                radial_nodes.append(node)

        radial_lines = []
        for idx, line in enumerate(self.lines.index):
            tmp = np.abs(self.ptdf[idx, :])
            tmp = np.around(tmp, decimals=3)
            if 1 in tmp:
                radial_lines.append(line)

        tmp = self.lines[((self.lines.node_i.isin(radial_nodes))| \
                          (self.lines.node_j.isin(radial_nodes)))& \
                          (self.lines.contingency)]
        if not tmp.empty:
            self.logger.info("Radial nodes are part of the contingency: " + \
                             ", ".join(list(tmp.index)))
            self.lines.contingency[((self.lines.node_i.isin(radial_nodes))| \
                                    (self.lines.node_j.isin(radial_nodes)))& \
                                    (self.lines.contingency)] = False
            self.logger.info("Contingency of radial nodes is set to False")

        tmp = self.lines.contingency[(self.lines.index.isin(radial_lines))& \
                                     (self.lines.contingency)]
        if not tmp.empty:
            self.logger.info("Radial lines are part of the contingency: " + \
                             ", ".join(list(tmp.index)))
            self.lines.contingency[(self.lines.index.isin(radial_lines))& \
                                   (self.lines.contingency)] = False
            self.logger.info("Contingency of radial lines is set to False")

    def loss_of_load(self, list_nodes):
        """
        see if loss of load breaches security domain
        input in the form list_nodes = [ ["n1","n2"], ["n1"], ["n2","n5","n7"]]
        """
        # get slack zones, loss of load is distributed equally in slack zone
        if self.mult_slack:
            slack_zones = self.slack_zones()
        else:
            slack_zones = [list(self.nodes.index)]
        # update injection vector
        for nodes in list_nodes:
            inj = self.nodes.net_injection.copy()
            for node in nodes:
                sz_idx = [x for x in range(0, len(slack_zones)) if node in slack_zones[x]][0]
                inj[inj.index.isin(slack_zones[sz_idx])] += inj[node]/(len(slack_zones[sz_idx])-1)
                inj[node] = 0
            #calculate resulting line flows
            flow = np.dot(self.ptdf, inj)
            f_max = self.lines.maxflow.values
            if self.lines.index[abs(flow) > f_max].empty:
                self.logger.info("The loss of load at nodes: " + ", ".join(nodes) +
                                 "\nDoes NOT cause a security breach!")
            else:
                self.logger.info("The loss of load at nodes: " + ", ".join(nodes) +
                                 "\nCauses a security breach at lines: \n" +
                                 ", ".join(self.lines.index[abs(flow) > f_max]))

    def check_n_1(self):
        """Check N-1 security for injections in self.nodes"""
        overloaded_lines = {}
        n_1_flow = self.n_1_flows()
        for outage in n_1_flow: # Outage
            # compare n-1flow vector with maxflow vector -> bool vector of overloaded lines
            for ov_line in self.lines.index[np.abs(n_1_flow[outage]) >\
                                            self.lines.maxflow.values*1.05]:
                overloaded_lines[len(overloaded_lines)] = \
                {'Line': ov_line, 'Outage': outage,
                 'Flow': n_1_flow[outage][self.lines.index.get_loc(ov_line)],
                 'maxflow': self.lines.maxflow[ov_line]}
        return overloaded_lines

    def check_n_1_for_marketresult(self, injections, timeslice=None, threshold=3):
        """
        Checks Market Result for N-1 Security, optional timeslice as str,
        optional threshhold for overloaded lines from which further check is cancelled
        injections dataframe from gms method gams_symbol_to_df
        """
        timeslice = timeslice or ['t'+ "{0:0>4}".format(x+1) \
                                  for x in range(0, len(injections.t.unique()))]
        self.logger.info(f"Run N-1 LoadFlow Check from {timeslice[0]} to {timeslice[-1]}")

        all_overloaded_lines = {}
        nr_overloaded_lines = 0
        for i, time in enumerate(timeslice):
            ## Generate Fancy Progressbar
            sys.stdout.write("\r[%-35s] %d%%  - Overloaded Lines in %d Timesteps" % \
                             ('='*int(i*35/len(timeslice)),
                              int(i*101/len(timeslice)), nr_overloaded_lines))
            sys.stdout.flush()
            #Exit if more than 10 N-1 Breaches are detected
            if nr_overloaded_lines > threshold:
                break
                print("\n")
                self.logger.error(f"More than {threshold} N-1 breaches!")
            net_injections = injections[injections.t == time]
            # sort by the same order of nodes as self.nodes.index
            net_injections = net_injections.set_index("n").reindex(self.nodes.index).reset_index()

            self.update_net_injections(net_injections.INJ.values)
            overloaded_lines = self.check_n_1()
            if overloaded_lines != {}:
                all_overloaded_lines[time] = overloaded_lines
                nr_overloaded_lines += 1
        self.logger.info(f"Check Done: {nr_overloaded_lines} Overloaded Lines")

        return all_overloaded_lines

    def grid_representation(self, option, ntc):
        """Bundle all relevant grid information in one dict for the market model"""
        grid_rep = {}
        grid_rep["option"] = option
        grid_rep["mult_slacks"] = self.mult_slack
        grid_rep["slack_zones"] = self.slack_zones()
        if option == "nodal":
            ptdf_dict = {}
            for idx, line in enumerate(self.lines.index):
                ptdf_dict[line + "_pos"] = {"ptdf": list(self.ptdf[idx,:]), "ram": self.lines.maxflow[line]}
                ptdf_dict[line + "_neg"] = {"ptdf": list(-self.ptdf[idx,:]), "ram": self.lines.maxflow[line]}
            grid_rep["cbco"] = ptdf_dict
        elif option == "ntc":
            grid_rep["ntc"] = ntc
        elif "cbco" in option.split("_"):
            from pathlib import Path
            A,b = self.contingency_Ab("nodal", self.n_1_ptdf)
            cbco = CBCOModule(Path.cwd(), self.nodes, self.lines, A, b)
            info, cbco = cbco.main(use_precalc=False, only_convex_hull=False)
            grid_rep["info"] = info
            grid_rep["cbco"] = cbco
        return grid_rep

    def slack_zones(self):
        """
        returns number of nodes that are part of a synchronous area
        for each slack defined
        """
        ## Creates Slack zones, given that the given slacks are well suited
        ## Meaning one slack per zone, all zones have a slack.
        # Method: slack -> find Line -> look at ptdf
        # all non-zero elementes are in slack zone.
        slacks = self.nodes.index[self.nodes.slack]
        slack_zones = {}
        for slack in slacks:
            slack_line = self.lines.index[(self.lines.node_i == slack) \
                                          |(self.lines.node_j == slack)][0]
            line_index = self.lines.index.get_loc(slack_line)
            pos = self.ptdf[line_index, :] != 0
            tmp = list(self.nodes.index[pos])
            tmp.append(slack)
            slack_zones[slack] = tmp
        return slack_zones

    def check_n_0(self, flow):
        """ Check N-0 Loadflow"""
        self.logger.info("Run N-0 LoadFlow Check")
        for l_idx, elem in enumerate(flow):
            if elem > self.lines.maxflow[l_idx]*1.01:
                self.logger.info(self.lines.index[l_idx] + ' is above max capacity')
            elif elem < -self.lines.maxflow[l_idx]*1.01:
                self.logger.info(self.lines.index[l_idx] + ' is below min capacity')

    def n_1_flows(self, option="outage"):
        """ returns N-1 Flows, either by outage or by line"""
        # Outate Line -> resulting Flows on all other lines
        injections = self.nodes.net_injection.values
        contingency = self.n_1_ptdf
        n_1 = []
        for i, _ in enumerate(self.lines.index):
            n_1.append(np.dot(contingency[i+1], injections))
        n_1_stack = np.vstack(n_1)
        n_1_flows = {}
        if option == "outage":
            for i, outage in enumerate(self.lines.index):
                n_1_flows[outage] = n_1_stack[i, :]
        else:
            for i, line in enumerate(self.lines.index):
                n_1_flows[line] = n_1_stack[:, i]
        return n_1_flows

    def update_flows(self):
        """update flows in self.lines"""
        flows = self.calc_flows()
        self.lines.flow = flows
        return flows

    def calc_flows(self):
        """ claculates flows, without saving them to self"""
        injections = self.nodes.net_injection.values
        ptdf = self.ptdf
        flow = np.dot(ptdf, injections)
        return flow

    def update_net_injections(self, net_inj):
        """updates net injection in self.nodes from list"""
        try:
            self.nodes.net_injection = net_inj
        except ValueError:
            self.logger.exception("invalid net injections provided")

    def update_gsk(self, option="dict"):
        """
        update generation shift keys based on value in nodes
        option allows to return an array instead of dict for zonal ptdf calc
        """
        try:
            nodes_in_zone = {}
            for zone in self.nodes.zone.unique():
                nodes_in_zone[zone] = self.nodes.index[self.nodes.zone == zone]
            gsk_dict = {}
            for zone in list(self.nodes.zone.unique()):
                gsk_zone = {}
                sum_weight = 0
                for node in nodes_in_zone[zone]:
                    sum_weight += self.nodes.gsk[node]
                for node in nodes_in_zone[zone]:
                    if sum_weight == 0:
                        gsk_zone[node] = 1
                    else:
                        gsk_zone[node] = self.nodes.gsk[node]/sum_weight
                gsk_dict[zone] = gsk_zone
            ## Convert to GSK Array
            gsk_array = np.zeros((len(self.nodes), len(self.nodes.zone.unique())))
            gsk = self.nodes.zone.to_frame()
            for zone in list(self.nodes.zone.unique()):
                gsk[zone] = [0]*len(self.nodes)
                gsk[zone][gsk.zone == zone] = list(gsk_dict[zone].values())
            gsk = gsk.drop(["zone"], axis=1)
            gsk_array = gsk.values
            if option == "array":
                return gsk_array
            else:
                return gsk_dict
        except:
            self.logger.exception('error:update_gsk')

    def create_incedence_matrix(self):
        """Create incendence matrix"""
        incedence = np.zeros((len(self.lines), len(self.nodes)))
        for i, elem in enumerate(self.lines.index):
            incedence[i, self.nodes.index.get_loc(self.lines.node_i[elem])] = 1
            incedence[i, self.nodes.index.get_loc(self.lines.node_j[elem])] = -1
        return incedence

    def create_susceptance_matrices(self):
        """ Create Line (Bl) and Node (Bn) susceptance matrix """
        suceptance_vector = self.lines.b
        incedence = self.create_incedence_matrix()
        susceptance_diag = np.diag(suceptance_vector)
        line_susceptance = np.dot(susceptance_diag, incedence)
        node_susceptance = np.dot(np.dot(incedence.transpose(1, 0), susceptance_diag), incedence)
        return(line_susceptance, node_susceptance)

    def create_ptdf_matrix(self):
        """ Create ptdf Matrix """
        try:
            #Find slack
            slack = list(self.nodes.index[self.nodes.slack])
            slack_idx = [self.nodes.index.get_loc(s) for s in slack]
            line_susceptance, node_susceptance = self.create_susceptance_matrices()
            #Create List without the slack and invert it
            list_wo_slack = [x for x in range(0, len(self.nodes.index)) \
                            if x not in slack_idx]
            inv = np.linalg.inv(node_susceptance[np.ix_(list_wo_slack, list_wo_slack)])
            #sort slack back in to get nxn
            node_susc_inv = np.zeros((len(self.nodes), len(self.nodes)))
            node_susc_inv[np.ix_(list_wo_slack, list_wo_slack)] = inv
            #calculate ptdfs
            ptdf = np.dot(line_susceptance, node_susc_inv)
            return ptdf
        except:
            self.logger.exception('error:create_ptdf_matrix')

    def create_psdf_matrix(self):
        """
        Calculate psdf (phase-shifting distribution matrix, LxLL)
        meaning the change of p.u. loadflow
        on a line LL through a phase-shifting by 1rad at line L
        """
        line_susceptance, _ = self.create_susceptance_matrices()
        psdf = np.diag(self.lines.b) - np.dot(self.ptdf, line_susceptance.T)
        return psdf

    def shift_phase_on_line(self, shift_dict):
        """
        Shifts the phase on line l by angle a (in rad)
        Recalculates the ptdf matrix and replaces it as the attribute
        """
        shift = np.zeros(len(self.lines))
        for line in shift_dict:
            shift[self.lines.index.get_loc(line)] = shift_dict[line]
            # recalc the ptdf
        shift_matrix = np.multiply(self.psdf, shift)
        self.ptdf += np.dot(shift_matrix, self.ptdf)
        # subsequently recalc n-1 ptdfs
        self.lodf = self.create_lodf_matrix()
        self.n_1_ptdf = self.create_n_1_ptdf()

    def create_lodf_matrix(self):
        """ Load outage distribution matrix -> Line to line sensitivity """
        try:
            lodf = np.zeros((len(self.lines), len(self.lines)))
            incedence = self.create_incedence_matrix()
            for idx, _ in enumerate(self.lines.index):
                for idy, line in enumerate(self.lines.index):
                    ## Exception for lines that are not in the contingency
                    if line in self.lines.index[~self.lines.contingency]:
                        lodf[idx, idy] = 0
                    elif idx == idy:
                        lodf[idx, idy] = -1
                    else:
                        lodf[idx, idy] = (np.dot(incedence[idy, :], self.ptdf[idx, :]))/ \
                                       (1-np.dot(incedence[idy, :], self.ptdf[idy, :]))
            return lodf
        except:
            self.logger.exception("error in create_lodf_matrix ", sys.exc_info()[0])
            raise ZeroDivisionError('LODFError: Check Slacks, radial Lines/Nodes')


    def create_n_1_ptdf(self):
        """
        Create N-1 ptdfs - For each line add the resulting ptdf to list contingency
        first ptdf is N-0 ptdf
        """
        try:
            # add N-0 ptdf
            contingency = [self.ptdf]
            # add ptdf for every CO -> N-1 ptdf (list with lxn matrices with
            #length l+1 (number of lines plus N-0 ptdf))
            for l0_idx, _ in enumerate(self.lines.index):
                n_1_ptdf = np.zeros((len(self.lines), len(self.nodes)))
                for l1_idx, _ in enumerate(self.lines.index):
                    n_1_ptdf[l1_idx, :] = self.ptdf[l1_idx, :] + \
                                          np.dot(self.lodf[l1_idx, l0_idx],
                                                 self.ptdf[l0_idx, :])
                contingency.append(n_1_ptdf)
            return contingency
        except:
            self.logger.exception('error:create_n_1_ptdf')

    def create_zonal_ptdf(self, contingency):
        """
        Create Zonal ptdf -> creates both positive and negative line
        restrictions or ram. Depending on flow != 0
        """
        try:
            gsk = self.update_gsk(option="array")
            # Calculate zonal ptdf based on ram -> (if current flow is 0 the
            # zonal ptdf is based on overall
            # avalable line capacity (l_max)), ram is calculated for every n-1
            # ptdf matrix to ensure n-1 security constrained FB Domain
            # The right side of the equation has to be positive ,
            # therefore the distingtion.
            list_zonal_ptdf = []
            for ptdf in contingency:
                ram_array = self.update_ram(ptdf, option="array")
                ram_pos = ram_array[:, 0].reshape(len(ram_array[:, 0]), 1)
                ram_neg = ram_array[:, 1].reshape(len(ram_array[:, 1]), 1)
                z_ptdf = np.dot(ptdf, gsk)
                z_ptdf_pos = np.concatenate((z_ptdf, ram_pos), axis=1)
                z_ptdf_neg = np.concatenate((-z_ptdf, -ram_neg), axis=1)
                tmp = np.concatenate((z_ptdf_pos, z_ptdf_neg), axis=0)
                tmp = tmp.tolist()
                list_zonal_ptdf += tmp
            return list_zonal_ptdf
        except:
            self.logger.exception('error:create_zonal_ptdf')

    def update_ram(self, ptdf, option="dict"):
        """
        Update ram based on Lineflows from netinjections
        option to return either array or dict
        (array used in cbco to make things faster)
        """
        injections = self.nodes.net_injection.values
        ram_dict = []
        ram_array = []
        for idx, line in enumerate(ptdf):
            pos = self.lines.maxflow[idx] - np.dot(line, injections)
            neg = -self.lines.maxflow[idx] - np.dot(line, injections)
            if pos < 0:
                ram_dict.append({'pos': 0.1, 'neg': neg})
                ram_array.append([0.1, neg])
            elif neg > 0:
                ram_dict.append({'pos': pos, 'neg': 0.1})
                ram_array.append([pos, 0.1])
            else:
                ram_dict.append({'pos': pos, 'neg': neg})
                ram_array.append([pos, neg])
        if option == "array":
            return np.asarray(ram_array)
        else:
            return ram_dict

    def slack_zones_index(self):
        """returns the indecies of nodes per slack_zones
        (aka control area/synchronious area) in the A matrix"""
        slack_zones = self.slack_zones()
        slack_zones_idx = []
        for slack in slack_zones:
            slack_zones_idx.append([self.nodes.index.get_loc(node) \
                                    for node in slack_zones[slack]])
        slack_zones_idx.append([x for x in range(0, len(self.nodes))])
        return slack_zones_idx

    def contingency_Ab(self, option, contingency=None):
        """ Bring the N-1 PTDF list in the form of inequalities A x leq b
            where A are the ptdfs, b the ram and x the net injections
            returns lists
        """
        contingency = contingency or self.n_1_ptdf
        if option == 'zonal':
            zonal_contingency = self.create_zonal_ptdf(contingency)
            A = []
            b = []
            for i, equation in enumerate(zonal_contingency):
                ### Check if RAM > 0
                if equation[-1] != 0:
                    A.append(equation[:-1])
                    b.append(equation[-1])
                else:
                    self.logger.debug('zonal:cbco not n-1 secure')
                    A.append(equation[:-1])
                    b.append(1)
        else:
            A = []
            b = []
            for ptdf in contingency:
                ram_array = self.update_ram(ptdf, option="array")
                ram_pos = ram_array[:, 0]
                ram_neg = ram_array[:, 1]
                tmp_A = np.concatenate((ptdf, -ptdf), axis=0)
                tmp_b = np.concatenate((ram_pos, -ram_neg), axis=0)
                A += tmp_A.tolist()
                b += tmp_b.tolist()
        return A, b
    
    def create_eq_list_zptdf(self, list_zonal_ptdf,
                             domain_x=None, domain_y=None, gsk_sink=None):
        """
        from zonal ptdf calculate linear equations ax = b to plot the FBMC domain
        nodes/Zones that are not part of the 2D FBMC are summerized using GSK sink
        """
        try:
            domain_x = domain_x or []
            domain_y = domain_y or []
            gsk_sink = gsk_sink or []

            nodes = self.nodes
            list_zones = list(nodes.zone.unique())
            # create 2D equation from zonal ptdfs
            if len(domain_x) == 1:
                sink_domain = list(list_zones)
                sink_domain.remove(domain_x[0])
                sink_domain.remove(domain_y[0])
            list_A = []
            list_b = []
            for ptdf in list_zonal_ptdf:
                if len(domain_x) == 0:
                    list_A.append(ptdf[:-1])
                    list_b.append(ptdf[-1])
                elif len(domain_x) == 1:
                    ptdf_sink = 0
                    for zones in sink_domain:
                        ptdf_sink += ptdf[list_zones.index(zones)]*gsk_sink[zones]
                    list_A.append([(ptdf[list_zones.index(domain_x[0])] - ptdf_sink),
                                   (ptdf[list_zones.index(domain_y[0])] - ptdf_sink)])
                    list_b.append(ptdf[len(list_zones)])
                elif len(domain_x) == 2:
                    list_A.append([(ptdf[list_zones.index(domain_x[0])] \
                                    - ptdf[list_zones.index(domain_x[1])]),
                                   (ptdf[list_zones.index(domain_y[0])] \
                                    - ptdf[list_zones.index(domain_y[1])])])
                    list_b.append(ptdf[len(list_zones)])
            #Clean reduce Ax=b only works if b_i != 0 for all i,
            #which should be but sometimes wierd stuff comes up
            #Therefore if b == 0, b-> 1 (or something small>0)
            for i, b in enumerate(list_b):
                if b == 0:
                    self.logger.debug('some b is not right ( possibly < 0) and \
                                      implies that n-1 is not given!', i)
                    list_b[i] = 1

            return(list_A, list_b)
        except:
            self.logger.exception('error:create_eq_list_zptdf')

    def plot_domain(self, contingency, domain_x, domain_y, gsk_sink=None, \
                    reduce=True, external_zonal_ptdf=False):
        """
        Creates coordinates for the FBMC Domain
        """
        try:
            gsk_sink = gsk_sink or {}
            ### Bring into 2 D form such that for each line
            ### the eqation is A*ptdf_A + B*ptdf_B < ram
            if external_zonal_ptdf:
                list_zonal_ptdf = contingency
            else:
                list_zonal_ptdf = self.create_zonal_ptdf(contingency)
            # create 2D equation from zonal ptdfs
            # This creates Domain X-Y
            A, b = self.create_eq_list_zptdf(list_zonal_ptdf, domain_x, domain_y, gsk_sink)
            # Reduce
            cbco_index = self.reduce_ptdf(A, b)
            if not reduce: # plot additional constrains
                # Limit the number of constraints to 3*number of lines (pos and neg)
                if len(A) > (3*2*len(self.lines)):
                    cbco_index = np.append(cbco_index, [x for x in range(0,3*2*len(self.lines))])
                else:
                    cbco_index = np.array([x for x in range(0,len(A))])

            A = np.take(np.array(A), cbco_index, axis=0)
            b = np.take(np.array(b), cbco_index, axis=0)
            Ab = np.concatenate((np.array(A), np.array(b).reshape(len(b), 1)), axis=1)
            self.logger.debug(f"Number of CBCOs {len(Ab)} in plot_domain")

            # Calculate two coordinates for a line plot -> Return X = [X1;X2], Y = [Y!,Y2]
            plot_eq = []
            for index in range(0, len(Ab)):
                xcoord = []
                ycoord = []
                for idx in range(-10000, 10001, 20000):
                    if Ab[index][1] != 0:
                        ycoord.append((Ab[index][2] - idx*(Ab[index][0])) / (Ab[index][1]))
                        xcoord.append(idx)
                    elif Ab[index][0] != 0:
                        ycoord.append(idx)
                        xcoord.append((Ab[index][2] - idx*(Ab[index][1])) / (Ab[index][0]))
                plot_eq.append([xcoord, ycoord])

            return plot_eq
        except:
            self.logger.exception('error:plot_domain')


    def plot_vertecies_of_inequalities(self, domain_x, domain_y, gsk_sink):
        """Plot Vertecies Representation of FBMC Domain"""

        self.nodes.net_injection = 0
        contingency = self.n_1_ptdf
        gsk_sink = gsk_sink or {}
        list_zonal_ptdf = self.create_zonal_ptdf(contingency)
        A, b = self.create_eq_list_zptdf(list_zonal_ptdf, domain_x, domain_y, gsk_sink)
        cbco_index = self.reduce_ptdf(A, b)

        if len(A) > (3*2*len(self.lines)):
            relevant_subset = np.append(cbco_index, [x for x in range(0,3*2*len(self.lines))])
        else:
            relevant_subset = np.array([x for x in range(0,len(A))])

        A = np.array(A)
        b = np.array(b).reshape(len(b), 1)

        vertecies_full = np.take(A, relevant_subset, axis=0)/np.take(b, relevant_subset, axis=0)
        vertecies_reduces = np.take(A, cbco_index, axis=0)/np.take(b, cbco_index, axis=0)

        x_max, x_min, y_max, y_min = tools.find_xy_limits([[vertecies_reduces[:,0], vertecies_reduces[:,1]]])

        fig = plt.figure()
        ax = plt.subplot()

        scale = 1.2
        ax.set_xlim(x_min*scale, x_max*scale)
        ax.set_ylim(y_min*scale, y_max*scale)

        for point in vertecies_full:
            ax.scatter(point[0], point[1], c='lightgrey')
        for point in vertecies_reduces:
            ax.scatter(point[0], point[1], c='r')


    def get_xy_hull(self, contingency, domain_x, domain_y, gsk_sink=None):
        """get x,y coordinates of the FBMC Hull"""
        try:
            gsk_sink = gsk_sink or {}
            list_zonal_ptdf = self.create_zonal_ptdf(contingency)

            # ptdf_x * X + ptdf_y *Y = B
            # Or in Matrix Form A*x = b where X = [X;Y]
            A, b = self.create_eq_list_zptdf(list_zonal_ptdf, domain_x, domain_y, gsk_sink)
            cbco_index = self.reduce_ptdf(A, b)

            ptdf = np.take(np.array(A), cbco_index, axis=0)
            ram = np.take(np.array(b), cbco_index, axis=0)
            self.logger.debug(f"Number of CBCOs {len(ram)} in get_xy_hull")
            ### Find all intersections between CO
            intersection_x = []
            intersection_y = []
            for idx in range(0, len(ptdf)):
                for idy in range(0, len(ptdf)):
                    ### x*ptdf_A0 +  y*ptdf_A1 = C_A ----- x*ptdf_B0 + y*ptdf_B1 = C_B
                    ### ptdf[idx,0] ptdf[idx,1] = ram[idx] <-> ptdf[idy,0] ptdf[idy,1] = ram[idy]
                    if idx != idy:
                            # A0 close to 0
                        if np.square(ptdf[idx, 0]) < 1E-9 and np.square(ptdf[idy, 0]) > 1E-9:
                            intersection_x.append((ram[idy]*ptdf[idx, 1] - ram[idx]*ptdf[idy, 1])\
                                     /(ptdf[idx, 1]*ptdf[idy, 0]))
                            intersection_y.append(ram[idx]/ptdf[idx, 1])
                            ## A1 close to 0
                        elif np.square(ptdf[idx, 1]) < 1E-9 and np.square(ptdf[idy, 1]) > 1E-9:
                            intersection_x.append(ram[idx]/ptdf[idx, 0])
                            intersection_y.append((ram[idy]*ptdf[idx, 0] - ram[idx]*ptdf[idy, 0]) \
                                     /(ptdf[idx, 0]*ptdf[idy, 1]))

                        elif (ptdf[idx, 1]*ptdf[idy, 0] - ptdf[idy, 1]*ptdf[idx, 0]) != 0 \
                        and (ptdf[idx, 0]*ptdf[idy, 1] - ptdf[idy, 0]*ptdf[idx, 1]):
                            intersection_x.append((ram[idx]*ptdf[idy, 1] - ram[idy]*ptdf[idx, 1]) \
                                    / (ptdf[idx, 0]*ptdf[idy, 1] - ptdf[idy, 0]*ptdf[idx, 1]))
                            intersection_y.append((ram[idx]*ptdf[idy, 0] - ram[idy]*ptdf[idx, 0]) \
                                    / (ptdf[idx, 1]*ptdf[idy, 0] - ptdf[idy, 1]*ptdf[idx, 0]))

            hull_x = []
            hull_y = []
            ### Filter intersection points for those which define the FB Domain
            for idx in range(0, len(intersection_x)):
                temp = 0
                for idy in range(0, len(ptdf)):
                    if ptdf[idy, 0]*intersection_x[idx] +\
                        ptdf[idy, 1]*intersection_y[idx] <= ram[idy]*1.0001:
                        temp += 1
                    if temp >= len(ptdf):
                        hull_x.append(intersection_x[idx])
                        hull_y.append(intersection_y[idx])

            ### Sort them Counter Clockwise to plot them
            list_coord = []
            for idx in range(0, len(hull_x)):
                radius = np.sqrt(np.power(hull_y[idx], 2) + np.power(hull_x[idx], 2))
                if hull_x[idx] >= 0 and hull_y[idx] >= 0:
                    list_coord.append([hull_x[idx], hull_y[idx],
                                       np.arcsin(hull_y[idx]/radius)*180/(2*np.pi)])
                elif hull_x[idx] < 0 and hull_y[idx] > 0:
                    list_coord.append([hull_x[idx], hull_y[idx],
                                       180 - np.arcsin(hull_y[idx]/radius)*180/(2*np.pi)])
                elif hull_x[idx] <= 0 and hull_y[idx] <= 0:
                    list_coord.append([hull_x[idx], hull_y[idx],
                                       180 - np.arcsin(hull_y[idx]/radius)*180/(2*np.pi)])
                elif hull_x[idx] > 0 and hull_y[idx] < 0:
                    list_coord.append([hull_x[idx], hull_y[idx],
                                       360 + np.arcsin(hull_y[idx]/radius)*180/(2*np.pi)])

            from operator import itemgetter
            list_coord = sorted(list_coord, key=itemgetter(2))
            ## Add first element to draw complete circle
            list_coord.append(list_coord[0])
            list_coord = np.array(list_coord)
            list_coord = np.round(list_coord, decimals=3)
            unique_rows_idx = [x for x in range(0, len(list_coord)-1) \
                               if not np.array_equal(list_coord[x, 0:2], list_coord[x+1, 0:2])]
            unique_rows_idx.append(len(list_coord)-1)
            list_coord = np.take(list_coord, unique_rows_idx, axis=0)
            return(list_coord[:, 0], list_coord[:, 1], intersection_x, intersection_y)
        except:
            self.logger.exception('error:get_xy_hull')

    def plot_fbmc(self, domain_x, domain_y, gsk_sink=None):
        """
        Combines previous functions to actually plot the FBMC Domain with the
        hull
        """
        try:
            gsk_sink = gsk_sink or {}
            self.nodes.net_injection = 0
            contingency = self.n_1_ptdf
#
#            self = MT.grid
#
#            domain_x = ['DE']
#            domain_y = ['DK-East']
#            gsk_sink={"DK-West": 0.5, "SE": 0.5}

            fig = plt.figure()
            ax = plt.subplot()
            plot = self.plot_domain(contingency, domain_x, domain_y, \
                                           gsk_sink, reduce=False, external_zonal_ptdf=False)

            hull_x, hull_y, xcoord, ycoord = self.get_xy_hull(contingency, domain_x, domain_y, gsk_sink)
            x_max, x_min, y_max, y_min = tools.find_xy_limits([[hull_x, hull_y]])

            scale = 1.5
            ax.set_xlim(x_min*scale, x_max*scale)
            ax.set_ylim(y_min*scale, y_max*scale)

            title = 'FBMC Domain between: ' + "-".join(domain_x) + ' and ' + \
            "-".join(domain_y) + '\n Number of CBCOs: ' + str(len(hull_x)-1)

            for elem in plot:
                ax.plot(elem[0], elem[1], c='lightgrey', ls='-')

            ax.plot(hull_x, hull_y, 'r--', linewidth=2)
            ax.set_title(title)
            ax.scatter(xcoord, ycoord)
            fig.show()
        except:
            self.logger.exception('error:plot_fbmc', sys.exc_info()[0])

    def lineloading_timeseries(self, injections, line):
        """
        Plots Line Loading (N-0; N-1) for all timeslices in inj dataframe
        inj dataframe from gms method gams_symbol_to_df
        """
        sys.stdout.write("\n")
        sys.stdout.flush()
        line_loadings = {}

        for i, time in enumerate(injections.t.unique()):
            ## Generate Fancy Progressbar
            sys.stdout.write("\r[%-35s] %d%%" % \
                             ('='*int(i*35/len(injections.t.unique())),
                              int(i*101/len(injections.t.unique()))))
            sys.stdout.flush()
            self.update_net_injections(injections.INJ[injections.t == time].values)
            self.update_flows()
            flow_n0 = abs(self.lines.flow[line])/self.lines.maxflow[line]
            flow_n0_20 = abs(self.lines.flow[line])/self.lines.maxflow[line]*1.2
            flow_n0_40 = abs(self.lines.flow[line])/self.lines.maxflow[line]*1.4
            flow_n0_60 = abs(self.lines.flow[line])/self.lines.maxflow[line]*1.6
            flow_n0_80 = abs(self.lines.flow[line])/self.lines.maxflow[line]*1.8
            flow_n0_100 = abs(self.lines.flow[line])/self.lines.maxflow[line]*2
            flow_n1 = self.n_1_flows(option="lines")
            flow_max_n1 = max(abs(flow_n1[line]))/self.lines.maxflow[line]
            line_loadings[time] = {"N-0": flow_n0, "N-1": flow_max_n1,
#                                   "N-1 + 20%": flow_n0_20, "N-1 + 40%": flow_n0_40,
#                                   "N-1 + 60%": flow_n0_60, "N-1 + 80%": flow_n0_80,
#                                   "N-1 + 100%": flow_n0_100
                                   }

        return pd.DataFrame.from_dict(line_loadings, orient="index")
