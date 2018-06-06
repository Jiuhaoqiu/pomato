import sys
import logging
import numpy as np
import pandas as pd
import scipy.io as sio

from pomato.resources import MPCOLNAMES

pd.options.mode.chained_assignment = None  # default='warn'

class DataManagement(object):
    """Data Set Class"""
    def __init__(self):
            # import logger
            self.logger = logging.getLogger('Log.MarketModel.DataManagement')
            self.logger.info("Initializing DataObject")
            
            # init as empty
            self.is_empty = True
            self.source = None


    def read_xls_data(self, xls_file, year, co2_price):
        self.logger.info("Reading Data from Excel File")
        self._clear_all_data()
        try:
            xls = pd.ExcelFile(xls_file)
            self.lines = xls.parse('lines', index_col=0)
            self.nodes = xls.parse('nodes', index_col=0)
            self.zones = xls.parse('zones', index_col=0)
            self.heatareas = xls.parse('heatareas', index_col=0)
            plants_list = [xls.parse('plants_el', index_col=0),
                           xls.parse('plants_heat', index_col=0),
                           xls.parse('plants_stor', index_col=0),
                           xls.parse('plants_res', index_col=0)]
            self.plants = pd.concat(plants_list)
            self.dclines = xls.parse('dc_lines', index_col=0)
            self.tech = xls.parse('technology')
            self.fuel = xls.parse('fuel', index_col=0)
            self.fuelmix = xls.parse('fuelmix', index_col=0)
            self.demand_el = xls.parse('demand_el', index_col=0)
            self.demand_h = xls.parse('demand_h', index_col=0)
            self.timeseries = xls.parse('timeseries', index_col=0)
            self.availability = pd.DataFrame(index=self.demand_el.index)
            self.ntc = xls.parse('ntc')
            self.year = year
            self.co2_price = co2_price

            ## TODO: Everything below here should be a new function so that "raw data"
            ##       and data that has been checked / altered can be differentiated
            ## Calculate Missing Values
            self._clean_names()
            self.efficiency()
            self.marginal_costs()
            self.availability_per_plant()
            self.demand_per_zone()
            self.line_susceptance()
            ## Check for NaNs
            self.unique_mc()
            self._check_data()

            self._check_netinjections()

            self.is_empty = False
            self.source = 'xls_data'
            self.logger.info("DataSet initialized!")
        except:
            self.logger.exception("Error in DataSet!")
            raise NameError("Error in DataSet!", sys.exc_info()[0])

    def read_matpower_case(self, casefile):
        self.logger.info("Reading MatPower Casefile")
        self._clear_all_data()
        case_raw = sio.loadmat(casefile)
        mpc = case_raw['mpc']
        bus = mpc['bus'][0,0]
        gen = mpc['gen'][0,0]
        baseMVA = mpc['baseMVA'][0,0]
        branch = mpc['branch'][0,0]
        gencost = mpc['gencost'][0,0]
        try:
            busname = mpc['bus_name'][0,0]
        except:
            busname = np.array([])
        docstring = mpc['docstring'][0,0]
        n = int(gencost[0,3])
        for i in range(n):
            MPCOLNAMES['gencost_keys'] = np.append(MPCOLNAMES['gencost_keys'], 'x{}'.format(n-i-1))
        bus_df = pd.DataFrame(bus, columns=MPCOLNAMES['bus_keys'])
        gen_df = pd.DataFrame(gen, columns=MPCOLNAMES['gen_keys'])
        branch_df = pd.DataFrame(branch, columns=MPCOLNAMES['branch_keys'])
        gencost_df = pd.DataFrame(gencost, columns=MPCOLNAMES['gencost_keys'])
        caseinfo = docstring[0]

        mpc_buses = {
                'idx': bus_df['bus_i'],
                'zone': bus_df['area'],
                'Pd': bus_df['Pd'],
                'Qd': bus_df['Qd'],
                'baseKV': bus_df['baseKV']
                }
        if busname.any():
            b_name = []
            for b in busname:
                b_name.append(b[0][0])
            b_name = np.array(b_name)
            mpc_buses['name'] = b_name

        lineidx = ['l{}'.format(i) for i in range(0,len(branch_df.index))]
        mpc_lines = {
                'idx': lineidx,
                'node_i': branch_df['fbus'],
                'node_j': branch_df['tbus'],
                'maxflow': branch_df['rateA'],
                'b': branch_df['b'],
                'r': branch_df['r'],
                'x': branch_df['x']
                }
        mpc_lines = self._mpc_data_pu_to_real(mpc_lines, mpc_buses['baseKV'][0], baseMVA[0][0])

        ng = len(gen_df.index)
        genidx = ['g{}'.format(i) for i in range(ng)]
        print(type(gencost_df['x2']))
        mpc_generators = {
                    'idx': genidx, 
                    'g_max': gen_df['Pmax'],
                    'g_max_Q': gen_df['Qmax'],
                    'node': gen_df['bus'],
                    'apf': gen_df['apf'],
                    'mc': gencost_df['x2'][list(range(0,ng))],
                    'mc_Q': np.zeros(ng)
                    }
        if len(gencost_df.index) == 2*ng:
            mpc_generators['mc_Q'] = gencost_df['x2'][list(range(ng,2*ng))].tolist

        self.lines = pd.DataFrame(mpc_lines)
        self.lines.set_index('idx')
        self.nodes = pd.DataFrame(mpc_buses)
        self.nodes.set_index('idx')
        self.plants = pd.DataFrame(mpc_generators)
        self.plants.set_index('idx')

        self.is_empty = False
        self.source = 'mpc_case'


    def _mpc_data_pu_to_real(self, lines,  base_kv, base_mva):
        v_base = base_kv * 1e3
        s_base = base_mva * 1e6
        z_base = np.power(v_base,2)/s_base
        lines['r'] = np.multiply(lines['r'], z_base)
        lines['x'] = np.multiply(lines['x'], z_base)
        lines['b'] = np.divide(lines['b'], z_base)
        return lines

    def _clear_all_data(self):
        attr = list(self.__dict__.keys())
        attr.remove('logger')
        for at in attr:
            delattr(self, at)
        self.is_empty = True

    def _clean_names(self):
        """
        Julia does not play well with "-" in plant names
        GAMS Does not like special characters
        use this function to find and replace corresponding chars
        """
        self.logger.info("Cleaning Names...")
        # replace the follwing chars
        char_dict = {"ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE",
                     "å": "a", "Å": "A", "-": "_", "/": "_"}

        # Replace in index of the DataFrame
        to_check = ['plants', 'nodes', 'lines', 'heatareas']
        for attr in to_check:
            if attr in list(self.__dict__.keys()):
                for i in char_dict:
                    self.__dict__[attr].index = self.__dict__[attr].index.str.replace(i, char_dict[i])

        # replace in the dataframe
        try:
            self.plants.heatarea.replace(char_dict, regex=True, inplace=True)
            # self.nodes.replace(char_dict, regex=True, inplace=True)
            # self.lines.replace(char_dict, regex=True, inplace=True)
        except:
            pass


    def _check_netinjections(self):
        """make net injections if there are imported ones, warning"""
        if not self.nodes[self.nodes.net_injection != 0].empty:
            self.nodes.net_injection = 0
            self.logger.warning("Reset Net Injections to zero!")

    def _check_data(self):
        """ checks if dataset contaisn NaNs"""
        self.logger.info("Checking Data...")
        ## Heatarea contains NaN, but that's alright
        data = vars(self)
        data_nan = {}
        for i, df_name in enumerate(data):
            if isinstance(data[df_name], pd.DataFrame):
                for col in data[df_name].columns:
                    if not data[df_name][col][data[df_name][col].isnull()].empty:
                        data_nan[i] = {"df_name": df_name, "col": col}
                        self.logger.warning("DataFrame " + df_name +
                                         " contains NaN in column " + col)
        return data_nan

    def line_susceptance(self):
        """Calculate the efficiency for plants that dont have it maunually set"""
        tmp = self.lines[['length', 'type', 'b']][self.lines.b.isnull()]
        tmp.b = self.lines.length/(self.lines.type)
        self.lines.b[self.lines.b.isnull()] = tmp.b

    def efficiency(self):
        """Calculate the efficiency for plants that dont have it maunually set"""
        tmp = self.plants[['eta', 'fuel_mix', 'tech', 'commissioned']][self.plants.eta.isnull()]
        tmp = pd.merge(tmp, self.tech[['fuel_mix', 'tech', 'eta_c', 'eta_slope']],
                       how='left', on=['tech', 'fuel_mix']).set_index(tmp.index)
        tmp.eta = (tmp.eta_c + tmp.eta_slope*(self.year - tmp.commissioned))
#        self.plants.eta[self.plants.eta.isnull()] = tmp.eta
        self.plants.eta[self.plants.eta.isnull()] = 0.5

    def marginal_costs(self):
        """Calculate the marginal costs for plants that don't have it manually set"""
        fuelmix_cost_dict = {}
        for flmx in self.fuelmix.index:
            fuelmix_cost_dict[flmx] = 0
            for ufuel in self.fuelmix:
                fuelmix_cost_dict[flmx] += self.fuelmix[ufuel][flmx]*self.fuel.fuel_cost[ufuel] + \
                                           self.fuel.CO2[ufuel]*self.co2_price

        fuelmix_cost = pd.DataFrame.from_dict(data=fuelmix_cost_dict, orient='index')
        fuelmix_cost.columns = ["fuelmix_cost"]
        tmp = self.plants[['mc', 'fuel_mix', 'tech', 'eta']][self.plants.mc.isnull()]
        tmp = pd.merge(tmp, self.tech[['fuel_mix', 'tech', 'om']],
                       how='left', on=['tech', 'fuel_mix']).set_index(tmp.index)

        tmp = pd.merge(tmp, fuelmix_cost, how='left', left_on='fuel_mix', right_index=True)
        #### MC = fuel_cost / (eff_c + eff_slope*year) + emission_factor* co2_price + om
        tmp.mc = tmp.fuelmix_cost / tmp.eta + tmp.om
        self.plants.mc[self.plants.mc.isnull()] = tmp.mc

    def unique_mc(self):
        """make mc's unique by adding a small amount"""
        for marginal_cost in self.plants.mc:
            self.plants.mc[self.plants.mc == marginal_cost] = self.plants.mc[self.plants.mc == marginal_cost] + \
            [int(x)*1E-1 for x in range(0, len(self.plants.mc[self.plants.mc == marginal_cost]))]

    def demand_per_zone(self):
        """ Calculate the demand per zone based on the general timeseries"""
        for node in self.nodes.index:
            self.demand_el[node] = self.demand_el[self.nodes.zone[node]] *\
                                   self.zones[self.year][self.nodes.zone[node]]*\
                                   self.nodes.demand_share[node]
        for elm in self.heatareas.index:
            self.demand_h[elm] = self.demand_h.profile1*self.heatareas[self.year][elm]

    def availability_per_plant(self):
        """Calculate the availability for generation that relies on timeseries"""

        # availability p and t in % of max capacity - only for tech that have
        # timeseries, the timeseries from ramses are in MW/TWh,
        # -> hourly ava [MW_available/ MW_installed] =
        # RAMSES_VALUE[MW_available/TWh] * flh [h] * [1 TW_inst/ 1E6 MW_inst]

        ts_tech = ['pv', 'wind_off', 'wind_on', "pvheat", "iheat"]
        flh = {'pv': 1000, 'wind_off': 4500, 'wind_on':2850, "pvheat": 1220, "iheat":3500}
        for elm in self.plants.index[self.plants.tech.isin(ts_tech)]:

            ts_zone = self.timeseries.zone == self.nodes.zone[self.plants.node[elm]]
            self.availability[elm] = self.timeseries[self.plants.tech[elm]][ts_zone].values
            self.availability[elm] = self.availability[elm]*flh[self.plants.tech[elm]]/1E6

        for elm in self.plants.index[self.plants.tech == "dem"]:
            ts_zone = self.timeseries.zone == self.nodes.zone[self.plants.node[elm]]
            self.availability[elm] = self.timeseries[self.plants.tech[elm]][ts_zone].values

    def add_line(self, node_i, node_j, typ, length, name=None):
        """ adding line to Lines DataFrame"""
        # Order of data
        #'node_i', 'node_j', 'type', 'length', 'name',
        #'contingency', 'capacity', 'flow', comm, decomm 'b'

        name = name or node_j + "_" + node_i
        capacity = {400: 1870, 220: 200, 150: 400}
        idx = 'l'+ "{0:0>3}".format(len(self.lines.index)+1)
        line_data = [node_i, node_j, typ, length, name,
                     True, capacity[typ], 0, 2010, 2017, length/typ]

        self.lines = self.lines.append(pd.DataFrame(data=[line_data], index=[idx],
                                                    columns=self.lines.columns))

