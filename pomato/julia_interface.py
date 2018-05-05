"""
    This is the Julia Interface. It Does:
    Read and save the relevant data into julia/data
"""
import logging
import subprocess
import json
import datetime
import pandas as pd

class JuliaInterface(object):
    """ Class to interface the Julia model with the python Market and Grid Model"""
    def __init__(self, wdir, DATA, opt_setup, grid_representation, model_horizon):
        # Import Logger
        self.logger = logging.getLogger('Log.MarketModel.JuliaInterface')
        self.wdir = wdir
        self.jdir = wdir.joinpath("julia")
        self.create_folders(self.wdir)

        self.opt_setup = opt_setup
        self.model_horizon = ['t'+ "{0:0>4}".format(x) for x in model_horizon]

        self.grid_representation = grid_representation
        self.nodes = DATA.nodes[["name", "zone", "slack"]]
        self.zones = DATA.zones
        self.plants = DATA.plants[['mc', 'tech', 'node', 'eta', 'g_max', 'h_max',
                                   'heatarea']]
        self.heatareas = DATA.heatareas

        self.demand_el = DATA.demand_el[DATA.demand_el.index.isin(self.model_horizon)]
        self.demand_h = DATA.demand_h[DATA.demand_h.index.isin(self.model_horizon)]
        self.availability = DATA.availability
        self.dclines = DATA.dclines[["node_i", "node_j", "capacity"]]
        self.ntc = DATA.ntc

        self.data_to_csv()
        self.data_to_json()

        self.results = {}

    def create_folders(self, wdir):
        """ create folders for bokeh interface"""
        if not wdir.joinpath("julia").is_dir():
            wdir.joinpath("julia").mkdir()
        if not self.jdir.joinpath("data").is_dir():
            self.jdir.joinpath("data").mkdir()
        if not self.jdir.joinpath("results").is_dir():
            self.jdir.joinpath("results").mkdir()
        if not self.jdir.joinpath("data").joinpath("json").is_dir():
            self.jdir.joinpath("data").joinpath("json").mkdir()

    def run(self):
        """Run the julia Programm via command Line"""
        args = ["julia", str(self.jdir.joinpath("main.jl")), str(self.jdir)]

        t_start = datetime.datetime.now()
        self.logger.info("Start-Time: " + t_start.strftime("%H:%M:%S"))
        with open(self.wdir.joinpath('julia.log'), 'w') as log:
            # shell=false needed for mac (and for Unix in general I guess)
            with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE) as programm:
                for line in programm.stdout:
                    log.write(line.decode())
                    self.logger.info(line.decode().strip())
#                    self.logger.info(line.decode())

        if programm.returncode == 1:
            ## have to rerun it to catch the error message :(
            ## there might be a better option
            self.logger.error("error in Julia Code!\n")
            programm = subprocess.Popen(args, shell=False, stderr=subprocess.PIPE)
            _, stderr = programm.communicate()
            self.logger.error(stderr.decode())
            ## Write Log file
            with open(self.wdir.joinpath('julia.log'), 'a') as log:
                log.write(stderr.decode())
            self.logger.info("julia.log saved!")

        t_end = datetime.datetime.now()
        self.logger.info("End-Time: " + t_end.strftime("%H:%M:%S"))
        self.logger.info("Total Time: " + str((t_end-t_start).total_seconds()) + " sec")

        if programm.returncode == 0:
            self.results["G"] = pd.read_json(self.jdir.joinpath("results").joinpath("G.json"),
                                             orient="index").sort_index()
            self.results["H"] = pd.read_json(self.jdir.joinpath("results").joinpath("H.json"),
                                             orient="index").sort_index()
            self.results["D_es"] = pd.read_json(self.jdir.joinpath("results").joinpath("D_es.json"),
                                                orient="index").sort_index()
            self.results["d_el"] = pd.read_json(self.jdir.joinpath("results").joinpath("d_el.json"),
                                                orient="index").sort_index()
            self.results["L_es"] = pd.read_json(self.jdir.joinpath("results").joinpath("L_es.json"),
                                                orient="index").sort_index()
            self.results["D_hs"] = pd.read_json(self.jdir.joinpath("results").joinpath("D_hs.json"),
                                                orient="index").sort_index()
            self.results["L_hs"] = pd.read_json(self.jdir.joinpath("results").joinpath("L_hs.json"),
                                                orient="index").sort_index()
            self.results["D_ph"] = pd.read_json(self.jdir.joinpath("results").joinpath("D_ph.json"),
                                                orient="index").sort_index()
            self.results["D_d"] = pd.read_json(self.jdir.joinpath("results").joinpath("D_d.json"),
                                               orient="index").sort_index()
            self.results["EX"] = pd.read_json(self.jdir.joinpath("results").joinpath("EX.json"),
                                              orient="index").sort_index()
            self.results["INFEAS_H"] = pd.read_json(self.jdir.joinpath("results").joinpath("INFEAS_H.json"),
                                                    orient="index").sort_index()
            self.results["INFEAS_EL"] = pd.read_json(self.jdir.joinpath("results").joinpath("INFEAS_EL.json"),
                                                     orient="index").sort_index()
            self.results["INJ"] = pd.read_json(self.jdir.joinpath("results").joinpath("INJ.json"),
                                               orient="index").sort_index()
            self.results["F_DC"] = pd.read_json(self.jdir.joinpath("results").joinpath("F_DC.json"),
                                                orient="index").sort_index()
            self.results["INFEAS_LINES"] = pd.read_json(self.jdir.joinpath("results").joinpath("INFEAS_LINES.json"),
                                                        orient="index").sort_index()
            self.results["EB_nodal"] = pd.read_json(self.jdir.joinpath("results").joinpath("EB_nodal.json"),
                                                    orient="index").sort_index()
            self.results["EB_zonal"] = pd.read_json(self.jdir.joinpath("results").joinpath("EB_zonal.json"),
                                                    orient="index").sort_index()

            with open(self.jdir.joinpath("results").joinpath("misc_result.json"), "r") as jsonfile:
                data = json.load(jsonfile)
            self.results["COST"] = data["Objective Value"]
            self.check_for_infeas()

    def check_for_infeas(self):
        """
        checks for infeasiblities in electricity/heat energy balances
        returns nothing
        """
        self.logger.info("Check for infeasiblities in electricity energy balance...")
        infeas = self.return_results("INFEAS_EL")
        infeas_pos = infeas[infeas.INFEAS_EL_POS >= 1E-6]
        infeas_neg = infeas[infeas.INFEAS_EL_NEG >= 1E-6]

        if not (infeas_pos.empty and infeas_neg.empty):
            nr_n = len(infeas_neg.groupby(["n"]).count())
            nr_t = len(infeas_neg.groupby(["t"]).count())
            self.logger.info("Negative infeasibilities in " + str(nr_t) +
                             " timesteps and at " + str(nr_n) + " different nodes")
            nr_n = len(infeas_pos.groupby(["n"]).count())
            nr_t = len(infeas_pos.groupby(["t"]).count())
            self.logger.info("Positive infeasibilities in " + str(nr_t) +
                             " timesteps and at " + str(nr_n) + " different nodes")

        self.logger.info("Check for infeasiblities in heat energy balance...")
        infeas = self.return_results("INFEAS_H")
        infeas_pos = infeas[infeas.INFEAS_H_POS >= 1E-6]
        infeas_neg = infeas[infeas.INFEAS_H_NEG >= 1E-6]

        if not (infeas_pos.empty and infeas_neg.empty):
            nr_n = len(infeas_neg.groupby(["ha"]).count())
            nr_t = len(infeas_neg.groupby(["t"]).count())
            self.logger.info("Negative infeasibilities in " + str(nr_t) +
                             " timesteps and at " + str(nr_n) + " different nodes")
            nr_n = len(infeas_pos.groupby(["ha"]).count())
            nr_t = len(infeas_pos.groupby(["t"]).count())
            self.logger.info("Positive infeasibilities in " + str(nr_t) +
                             " timesteps and at " + str(nr_n) + " different nodes")

        if self.opt_setup["opt"] in ["cbco_nodal", "cbco_zonal"]:
            self.logger.info("Check for infeasiblities on Lines...")
            infeas_lines = self.return_results("INFEAS_LINES")
            infeas_lines = infeas_lines[infeas_lines.INFEAS_LINES >= 1E-6]
            if not infeas_lines.empty:
                nr_cb = len(infeas_lines.groupby(["cb"]).count())
                nr_t = len(infeas_lines.groupby(["t"]).count())
                self.logger.info("Infeasibilities in " + str(nr_t) +
                                 " timesteps and at " + str(nr_cb) + " different cbcos")

    def data_to_json(self):
        """Export Data to json files in the jdir + json_path"""
        json_path = self.jdir.joinpath('data').joinpath('json')
        self.plants.to_json(str(json_path.joinpath('plants.json')), orient='index')
        self.nodes.to_json(str(json_path.joinpath('nodes.json')), orient='index')
        self.zones.to_json(str(json_path.joinpath('zones.json')), orient='index')
        self.heatareas.to_json(str(json_path.joinpath('heatareas.json')), orient='index')
        self.demand_el.to_json(str(json_path.joinpath('demand_el.json')), orient='index')
        self.demand_h.to_json(str(json_path.joinpath('demand_h.json')), orient='index')
        self.availability.to_json(str(json_path.joinpath('availability.json')), orient='index')
        self.ntc.to_json(str(json_path.joinpath('ntc.json')), orient='split')
        self.dclines.to_json(str(json_path.joinpath('dclines.json')), orient='index')

    def data_to_csv(self):
        """Export Data to csv files file in the jdir + \\data"""
        csv_path = self.jdir.joinpath('data')
        self.plants.to_csv(str(csv_path.joinpath('plants.csv')), index_label='index')
        self.nodes.to_csv(str(csv_path.joinpath('nodes.csv')), index_label='index')
        self.zones.to_csv(str(csv_path.joinpath('zones.csv')), index_label='index')
        self.heatareas.to_csv(str(csv_path.joinpath('heatareas.csv')), index_label='index')
        self.demand_el.to_csv(str(csv_path.joinpath('demand_el.csv')), index_label='index')
        self.demand_h.to_csv(str(csv_path.joinpath('demand_h.csv')), index_label='index')
        self.availability.to_csv(str(csv_path.joinpath('availability.csv')), index_label='index')
        self.ntc.to_csv(str(csv_path.joinpath('ntc.csv')), index=False)
        self.dclines.to_csv(str(csv_path.joinpath('dclines.csv')), index_label='index')

        try:
            with open(csv_path.joinpath('cbco.json'), 'w') as file:
                json.dump(self.grid_representation["cbco"], file)
        except:
            self.logger.warning("CBCO.json not found - Check if relevant for the model")
        try:
            with open(csv_path.joinpath('ptdf.json'), 'w') as file:
                json.dump(self.grid_representation["ptdf"], file)
        except:
            self.logger.warning("ptdf.json not found - Check if relevant for the model")
        try:
            with open(csv_path.joinpath('slack_zones.json'), 'w') as file:
                json.dump(self.grid_representation["slack_zones"], file)
        except:
            self.logger.warning("slack_zones.json not found - Check if relevant for the model")
        try:
            with open(csv_path.joinpath('opt_setup.json'), 'w') as file:
                json.dump(self.opt_setup, file)
        except:
            self.logger.warning("opt_setup.json not found - Check if relevant for the model")

    def price(self):
        """returns nodal electricity price"""
        eb_nodal = self.results["EB_nodal"]
        eb_nodal = pd.merge(eb_nodal, self.nodes.zone.to_frame(),
                            how="left", left_on="n", right_index=True)
        eb_nodal.EB_nodal[abs(eb_nodal.EB_nodal) < 1E-3] = 0

        eb_zonal = self.results["EB_zonal"]
        eb_zonal.EB_zonal[abs(eb_zonal.EB_zonal) < 1E-3] = 0

        price = pd.merge(eb_nodal, eb_zonal, how="left",
                         left_on=["t", "zone"], right_on=["t", "z"])

        price["marginal"] = -(price.EB_zonal + price.EB_nodal)

        return price[["t", "n", "z", "marginal"]]
    def return_results(self, symb):
        """interface method to allow access to results alalog to the gms class"""
        try:
            return self.results[symb]
        except:
            self.logger.error("Symbol not in Results")
