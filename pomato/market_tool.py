import logging
import json
import pandas as pd
from pathlib import Path


from . import MPCASES
from .data_management import DataManagement
from .grid_model import GridModel



def _logging_setup(wdir):
    # Logging setup
    logger = logging.getLogger('Log.MarketModel')
    logger.setLevel(logging.INFO)
    if len(logger.handlers) < 2:
        # create file handler which logs even debug messages
        file_handler = logging.FileHandler(wdir.joinpath('market_tool.log'))
        file_handler.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                      "%d.%m.%Y %H:%M")
        file_handler.setFormatter(fh_formatter)
        # Only message in Console
        ch_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(ch_formatter)
        # add the handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


class MarketTool(object):
    """ Main Class"""
    def __init__(self, **kwargs):
        self.wdir = Path.cwd()
        self.logger = _logging_setup(self.wdir)

        self._parse_kwargs(kwargs)
        self.logger.info("Market Tool Initialized")

        self.data = DataManagement()
        self.grid = GridModel()
        
        ## TODO, [rom]
        ## Not sure what does do, but all objectes should be initalized and 
        ## then filled when needed
         ## Core Attributes
        self.market_model = None
        self.bokeh_plot = None
        # Hidden option
        self._gams = False

    def load_data_from_file(self, filename, year=2017, co2_price=6, autobuild_gridmodel=True):
        datafile = Path(filename)
        if datafile.is_file():
            self.data.read_xls_data(datafile, year, co2_price)
            # Legacy, not yet properly integrated
            # e.g. no empty init possible 
            # wait for richards major commit
            # self.grid = GridModel(self.data.nodes, self.data.lines) 
            if autobuild_gridmodel:
                try: 
                    self.grid.build_grid_model(self.data.nodes, self.data.lines)
                except:
                    self.looger.info("Grid Model has not been build")
        else:
            self.logger.exception("File {} can not be found!".format(filename))

    def load_matpower_case(self, casename, autobuild_gridmodel=False):
        case = Path(casename)
        if case.is_file():
            self.data.read_matpower_case(case)
        elif casename in MPCASES:
            case = Path('mp_casedata/{}.mat'.format(casename))
            self.data.read_matpower_case(case)
        else:
            self.logger.exception("MP Case {} can not be found!".format(casename))
            return
        if autobuild_gridmodel:
            try: 
                self.grid.build_grid_model(self.data.nodes, self.data.lines)
            except:
                self.looger.info("Grid Model has not been build")

    def clear_data(self):
        self.logger.info("Resetting Data Object")
        self.data = DataManagement()
        

    def _parse_kwargs(self, kwargs):
        options = [
                    'opt_file'  # define location of a .json options file
                    ]
        for ka in kwargs.keys():
            if not(ka in options):
                self.logger.warn("Unknown keyword: {}".ka)
        
        if 'opt_file' in kwargs.keys():
            with open(self.wdir.joinpath(kwargs['opt_file'])) as ofile:
                self.opt_setup = json.load(ofile)
                opt_str = "Optimization Options:" + json.dumps(self.opt_setup, indent=2) + "\n"
            self.logger.info(opt_str)

    @property
    def grid_representation(self):
        """Grid Representation as property, get recalculated when accessed via dot"""
        return self.grid.grid_representation(self.opt_setup["opt"], self.data.ntc)

    def init_market_model(self):
        """init market model"""
        if not self._gams:
            self.market_model = julia.JuliaInterface(self.wdir, self.data, self.opt_setup,
                                                     self.grid_representation, self.model_horizon)
        else:
            self.market_model = gms.GamsModel(self.wdir, self.data, self.opt_setup,
                                              self.grid_representation, self.model_horizon)

    def init_bokeh_plot(self, add_grid=True, name="default"):
        """init boke plot (saves market result and grid object)"""
        self.bokeh_plot = bokeh.BokehPlot(self.wdir, self.data)
        self.bokeh_plot.add_market_result(self.market_model, name)

        if add_grid:
            self.bokeh_plot.add_grid_object(self.grid)

    def check_n_1_for_marketresult(self):
        """Interface with grid model check n-1 method"""
        overloaded_lines = \
        self.grid.check_n_1_for_marketresult(self.market_model.return_results("INJ"),
                                             self.market_model.model_horizon,
                                             threshold=1000)
        return overloaded_lines
