import numpy as np
import pkg_resources as pkr


JULIA_PATH = pkr.resource_filename('pomato', 'julia-files/')

MPCASE_PATH = pkr.resource_filename('pomato', 'mp_casedata/')


MPCASES = ['case118', 'case2737sop', 'case3012wp', 'case4gs', 'case9', 'case1354pegase', 
        'case2746wop', 'case30Q', 'case5', 'case9241pegase', 'case14', 'case2746wp', 
        'case30pwl', 'case57', 'case96', 'case2383wp', 'case2869pegase', 'case3120sp', 
        'case6ww', 'case9Q', 'case24_ieee_rts', 'case30', 'case3375wp', 'case79', 
        'case9target', 'case2736sp', 'case300', 'case39', 'case89pegase', 'case_ieee30']


MPCOLNAMES = {'bus_keys': np.array(['bus_i', 'b_type', 'Pd', 'Qd', 'Gs', 'Bs', 'area', 
                                        'Vm', 'Va', 'baseKV', 'zone', 'Vmax', 'Vmin']),
                'gen_keys': np.array(['bus', 'Pg', 'Qg', 'Qmax', 'Qmin', 'Vg', 'mBase', 
                                    'status', 'Pmax', 'Pmin', 'Pc1', 'Pc2', 'Qc1min', 
                                    'Qc1max', 'Qc2min', 'Qc2max', 'ramp_agc', 'ramp_10', 
                                    'ramp_30', 'ramp_q', 'apf']),
                'branch_keys': np.array(['fbus', 'tbus', 'r', 'x', 'b', 'rateA', 'rateB',  
                                    'rateC', 'ratio', 'angle', 'status', 'angmin', 'angmax']),
                'gencost_keys': np.array(['model', 'startup', 'shutdown', 'n'])}
