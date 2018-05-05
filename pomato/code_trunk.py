
    def alt_reduce(self, contingency):
        self = MT.grid
        from scipy.spatial import HalfspaceIntersection
        A = []
        b = []
        for ptdf in contingency:
            ram_array = self.update_ram(ptdf, option="array")
            ram_pos = ram_array[:, 0]
            ram_neg = ram_array[:, 1]
            tmp_A = np.concatenate((-ptdf, ptdf), axis=0)
            tmp_b = np.concatenate((-ram_pos, ram_neg), axis=0)
            A += tmp_A.tolist()
            b += tmp_b.tolist()

        A = np.array(A)
        b = np.array(b).reshape(len(b), 1)
#        len(self.nodes)
        Ab = np.concatenate((A, b), axis=1)

        test = HalfspaceIntersection(Ab, np.zeros(len(self.nodes)), qhull_options="QJ")
#        dir(test)
        nr = test.dual_vertices
        return(Ab, nr)


### TEST of how fast gdx read in is for different ways of getting it into python
from gams import * #GamsWorkspace, GamsSet, GamsParameter, GamsOptions
import pandas as pd
import numpy as np
import timeit

pd.options.mode.chained_assignment = None  # default='warn'
## Load Result gdx and GRID Object
wdir = os.path.abspath("")
ws = GamsWorkspace(working_directory=wdir)
gams_db = ws.add_database_from_gdx("output.gdx")


%%timeit
g = []
for t in gams_db['t']:
    tmp = 0
    for p in gams_db['co']:
        tmp += gams_db["G"].find_record(keys=[p.get_keys()[0],t.get_keys()[0]]).level
    g.append(tmp)


%%timeit
g = []
for t in gams_db['t']:
    tmp = 0
    for p in gams_db['co']:
        tmp += gams_db["G"].find_record(keys=[p.get_keys()[0],t.get_keys()[0]]).level
    g.append(tmp)


%%timeit
l = []
for i in gams_db["G"]:
    l.append( i.keys + [i.level])
df = pd.DataFrame(columns=["plant", "t", "G"], data = l)
df.groupby(["t"]).sum()

%%timeit
t = 1
nodes = [Nodes.index[x] for x in range(13,26)]
t = 't'+ "{0:0>4}".format(int(t))
fuel = {}
fuel["type"] = ["gen", "dem"]
for f in set(Tech.fuel_mix):
    fuel[f] = []
for f in set(Tech.fuel_mix):
    tmp = 0
    for n in nodes:
        for p in Plants.index[(Plants.fuel_mix==f)&(Plants.g_max>0)&(Plants.node==n)]:
            tmp += gams_db["G"].find_record(keys=[p,t]).level
    fuel[f] = [tmp, 0]
tmp = 0
for n in nodes:
    try:
        tmp += gams_db["d_el"].find_record(keys=[n,t]).value
    except:
        tmp += 0
fuel["Dem"] = [0, tmp]

tmp = GMS.gams_symbol_to_df("G")
tmp = pd.merge(df_G, Plants[["node","fuel_mix"]], how="left", left_on="p", right_index=True)
DF_Fuel = tmp.groupby(["t","fuel_mix","node"], as_index=False).sum()

%%timeit
t = 1
nodes = [Nodes.index[x] for x in range(13,26)]
t = 't'+ "{0:0>4}".format(int(t))
tmp = fueldf[(fueldf.t==t)&(fueldf.node.isin(nodes))].groupby("fuel_mix").sum()
tmp["D"] = 0
tmp = tmp.append(pd.DataFrame([[0, gams_db["d_el"].find_record(keys=[n,t]).value]], index=["Dem"], columns=["G","D"]))



### OLD way of excecuting the market tool
def manual_exe():

    WDIR = os.path.abspath("")
    DATA = dm.DataManagement(WDIR, "\\data\\test_data.xlsx", co2_price=6)

#    DATA.add_line("SON", "SHE", 150, 26.4)
#    DATA.add_line("BBR", "KAS", 150, 28)
#    DATA.add_data_centers()


    OPT_SETUP =  {"opt": 'cbco_nodal', # dispatch, ntc, nodal, cbco_nodal, cbco_zonal
                  "infeas_heat": True, # Allow Infeasibilities in Heat EB
                  "infeas_el": True, # Allow Infeasibilities in EL EB
                  "infeas_lines": True, # Allow Infeasibilities on Lines
                  }

    add_cbco = {"cbco": [["l057", "l054"],
                     ["l137", "l112"],
                     ["l111", "l112"],
                     ["l113", "l112"],
            ]}
#    add_cbco = {}
    GRID = grm.GridModel(DATA.nodes, DATA.lines)
    GRID_REP = GRID.gms_grid_rep(OPT_SETUP["opt"], DATA.ntc, add_cbco=None)

#    GRID.shift_phase_on_line({"l139": -5, "l304": -5})
#
    JL = julia.JuliaInterface(WDIR, DATA, OPT_SETUP, GRID_REP, model_horizon=range(200,300))
    JL.data_to_json()
    JL.run()

    GMS = gms.GamsModel(WDIR, DATA, OPT_SETUP, GRID_REP, model_horizon=range(200,300))
    GMS.run()

    OBJ = {"julia": JL.results["Obj"], "GAMS": GMS.gams_symbol_to_df("COST").COST.values[0]}
    OBJ["abs_delta"] = abs(OBJ["julia"] - OBJ["GAMS"])
    OBJ["rel_delta"] = abs(OBJ["julia"] - OBJ["GAMS"])/max(OBJ["julia"], OBJ["GAMS"])

    bokeh_plot = bokeh.BokehPlot(WDIR, DATA)
    bokeh_plot.add_market_result_from_gdx(GMS, "gams")
    bokeh_plot.add_market_result_from_julia(JL, "julia")
    bokeh_plot.add_grid_object(GRID)
    bokeh_plot.start_server()
    bokeh_plot.stop_server()

#    G = pd.merge(JL.results["G"], GMS.gams_symbol_to_df("G"), how="outer", on=["p", "t"]).fillna(0)
#    H = pd.merge(JL.results["H"], GMS.gams_symbol_to_df("H"), on=["p", "t"]).fillna(0)
##    dd = pd.merge(JL.results["D_d"], GMS.gams_symbol_to_df("D_d"), on=["d", "t"]).fillna(0)
#    ph = pd.merge(JL.results["D_ph"], GMS.gams_symbol_to_df("D_ph"), on=["ph", "t"]).fillna(0)
##    es = pd.merge(JL.results["D_es"], GMS.gams_symbol_to_df("D_es"), on=["es", "t"]).fillna(0)
#    hs = pd.merge(JL.results["D_hs"], GMS.gams_symbol_to_df("D_hs"), on=["hs", "t"]).fillna(0)
#
#    INJ = pd.merge(JL.results["INJ"], GMS.gams_symbol_to_df("INJ"), on=["n", "t"]).fillna(0)
#    F_DC = pd.merge(JL.results["F_DC"], GMS.gams_symbol_to_df("F_DC"), on=["dc", "t"]).fillna(0)
#    EX = pd.merge(JL.results["EX"], GMS.gams_symbol_to_df("EX"), on=["z", "zz", "t"]).fillna(0)

#    prices_gms = GMS.nodal_prices()
#    prices_jl = JL.price()
#    prices = pd.merge(prices_jl, prices_gms, how="outer", on=["t", "n", "z"]).fillna(0)
#    prices["diff"] = prices.marginal_x - prices.marginal_y
#
    TIMESLICE = ['t'+ "{0:0>4}".format(x) for x in range(200,300)]
#    ol = GRID.check_n_1_for_marketresult(GMS.gams_symbol_to_df("INJ"), TIMESLICE, threshold=1000)
    ol = GRID.check_n_1_for_marketresult(JL.results["INJ"], TIMESLICE, threshold=1000)
#
#    GRID.lineloading_timeseries(GMS.gams_symbol_to_df("INJ"), "l117").plot()

#    GMS.plot_generation_area(option="fuel")
#    GRID.plot_fbmc(["DK-East"], ["DK-West"])





    def update_net_injection(self, Nodes, Plants, time):
    #        gamsdb = gamsdb
    #        time = 't0001'
        # Update Net Injection
        Nodes.net_injection = Nodes.net_injection.astype(float)
        for n in Nodes.index:
            dem = gamsdb["d_el"].find_record([n, time])
            Nodes.set_value(n, 'net_injection', -dem.value)
            Nodes.net_injection[n] = -dem.value

        for p in gamsdb["co"]:
            gen = gamsdb["G"].find_record(keys=[p.get_keys()[0], time])
            Nodes.set_value(Plants.node[p.keys[0]], 'net_injection', \
                            Nodes.net_injection[Plants.node[p.keys[0]]] + gen.level)
        for p in gamsdb["es"]:
            stor = gamsdb["D_es"].find_record(keys=[p.get_keys()[0], time])
            Nodes.set_value(Plants.node[p.keys[0]], 'net_injection', \
                            Nodes.net_injection[Plants.node[p.keys[0]]] - stor.level)
        for p in gamsdb["hp"]:
            stor = gamsdb["D_hp"].find_record(keys=[p.get_keys()[0], time])
            Nodes.set_value(Plants.node[p.keys[0]], 'net_injection', \
                            Nodes.net_injection[Plants.node[p.keys[0]]] - stor.level)

    def update_gsk(self, Nodes, Plants, option, time):
        # Update GSK based on the marginal price per zone/node
        # -> marginal Plants provide increased Generation
        ref_price = {}
        if option in ['Nodal', 'CBCO']:
            for n in Nodes.index:
                price = -gamsdb["EB_Nodal"].find_record(keys=[n, time]).marginal
                ref_price[n] = -price
        else:
            for z in Nodes.zone:
                price = -gamsdb["EB_Dispatch_NTC"].find_record(keys=[z, time]).marginal
                ref_price[z] = -price

        Nodes.loc[Nodes.index, 'gsk'] = 1
        for g in Plants.index:
            if option in ['Nodal', 'CBCO'] \
            and Plants.mc[g] <= ref_price[Plants.node[g]]*1.1:
                Nodes.loc[Nodes.index == Plants.node[g], 'gsk'] += 1

            elif option not in ['Nodal', 'CBCO'] \
            and Plants.mc[g] <= ref_price[Nodes.zone[Plants.node[g]]]*1:
                Nodes.loc[Nodes.index == Plants.node[g], 'gsk'] += 1
    #        Nodes.loc[Nodes.index == 'C3', 'gsk'] += 15
    #
    #        print('Nodes Updated with net injections and GSK')
#

#nodes_tva_xls = pd.ExcelFile(wdir + "\\nodes_tva.xlsx")
#lines_tva_xls = pd.ExcelFile(wdir + "\\lines_tva.xlsx")
#
#Lines = lines_tva_xls.parse('Sheet1', index_col = 0)
#Nodes = nodes_tva_xls.parse('Sheet1', index_col = 1, parse_cols=tools.a2i("G"))
#
#naming_nodes = {'lat_anders': "node_lat",
#                "lon_anders": "node_lon",
#                }
#Nodes = Nodes.rename(columns = naming_nodes)
#
#for l in Lines.index:
#    if Lines.node_i[l] not in Nodes.index and \
#        Lines.node_j[l] not in Nodes.index:
#            print(1)
#            Lines = Lines.drop(l)







