# if pomato is not yet added to PYPATH
import sys, os
module_in_path = os.getcwd() + '/pomato'
sys.path.append(module_in_path)
from pomato.market_tool import MarketTool


if __name__ == "__main__":

    mato = MarketTool(opt_file="opt_setup.json")
    mato.load_matpower_case('case118')
    print(mato.data.nodes)
    print("OK")

    # MT = MarketTool(wdir, "diw_demo.xlsx", "opt_setup.json", 2017,
    #                  6, model_horizon=range(101, 200))

    # MT.data.nodes.net_injection = 0
    # MT.opt_setup["opt"] = "cbco_nodal"

    # MT.init_market_model()
    # MT.market_model.run()
    # o_lines = MT.check_n_1_for_marketresult()

    # net_injections = MT.market_model.return_results("INJ")
    # MT.grid.lineloading_timeseries(net_injections, "l002").plot()
    # MT.init_bokeh_plot(name="asd")
    # MT.bokeh_plot.start_server()
    #  # MT.bokeh_plot.stop_server()

# ++++++++++++++++++++++++

    # MT.grid.plot_fbmc(["DE"], ["DK-East"], gsk_sink={"DK-West": 0.5, "SE": 0.5})
    # MT.grid.plot_vertecies_of_inequalities(["DE"], ["DK-East"], gsk_sink={"DK-West": 0.5, "SE": 0.5})

#    self = MT
#    add_cbco = []
#    for t in o_lines:
#        for cbco in o_lines[t]:
#           if not [o_lines[t][cbco]["Line"], o_lines[t][cbco]["Outage"]] in add_cbco:
#               add_cbco.append([o_lines[t][cbco]["Line"], o_lines[t][cbco]["Outage"]])
#
#    MT.data.nodes.net_injection = 0
#    MT.grid.cbco_index = None
#    MT.grid.add_cbco = MT.grid.add_cbco + add_cbco
##    MT.grid.add_cbco =  add_cbco
#    MT.opt_setup["infeas_lines"] = False
#    MT.init_market_model()
#
#    MT.market_model.run()
#    o_lines = MT.check_n_1_for_marketresult()


#    cbco_2 = MT.grid.cbco_index
#    MT.grid.cbco_option ="gams"
#    MT.cbco_index = None
#
#
#    a = list(cbco_2)
#    b = list(cbco_1)
#
#    c = [x for x in a if x not in b]
#    d = [x for x in b if x not in a]

    #%%
#    from scipy.spatial import ConvexHull
#    from sklearn.decomposition import PCA
#    import numpy as np
#    from scipy.spatial import HalfspaceIntersection as HSI
#    import gams_cbco_reduction as cbco_reduction
#    wdir = Path.cwd()
#    MT = MarketTool(wdir, "test_data_nodc.xlsx", "opt_setup.json", 2017,
#                     6, model_horizon=range(101, 200))
#
#    MT.grid_representation
#    MT._gams = True
#    MT.data.nodes.net_injection = 0
#    MT.init_market_model()
#    indexed_ch = MT.grid.cbco_index
#    A, b = MT.grid.contingency_Ab("nodal", MT.grid.n_1_ptdf)
#    np.savetxt("A2.csv", np.asarray(A), delimiter=",")
#    np.savetxt("b2.csv", np.asarray(b), delimiter=",")


#    A.append([1 for x in range(0, len(MT.grid.nodes))])
#    b.append(0)
#
#    #%%
#    A = np.array(A)
#    b = np.array(b).reshape(len(b), 1)
#
#    l = len(MT.grid.lines)*2 + 1
#
#    A = A[-l:]
#    b = b[-l:]
#    Ab = np.concatenate((A,[-(x+1e-3) for x in b]), axis=1)
##    Ab = np.concatenate((A,-b), axis=1)
#
#
##
##    indexes = [x for x in range(0,165)] + [x for x in range(l,l+165)]
##
##    indexes = MT.grid.cbco_index
##    len(indexes)
##    indexes = [x for x in range(0,144)]
#    cbco = cbco_reduction.LPReduction(Path.cwd(), A, b)
##    cbco.algorithm()
#    test = cbco.feasible_point()
#    feasible_point = []
#    for i in range(0, len(MT.grid.nodes)):
#        try:
#            feasible_point.append(test[str(i)])
#        except:
#            feasible_point.append(0)
#    feasible_point = np.array(feasible_point)
#
#    test = HSI(Ab, feasible_point, qhull_options="QJ")
#
#    dir(test)
#    len(test.dual_vertices)
#    indexes = test.dual_vertices[:-1]

    #%%


#     for i in indexed_ch:
#         if i not in indexes:
#             print(i)


    #%%
#    MT.data.nodes.net_injection = 0
#
#    A, b = MT.grid.contingency_Ab("nodal", MT.grid.n_1_ptdf)
#    A.append([1 for x in range(0, len(MT.grid.nodes))])
#    b.append(0.1)
#    indexes = MT.grid.cbco_index
#    for i in range(0,5):
#        b_dash = np.take(b, indexes).reshape(len(indexes), 1)
#        A_dash = np.take(A, indexes, axis=0)
#        #    A = np.array(A)
#        #    b = np.array(b).reshape(len(b), 1)
#        D = A_dash/b_dash
#        model = PCA(n_components=6).fit(D)
#        D_t = model.transform(D)
#
#        k = ConvexHull(D_t, qhull_options="QJ") #, qhull_options = "QJ")
#        print(len(k.vertices))
#
#        if (len(b_dash)-1) in list(k.vertices):
#            cbco_rows = list(k.vertices)
#            cbco_rows.remove((len(b_dash)-1))
#            indexes = np.take(indexes, np.array(cbco_rows))
#        else:
#            indexes = np.take(indexes, k.vertices)
#
#
##    k.simpleces
#    nr = k.simplices

#    test = cbco.LPReduction(wdir, A, b)
#    test.algorithm()
#    nr1 = test.cbco_rows
#    nr = MT.grid.reduce_ptdf_gams(A, b)
#    test.cbco_counter

