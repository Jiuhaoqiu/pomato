from pomato.market_tool import MarketTool

mato = MarketTool(opt_file="opt_setup.json", model_horizon=range(200,300))
mato.load_data_from_file('example_data/diw_demo.xlsx')
# mato.load_matpower_case('case118')

mato.init_market_model()
mato.run_market_model()

mato.plot_grid_object()

print("OK")
