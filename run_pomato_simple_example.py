# if pomato is not yet added to PYPATH
import sys, os
module_in_path = os.getcwd() + '/pomato'
sys.path.append(module_in_path)

from pomato.market_tool import MarketTool

mato = MarketTool(opt_file="opt_setup_example.json", model_horizon=range(200,300))
mato.load_data_from_file('test_data/diw_demo.xlsx')
# mato.load_matpower_case('case118')

mato.init_market_model()
mato.market_model.run()

print("OK")
