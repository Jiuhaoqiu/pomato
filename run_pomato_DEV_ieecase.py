from pomato.market_tool import MarketTool
import pandas as pd

mato = MarketTool(opt_file="opt_setup.json")
mato.load_matpower_case('case2383wp', autobuild_gridmodel=True)
# mato.load_matpower_case('case30', autobuild_gridmodel=True)
# mato.load_data_from_file('example_data/diw_demo.xlsx', autobuild_gridmodel=True)
# print(mato.data.nodes)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(mato.data.lines)

mato.grid_representation


print("OK")
