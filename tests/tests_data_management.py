import sys, os
module_in_path = '/Users/Balu/tubcloud/Market_Tool/pomato_repo/'
sys.path.append(module_in_path)

from pomato.data_management import DataManagement
# from pomato.market_tool import MarketTool
import pickle
import unittest


class TestMPDataLoad(unittest.TestCase):

    def test_mpc_with_name(self):
        reference_data = pickle.load(open("plants9Q.p", "rb"))
        dm = DataManagement()
        casefile = './mp_casedata/case9Q.mat'
        dm.read_matpower_case(casefile)
        self.assertEqual(dm.plants, reference_data)

if __name__ == '__main__':
    unittest.main()
