
import logging
import subprocess
import json
import datetime
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
#import gams_cbco_reduction as cbco_reduction

class CBCOModule(object):
    """ Class to do all calculations in connection with cbco calculation"""
    def __init__(self, wdir, nodes, lines, A, b, add_cbco=None):
        # Import Logger
        self.logger = logging.getLogger('Log.MarketModel.CBCOModule')

        self.wdir = wdir
        self.nodes = nodes
        self.lines = lines
        self.A = np.array(A)
        self.b = np.array(b).reshape(len(b), 1)
        self.create_folders(wdir)
        
        self.cbco_index = np.array([], dtype=np.int8)
        ##add specific cbco manually (if add_cbco)
        if add_cbco:
            self.add_to_cbco_index(self.return_index_from_cbco(add_cbco))

    def __getstate__(self):
        """
        Method to remove logger attribute from __dict__
        needed when pickeled
        """
        d = dict(self.__dict__)
        del d["logger"]
        return d

    def __setstate__(self, d):
        """
        Method updates self with modified __dict__ without logger
        needed when pickeled
        """
        self.__dict__.update(d) # I *think* this is a safe way to do it


    def main(self, use_precalc=False, only_convex_hull=True):
        if use_precalc:
            try:
                self.logger.info("Using cbco indices from pre-calc")
                precalc_cbco = np.genfromtxt(self.wdir.joinpath("julia/cbco_data/cbco.csv"), delimiter=',')
                precalc_cbco = self.cbco_index_positive_to_full_Ab(precalc_cbco)
                self.cbco_index = np.array(precalc_cbco, dtype=np.int8)
                self.logger.info("Number of CBCOs from pre-calc: " + str(len(self.cbco_index)))
            except FileNotFoundError:
                self.logger.warning("FileNotFound: No Precalc available")
                self.logger.warning("Running nomal CBCO Algorithm - ConvexHull only")
                use_precalc = False
                only_convex_hull=True
        
        if not use_precalc:
            self.cbco_algorithm(only_convex_hull)
        info = {}
        for n in self.cbco_index:
            info[n] = self.create_cbcomodule_return(n)
        cbco = {}
        for i in self.cbco_index: # range(0, len(b)): #
            cbco['cbco'+ "{0:0>4}".format(i+1)] = {'ptdf': list(self.A[i]), 'ram': int(self.b[i])}
        return(info, cbco)

    def add_to_cbco_index(self, add_cbco):
        """adds the indecies of the manulally added cbco to cbco_index"""
        # make sure its np.array
        if not isinstance(add_cbco, np.ndarray):
            add_cbco = np.array(add_cbco, dtype=np.int8)
        self.cbco_index = np.union1d(self.cbco_index, add_cbco)

    def create_folders(self, wdir):
        """ create folders for julia cbco_analysis"""
        if not wdir.joinpath("julia").is_dir():
            wdir.joinpath("julia").mkdir()
        if not wdir.joinpath("julia/cbco_data").is_dir():
            wdir.joinpath("julia/cbco_data").mkdir()

    def julia_cbco_interface(self, A, b, cbco_index):
        ## save A,b to csv
        ## save cbco_index for starting set A', and b' as csv
        np.savetxt(self.wdir.joinpath("julia/cbco_data/A.csv"), np.asarray(A), delimiter=",")
        np.savetxt(self.wdir.joinpath("julia/cbco_data/b.csv"), np.asarray(b), delimiter=",")
        ## fmt='%i' is needed to save as integer
        np.savetxt(self.wdir.joinpath("julia/cbco_data/cbco_index.csv"), cbco_index.astype(int),
                   fmt='%i', delimiter=",")

        args = ["julia", str(self.wdir.joinpath("julia/cbco.jl")), str(self.wdir.joinpath("julia"))]
        t_start = datetime.datetime.now()
        self.logger.info("Start-Time: " + t_start.strftime("%H:%M:%S"))
        with open(self.wdir.joinpath('cbco_reduction.log'), 'w') as log:
            # shell=false needed for mac (and for Unix in general I guess)
            with subprocess.Popen(args, shell=False, stdout=subprocess.PIPE) as programm:
                for line in programm.stdout:
                    log.write(line.decode())
                    self.logger.info(line.decode().strip())

        if programm.returncode == 0:
            tmp_cbco = np.genfromtxt(self.wdir.joinpath("julia/cbco_data/cbco.csv"), delimiter=',')
#            tmp_cbco = self.add_negative_constraints(tmp_cbco)
            return np.array(tmp_cbco, dtype=np.int8)
        else:
            self.logger.critical("Error in Julia code")

    def cbco_algorithm(self, only_convex_hull):
        """
        Creating Ax = b Based on the list of N-1 ptdfs and ram
        Reduce it by:
        1) using the convex hull method to get a subset A' from A, where A'x<=b'
           is a non rendundant system of inequalities
        2) Using the julia algorithm to check all linear inequalities of A
           against A' and add those which are non-redundant but missed bc of
           the low dim of the convexhull problem
        """
        try:
            self.add_to_cbco_index(self.reduce_Ab_convex_hull())
            self.logger.info("Number of CBCOs from ConvexHull Method: " + str(len(self.cbco_index)))
            
            if not only_convex_hull:
                self.logger.info("Running Julia CBCO-Algorithm...")
                
                A, b = self.return_positive_cbcos()
                cbco_index = self.cbco_index_full_to_positive_Ab(self.cbco_index)
                self.logger.info("Number of positive CBCOs for Julia CBCO-Algorithm: " + str(len(cbco_index)))
                
                cbco_index = self.julia_cbco_interface(A, b, cbco_index)
                self.cbco_index = self.cbco_index_positive_to_full_Ab(cbco_index)
                self.logger.info("Number of CBCOs after Julia CBCO-Algorithm: " + str(len(self.cbco_index)))

        except:
            self.logger.exception('e:cbco')

    def reduce_Ab_convex_hull(self):
        """
        Given an system Ax = b, where A is a list of ptdf and b the corresponding ram
        Reduce will find the set of ptdf equations which constrain the solution domain
        (which are based on the N-1 ptdfs)
        """
        try:
#            self = MT.grid
            D = self.A/self.b
            if len(self.A[0]) > 4:
                model = PCA(n_components=6).fit(D)
                D_t = model.transform(D)
                k = ConvexHull(D_t, qhull_options="QJ")
            else:
                k = ConvexHull(D) #, qhull_options = "QJ")

            return k.vertices #np.array(cbco_rows)
        except:
            self.logger.exception('error:reduce_ptdf')

#    pos = [x for x in range(1,10)]
#    neg = [-x for x in range(1,10)]
#    
#    test = pos + neg + pos + neg
#    array_cbco_index = np.array(test)
#    array_cbco_index = np.array([0, 9, 18, 27])
#    array_cbco_index = np.array(tmp)
#    lines = 9
#    np.array(test)[18]
    
    def cbco_index_full_to_positive_Ab(self, array_cbco_index):
        """Get the corresponding indecies for the postive Ab Matrix from the 
        cbco indecies for the full Ab matrix
        """
        ## Only pos constraints
        idx_pos = []
        for i in array_cbco_index:
#            idx_pos.append(int(i/lines)%2==0)
            idx_pos.append(int(i/len(self.lines))%2==0)
        idx_pos = np.array(idx_pos)
        
        ## Map to the only-positive indices
        tmp = []
        for i in array_cbco_index[idx_pos]:
#            tmp.append(i - ((lines)*(int((i-1)/(lines)))))
             tmp.append(i - (len(self.lines)*(int((i-1)/len(self.lines)))))

        return np.array(tmp)
    
    def cbco_index_positive_to_full_Ab(self, array_cbco_index):
        """Get the corresponding indecies for the full Ab Matrix from the 
        cbco indecies Ab matrix for only positve constraints 
        """
        ## Map to full matrix indices
        idx = []
        for i in array_cbco_index:
            idx.append(i + (len(self.lines)*(int((i-1)/len(self.lines)))))
#            idx.append(i + (len(self.lines)*(int(i/len(self.lines)) - 1)))
        idx = np.array(idx)
        ## add corresponding negative constraints
        tmp = []
        for i in idx:
            tmp.append(i)
            tmp.append(i + (len(self.lines)))
        return np.array(tmp)
    
    def return_positive_cbcos(self):
        """return A', b', where they are the posotive cbcos from A,b
            optional: return an array of cbcos, that belong to a positive constraint"""
        idx_pos = []
        for i in range(0, len(self.b)):
            idx_pos.append(int(i/len(self.lines))%2==0)
        idx_pos = np.array(idx_pos)
        return self.A[idx_pos], self.b[idx_pos]

    def return_index_from_cbco(self, additional_cbco):
        """Creates a list of cbco_indecies for list of [cb,co], both pos/neg constraints"""
        cbco_index = []
        for [line, out] in additional_cbco:
            cbco_index.append((self.lines.index.get_loc(out) + 1)*2*len(self.lines) + \
                               self.lines.index.get_loc(line))
            cbco_index.append((self.lines.index.get_loc(out) + 1)*2*len(self.lines) + \
                               len(self.lines) + self.lines.index.get_loc(line))
        return cbco_index

    def create_cbcomodule_return(self, index):
        """translates cbco index from reduce method to cb-co from lines data"""
        # Ordering: L+1 N-1 ptdfs, [N-0, N-1_l1, N-1_l2 .... N-1_lL]
        # Each ptdf contains L*2 equations ptdf_1, ... , ptdf_N, ram
        # 0-L constraints have positive ram, L+1 - 2*L Negative ram
        pos_or_neg = {0: 'pos', 1: 'neg'}
        if index/(len(self.lines)*2) < 1: # N-0 ptdf
            info = {'Line': self.lines.index[int(index%(len(self.lines)))],
                    '+/-':  pos_or_neg[int(index/len(self.lines))%2],
                    'Outage': 'N-0'}
        else: # N-1 ptdfs
            info = {'Line': self.lines.index[int(index%(len(self.lines)))],
                    '+/-':  pos_or_neg[int(index/len(self.lines))%2],
                    'Outage': self.lines.index[int(index/(len(self.lines)*2))-1]}
        return info

    #### OLD CODE
    def reduce_ptdf_gams(self, A, b):
        """ Equiv to reduce_ptdf but using gams algorithm"""
        from pathlib import Path
        self.logger.critical("DEPRICATED")
        #add slack to cbco calc sum INJ = 0 for all slack zones
        slack_zones_idx = self.slack_zones_index()
        slack_cbco = []
        for nodes_idx in slack_zones_idx:
            add = np.zeros(len(self.nodes))
            add[nodes_idx] = 1
            add = add.reshape(1, len(self.nodes))
            A = np.concatenate((A, add), axis=0)
            b = np.append(b, [0.00001])
            b = b.reshape(len(b), 1)
            slack_cbco.append(len(b)-1)
        cbco = cbco_reduction.LPReduction(Path.cwd(), A, b)
        cbco.algorithm()
        cbco_rows = cbco.cbco_rows
        for index in slack_cbco:
            if index in cbco_rows:
                cbco_rows.remove(index)
        return np.array(cbco_rows)