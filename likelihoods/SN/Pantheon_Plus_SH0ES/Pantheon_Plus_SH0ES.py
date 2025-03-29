import numpy as np
import scipy.linalg as la
import numexpr as ne
import pandas as pd
from pandas import read_table
import os,sys

try:
    import numexpr as ne
except ImportError:
    raise io_mp.MissingLibraryError(
        "This likelihood has intensive array manipulations. You "
        "have to install the numexpr Python package. Please type:\n"
        "pip install numexpr")
    

try:
    from cobaya.likelihood import Likelihood
    print('Importiong Pantheon+ as cobaya likelihood')
except:
    class Likelihood:  # dummy class to inherit if cobaya is missing
        print('dummy class to inherit')
        pass
    

class Pantheon_Plus_SH0ES(Likelihood):
    
    name: str = "Pantheon_Plus_SH0ES"
    
    def initialize(self):

        current_path = os.path.abspath(__file__)
        like_path= os.path.abspath(os.path.join(current_path, os.pardir))
        self.path_covmat = like_path + '/data/Pantheon+SH0ES_STAT+SYS.cov'
        self.path_lc = like_path + '/data/Pantheon+SH0ES.dat'

        self.z_min=0.023
        
        #Reading the covariance matrix
        with open(self.path_covmat, 'r') as text:
            length = int(text.readline())
        self.C00 = read_table(self.path_covmat).to_numpy().reshape((length, length))
         
        #Reading ligth curve params
        with open(self.path_lc, 'r') as text:
            clean_first_line = text.readline()[1:].strip()
            names = [e.strip().replace('3rd', 'third')
                     for e in clean_first_line.split()]
        self.light_curve_params = read_table(self.path_lc, sep=' ', names=names, header=0, index_col=False)
        
        C00 = self.C00
        covm = ne.evaluate("C00")
        mask=[]
        
        sn = self.light_curve_params
        true_size=0
        ignored = 0
        for ii in range(len(self.light_curve_params.zHD)):
                if self.light_curve_params.zHD[ii]>self.z_min or self.light_curve_params.IS_CALIBRATOR[ii] > 0:
                        true_size+=1
                        mask.append(1)
                else:
                        ignored+=1
                        mask.append(0)
        self.true_size = true_size
        newcovm = np.zeros((true_size,true_size), 'float64')
        newcovm=covm[np.array(mask).astype(bool)].T[np.array(mask).astype(bool)].T
        self.cov = la.cholesky(newcovm, lower=True, overwrite_a=True)
        
    def get_requirements(self):
        """
         return dictionary specifying quantities calculated by a theory code are needed
        """
        reqs = {"angular_diameter_distance": {"z": self.light_curve_params.zHD[self.light_curve_params.zHD>self.z_min]} , 'M':None}

        return reqs
    

    def logp(self, **params_values):
        
        M = self.provider.get_param("M")
        
        redshifts = self.light_curve_params.zHD
        size = redshifts.size
        
        moduli = np.empty((self.true_size, ))
        Mb_obs = np.empty((self.true_size, ))
        good_z = 0
        
        for index, row in self.light_curve_params.iterrows():
            z_cmb = row['zHD']
            z_hel = row['zHEL']
            Mb_corr = row['m_b_corr']
            if row['IS_CALIBRATOR'] == 1:
                moduli[good_z] = row['CEPH_DIST']
                Mb_obs[good_z] = Mb_corr
                good_z+=1
            else:
                if z_cmb > self.z_min:
                    moduli[good_z] = 5 * np.log10((1+z_cmb)*(1+z_hel)*self.provider.get_angular_diameter_distance(z_cmb)) + 25
                    Mb_obs[good_z] = Mb_corr
                    good_z+=1
                else:
                    pass
                
        residuals = np.empty((self.true_size,))
        #sn = self.light_curve_params
        residuals = Mb_obs - M
        residuals -= moduli
        residuals = la.solve_triangular(self.cov, residuals, lower=True, check_finite=False)
        chi2 = (residuals**2).sum()
        #print(chi2)
        return -0.5 * chi2
