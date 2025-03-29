import numpy as np
import scipy.linalg as la
import pandas as pd
from pandas import read_table
import os,sys


try:
    from cobaya.likelihood import Likelihood
    print('importing DESI-BAO-DR2-LRG2')
except:
    class Likelihood(object):  # dummy class to inherit if cobaya is missing
        print('dummy class to inherit')
        pass
    

class desi_lrg2(Likelihood):
    
    name: str = "desi_dr2_lrg2"
    
    def initialize(self):
        
        #-----------------------
        
        self.z_eff=0.706
        
        #-----------------------
        
        self.data_DM_over_DH= 0.892
        self.error_DM_over_DH= 0.021
        
        self.data_DV= 16.048
        self.error_DV=0.110
        
        self.r_VMH=-0.001
        
        b_11 = self.error_DV**2
        b_12 = self.r_VMH * self.error_DV * self.error_DM_over_DH
        b_21 = self.r_VMH * self.error_DV * self.error_DM_over_DH
        b_22 = self.error_DM_over_DH**2
        
        self.covmat_VMH = np.array([[b_11,b_12],[b_21,b_22]])
        
        #-----------------------
    
        
    def get_requirements(self):
        """
         return dictionary specifying quantities calculated by a theory code are needed
        """
        reqs = {"angular_diameter_distance": {"z": self.z_eff}, "Hubble": {"z": self.z_eff}, 'rdrag' :None}

        return reqs
    
    
    def logp(self, **params_values):
        
        data_array = np.array([], 'float64')
        
        chi2 = 0.
        
        rs = self.provider.get_param("rdrag")
            
        da_th = self.provider.get_angular_diameter_distance(self.z_eff)
        
        H_th= self.provider.get_Hubble(self.z_eff,units="km/s/Mpc")
        
        dr=( self.z_eff * (2.998 * 10**5) ) / H_th
        
        #theory DM
        DM_th = da_th*(1. + self.z_eff)/rs
        
        #theory DH
        DH_th = (2.998 * 10**5)/H_th/rs
        
        #theory DM/DH
        DM_over_DH_th = DM_th / DH_th
        
        #theory DV
        DV_th =( (da_th*da_th*(1 + self.z_eff)*(1 + self.z_eff)*dr)**(1/3) ) /rs
        
        #chi2 for DV and DM/DH
        x=[ [DV_th-self.data_DV] , [DM_over_DH_th-self.data_DM_over_DH] ]
        data_array = np.append(data_array,x)
            
        chi2 = np.dot(np.dot(data_array,np.linalg.inv(self.covmat_VMH)),data_array)
                
        loglike = - 0.5*chi2

        return loglike
