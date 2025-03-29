import numpy as np
import scipy.linalg as la
import pandas as pd
from pandas import read_table
import os,sys


try:
    from cobaya.likelihood import Likelihood
    print('Importiong DESI-BAO LRG')
except:
    class Likelihood:  # dummy class to inherit if cobaya is missing
        print('dummy class to inherit')
        pass
    

class desi_lrg(Likelihood):
    
    name: str = "desi_lrg"
    
    def initialize(self):
        
        self.z_eff=[0.51,0.71]
        
        self.z_051=0.51
        self.z_071=0.706
        
        self.data_DM_z_051 = 13.62
        self.data_DH_z_051= 20.98
        
        self.data_DM_z_071 = 16.85
        self.data_DH_z_071 = 20.08
        
        self.covmat_z_051 = [[0.0625,-0.06786],[-0.06786,0.3721]]
        self.covmat_z_071 = [[0.1024,-0.08064],[-0.08064,0.360]]
    
        
    def get_requirements(self):
        """
         return dictionary specifying quantities calculated by a theory code are needed
        """
        reqs = {"angular_diameter_distance": {"z": self.z_eff}, "Hubble": {"z": self.z_eff}, 'rdrag' :None}

        return reqs
    
    
    def logp(self, **params_values):
        
        data_array_z_051 = np.array([], 'float64')
        data_array_z_071 = np.array([], 'float64')
        
        chi2_z_051 = 0.
        chi2_z_071 = 0.
        
        rs = self.provider.get_param("rdrag")
            
        da_z_051 = self.provider.get_angular_diameter_distance(self.z_eff[0])
        H_z_051 = self.provider.get_Hubble(self.z_eff[0],units="km/s/Mpc")
        
        da_z_071 = self.provider.get_angular_diameter_distance(self.z_eff[1])
        H_z_071 = self.provider.get_Hubble(self.z_eff[1],units="km/s/Mpc")
        
        DM_z_051=da_z_051*(1. + self.z_051)/rs 
        DH_z_051= (2.998 * 10**5)/H_z_051/rs
        print('DM_z_051=',DM_z_051)
        print('DH_z_051=',DH_z_051)
        
        DM_z_071=da_z_071*(1. + self.z_071)/rs 
        DH_z_071= (2.998 * 10**5)/H_z_071/rs
        print('DM_z_071=',DM_z_071)
        print('DH_z_071=',DH_z_071)
        
        x_z_051=[ [DM_z_051-self.data_DM_z_051] , [DH_z_051-self.data_DH_z_051] ]
        x_z_071=[ [DM_z_071-self.data_DM_z_071] , [DH_z_071-self.data_DH_z_071] ]
        #print('delta_051=',x_z_051)
        #print('delta_071=',x_z_071)
    
        data_array_z_051 = np.append(data_array_z_051, x_z_051)
        data_array_z_071 = np.append(data_array_z_071, x_z_071)
            
        chi2_z_051 = np.dot(np.dot(data_array_z_051,np.linalg.inv(self.covmat_z_051)),data_array_z_051)
        chi2_z_071 = np.dot(np.dot(data_array_z_071,np.linalg.inv(self.covmat_z_071)),data_array_z_071)
        
        loglike_z_051 = - 0.5*chi2_z_051
        loglike_z_071 = - 0.5*chi2_z_071
        
        loglike=loglike_z_051+loglike_z_071
        print('loglike=',loglike,'\n')
        return loglike
