import numpy as np
import scipy.linalg as la
import pandas as pd
from pandas import read_table
import os,sys


try:
    from cobaya.likelihood import Likelihood
    print('Importiong DESI-BAO LRG+ELG')
except:
    class Likelihood:  # dummy class to inherit if cobaya is missing
        print('dummy class to inherit')
        pass
    

class desi_lrg_elg(Likelihood):
    
    name: str = "desi_lrg_elg"
    
    def initialize(self):
        
        self.z_eff=0.93
        
        self.data_DM_z_093 = 21.71
        self.data_DH_z_093= 17.88
        
        self.covmat_z_093 = [[0.0784,-0.038122],[-0.038122,0.1225]]
    
        
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
            
        da = self.provider.get_angular_diameter_distance(self.z_eff)
        H = self.provider.get_Hubble(self.z_eff,units="km/s/Mpc")
        
        DM=da*(1. + self.z_eff)/rs
        DH= (2.998 * 10**5)/H/rs
        
        print('DM_z_093=',DM)
        print('DH_z_093=',DH)
        
        x=[ [DM-self.data_DM_z_093] , [DH-self.data_DH_z_093] ]
        print('delta_093=',x)
    
        data_array = np.append(data_array, x)
            
        chi2 = np.dot(np.dot(data_array,np.linalg.inv(self.covmat_z_093)),data_array)
        
        loglike = - 0.5*chi2
        print('loglike=',loglike,'\n')
        return loglike
