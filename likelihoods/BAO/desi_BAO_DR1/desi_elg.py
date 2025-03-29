import numpy as np
import scipy.linalg as la
import pandas as pd
from pandas import read_table
import os,sys


try:
    from cobaya.likelihood import Likelihood
    print('Importiong DESI-BAO ELG')
except:
    class Likelihood:  # dummy class to inherit if cobaya is missing
        print('dummy class to inherit')
        pass
    

class desi_elg(Likelihood):
    
    name: str = "desi_elg"
    
    def initialize(self):
        
        self.z_eff=1.317
        
        self.data_DM_z_132 = 27.79
        self.data_DH_z_132= 13.82
        
        self.covmat_z_132 = [[0.4761,-0.128671],[-0.128671,0.1764]]
    
        
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
        print('DM_z_132',DM)
        print('DH_z_132',DH)
        
        x=[ [DM-self.data_DM_z_132] , [DH-self.data_DH_z_132] ]
        #print('delta_z_132',x)

    
        data_array = np.append(data_array, x)
            
        chi2 = np.dot(np.dot(data_array,np.linalg.inv(self.covmat_z_132)),data_array)
        
        loglike = - 0.5*chi2
        print('loglike=',loglike,'\n')
        return loglike
