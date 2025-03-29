import numpy as np
import scipy.linalg as la
import pandas as pd
from pandas import read_table
import os,sys

try:
    from cobaya.likelihood import Likelihood
    print('Importing DESI-BAO-DR2-BGS')
except:
    class Likelihood:  # dummy class to inherit if cobaya is missing
        print('dummy class to inherit')
        pass
    

class desi_bgs(Likelihood):
    
    name: str = "desi_dr2_bgs"
    
    def initialize(self):
        
        self.z_eff= 0.295
        
        self.data_DV = 7.944
        
        self.error_DV = 0.075
    
        
    def get_requirements(self):
        """
         return dictionary specifying quantities calculated by a theory code are needed
        """
        reqs = {"angular_diameter_distance": {"z": self.z_eff}, "Hubble": {"z": self.z_eff}, 'rdrag' :None}

        return reqs
    
    
    def logp(self, **params_values):
        
        chi2 = 0.
        
        rs = self.provider.get_param("rdrag")
            
        da = self.provider.get_angular_diameter_distance(self.z_eff)
        
        H = self.provider.get_Hubble(self.z_eff,units="km/s/Mpc")
        
        dr=( self.z_eff * (2.998 * 10**5) ) / H
        
        DV= ( (da*da*(1 + self.z_eff)*(1 + self.z_eff)*dr)**(1/3) ) /rs
        
        chi2 = (DV-self.data_DV)**2 / (self.error_DV**2)
        
        loglike = - 0.5*chi2
        
        return loglike
