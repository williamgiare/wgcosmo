import numpy as np
import scipy.linalg as la
import pandas as pd
from pandas import read_table
import os,sys

try:
    from cobaya.likelihood import Likelihood
    print('Importiong BAO_DES-y6 as cobaya likelihood')
except:
    class Likelihood:  # dummy class to inherit if cobaya is missing
        print('dummy class to inherit')
        pass

class DES_Y6_BAO(Likelihood):
    
    name: str = "DES_Y6_BAO"
    
    def initialize(self):
        
        self.z = 0.85
        self.data = 19.51
        self.error = 0.41
        
            
    def get_requirements(self):
        """
         return dictionary specifying quantities calculated by a theory code are needed
        """
        reqs = {"angular_diameter_distance": {"z": self.z}, 'rdrag' :None}

        return reqs
    
    
    def logp(self, **params_values):

        chi2 = 0.
        
        rs = self.provider.get_param("rdrag")
        
        dM_over_rs = ( (1 + self.z) * self.provider.get_angular_diameter_distance(self.z) ) / rs
       # print('theory=', dM_over_rs)
        chi2 = (self.data-dM_over_rs)**2 / (self.error)**2
        
        loglike = - 0.5*chi2
        #print('loglike=', loglike) 
        return loglike
        
