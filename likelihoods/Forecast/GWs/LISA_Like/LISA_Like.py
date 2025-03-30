import numpy as np
import scipy.linalg as la
import pandas as pd
from pandas import read_table
import os,sys

try:
    from cobaya.likelihood import Likelihood
    print('Importiong LISA_Like as cobaya likelihood')
except:
    class Likelihood:  # dummy class to inherit if cobaya is missing
        print('dummy class to inherit')
        pass


class LISA_Like(Likelihood):
    
    name: str = "LISA_Like"
    
    def initialize(self):
        
        current_path = os.path.abspath(__file__)
        like_path= os.path.abspath(os.path.join(current_path, os.pardir))
        self.LISA_path=like_path +'/data/LISA.txt'
        
        LISA_GWs = pd.read_csv(self.LISA_path, sep='\s+', header=None, names=['z', 'DL', 'dDL']).sort_values(by='z')
        
        self.z = np.array([], 'float64')
        self.data = np.array([], 'float64')
        self.error = np.array([], 'float64')
        
        self.z=LISA_GWs['z']
        self.data=LISA_GWs['DL']
        self.error=LISA_GWs['dDL']
        self.num_GW=len(self.z)

    def get_requirements(self):
        """
         return dictionary specifying quantities calculated by a theory code are needed
        """
        reqs = {"angular_diameter_distance": {"z": self.z}}

        return reqs
    
    def logp(self, **params_values):
        
        data_array = np.array([], 'float64')
        chi2 = 0.
        
        for i in range(self.num_GW):
            da = self.provider.get_angular_diameter_distance(self.z[i])
            dl = da*(1+self.z[i])**2
            x = (self.data[i]-dl)**2 / (self.error[i])**2
            data_array = np.append(data_array, x)
            
        chi2 = np.sum(data_array)
        #print(chi2)
        loglike = - 0.5*chi2
        
        return loglike
