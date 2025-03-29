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
        "(sudo) pip install numexpr --user")
    

try:
    from cobaya.likelihood import Likelihood
    print('Importiong DESI_Like as cobaya likelihood')
except:
    class Likelihood:  # dummy class to inherit if cobaya is missing
        print('dummy class to inherit')
        pass
    
class DESI_Like(Likelihood):
    
    name: str = "DESI_Like"
    
    def initialize(self):
        
        #self.DESI_path='./data/DESI.txt'
        
        current_path = os.path.abspath(__file__)
        like_path= os.path.abspath(os.path.join(current_path, os.pardir))
        self.DESI_path=like_path +'/data/DESI.txt'
        
        DESI_BAO = pd.read_csv(self.DESI_path, sep=',', header=None, names=['z','DA','dDA','theta', 'dtheta'], skiprows=1).sort_values(by='z')

        self.z = np.array([], 'float64')
        self.data = np.array([], 'float64')
        self.error = np.array([], 'float64')
        
        self.z=DESI_BAO['z']
        self.data=DESI_BAO['theta']
        self.error=DESI_BAO['dtheta']
        self.num_BAO=len(self.z)
    
        
    def get_requirements(self):
        """
         return dictionary specifying quantities calculated by a theory code are needed
        """
        reqs = {"angular_diameter_distance": {"z": self.z}, 'rdrag' :None}

        return reqs
    
    
    def logp(self, **params_values):
        
        data_array = np.array([], 'float64')
        chi2 = 0.
        for i in range(self.num_BAO):
            da = self.provider.get_angular_diameter_distance(self.z[i])
            rs = self.provider.get_param("rdrag")           
            theta = rs/(da*(1 + self.z[i]))*(180/np.pi)
            x = (self.data[i]-theta)**2 / (self.error[i])**2
            data_array = np.append(data_array, x)
        chi2 =np.sum(data_array)
        loglike = - 0.5*chi2
        
        return loglike