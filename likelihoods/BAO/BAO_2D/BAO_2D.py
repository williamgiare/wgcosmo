import numpy as np
import scipy.linalg as la
import pandas as pd
from pandas import read_table
import os,sys
    

try:
    from cobaya.likelihood import Likelihood
    print('Importiong BAO-2D as cobaya likelihood')
except:
    class Likelihood:  # dummy class to inherit if cobaya is missing
        print('dummy class to inherit')
        pass
    
class BAO_2D(Likelihood):
    
    name: str = "BA02D_MM"
    
    def initialize(self):
        
        self.covmath_path='/users/sm1wg/ExLike/BAO_2D/BAO_2D_CovMat.txt'
        self.bao_data='/users/sm1wg/ExLike/BAO_2D/BAO_2D_data.txt'
    
        self.covmat = np.loadtxt(self.covmath_path,unpack=True)
        
        self.z = np.array([], 'float64')
        self.data = np.array([], 'float64')
        self.error = np.array([], 'float64')
        
        with open(self.bao_data, 'r') as f:
            for i, line in enumerate(f):
                if line.strip() and line.find('#') == -1:
                    this_line = line.split()
                    self.z = np.append(self.z, float(this_line[0]))
                    self.data = np.append(self.data, float(this_line[2]))
                    self.error = np.append(self.error, float(this_line[5]))
        
        self.num_BAO = np.shape(self.z)[0]
        
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
            x = (self.data[i]-theta)
            data_array = np.append(data_array, x)
        invcov = np.linalg.inv(self.covmat)
        chi2 = np.dot(np.dot(data_array,invcov),data_array)
        #print(chi2)
        loglike = - 0.5*chi2
        
        return loglike
    
