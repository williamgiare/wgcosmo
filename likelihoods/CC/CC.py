import numpy as np
import scipy.linalg as la
import pandas as pd
from pandas import read_table
import os,sys

try:
    from cobaya.likelihood import Likelihood
    print('Importiong CC  as cobaya likelihood')
except:
    class Likelihood:  # dummy class to inherit if cobaya is missing
        print('dummy class to inherit')
        pass
        

class CC(Likelihood):
    
    name: str = "CC"
    
    def initialize(self):
        
        
        current_path = os.path.abspath(__file__)
        like_path= os.path.abspath(os.path.join(current_path, os.pardir))
        self.CC_path=like_path +'/data/CC.txt'
        self.CovMat_path=like_path +'/data/CovMat.txt'
        
        
        CC = pd.read_csv(self.CC_path, sep=',', header=None, names=['z','Hz','errHz','stat_contr', 'met_contr'], skiprows=1).sort_values(by='z')
        self.covmat = np.loadtxt(self.CovMat_path,unpack=True)

        self.z = np.array([], 'float64')
        self.data = np.array([], 'float64')
        self.error = np.array([], 'float64')
        
        self.z=CC['z']
        self.data=CC['Hz']
        self.error=CC['errHz']
        self.num_CC=len(self.z)
    
        
    def get_requirements(self):
        """
         return dictionary specifying quantities calculated by a theory code are needed
        """
        reqs = {"Hubble": {"z": self.z}}

        return reqs
    
    
    def logp(self, **params_values):
        
        data_array = np.array([], 'float64')
        chi2 = 0.
        
        for i in range(self.num_CC):
            theo = self.provider.get_Hubble(self.z[i],units="km/s/Mpc")
            #print(theo)
            x = (self.data[i]-theo)
            data_array = np.append(data_array, x)
        invcov = np.linalg.inv(self.covmat)
        chi2 = np.dot(np.dot(data_array,invcov),data_array)
        loglike = - 0.5*chi2
        #print(loglike)        

        return loglike
