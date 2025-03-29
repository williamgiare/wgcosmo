###########################################################################################################################
# SUSPICIOUSNESS
# William Giarè
# V3 (February 2023)
###########################################################################################################################
import pandas as pd
import numpy as np
import sys, os
from scipy.special import gamma, factorial, erfinv
import scipy.integrate as integrate
from scipy.integrate import quad
import re
from IPython.display import display


def integrand(x,d):
    A=1/(2**(d/2)*gamma(d/2))
    return A*np.exp(-x/2) * x**(d/2 - 1)


def get_chain(root,params,fburn=0.3,verbose=False):
    
    _params=[0]*len(params)
    n_chains=0

    
    keep_params= params
    
  
    if os.path.isfile(root+'.1.txt'):
        if verbose:
            print("Reading chain:",root+'.1.txt')
        chain_1=pd.read_csv(root+'.1.txt', sep='\s+')
        chain_1=pd.read_csv(root+'.1.txt', sep='\s+',header=None, skiprows=int(fburn*len(chain_1)+1), names=chain_1.keys()[1:len(chain_1.keys())])
        chain=chain_1
        n_chains+=1

    if os.path.isfile(root+'.2.txt'):
        if verbose:
            print("Reading chain:",root+'.2.txt')
        chain_2=pd.read_csv(root+'.2.txt', sep='\s+')
        chain_2=pd.read_csv(root+'.2.txt', sep='\s+',header=None, skiprows=int(fburn*len(chain_2)+1), names=chain_2.keys()[1:len(chain_2.keys())])
        chain=pd.concat([chain,chain_2], axis=0)
        n_chains+=1

    if os.path.isfile(root+'.3.txt'):
        if verbose:
            print("Reading chain:",root+'.3.txt')
        chain_3=pd.read_csv(root+'.3.txt', sep='\s+')
        chain_3=pd.read_csv(root+'.3.txt', sep='\s+',header=None, skiprows=int(fburn*len(chain_3)+1), names=chain_3.keys()[1:len(chain_3.keys())])
        chain=pd.concat([chain,chain_3], axis=0)
        n_chains+=1

    if os.path.isfile(root+'.4.txt'):
        if verbose:
            print("Reading chain:",root+'.3.txt')
        chain_4=pd.read_csv(root+'.4.txt', sep='\s+')
        chain_4=pd.read_csv(root+'.4.txt', sep='\s+',header=None, skiprows=int(fburn*len(chain_4)+1), names=chain_4.keys()[1:len(chain_4.keys())])
        chain=pd.concat([chain,chain_4], axis=0)
        n_chains+=1
    
    if os.path.isfile(root+'.5.txt'):
        print("The code can read only up to 4 chains, all the pthers will be ignored. However adding more chains is trivial, see sampler.py")
        

    if n_chains==0:
        logging.error("No chains found with root:", root)
        
    else:
         if verbose==True:
            print("removing burn-in:",fburn)
            print("\nConsidering only the following params:",end=' ')
            print(keep_params)
            print("\n")
    
    return chain[keep_params]



def get_sus(root_A , root_B, params, fburn=0.3, verbose=True, get_results=False, get_latex=False, model=None):
    
    d=int(len(params))
    
    if verbose:
        print("------------------------------------------------------------------------------------------------------------------")
        print("Quantifying the global parameter tensions for the two entry MCMC chains by means of the Suspiciousness Statistic")
        print("                             (see arXiv:2007.08496 and arXiv:2209.14054)                                          ")
        print("------------------------------------------------------------------------------------------------------------------")
        print("\n\n")

    chain_A = get_chain(root_B , params, verbose=verbose, fburn=fburn)
    chain_B = get_chain(root_A , params, verbose=verbose, fburn=fburn)

    Chi2=np.dot((chain_A.mean()-chain_B.mean()),np.dot(np.linalg.inv(chain_A.cov()+chain_B.cov()),(chain_A.mean()-chain_B.mean())))
    logS=(d/2)-(Chi2/2)
    IntS=quad(integrand,Chi2, 10000, args=d)
    p=IntS[0]
    sigma=2**0.5 * erfinv(1-p)
    
    if verbose==True:
        print("------------------------------")
        print("SuStat")
        print("------------------------------")
        print("Param-space dimension:",d)
        print("Chi2=","{0:.3g}".format(Chi2))
        print("p=","{0:.3g}".format(p))
        print("logS=","{0:.3g}".format(logS))
        print("sigma=","{0:.3g}".format(sigma))
        print("------------------------------")
    if get_results==True:
        return Chi2, logS, sigma
    if get_latex == True:
        print(model,"&","$",d,"$ &","$","{0:.3g}".format(Chi2),"$ &","$","{0:.3g}".format(p),"$ &","$","{0:.3g}".format(logS),"$ &","$","{0:.3g}".format(sigma),"\,\sigma$ \\\\")
        return sigma
