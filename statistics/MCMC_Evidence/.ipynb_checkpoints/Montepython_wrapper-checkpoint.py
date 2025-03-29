
#############################################################################################

# MCMC EVIDENCE

"""
Authors: William Giarè
Description :
Python implementation of the evidence estimation from MCMC chains with Cobaya
"""
#############################################################################################

from __future__ import print_function
import sys, os
sys.path.insert(0,os.path.realpath(os.path.join(os.getcwd(),'..')))
from getdist import plots, loadMCSamples, MCSamples
import getdist
import matplotlib.pyplot as plt
import IPython
from matplotlib import rc
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import argparse
import time
from MCEvidence import get_prior_volume,MCEvidence

def BayesianEvidence(root):

    args = argparse.Namespace(root_name=root, kmax=2, idchain=0, 
                              ndim=None, paramsfile='', burnlen=0.3, thinlen=0, verbose=1, priorvolume=None,
                              allparams=False, cross=False)
    
    cosmo = not args.allparams
    prior_volume = get_prior_volume(args,cosmo=cosmo)


    method=args.root_name
    kmax=args.kmax
    idchain=args.idchain
    ndim=args.ndim
    burnlen=args.burnlen
    thinlen=args.thinlen
    verbose=args.verbose
    split = args.cross

    mce=MCEvidence(method,split=split, ndim=ndim,priorvolume=prior_volume,
                   idchain=idchain,
                   kmax=kmax,verbose=verbose,burnlen=burnlen,
                   thinlen=thinlen)
    return mce.evidence() 


def match_CosmoMC_chains(root, params,verbose=False):
    
    _params=[0]*len(params)
    n_chains=0
    
    for i in range(len(params)):
        if ":" in params[i]:
            _params[i]=params[i].split(":")[0]
        else:
             _params[i]=params[i]
    
    keep_params=['weight', 'loglike'] + _params
    
  
    if os.path.isfile(root+'.1.txt'):
        chain_1=pd.read_csv(root+'.1.txt', sep='\s+')
        chain_1=pd.read_csv(root+'.1.txt', sep='\s+',header=None, skiprows=1, names=chain_1.keys()[1:len(chain_1.keys())])
        chain_1["loglike"]=0.5*chain_1["chi2"]
        chain_1[keep_params].to_csv(root+'_BE.1.txt', header=False, index=False, sep=' ')
        n_chains+=1 

    if os.path.isfile(root+'.2.txt'): 
        chain_2=pd.read_csv(root+'.2.txt', sep='\s+')
        chain_2=pd.read_csv(root+'.2.txt', sep='\s+',header=None, skiprows=1, names=chain_2.keys()[1:len(chain_2.keys())])
        chain_2["loglike"]=0.5*chain_2["chi2"]
        chain_2[keep_params].to_csv(root+'_BE.2.txt', header=False, index=False, sep=' ')
        n_chains+=1

    if os.path.isfile(root+'.3.txt'):
        chain_3=pd.read_csv(root+'.3.txt', sep='\s+')
        chain_3=pd.read_csv(root+'.3.txt', sep='\s+',header=None, skiprows=1, names=chain_3.keys()[1:len(chain_3.keys())])
        chain_3["loglike"]=0.5*chain_3["chi2"]
        chain_3[keep_params].to_csv(root+'_BE.3.txt', header=False, index=False, sep=' ')
        n_chains+=1

    if os.path.isfile(root+'.4.txt'):
        chain_4=pd.read_csv(root+'.4.txt', sep='\s+')
        chain_4=pd.read_csv(root+'.4.txt', sep='\s+',header=None, skiprows=1, names=chain_4.keys()[1:len(chain_4.keys())])
        chain_4["loglike"]=0.5*chain_4["chi2"]
        chain_4[keep_params].to_csv(root+'_BE.4.txt', header=False, index=False, sep=' ')
        n_chains+=1
    
    if os.path.isfile(root+'.5.txt'):
        chain_5=pd.read_csv(root+'.5.txt', sep='\s+')
        chain_5=pd.read_csv(root+'.5.txt', sep='\s+',header=None, skiprows=1, names=chain_5.keys()[1:len(chain_5.keys())])
        chain_5["loglike"]=0.5*chain_5["chi2"]
        chain_5[keep_params].to_csv(root+'_BE.5.txt', header=False, index=False, sep=' ')
        n_chains+=1
        

    if n_chains==0:
        logging.error("No chains found with root:", root)
        
    else:
         if verbose==True:
            print(n_chains, "chains are produced to mach the CosmoMC outpu, with the following columns:\n")
            print(keep_params)
            print("\n")
    
    

def get_dot_ranges(root, params, verbose=False):
    
    samp=getdist.loadMCSamples(root);
    
    lower=[0]*len(params)
    upper=[0]*len(params)
    _params=[0]*len(params)

    
    for i in range(len(params)):
        
        if ":" in params[i]:
            info=params[i].split(":",1)[1]
            _params[i]=params[i].split(":")[0]
            lower[i]=info.split("/")[0]
            upper[i]=info.split("/",1)[1]
            
            if verbose==True:
                print(params[i].split(":")[0], end=" ")
                print(info.split("/")[0], end=" ")
                print(info.split("/",1)[1])
            
        else:
            _params[i]=params[i]
            lower[i]=samp.ranges.getLower(params[i])
            upper[i]=samp.ranges.getUpper(params[i])
            
            if verbose==True:
                print(params[i], end=" ")
                print(samp.ranges.getLower(params[i]), end=" ")
                print(samp.ranges.getUpper(params[i]))
            
    for i in range(len(params)):
        
        if _params[i]=="ombh2" or _params[i]=="omega_b":
            _params[i]="omegabh2"
            
        if _params[i]=="omch2" or _params[i]=="omega_c":
            _params[i]="omegach2"
            
        if _params[i]=="theta_MC_100" or _params[i]=="theta_s_1e2":
            _params[i]= "theta"
            
        if _params[i]=="omk" or _params[i]=="omega_k":
            _params[i]="omegak"
        
        if _params[i]=="tau_reio":
            _params[i]="tau"
        
        if _params[i]=="n_s":
            _params[i]="ns"
        
    
    x = {'Param': _params, 'min': lower, 'max':upper}
    df = pd.DataFrame(data=x)
    df.to_csv(root+'_BE.ranges', header=False, index=False, sep=' ')
    
    return

def cleaning_up(root):
    if os.path.isfile(root+'_BE.1.txt'):
        os.remove(root+'_BE.1.txt')
    if os.path.isfile(root+'_BE.2.txt'):
        os.remove(root+'_BE.2.txt')
    if os.path.isfile(root+'_BE.3.txt'):    
        os.remove(root+'_BE.3.txt')
    if os.path.isfile(root+'_BE.4.txt'):
        os.remove(root+'_BE.4.txt')
    if os.path.isfile(root+'_BE.5.txt'):
        os.remove(root+'_BE.5.txt')    
    if os.path.isfile(root+'_BE.ranges'):    
        os.remove(root+'_BE.ranges')

def MCMC_Evidence(root, params, verbose=False, get_results=False, labels=False):
    
    print("===========================================================")
    print("RUNNING MCMC EVIDENCE FOR COBAYA")
    print("===========================================================")
    
    if verbose:
        print("\n")
        print("1) Matching CosmoMC chains output\n")

    match_CosmoMC_chains(root, params, verbose=verbose)

    if verbose:
        print("2) producing a file .ranges with the following limits\n")
        
    get_dot_ranges(root,params, verbose=verbose)
    
    if verbose:
        print("\n")
    
    if verbose:
        print("3) Running MCEvidence for Cobaya")
        time.sleep(1)

    lnB=BayesianEvidence(root+'_BE')[0]
    
    if labels==False:
        print("===========================================================")
        print("ln(B)[k=1] =", lnB)
        print("===========================================================")
        print("\n")
    else:
        print("===========================================================")
        print("[",labels,"]" "         ln(B)[k=1] =", lnB)
        print("===========================================================")
        print("\n")
        
    if verbose:
        print("4) cleaning-up everything...")
    cleaning_up(root)
    
    if get_results==True:
        return lnB
