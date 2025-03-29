from __future__ import print_function
import sys, os
from getdist import plots, loadMCSamples, MCSamples
import getdist
import IPython
import seaborn as sns
import pandas as pd
import numpy as np
import logging

###########################################################################################################################

# TABLES

###########################################################################################################################

def get_limit(root, param, limit=1, expected_marker="=", get_value=True, both=False ):
    
    samples_A = getdist.loadMCSamples(root, settings={'ignore_rows':0.5,
                                                     });
    

    if get_value==True:
        
        if both==False:
            if(expected_marker in samples_A.getInlineLatex(param, limit=limit)):
                return samples_A.getInlineLatex(param, limit=limit).split(expected_marker,1)[1];
            else:
                expected_marker= "<"
                if(expected_marker in samples_A.getInlineLatex(param,limit=limit)):
                    return "<"+samples_A.getInlineLatex(param, limit=limit).split(expected_marker,1)[1];
                else:
                    expected_marker= ">"
                    if(expected_marker in samples_A.getInlineLatex(param,limit=limit)):
                        return ">"+samples_A.getInlineLatex(param, limit=limit).split(expected_marker,1)[1];
                    
        if both==True:
            if(expected_marker in samples_A.getInlineLatex(param, limit=1)):
                if expected_marker in samples_A.getInlineLatex(param, limit=2):
                    return samples_A.getInlineLatex(param, limit=1).split(expected_marker,1)[1]+"\\, ("+samples_A.getInlineLatex(param, limit=2).split(expected_marker,1)[1]+" )";
                else:
                    expected_marker = "<"
                    if expected_marker in samples_A.getInlineLatex(param, limit=2):
                        return samples_A.getInlineLatex(param, limit=1).split("=",1)[1]+"\\, ("+"<"+samples_A.getInlineLatex(param, limit=2).split(expected_marker,1)[1]+" )";
                    else:
                        expected_marker = ">"
                        if expected_marker in samples_A.getInlineLatex(param, limit=2):
                            return samples_A.getInlineLatex(param, limit=1).split("=",1)[1]+"\\, ("+">"+samples_A.getInlineLatex(param, limit=2).split(expected_marker,1)[1]+" )";
                        
                    
            else:
                expected_marker= "<"
                if(expected_marker in samples_A.getInlineLatex(param,limit=limit)):
                    return "<"+samples_A.getInlineLatex(param, limit=1).split(expected_marker,1)[1]+"\\, ("+"<"+samples_A.getInlineLatex(param, limit=2).split(expected_marker,1)[1]+" )";
                else:
                    expected_marker= ">"
                    if(expected_marker in samples_A.getInlineLatex(param,limit=limit)):
                        return ">"+samples_A.getInlineLatex(param, limit=1).split(expected_marker,1)[1]+"\\, ("+">"+samples_A.getInlineLatex(param, limit=2).split(expected_marker,1)[1]+" )";
            
            
    
    else:
        if(expected_marker in samples_A.getInlineLatex(param,limit=limit)):
            return samples_A.getInlineLatex(param, limit=limit).split(expected_marker)[0]; 
        else:
            expected_marker= "<"
            if(expected_marker in samples_A.getInlineLatex(param,limit=limit)):
                return samples_A.getInlineLatex(param, limit=limit).split(expected_marker)[0];
            else:
                expected_marker= ">"
                if(expected_marker in samples_A.getInlineLatex(param,limit=limit)):
                    return samples_A.getInlineLatex(param, limit=limit).split(expected_marker)[0];
                
                

def get_limits_for_param(roots, param, limit=1):
    
    i=0
    both=False
    

    if limit=="both" or limit=="Both" or limit=="b" or limit=="B":
        limit=1
        both=True 
    else:
        if limit=="1" or limit=="2" or limit=="3":
            limit=int(limit)
        else:
            logging.error("limit can only be: 1 = 68 CL | 2 = 95 CL | 3 = 95CL | both = 68 CL (and 95 CL)")
            logging.error("use the following entry, paraname:limit")
            logging.error("e.g., ns:1 for getting 68 CL limmits on ns")
            sys.exit(1)

    
    print("$",get_limit(roots[i], param, get_value=False),"$","&",end = ' ')
    
    for i in range(len(roots)):
        if i==int(len(roots)-1):
            print("$",get_limit(roots[i], param, limit=limit, both=both),"$","\\\ " )
            
        else:
            print("$",get_limit(roots[i], param, limit=limit, both=both),"$","&", end = ' ')

    return



def get_table(roots, params, col_labels=False, caption="Caption TBW", label="tab.label", info=False):
    
        
    if info==True:
        print("Table requested with the following entries\n")
        print("number of columns: ", len(roots)+1 )
        if col_labels==False:
            print("column labels: unkwown")
        else:
            print("column labels:", end=" ")
            print("| Parameter | ", end=" ")
            for k in range(len(col_labels)):
                print(col_labels[k], "|", end=" ")
            print("\n")
        print("caption:", caption)
        print("table label:", label)
        print("\n")
        print("Parameters with limits:")
        for k in range(len(params)):
            print("parmater:", params[k].split(":")[0], "limit:", params[k].split(":",1)[1])
        print("\n")

    print("Please, copy what follows on a latex document to get the table")
    print("\n")
    print("%====================================================")
    print("\n")
    print("\\begin{table*}")
    print("\\begin{center}")
    print("\\renewcommand{\\arraystretch}{1.5}")
    print("\\resizebox{\\textwidth}{!}{" )
    print("\\begin{tabular}{l c c c c c c c c c c c c c c c }")
    print("\\hline")
    
    if col_labels != False:
        print("\\textbf{Parameter} &", end = ' ')
        for i in range(len(col_labels)):
            if i==int(len(col_labels)-1):
                print("\\textbf{",col_labels[i],"}","\\\ ")
            else:
                print("\\textbf{",col_labels[i],"}","&",end = ' ')
        
        print("\\hline\\hline")
    print("")
    
    for i in range(len(params)):
        get_limits_for_param(roots, params[i].split(":")[0], limit=params[i].split(":",1)[1]);    
    
    print("")
    print("\\hline \\hline")
    print("\\end{tabular} }")
    print("\\end{center}")
    print("\\caption{",caption,"}")
    print("\\label{",label,"}")
    print("\\end{table*}")
    print("\n")
    print("%====================================================")
    
    return

###########################################################################################################################
###########################################################################################################################


