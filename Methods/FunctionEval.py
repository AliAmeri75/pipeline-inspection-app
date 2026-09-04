# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 11:12:39 2018

@author: abtahi1
"""
def Function_Evaluation(Eq,Name_var,currentVal_var):
    for l in range(0,len(Name_var)):
        exec("%s = %f" % (Name_var[l],currentVal_var[l]))
    LSF=eval(Eq)
    return LSF
