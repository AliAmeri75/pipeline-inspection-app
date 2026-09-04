# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 11:18:02 2018

@author: abtahi1
"""
def DGx(Eq,Name_var,x0,delta_h=0.001):
    '''
    Derivation of G with respect to G at x0 with finite difference method
    delta_h=0.001 by default
    '''
    from Methods import FunctionEval
    import numpy as np
    n=len(Name_var)
    deriv=np.zeros((n))
    for i in range(n):
        delta=np.zeros((n))
        delta[i]=delta_h
        x2=x0+delta
        deriv[i]=(FunctionEval.Function_Evaluation(Eq,Name_var,x2)-FunctionEval.Function_Evaluation(Eq,Name_var,x0))/delta_h
    return deriv