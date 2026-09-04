# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:58:37 2018

@author: abtahi1
"""
#FOSM
def FOSM(Eq,Type_var,Name_var,Mean_var,std_var,Correl_var1,Correl_var2,Correl_factor,delta_h=0.001):
    '''
    Equation,Name of variables, Mean, Covariance matrix
    The nonlinear functions are approximated by their first order tailor series
    '''
    #Correlation Matrix
    import numpy as np
    n=len(Type_var)
    Rx=np.identity(n)
    for i in range(len(Correl_factor)):
        corrA=Correl_var1[i]
        corrB=Correl_var2[i]
        posiA=Name_var.index(corrA)
        posiB=Name_var.index(corrB)
        Rx[posiA,posiB]=Correl_factor[i]
        Rx[posiB,posiA]=Correl_factor[i]
    Dx=np.zeros((n,n))
    for i in range(n):
        Dx[i,i]=std_var[i]
    aa=np.matmul(Dx,Rx)
    C=np.matmul(aa,Dx)
    
    
    #Mean of the function
    from Methods import FunctionEval
    Z_mean=FunctionEval.Function_Evaluation(Eq,Name_var,Mean_var)
    #Gradient of the function with finite difference method

    from Methods import Gradient
    deriv=Gradient.DGx(Eq,Name_var,Mean_var,delta_h)
    #Standard deviation of the function
    a1=np.transpose(deriv)
    a=np.matmul(a1,C)
    Z_std=(np.matmul(a,deriv))**0.5    
    #Reliability Index
    Beta_FOSM=Z_mean/Z_std
    return Beta_FOSM

#Eq='1-X1*X2'
#import numpy as np
#Name_var=['X1','X2']
#Mean_var=[5,4]
#C=np.zeros((2,2))
#C[0,0]=4
#C[1,1]=1
#delta_h=0.001
#print(FOSM(Eq,Name_var,Mean_var,C,delta_h))