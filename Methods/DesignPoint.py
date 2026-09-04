# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 16:10:13 2018

@author: abtahi1
"""

#iHLRF
def iHLRF(Eq,Name_var,Type_var,Mean_var,std_var,Rx,um,flagTrans,b=0.5,k=10,gamma=2,eta=10,delta_h=0.001):
    import numpy as np
    from Methods import FunctionEval
    from Methods import Gradient
    from Methods import Transformer
    from Methods import Merit
    FlagMerit=False
    k1=0
    if flagTrans==True:
        outU1=Transformer.NoCorrelationUtoX(um,Type_var,Mean_var,std_var)
    else:
        outU1=Transformer.NATAFUtoX(um,Type_var,Mean_var,std_var,Rx)
    #Function
    Gum=FunctionEval.Function_Evaluation(Eq,Name_var,outU1[0])        
    #Gradient
    Grdum=np.matmul(Gradient.DGx(Eq,Name_var,outU1[0],delta_h),outU1[1])       
    alpha=[-x/np.linalg.norm(Grdum) for x in Grdum]
    Delta=Gum/np.linalg.norm(Grdum)+np.matmul(np.transpose(alpha),um)
    dm=[(Delta*x-y) for x,y in zip(alpha,um)]  
    if flagTrans==True:
        while (FlagMerit==False):
            #Armijo Rule:        
            Sm=b**k1    
            um2=[x+Sm*y for x,y in zip(um,dm)]    
            if flagTrans==True:
                outU2=Transformer.NoCorrelationUtoX(um2,Type_var,Mean_var,std_var)
            else:
                outU2=Transformer.NATAFUtoX(um2,Type_var,Mean_var,std_var,Rx)
            #Function
            Gum2=FunctionEval.Function_Evaluation(Eq,Name_var,outU2[0])
            Grdum2=np.matmul(Gradient.DGx(Eq,Name_var,outU2[0],delta_h),outU2[1])    
            #Check the Step Size
            FlagMerit=Merit.MeritChecker(um,Gum,Grdum,um2,Gum2,Grdum2,Sm,dm,gamma,eta)
            k1=k1+1
            if k1>k:
                break
    else:
        #************************************************ Fixed Step Size for Correlated ones ********** (Need Improvement)
        Sm=b     
        um2=[x+Sm*y for x,y in zip(um,dm)]  
    return um2