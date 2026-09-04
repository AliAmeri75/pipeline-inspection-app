# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:58:37 2018

@author: abtahi1
"""

def FORM(Name_var,Type_var,Mean_var,std_var,Correl_var1,Correl_var2,Correl_factor,Eq,Iteration,delta_h=0.001,e1=0.001,e2=0.001,b=0.5,k=10,gamma=2,eta=10):
    import numpy as np
    from Models import Normal       
    from Methods import Transformer
    from Methods import FunctionEval
    from Methods import Gradient
    from Methods import DesignPoint
    from Methods import Convergence
    #Correlation Matrix
    n=len(Type_var)
    Rx=np.identity(n)
    for i in range(len(Correl_factor)):
        corrA=Correl_var1[i]
        corrB=Correl_var2[i]
        posiA=Name_var.index(corrA)
        posiB=Name_var.index(corrB)
        Rx[posiA,posiB]=Correl_factor[i]
        Rx[posiB,posiA]=Correl_factor[i]

    flagTrans=True#Uncorrelated
    for i in range(n):
        for j in range(i):
            if i!=j:
                if Rx[i,j]!=0:
                    #Correlated
                    flagTrans=False
    x0=Mean_var
    #FORM Procedure
    #Step 1: set m=1       
    #Step 2: Select the start point
    if flagTrans==True:
        u0=Transformer.NoCorrelationXtoU(x0,Type_var,Mean_var,std_var)
    else:
        u0=Transformer.NATAFXtoU(x0,Type_var,Mean_var,std_var,Rx)
    um=u0[0]#Start point u0
    #Step 6: G0
    G0=FunctionEval.Function_Evaluation(Eq,Name_var,x0)
    CC=[False,False]
    Ni=1
    while (CC[0]==False or CC[1]==False):
        Ni=Ni+1
        if Ni>Iteration:
            print('Maximum number of iteration has been reached without convergence')
            break
        #Step 3: Transformation
        if flagTrans==True:
            out=Transformer.NoCorrelationUtoX(um,Type_var,Mean_var,std_var)
        else:
            out=Transformer.NATAFUtoX(um,Type_var,Mean_var,std_var,Rx)
        #Step 4: LSF Evaluation
        Gum=FunctionEval.Function_Evaluation(Eq,Name_var,out[0])
        #Step 5: Gradient
        Grdum=np.matmul(Gradient.DGx(Eq,Name_var,out[0],delta_h),out[1])
        #Step 7: check the convergence
        CC=Convergence.Checker(Gum,G0,Grdum,um,e1,e2)
        #Step 8: Next point
        u2=DesignPoint.iHLRF(Eq,Name_var,Type_var,Mean_var,std_var,Rx,um,flagTrans,b,k,gamma,eta,delta_h)
        um=u2    
       
    #Step 9: Reliability index and probability of failure
    Beta_FORM=np.linalg.norm(um)
    Pf=Normal.Normal_CDF(-Beta_FORM,0,1)
    DesignPointSN=um
    if flagTrans==True:
        out=Transformer.NoCorrelationUtoX(DesignPointSN,Type_var,Mean_var,std_var)
    else:
        out=Transformer.NATAFUtoX(DesignPointSN,Type_var,Mean_var,std_var,Rx)
    DesignPointOrig=out[0]
    
    return Beta_FORM,Pf,DesignPointSN,DesignPointOrig

##Type_var=['Lognormal','Lognormal','Uniform']
##Mean_var=[500,2000,5]
##std_var=[100,400,0.5]
##delta_h=0.001
##e1=0.001
##e2=0.001
##b=0.5
##k=10
##gamma=2
##eta=10
##Name_var=['x1','x2','x3']
##Correl_var1=[]
##Correl_var2=[]
##Correl_factor=[]
##Eq='1.0-x2/(1000.0*x3) - (x1/(200.0*x3))**2'
##Iteration=22
##x0=Mean_var  
##
## 
##a=FORM(Name_var,Type_var,Mean_var,std_var,Correl_var1,Correl_var2,Correl_factor,Eq,Iteration,x0)    
##print('Reliability Index: ',a[0])
##print('Probability of Failure: ',a[1])   
##print('Design Point in Standard Normal Space: ',a[2])
##print('Design Point in Original Space: ',a[3])


  
