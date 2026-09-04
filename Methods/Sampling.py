# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:31:57 2018

@author: abtahi1
"""

#Sampling
def MC(Name_var,Type_var,Mean_var,std_var,Correl_var1,Correl_var2,Correl_factor,Eq,Num,Seed=0,CoV=0.02):
    import numpy as np     
    from Methods import Transformer
    from Methods import FunctionEval
    from Models import Normal
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
    
    #Generate strandard normal random numbers
    SNsamples=np.zeros((Num,n))
    cov=np.identity(n)
    mu=np.zeros(n)
    np.random.seed(Seed)
    SNsamples=np.random.multivariate_normal(mu,cov,Num)
    #Transfer samples to the original space
    OriginSamples=np.zeros((Num,n))
    for i in range(Num):
        if flagTrans==True:
            OriginSamples[i,:]=Transformer.NoCorrelationUtoX(SNsamples[i,:],Type_var,Mean_var,std_var)[0]
        else:
            OriginSamples[i,:]=Transformer.NATAFUtoX(SNsamples[i,:],Type_var,Mean_var,std_var,Rx)[0]
    
    #Function Evaluation
    g=np.zeros(Num)
    Ig=np.zeros(Num)
    for i in range(Num):
        g[i]=FunctionEval.Function_Evaluation(Eq,Name_var,OriginSamples[i,:])
        if g[i]<=0:
            Ig[i]=1
    #Probability of failure
    Pf=np.sum(Ig)/Num
    if Pf==0:
        print('Error: No failure point with this number of samples!')
        Beta_Sampling='NaN'
    else:
        Beta_Sampling=-Normal.Normal_CDFInverse(Pf,0,1)

    return Beta_Sampling,Pf

Type_var=['Lognormal','Lognormal','Uniform']
Mean_var=[500,2000,5]
std_var=[100,400,0.5]
Name_var=['x1','x2','x3']
Correl_var1=['x1','x1','x2']
Correl_var2=['x2','x3','x3']
Correl_factor=[0.3,0.2,0.2]
Eq='1.0-x2/(1000.0*x3)-(x1/(200.0*x3))**2'
Num=1000
Seed=0
 
a=MC(Name_var,Type_var,Mean_var,std_var,Correl_var1,Correl_var2,Correl_factor,Eq,Num)
print(a[0])
print(a[1])

