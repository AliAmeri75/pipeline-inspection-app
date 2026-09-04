# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:58:37 2018

@author: abtahi1
"""
#No Correlation Transformation
def NoCorrelationXtoU(input_var,Type_var,Mean_var,std_var):
    '''
    from x in non-normal to u in standard normal
    '''
    import numpy as np
    from Models import Normal
    n=len(input_var)
    U=np.zeros(n)
    CDF_var=np.zeros(n)
    PDF_var=np.zeros(n)
    for i in range(n):     
        if Type_var[i]=='Normal':
            CDF_var[i]=Normal.Normal_CDF(input_var[i],Mean_var[i],std_var[i])
            PDF_var[i]=Normal.Normal_PDF(input_var[i],Mean_var[i],std_var[i])
        if Type_var[i]=='Lognormal':
            from Models import Lognormal
            CDF_var[i]=Lognormal.Lognormal_CDF(input_var[i],Mean_var[i],std_var[i])
            PDF_var[i]=Lognormal.Lognormal_PDF(input_var[i],Mean_var[i],std_var[i])
        if Type_var[i]=='Uniform':
            from Models import Uniform
            CDF_var[i]=Uniform.Uniform_CDF(input_var[i],Mean_var[i],std_var[i])
            PDF_var[i]=Uniform.Uniform_PDF(input_var[i],Mean_var[i],std_var[i])
        if Type_var[i]=='Gamma':
            from Models import Gamma
            CDF_var[i]=Gamma.Gamma_CDF(input_var[i],Mean_var[i],std_var[i])
            PDF_var[i]=Gamma.Gamma_PDF(input_var[i],Mean_var[i],std_var[i])
        if Type_var[i]=='Gumbel':
            from Models import Gumbel
            CDF_var[i]=Gumbel.Gumbel_CDF(input_var[i],Mean_var[i],std_var[i])
            PDF_var[i]=Gumbel.Gumbel_PDF(input_var[i],Mean_var[i],std_var[i])
    for i in range(n):
        U[i]=Normal.Normal_CDFInverse(CDF_var[i],0,1)
    #Jacobian Ju,x
    Jux=np.zeros((n,n))
    for i in range(n):
        Jux[i,i]=PDF_var[i]/Normal.Normal_PDF(U[i],0,1)
    
    return U,Jux

def NoCorrelationUtoX(input_U,Type_var,Mean_var,std_var):
    '''
    from u in standard normal to x in non-normal
    '''
    import numpy as np
    from Models import Normal
    n=len(input_U)
    x=np.zeros(n)
    pdfx=np.zeros(n)
    for i in range(n):
        arg=Normal.Normal_CDF(input_U[i],0,1)
        if Type_var[i]=='Normal':
            x[i]=Normal.Normal_CDFInverse(arg,Mean_var[i],std_var[i])
            pdfx[i]=Normal.Normal_PDF(x[i],Mean_var[i],std_var[i])
        if Type_var[i]=='Lognormal':
            from Models import Lognormal
            x[i]=Lognormal.Lognormal_CDFInverse(arg,Mean_var[i],std_var[i])
            pdfx[i]=Lognormal.Lognormal_PDF(x[i],Mean_var[i],std_var[i])
        if Type_var[i]=='Uniform':
            from Models import Uniform
            x[i]=Uniform.Uniform_CDFInverse(arg,Mean_var[i],std_var[i])
            pdfx[i]=Uniform.Uniform_PDF(x[i],Mean_var[i],std_var[i])
        if Type_var[i]=='Gamma':
            from Models import Gamma
            x[i]=Gamma.Gamma_CDFInverse(arg,Mean_var[i],std_var[i])
            pdfx[i]=Gamma.Gamma_PDF(x[i],Mean_var[i],std_var[i])
        if Type_var[i]=='Gumbel':
            from Models import Gumbel
            x[i]=Gumbel.Gumbel_CDFInverse(arg,Mean_var[i],std_var[i])
            pdfx[i]=Gumbel.Gumbel_PDF(x[i],Mean_var[i],std_var[i])
    #Jacobian Jx,u
    Jxu=np.zeros((n,n))
    for i in range(n):
        Jxu[i,i]=Normal.Normal_PDF(input_U[i],0,1)/pdfx[i]
    
    return x,Jxu

def Correlation_modification(D1,D2,ro):
    '''
    CoV of first distribution
    CoV of second distribution
    Note that these should be in the same order as types of distributions
    Gumbel: Type I Largest value
    '''
    import math as m
    LL=m.log(1+ro*D1*D2)/(ro*m.sqrt(m.log(1+D1**2)*m.log(1+D2**2)))
    GG=1.002+0.022*ro-0.012*(D1+D2)+0.001*ro**2+0.125*(D1**2+D2**2)-0.077*ro*(D1+D2)+0.014*D1*D2
    UU=1.047-0.047*ro**2
    NL=D2/m.sqrt(m.log(1+D2**2))
    LN=D1/m.sqrt(m.log(1+D1**2))
    NG=1.001-0.007*D2+0.118*D2**2
    GN=1.001-0.007*D1+0.118*D1**2
    LU=1.019+0.014*D1+0.01*ro**2+0.249*D1**2
    UL=1.019+0.014*D2+0.01*ro**2+0.249*D2**2
    UG=1.023-0.007*D2+0.002*ro**2+0.127*D2**2
    GU=1.023-0.007*D1+0.002*ro**2+0.127*D1**2
    LG=1.001+0.033*ro+0.004*D1-0.016*D2+0.002*ro**2+0.223*D1**2+0.13*D2**2-0.104*ro*D1+0.029*D1*D2-0.119*ro*D2
    GL=1.001+0.033*ro+0.004*D2-0.016*D1+0.002*ro**2+0.223*D2**2+0.13*D1**2-0.104*ro*D2+0.029*D2*D1-0.119*ro*D1
    LGu=1.029+0.001*ro+0.014*D1+0.004*ro**2+0.233*D1**2-0.197*ro*D1
    UGu=1.055+0.015*ro**2
    GGu=1.031+0.001*ro-0.007*D1+0.003*ro**2+0.131*D1**2-0.132*ro*D1
    GuL=1.029+0.001*ro+0.014*D2+0.004*ro**2+0.233*D2**2-0.197*ro*D2
    GuU=1.055+0.015*ro**2
    GuG=1.031+0.001*ro-0.007*D2+0.003*ro**2+0.131*D2**2-0.132*ro*D2
    GuGu=1.064-0.069*ro+0.005*ro**2

    F={'Normal':{'Normal':1,'Lognormal':NL,'Uniform':1.023,'Gamma':NG,'Gumbel':1.031},'Lognormal':{'Normal':LN,'Lognormal':LL,'Uniform':LU,'Gamma':LG,'Gumbel':LGu},'Uniform':{'Normal':1.023,'Lognormal':UL,'Uniform':UU,'Gamma':UG,'Gumbel':UGu},'Gamma':{'Normal':GN,'Lognormal':GL,'Uniform':GU,'Gamma':GG,'Gumbel':GGu},'Gumbel':{'Normal':1.031,'Lognormal':GuL,'Uniform':GuU,'Gamma':GuG,'Gumbel':GuGu}}
    return F


#NATAF Transformer
def NATAFXtoU(input_var,Type_var,Mean_var,std_var,Rx):
    '''
    from x in non-normal to u in standard normal
    X is non-normal
    Z is standard normal but dependent
    U is standard normal independent
    CDF_var=F(xi)
    Rx=correlation matrix
    '''
    import numpy as np
    from Methods import Cholesky as ch
    from Models import Normal
    n=len(input_var)
    CDF_var=np.zeros(n)
    PDF_var=np.zeros(n)
    for i in range(n):     
        if Type_var[i]=='Normal':
            CDF_var[i]=Normal.Normal_CDF(input_var[i],Mean_var[i],std_var[i])
            PDF_var[i]=Normal.Normal_PDF(input_var[i],Mean_var[i],std_var[i])
        if Type_var[i]=='Lognormal':
            from Models import Lognormal
            CDF_var[i]=Lognormal.Lognormal_CDF(input_var[i],Mean_var[i],std_var[i])
            PDF_var[i]=Lognormal.Lognormal_PDF(input_var[i],Mean_var[i],std_var[i])
        if Type_var[i]=='Uniform':
            from Models import Uniform
            CDF_var[i]=Uniform.Uniform_CDF(input_var[i],Mean_var[i],std_var[i])
            PDF_var[i]=Uniform.Uniform_PDF(input_var[i],Mean_var[i],std_var[i])
        if Type_var[i]=='Gamma':
            from Models import Gamma
            CDF_var[i]=Gamma.Gamma_CDF(input_var[i],Mean_var[i],std_var[i])
            PDF_var[i]=Gamma.Gamma_PDF(input_var[i],Mean_var[i],std_var[i])
        if Type_var[i]=='Gumbel':
            from Models import Gumbel
            CDF_var[i]=Gumbel.Gumbel_CDF(input_var[i],Mean_var[i],std_var[i])
            PDF_var[i]=Gumbel.Gumbel_PDF(input_var[i],Mean_var[i],std_var[i])            
  
    U=np.zeros(n)
    Rz=np.identity(n)
    for i in range(n):
        for j in range(i):
            if i!=j:
                D1=float(std_var[i]/Mean_var[i])
                D2=float(std_var[j]/Mean_var[j])
                F=Correlation_modification(D1,D2,Rx[i,j])[Type_var[i]][Type_var[j]]
                roz=F*Rx[i,j]
                Rz[i,j]=roz
    L0=ch.CholeskyDecomposition(Rz)
    b=np.zeros(n)
    a=np.zeros(n)
    for i in range(n):
        arg=Normal.Normal_CDFInverse(CDF_var[i],0,1)
        b[i]=(Normal.Normal_PDF(arg,0,1))/(PDF_var[i])
        a[i]=input_var[i]-b[i]*Normal.Normal_CDFInverse(CDF_var[i],0,1)
    D=np.zeros((n,n))
    for i in range(n):
        D[i,i]=b[i]
    co=np.zeros((n,n))
    co=np.matmul(np.linalg.inv(L0[0]),np.linalg.inv(D))
    U=np.matmul(co,(input_var-a))   
    
    #Jacobian
    Jux=np.zeros((n,n))
    d=np.zeros((n,n))
    for i in range(n):
        d[i,i]=PDF_var[i]/Normal.Normal_PDF(U[i],0,1)
    Jux=np.matmul(np.linalg.inv(L0[0]),d)
    
    return U,Jux


def NATAFUtoX(input_U,Type_var,Mean_var,std_var,Rx):
    '''
    from u in standard normal to x in non-normal
    X is non-normal
    Z is standard normal but dependent
    U is standard normal independent
    Rx=correlation matrix
    '''
    import numpy as np
    from Methods import Cholesky as ch
    from Models import Normal
    n=len(input_U)
    x=np.zeros(n)
    Rz=np.identity(n)
    for i in range(n):
        for j in range(i):
            if i!=j:
                D1=float(std_var[i]/Mean_var[i])
                D2=float(std_var[j]/Mean_var[j])
                F=Correlation_modification(D1,D2,Rx[i,j])[Type_var[i]][Type_var[j]]
                roz=F*Rx[i,j]
                Rz[i,j]=roz
    L0=ch.CholeskyDecomposition(Rz)
    Z=np.zeros(n)
    Z=np.matmul(L0[0],input_U)
    pdfx=np.zeros(n)
    for i in range(n):
        arg=Normal.Normal_CDF(Z[i],0,1)
        if Type_var[i]=='Normal':
            x[i]=Normal.Normal_CDFInverse(arg,Mean_var[i],std_var[i])
            pdfx[i]=Normal.Normal_PDF(x[i],Mean_var[i],std_var[i])
        if Type_var[i]=='Lognormal':
            from Models import Lognormal
            x[i]=Lognormal.Lognormal_CDFInverse(arg,Mean_var[i],std_var[i])
            pdfx[i]=Lognormal.Lognormal_PDF(x[i],Mean_var[i],std_var[i])
        if Type_var[i]=='Uniform':
            from Models import Uniform
            x[i]=Uniform.Uniform_CDFInverse(arg,Mean_var[i],std_var[i])
            pdfx[i]=Uniform.Uniform_PDF(x[i],Mean_var[i],std_var[i])
        if Type_var[i]=='Gamma':
            from Models import Gamma
            x[i]=Gamma.Gamma_CDFInverse(arg,Mean_var[i],std_var[i])
            pdfx[i]=Gamma.Gamma_PDF(x[i],Mean_var[i],std_var[i])
        if Type_var[i]=='Gumbel':
            from Models import Gumbel
            x[i]=Gumbel.Gumbel_CDFInverse(arg,Mean_var[i],std_var[i])
            pdfx[i]=Gumbel.Gumbel_PDF(x[i],Mean_var[i],std_var[i])
            
    #Jacobian
    Jxu=np.zeros((n,n))
    d=np.zeros((n,n))
    for i in range(n):
        d[i,i]=pdfx[i]/Normal.Normal_PDF(input_U[i],0,1)
    Jux=np.matmul(np.linalg.inv(L0[0]),d)
    Jxu=np.linalg.inv(Jux)
    
    return x,Jxu