# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:50:27 2018

@author: abtahi1
"""

#Lognormal distribution
def Lognormal_Samples(mean_value,std_value,seed_value,Num_samples):
    '''
    mean_value,std_value,seed_value,Num_samples
    '''
    import math
    from scipy.stats import lognorm
    lognorm.mean=mean_value
    lognorm.std=std_value
    zeta=math.log(lognorm.mean)-0.5*math.log(1+(lognorm.std/lognorm.mean)**2)
    scale=math.exp(zeta)
    s=(math.log((lognorm.std/lognorm.mean)**2+1))**0.5
    samples=lognorm.rvs(s,0,scale,Num_samples,random_state=seed_value)
    return samples

def Lognormal_PDF(x,mean_value,std_value):
    '''
    x,mean_value,std_value
    '''
    import math
    from scipy.stats import lognorm
    lognorm.mean=mean_value
    lognorm.std=std_value
    zeta=math.log(lognorm.mean)-0.5*math.log(1+(lognorm.std/lognorm.mean)**2)
    scale=math.exp(zeta)
    s=(math.log((lognorm.std/lognorm.mean)**2+1))**0.5
    return lognorm.pdf(x,s,0,scale)

def Lognormal_CDF(x,mean_value,std_value):
    '''
    x,mean_value,std_value
    '''
    import math
    from scipy.stats import lognorm
    lognorm.mean=mean_value
    lognorm.std=std_value
    zeta=math.log(lognorm.mean)-0.5*math.log(1+(lognorm.std/lognorm.mean)**2)
    scale=math.exp(zeta)
    s=(math.log((lognorm.std/lognorm.mean)**2+1))**0.5
    return lognorm.cdf(x,s,0,scale)

def Lognormal_CDFInverse(x,mean_value,std_value):
    '''
    x,mean_value,std_value
    '''
    import math
    from scipy.stats import lognorm
    lognorm.mean=mean_value
    lognorm.std=std_value
    zeta=math.log(lognorm.mean)-0.5*math.log(1+(lognorm.std/lognorm.mean)**2)
    scale=math.exp(zeta)
    s=(math.log((lognorm.std/lognorm.mean)**2+1))**0.5
    return lognorm.ppf(x,s,0,scale)