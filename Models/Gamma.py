# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 09:51:39 2018

@author: abtahi1
"""

#Gamma distribution
def Gamma_Samples(mean_value,std_value,seed_value,Num_samples):
    '''
    mean_value,std_value,seed_value,Num_samples
    '''
    from scipy.stats import gamma
    gamma.mean=mean_value
    gamma.std=std_value
    k=(gamma.mean**2)/(gamma.std**2)
    scale=(gamma.std**2)/(gamma.mean)
    samples=gamma.rvs(k,0,scale,Num_samples,random_state=seed_value)
    return samples

def Gamma_PDF(x,mean_value,std_value):
    '''
    x,mean_value,std_value
    '''
    from scipy.stats import gamma
    gamma.mean=mean_value
    gamma.std=std_value
    k=(gamma.mean**2)/(gamma.std**2)
    scale=(gamma.std**2)/(gamma.mean)
    return gamma.pdf(x,k,0,scale)

def Gamma_CDF(x,mean_value,std_value):
    '''
    x,mean_value,std_value
    '''
    from scipy.stats import gamma
    gamma.mean=mean_value
    gamma.std=std_value
    k=(gamma.mean**2)/(gamma.std**2)
    scale=(gamma.std**2)/(gamma.mean)
    return gamma.cdf(x,k,0,scale)

def Gamma_CDFInverse(x,mean_value,std_value):
    '''
    x,mean_value,std_value
    '''
    from scipy.stats import gamma
    gamma.mean=mean_value
    gamma.std=std_value
    k=(gamma.mean**2)/(gamma.std**2)
    scale=(gamma.std**2)/(gamma.mean)
    return gamma.ppf(x,k,0,scale)
