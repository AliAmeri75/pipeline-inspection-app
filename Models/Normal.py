# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:50:26 2018

@author: abtahi1
"""
#Normal distribution
def Normal_Samples(mean_value,std_value,seed_value,Num_samples):
    '''
    mean_value,std_value,seed_value,Num_samples
    '''
    from scipy.stats import norm
    norm.mean=mean_value
    norm.std=std_value
    samples=norm.rvs(norm.mean,norm.std,Num_samples,random_state=seed_value)
    return samples

def Normal_PDF(x,mean_value,std_value):
    '''
    x,mean_value,std_value
    '''
    from scipy.stats import norm
    norm.mean=mean_value
    norm.std=std_value
    return norm.pdf(x,norm.mean,norm.std)

def Normal_CDF(x,mean_value,std_value):
    '''
    x,mean_value,std_value
    '''
    from scipy.stats import norm
    norm.mean=mean_value
    norm.std=std_value
    return norm.cdf(x,norm.mean,norm.std)

def Normal_CDFInverse(p,mean_value,std_value):
    '''
    p,mean_value,std_value
    '''
    from scipy.stats import norm
    norm.mean=mean_value
    norm.std=std_value
    return norm.ppf(p,norm.mean,norm.std)
