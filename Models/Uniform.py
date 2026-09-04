# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 09:51:55 2018

@author: abtahi1
"""
#Uniform Distribution
def Uniform_Samples(mean_value,std_value,seed_value,Num_samples):
    '''
    mean_value,std_value,seed_value,Num_samples
    '''
    from scipy.stats import uniform
    uniform.mean=mean_value
    uniform.std=std_value
#    x=3
    a=uniform.mean-uniform.std*(3**0.5)
    b=uniform.mean+uniform.std*(3**0.5)
    loc=a
    scale=b-a
#    print(
    samples=uniform.rvs(loc,scale,Num_samples,random_state=seed_value)
    return samples

def Uniform_PDF(x,mean_value,std_value):
    '''
    x,mean_value,std_value
    '''
    from scipy.stats import uniform
    uniform.mean=mean_value
    uniform.std=std_value
    a=uniform.mean-uniform.std*(3**0.5)
    b=uniform.mean+uniform.std*(3**0.5)
    loc=a
    scale=b-a
    return uniform.pdf(x,loc,scale)

def Uniform_CDF(x,mean_value,std_value):
    '''
    x,mean_value,std_value
    '''
    from scipy.stats import uniform
    uniform.mean=mean_value
    uniform.std=std_value
    a=uniform.mean-uniform.std*(3**0.5)
    b=uniform.mean+uniform.std*(3**0.5)
    loc=a
    scale=b-a
    return uniform.cdf(x,loc,scale)

def Uniform_CDFInverse(x,mean_value,std_value):
    '''
    x,mean_value,std_value
    '''
    from scipy.stats import uniform
    uniform.mean=mean_value
    uniform.std=std_value
    a=uniform.mean-uniform.std*(3**0.5)
    b=uniform.mean+uniform.std*(3**0.5)
    loc=a
    scale=b-a
    return uniform.ppf(x,loc,scale)

