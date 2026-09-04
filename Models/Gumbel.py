# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 09:52:28 2018

@author: abtahi1
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 09:51:39 2018

@author: abtahi1
"""

#Gumbel distribution: Largest value with c=0 (right skewed) 
def Gumbel_Samples(mean_value,std_value,seed_value,Num_samples):
    '''
    mean_value,std_value,seed_value,Num_samples
    '''
    import math
    from scipy.stats import gumbel_r
    gumbel_r.mean=mean_value
    gumbel_r.std=std_value
    scale=gumbel_r.std*math.sqrt(6)/math.pi
    loc=gumbel_r.mean-scale*0.5772
    samples=gumbel_r.rvs(loc,scale,Num_samples,random_state=seed_value)
    return samples

def Gumbel_PDF(x,mean_value,std_value):
    '''
    x,mean_value,std_value
    '''
    import math
    from scipy.stats import gumbel_r
    gumbel_r.mean=mean_value
    gumbel_r.std=std_value
    scale=gumbel_r.std*math.sqrt(6)/math.pi
    loc=gumbel_r.mean-scale*0.5772
    return gumbel_r.pdf(x,loc,scale)

def Gumbel_CDF(x,mean_value,std_value):
    '''
    x,mean_value,std_value
    '''
    import math
    from scipy.stats import gumbel_r
    gumbel_r.mean=mean_value
    gumbel_r.std=std_value
    scale=gumbel_r.std*math.sqrt(6)/math.pi
    loc=gumbel_r.mean-scale*0.5772
    return gumbel_r.cdf(x,loc,scale)

def Gumbel_CDFInverse(x,mean_value,std_value):
    '''
    x,mean_value,std_value
    '''
    import math
    from scipy.stats import gumbel_r
    gumbel_r.mean=mean_value
    gumbel_r.std=std_value
    scale=gumbel_r.std*math.sqrt(6)/math.pi
    loc=gumbel_r.mean-scale*0.5772
    return gumbel_r.ppf(x,loc,scale)
