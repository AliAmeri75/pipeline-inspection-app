#Weibull Distribution
#Author: MohammadAli Ameri Fard Nasrand

def exponweib_Samples(mean_value,std_value,seed_value,Num_samples):
    '''
    mean_value,std_value,seed_value,Num_samples
    '''
    from scipy.stats import exponweib
    exponweib.mean=mean_value
    exponweib.std=std_value
    k=(exponweib.mean**2)/(exponweib.std**2)
    scale=(exponweib.std**2)/(exponweib.mean)
    samples=exponweib.rvs(k,0,scale,Num_samples,random_state=seed_value)
    return samples

def exponweib_PDF(x,mean_value,std_value):
    '''
    x,mean_value,std_value
    '''
    from scipy.stats import exponweib
    exponweib.mean=mean_value
    exponweib.std=std_value
    k=(exponweib.mean**2)/(exponweib.std**2)
    scale=(exponweib.std**2)/(exponweib.mean)
    return exponweib.pdf(x,k,0,scale)

def exponweib_CDF(x,mean_value,std_value):
    '''
    x,mean_value,std_value
    '''
    from scipy.stats import exponweib
    exponweib.mean=mean_value
    exponweib.std=std_value
    k=(exponweib.mean**2)/(exponweib.std**2)
    scale=(exponweib.std**2)/(exponweib.mean)
    return exponweib.cdf(x,k,0,scale)

def exponweib_CDFInverse(x,mean_value,std_value):
    '''
    x,mean_value,std_value
    '''
    from scipy.stats import exponweib
    exponweib.mean=mean_value
    exponweib.std=std_value
    k=(exponweib.mean**2)/(exponweib.std**2)
    scale=(exponweib.std**2)/(exponweib.mean)
    return exponweib.ppf(x,k,0,scale)
