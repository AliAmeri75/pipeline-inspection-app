# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 16:28:25 2018

@author: abtahi1
"""
def MeritFunction(um,Gum,Grdum,gamma=2,eta=10):
    import numpy as np
    c=(gamma*np.linalg.norm(um))/(np.linalg.norm(Grdum))+eta
    if c<(np.linalg.norm(um)/np.linalg.norm(Grdum)):
        c=(np.linalg.norm(um)/np.linalg.norm(Grdum))
    M=0.5*(np.linalg.norm(um))**2+c*abs(Gum)
#    a1=[c*np.sign(Gum)*x for x in Grdum]
#    GrdM=[x+y for x,y in zip(um,a1)]
    return M#,GrdM

def MeritChecker(um,Gum,Grdum,um2,Gum2,Grdum2,Sm,dm,gamma=2,eta=10):#,a=0.5):
#    import numpy as np
    flag=False
    M1=MeritFunction(um,Gum,Grdum,gamma,eta)
    M2=MeritFunction(um2,Gum2,Grdum2,gamma,eta)
#    arg=a*Sm*(np.matmul(np.transpose(M1[1]),dm))   
#    if ((M2[0]-M1[0])<=arg):
    if M2<M1:
        flag=True
    return flag

