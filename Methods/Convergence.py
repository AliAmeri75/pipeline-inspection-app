# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 13:09:42 2018

@author: abtahi1
"""

#Convergence
def Checker(Gum,G0,Grdum,um,e1=0.001,e2=0.001):
    import numpy as np
    flag1=False
    flag2=False
    arg1=abs(Gum/G0)
    if arg1<=e1:
        flag1=True
    alpha=[-x/(np.linalg.norm(Grdum)) for x in Grdum]
    a1=(np.matmul(np.transpose(alpha),um))
    n=len(um)
    a=np.zeros(n)
    for i in range(len(um)):
        a[i]=um[i]-a1*alpha[i]
    arg2=np.linalg.norm(a)
    if arg2<=e2:
        flag2=True     
    return flag1,flag2
