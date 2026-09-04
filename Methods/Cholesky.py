# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 14:00:21 2018

@author: abtahi1
"""
#Cholesky Decomposition
def CholeskyDecomposition(R):
    import numpy as np
    n,n = R.shape
    err = 0
    for k in range(n):
        if R[k][k] <= 0:
            err = k
            print('Error in Choleski decomposition: Matrix must be positive definite')
            break
        R[k][k] = np.sqrt(R[k][k])
        indx = list(range(k+1,n))
        for i in indx:
            R[i][k] = R[i][k] * R[k][k]**(-1)

        for j in range(k+1,n):
            indx = list(range(j,n))
            for i in indx:
                R[i][j] = R[i][j] - R[i][k]*R[j][k]
    L = np.tril(R)
    return L, err

#import numpy as np
#Rx=np.zeros((3,3))
#Rx[0,0]=1.0
#Rx[1,1]=1.0 
#Rx[2,2]=1.0
#Rx[0,1]=0.3
#Rx[1,0]=0.3
#Rx[0,2]=0.2
#Rx[2,0]=0.2
#Rx[1,2]=0.2
#Rx[2,1]=0.2
#print(CholeskyDecomposition(Rx)[0])