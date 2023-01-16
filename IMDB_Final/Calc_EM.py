import numpy as np
import pandas as pd
import random
import copy

import sys
import os


base_path = sys.argv[2]

K = [1,2,5]

split = int(sys.argv[1]) -1 #[This is a range of 1-60]
k = K[split//20]
cv = split%20

np.random.seed(10)

#%%

def calculate_lambda(W, W_old, W_hat, non_missing_elements):
    num = 0
    denom = 0
    for (i, j) in non_missing_elements:
        diff = W_old[i][j] - W_hat[i][j]
        num += (W[i][j] - W_old[i][j])*diff
        denom += diff**2

    return num/denom


"""
Assumes missing values are -inf
Initial imputation is (min value - 2)
"""

def calcTruncSVD(x,d):
    '''
    Objective: calculate the truncated SVD 
    
    Input - x: Matrix
            d: rank of the reduced matrix
            
    Output - svd1A: Truncated Matrix
             svd1: Left Side of SVD, U*sqrt(Sigma)
             svd1T: Right side of SVD, sqrt(Sigma)*V^T
    '''
    svd = np.linalg.svd(x)
    svd1 = np.dot(svd[0][:,:d],np.diag([x**.5 for x in svd[1][:d]]))
    svd1T = np.dot(np.diag([x**.5 for x in svd[1][:d]]),svd[2][:d,:])
    svd1A = np.dot(svd1,svd1T)
    return(svd1A)

def svd_missing_values(X, d, tol=10**-3, max_iters = 10):
    '''
    Objective: Run missing value svd
    
    Input -X: PMI matrix with -inf values,
           d: number of dimensions
           tol: tolerance for ending
           max_iters: maximum iterations
           
    Output - W the imputed matrix
    '''
    
    W = copy.copy(X)
    missing_elements = set()
    non_missing_elements = set()
    min_value = np.Inf
    frob_norm = 0
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            if not np.isfinite(W[i][j]):
                missing_elements.add((i, j))
            elif W[i][j] < min_value:
                min_value = W[i][j]
            if np.isfinite(W[i][j]):
                non_missing_elements.add((i, j))
                frob_norm += W[i][j]**2
    frob_norm = frob_norm ** 0.5
    

    #impute W
    for (i, j) in missing_elements:
        W[i][j] = min_value - 2

    #EM algorithm
    # changed below lines from neils code
    W_tm1 = calcTruncSVD(W,d)
    #Ut, S, Vt = lanczos.tochastic_svd(csc_matrix(W), d)
    #W_tm1 = tp(Ut) @ np.diag(S) @ Vt
    for (i, j) in missing_elements:
        W[i][j] = W_tm1[i][j]

    num_iters = 0
    while True:
        num_iters += 1
        # changed below line
        W_hat = calcTruncSVD(W,d)
        #Ut, S, Vt = lanczos(csc_matrix(W), d)
        #W_hat = tp(Ut) @ np.diag(S) @ Vt
        """
        U, S, Vt = svds(W, d)
        W_hat = U @ np.diag(S) @ Vt
        """

        l = calculate_lambda(W, W_tm1, W_hat, non_missing_elements)
        
        l = -l

        W_t = l * W_hat + (1-l) * W_tm1

        #The frobenius norm of W - W_new, calculated efficiently
        frob_difference = 0
        for (i, j) in missing_elements:
            temp = W[i][j]
            W[i][j] = W_t[i][j]
            frob_difference += (temp - W[i][j])**2

        frob_difference = frob_difference ** 0.5/frob_norm
        print("rel frob:", frob_difference)
        if frob_difference < tol or num_iters >= max_iters:
            #return tp(Ut), S, Vt
            #return W_hat
            return(W)
        W_tm1 = W_t

#%%

# Run EM_MVSVD

PMI = np.array(pd.read_csv(base_path + '/Matricies/PMI_k{}_cv{}_matrix.csv'.format(k,cv)))
PMI_missV = svd_missing_values(PMI,100,max_iters=1)
PMI_missV_df = pd.DataFrame(PMI_missV)
PMI_missV_df.to_csv(base_path + '/Matrices/EM_matrix_k{}_cv{}.csv'.format(k,cv),index=False)
print('Done with K: {}'.format(k))