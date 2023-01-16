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
    W = copy(X)
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

from scipy.optimize import minimize_scalar

def frob_diff(W1, W2, non_missing_elements):
    total = 0
    for (i, j) in non_missing_elements:
        total += (W1[i][j] - W2[i][j])**2

    #print("Frob Diff", total)
    return total

def gof(exp, obs):
    total = 0
    for i in range(len(exp)):
        for j in range(len(exp[0])):
            total += (obs[i][j] - exp[i][j])**2/exp[i][j]

    #print("GoF", total)
    return total

def calculate_alpha_zero_lambda(W, W_old, W_hat, non_missing_elements):
    num = 0
    denom = 0
    for (i, j) in non_missing_elements:
        diff = W_old[i][j] - W_hat[i][j]
        num += (W[i][j] - W_old[i][j])*diff
        denom += diff**2

    return num/denom

def svd_missing_values_zif(X, d,exp,alpha=1, tol=10**-3, max_iters = 10):
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
        
        #l = calculate_lambda(W, W_tm1, W_hat, non_missing_elements)
        
        if alpha == 0:
            l = -1*calculate_alpha_zero_lambda(W, W_tm1, W_hat, non_missing_elements)
        else:
            l = minimize_scalar(lambda l: (1-alpha)*frob_diff(W, l * W_hat + (1-l)*W_tm1, non_missing_elements)
                                + alpha * gof(exp, l * W_hat + (1-l)*W_tm1), bounds = (0, 1), method = 'bounded').x


        W_t = l * W_hat + (1-l) * W_tm1

        #The frobenius norm of W - W_new, calculated efficiently
        frob_difference = 0
        for (i, j) in missing_elements:
            temp = W[i][j]
            W[i][j] = W_t[i][j]
            frob_difference += (temp - W[i][j])**2

        frob_difference = frob_difference ** 0.5/frob_norm
        print("rel frob:", frob_difference)
        if frob_difference < tol or num_iters >= 0:
            #return tp(Ut), S, Vt
            #return W_hat
            return(W)
        W_tm1 = W_t

#%%
np.random.seed(10)
PMI = np.array(pd.read_csv(base_path + '/Matricies/PMI_k{}_cv{}_matrix.csv'.format(k,cv)))
PMI_est = np.array(pd.read_csv(base_path + '/Matrices/estimated_PMI_k{}_cv{}.csv'.format(k,cv)))

PMI_missV_frob = svd_missing_values_zif(PMI,100,PMI_est,5)
PMI_missV_frob_df = pd.DataFrame(PMI_missV_frob)
PMI_missV_frob_df.to_csv(base_path + '/Matrices/DD_matrix_k{}_cv{}.csv'.format(k,cv),index=False)
