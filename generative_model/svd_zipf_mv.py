import numpy as np
from numpy import transpose as tp
from numpy.linalg import svd
from scipy.sparse import csc_matrix
#from sparsesvd import sparsesvd as lanczos
from scipy.sparse.linalg import svds
from scipy.optimize import minimize_scalar
import sys


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

"""
Assumes missing values are -inf
Initial imputation is (min value - 2)
The parameter alpha imputes between DD-MVSVD (alpha = 1) and EM-MVSVD (alpha = 0).
"""
def svd_missing_values(W, d, exp, alpha = 1, tol=10**-3, max_iters = 100):
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
    #print("Imputing")
    for (i, j) in missing_elements:
        W[i][j] = min_value - 2

    #EM algorithm
    #print("First EM step")
    """
    Ut, S, Vt = lanczos(csc_matrix(W), d)
    W_tm1 = tp(Ut) @ np.diag(S) @ Vt
    """
    U, S, Vt = svd(W, full_matrices=False)
    W_tm1 = U[:,0:d] @ np.diag(S[0:d]) @ Vt[0:d]
    for (i, j) in missing_elements:
        W[i][j] = W_tm1[i][j]

    num_iters = 0
    while True:
        num_iters += 1
        #print("SVD")
        """
        Ut, S, Vt = lanczos(csc_matrix(W), d)
        W_hat = tp(Ut) @ np.diag(S) @ Vt
        """
        U, S, Vt = svd(W, full_matrices=False)
        W_hat = U[:,0:d] @ np.diag(S[0:d]) @ Vt[0:d]
        """
        U, S, Vt = svds(W, d)
        W_hat = U @ np.diag(S) @ Vt
        """

        #print("Calculating lambda")
        if alpha == 0:
            l = -1*calculate_alpha_zero_lambda(W, W_tm1, W_hat, non_missing_elements)
        else:
            l = minimize_scalar(lambda l: (1-alpha)*frob_diff(W, l * W_hat + (1-l)*W_tm1, non_missing_elements) + alpha * gof(exp, l * W_hat + (1-l)*W_tm1), bounds = (0, 1), method = 'bounded').x

        W_t = l * W_hat + (1-l) * W_tm1

        #The frobenius norm of W - W_new, calculated efficiently while replacing the missing elements of W
        #print("Computing frob")
        frob_difference = 0
        for (i, j) in missing_elements:
            temp = W[i][j]
            W[i][j] = W_t[i][j]
            frob_difference += (temp - W[i][j])**2

        frob_difference = frob_difference ** 0.5/frob_norm
        #print("rel frob:", frob_difference)
        if frob_difference < tol or num_iters >= max_iters:
            #return tp(Ut), S, Vt
            return U, S, Vt

        W_tm1 = W_t
