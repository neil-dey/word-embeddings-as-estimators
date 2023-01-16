
import numpy as np
import pandas as pd

import sys
import os
import tensorflow


base_path = sys.argv[2]

K = [1,2,5]
numWords = 3104


split = int(sys.argv[1]) -1 #[This is a range of 1-60]
k = K[split//20]
cv = split%20


# transform word2vec into a easy to look up matrix

model = tensorflow.keras.models.load_model(base_path+'/Matrices/w2v_model_{}_k{}_cv{}'.format(100,k,cv))
embedding = model.get_layer('embedding')
w2vX = np.zeros((numWords,100))
for word in range(numWords):
    a = embedding(word)
    w2vX[word] = np.array(a)

w2vX_df = pd.DataFrame(w2vX)
w2vX_df.to_csv(base_path + '/Embeddings/w2v_k{}_cv{}.csv'.format(k,cv),index=False)


def embedFromMatrix(x,d):
    '''
    Objective: Use the right singular values of a matrix as the embedding for a given matrix
    
    Input: x - matrix to be embedded
           d - dimensions of the embedding (how many singular values to use)
           
    Output: svd1 - embedding matrix
    '''
    svd = np.linalg.svd(x)
    svd1 = np.dot(svd[0][:,:d],np.diag([x**.5 for x in svd[1][:d]]))
    return(svd1)

# embed all matrices
EM = pd.read_csv(base_path + '/Matrices/EM_matrix_k{}_cv{}.csv'.format(k,cv))
DD = pd.read_csv(base_path + '/Matrices/DD_matrix_k{}_cv{}.csv'.format(k,cv))
SPPMI = pd.read_csv(base_path + '/Matrices/SPPMI_k{}_cv{}.csv'.format(k,cv))

EM_emb = embedFromMatrix(EM,100)
DD_emb = embedFromMatrix(DD,100)
SPPMI_emb = embedFromMatrix(SPPMI,100)

# In making the matrices the embeddings got misaligned (sorted due to common occurrence)
# we will now unsort them so they align with the uuid of the word
# ex: word 0 is in row 0 f the matrix

rankList = pd.read_csv(base_path + '/Matrices/PMI_reorder_cv{}.csv'.format(cv))
rankList = rankList.sort_values('rank')
rankList = rankList.reset_index()

reorder = list(rankList['index'])

def reorderMat(X,r):
    x = np.array(X)
    x = x[r,:]
    x = pd.DataFrame(x)
    return(x)

# save embeddings
EM_emb = reorderMat(EM_emb,reorder)
DD_emb = reorderMat(DD_emb,reorder)
SPPMI_emb = reorderMat(SPPMI_emb,reorder)

EM_emb.to_csv(base_path + '/Embeddings/EM_matrix_k{}_cv{}.csv'.format(k,cv),index=False)
DD_emb.to_csv(base_path + '/Embeddings/DD_matrix_k{}_cv{}.csv'.format(k,cv),index=False)
SPPMI_emb.to_csv(base_path + '/Embeddings/SPPMI_k{}_cv{}.csv'.format(k,cv),index=False)















