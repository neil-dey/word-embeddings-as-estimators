
import numpy as np
import pandas as pd

import os
import sys

base_path = sys.argv[2]


split = int(sys.argv[1]) -1 #[This is a range of 1-20
cv = split

#numWords = 3514
numWords = 3104
np.random.seed(10)

CVS_pos = pd.read_csv(base_path + '/pos_cvs.csv')
CVS_neg = pd.read_csv(base_path + '/neg_cvs.csv')

WP_df = pd.read_csv(base_path + '/WordPairs.csv')
NS_df = pd.read_csv(base_path + '/NegativeSamples.csv')

WP_ind_df = pd.read_csv(base_path + '/WordPairs_ind.csv')
NS_ind_df = pd.read_csv(base_path + '/NegativeSamples_ind.csv')

samp_pos = CVS_pos.iloc[cv,:]
samp_neg = CVS_neg.iloc[cv,:]


# now we need to grap Word pairs and Negative Samples then do it
All_obs = [i for (i, v) in zip(range(len(samp_pos)), samp_pos) if v]
neg_obs = [i for (i, v) in zip(range(len(samp_neg)), samp_neg) if v]
neg_obs = [x + 25000 for x in neg_obs]

All_obs.extend(neg_obs)

train_ind = WP_ind_df[WP_ind_df['obs'].isin(All_obs)]

# this now returns us a dataframe with obs and row where row is the rows of the WP matrix we want

wp_train = WP_df.iloc[train_ind['row'],:]

wp_train.loc[:,'Label'] = 1

wp_train = wp_train.astype(int)
coocMatrix = np.zeros((numWords,numWords))
for i in range(len(wp_train)):
    loc = wp_train.iloc[i,:]
    coocMatrix[loc[0],loc[1]] += 1
    if i%1000000 == 0:
        print('done with {}/31'.format(i//1000000))


condProbList = coocMatrix.sum(axis=1)
rankList = sorted(range(len(condProbList)), key=lambda k: condProbList[k])
rankList.reverse() # inverting the sort

rankList_df = pd.DataFrame(rankList,columns=['rank'])
rankList_df.to_csv(base_path + '/Matrices/PMI_reorder_cv{}.csv'.format(cv))

coocMatrix = coocMatrix[:,rankList] # reorder our matrices by the rank
coocMatrix = coocMatrix[rankList,:]
condProbList= coocMatrix.sum(axis=1)

ngrams = sum(condProbList) # this should be 37269078

K = [1,2,5]

coMatrix = coocMatrix*int(ngrams)
coMatrix = coMatrix/condProbList[:,None]
coMatrix = coMatrix/condProbList[None,:]
for k in K:
    PMI = np.log(coMatrix) - np.log(k)
    PMI_df = pd.DataFrame(PMI)
    PMI_df.to_csv(base_path + '/Matrices/PMI_k{}_cv{}_matrix.csv'.format(k,cv),index=False)
    
    SPPMI = np.array([[max(x,0) for x in y] for y in PMI])
    SPPMI_df = pd.DataFrame(SPPMI)
    SPPMI_df.to_csv(base_path + '/Matrices/SPPMI_k{}_cv{}.csv'.format(k,cv),index=False)






