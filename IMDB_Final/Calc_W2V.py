
import numpy as np
import pandas as pd

import os
import sys

import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.layers import Embedding

base_path = sys.argv[2]

K = [1,2,5]

split = int(sys.argv[1]) -1#[This is a range of 1-60]
k = K[split//20]
cv = split%20

np.random.seed(10)
numWords = 3104

CVS_pos = pd.read_csv(base_path + '/pos_cvs.csv')
CVS_neg = pd.read_csv(base_path + '/neg_cvs.csv')


WP_df = pd.read_csv(base_path + '/WordPairs.csv')
NS_df = pd.read_csv(base_path + '/NegativeSamples.csv')

WP_ind_df = pd.read_csv(base_path + '/WordPairs_ind.csv')
NS_ind_df = pd.read_csv(base_path + '/NegativeSamples_ind.csv')

samp_pos = CVS_pos.iloc[cv,:]
samp_neg = CVS_neg.iloc[cv,:]

# now we need to grap Word pairs and Negative Samples then do it
A_obs = [i for (i, v) in zip(range(len(samp_pos)), samp_pos) if v]
B_obs = [i for (i, v) in zip(range(len(samp_neg)), samp_neg) if v]
B_obs = [x + 25000 for x in B_obs]

A_obs.extend(B_obs)

train_ind = WP_ind_df[WP_ind_df['obs'].isin(A_obs)]
train_ind2 = NS_ind_df[NS_ind_df['obs'].isin(A_obs)]

# this now returns us a dataframe with obs and row where row is the rows of the WP matrix we want

wp_train = WP_df.iloc[train_ind['row'],:]
ns_train = NS_df.iloc[train_ind2['row'],:]

ns_sub = ns_train[[i%5 < k for i in range(len(ns_train))]]
ns_sub['Label'] = 0
wp_train['Label'] = 1

train_data = pd.concat([wp_train,ns_sub])
train_data.columns = ['Context','Target','Label']
np.random.seed(10)
train_data = train_data.sample(frac=1).reset_index(drop=True) # this is it for word2vec

def trainW2V(data_train_final_df,s = 100,e = 5,k = 0):
    vocab_size = numWords
    vector_dim = s
    
    input_target = Input((1,))
    input_context = Input((1,))
    
    embedding = Embedding(vocab_size, vector_dim, input_length=1, name='embedding')
    target = embedding(input_target)
    target = Reshape((vector_dim, 1))(target)
    context = embedding(input_context)
    context = Reshape((vector_dim, 1))(context)
    
    #similarity = keras.layers.Dot(1,normalize=True)([target,context])
    dot_prod = tensorflow.keras.layers.Dot(1)([target,context])
    dot_prod = Reshape((1,))(dot_prod)
    output = Dense(1,activation='sigmoid')(dot_prod)
    
    model = Model(inputs=[input_target, input_context], outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    
    labels = np.array(data_train_final_df.Label)
    word_target = np.array(data_train_final_df.Target, dtype="int32")
    word_context = np.array(data_train_final_df.Context, dtype="int32")
    hist = model.fit([word_target,word_context],labels,epochs = e,batch_size =100,use_multiprocessing = True)
        
    model.save(base_path+'/Matrices/w2v_model_{}_k{}_cv{}'.format(vector_dim,k,cv))

trainW2V(train_data,100,10,k)