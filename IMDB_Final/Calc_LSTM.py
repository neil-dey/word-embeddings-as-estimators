
import numpy as np
import pandas as pd

import os
import sys

import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam


# load in data and set max length
base_path = sys.argv[2]

K = [1,2,5]
models = ['w2v2_','SPPMI_','EM_matrix_','DD_matrix_','indep']
maxLen = 500 # set maximum length of lstm

np.random.seed(10)

split = int(sys.argv[1]) -1#[This is a range of 1-300]

model = split//60
k = K[split%60//20]
cv = split%20

numWords = 3104

pos_text_post = pd.read_csv(base_path + '/positiveReviews.csv')
pos_text_post = [x[0].split(',') for x in pos_text_post.values]
pos_text_post = [[int(y) if y != 'nan' else 0 for y in x] for x in pos_text_post]

neg_text_post = pd.read_csv(base_path + '/negativeReviews.csv')
neg_text_post = [str(x[0]).split(',') for x in neg_text_post.values]
neg_text_post = [[int(y) if y != 'nan' else 0 for y in x] for x in neg_text_post]

CVS_pos = pd.read_csv(base_path + '/pos_cvs.csv')
CVS_neg = pd.read_csv(base_path + '/neg_cvs.csv')


samp_pos = CVS_pos.iloc[cv,:]
samp_neg = CVS_neg.iloc[cv,:]

# this retrieves test/train data
train_pos =  [i for (i, v) in zip(pos_text_post, samp_pos) if v]

train_neg =  [i for (i, v) in zip(neg_text_post, samp_neg) if v]

test_pos = [i for (i, v) in zip(pos_text_post, samp_pos) if not v]
test_neg = [i for (i, v) in zip(neg_text_post, samp_neg) if not v]

labels = [1]*12500
labels.extend([0]*12500)
labels = np.array(labels)

train_pos = [[x+1 for x in y] for y in train_pos]
train_neg = [[x+1 for x in y] for y in train_neg]

lstm_train_pos = tensorflow.keras.preprocessing.sequence.pad_sequences(train_pos,maxlen=500)
lstm_train_neg = tensorflow.keras.preprocessing.sequence.pad_sequences(train_neg,maxlen=500)
lstm_train = np.vstack([lstm_train_pos,lstm_train_neg])

test_pos = [[x+1 for x in y] for y in test_pos]
test_neg = [[x+1 for x in y] for y in test_neg]

lstm_test_pos = tensorflow.keras.preprocessing.sequence.pad_sequences(test_pos,maxlen=500)
lstm_test_neg = tensorflow.keras.preprocessing.sequence.pad_sequences(test_neg,maxlen=500)
lstm_test = np.vstack([lstm_test_pos,lstm_test_neg])

#%%
# for regular models load the embedding layer else make a custom embedding layer

if model < 4:

    encoding = np.array(pd.read_csv(base_path + '/Embeddings/'+ models[model] +'k{}_cv{}.csv'.format(k,cv)))
    encoding = np.vstack([np.zeros((1,100)),encoding])

    movie_input = Input(shape= (None,))
    base = Embedding(numWords+1,100,embeddings_initializer=tensorflow.keras.initializers.Constant(encoding),trainable=False)(movie_input)
    base = Bidirectional(LSTM(100, return_sequences=True))(base)
    base = Bidirectional(LSTM(100))(base)
    base = Dense(1,activation='sigmoid')(base)
    base_model = Model(inputs=movie_input,outputs=base,name='base_model')
    base_model.summary()

    base_model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy'])
    base_model.summary()
else:
    movie_input = Input(shape= (None,))
    base = Embedding(numWords +1,100)(movie_input)
    base = Bidirectional(LSTM(100, return_sequences=True))(base)
    base = Bidirectional(LSTM(100))(base)
    base = Dense(1,activation='sigmoid')(base)
    base_model = Model(inputs=movie_input,outputs=base,name='base_model')
    base_model.summary()

    base_model.compile(optimizer = 'adam',loss = 'binary_crossentropy',metrics=['accuracy'])
    base_model.summary()




#%%
# fit our model, evaluate on the test dataset then save accuracy to results folder

base_model.fit(lstm_train,labels,batch_size=100,epochs=10,verbose=1,shuffle=True)
base_model.save(base_path + '/LSTM/'+models[model]+'_k{}_cv{}'.format(k,cv))
pred_temp = base_model.evaluate(lstm_test,labels)

pred_temp = pd.DataFrame(np.array(pred_temp),columns=['metrics'])
pred_temp.to_csv(base_path + '/Results/'+models[model]+'_acc_k{}_cv{}.csv'.format(k,cv),index=False)

print(pred_temp[1])

