import numpy as np
import pandas as pd

from copy import copy
import matplotlib.pyplot as plt
import random
import scipy
import os
import sys

import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam


#cutoff at 300
#3103 words

#%% find file names, set up initial source file
base_path = sys.argv[1]

# read in their training data
pos_names_train = os.listdir(base_path + '/train/pos/')
neg_names_train = os.listdir(base_path + '/train/neg/')

# read in their test data
pos_names_test = os.listdir(base_path + '/test/pos/')
neg_names_test = os.listdir(base_path + '/test/neg/')

# initialize lists with size equal to num of documents
pos_text = ['']*(len(pos_names_train) + len(pos_names_test))
neg_text = ['']*(len(neg_names_train) + len(neg_names_test))


#%%

# list of puncturation  to delete
punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~''' 

def processText(x):
    '''
    Objective: take input text, make lower case, remove end line indicators, and delete puncutuation
    
    Input: Text
    Output: Reformatted text
    '''
    x = x.lower()
    x = x.replace('/><br','')
    for char in x:
        if char in punc:
           x = x.replace(char,'')
    return(x)



#%% read in positive and negative reviews

# list of bad indicies
bad_pos = []

# j indicates test/train
for j in range(2):
    # I indicates document index (12.5k documents in positive test and train)
    for i in range(12500):
        # try to read the test, if something is wrong say so and append the file index
        try:
            # load in text
            if j == 0:
                temp_text = pd.read_csv(base_path + '/train/pos/' + pos_names_train[i],delim_whitespace=True,header=None,engine='python',error_bad_lines=False) # puts it into a column each word
            else:
                temp_text = pd.read_csv(base_path + '/test/pos/' + pos_names_test[i],delim_whitespace=True,header=None,engine='python',error_bad_lines=False)
            
            # change text from dataframe to list, run preprocessing and delete blank spaces.
            temp_list = temp_text.values.tolist()[0]
            temp_list = [processText(str(x)) for x in temp_list]
            temp_list = [x for x in temp_list if x != '' and x != ' ']
            
            # insert text into list of documents
            pos_text[i + j*12500] = temp_list
        except:
            print('{} is bad'.format(i))
            bad_pos.append(i + j*12500)
        # indicator tracking progress
        if i % 500 == 0:
            print('Done with {}'.format(i))

# read in negative reviews
# list of bad indicies
bad_neg = []

for j in range(2):
    for i in range(12500):
        try:
            if j == 0:
                temp_text = pd.read_csv(base_path + '/train/neg/' + neg_names_train[i],delim_whitespace=True,header=None,engine='python',error_bad_lines=False)
            else:
                temp_text = pd.read_csv(base_path + '/test/neg/' + neg_names_test[i],delim_whitespace=True,header=None,engine='python',error_bad_lines=False)# puts it into a column each word
            temp_list = temp_text.values.tolist()[0]
            temp_list = [processText(str(x)) for x in temp_list]
            temp_list = [x for x in temp_list if x != '' and x != ' ']
            neg_text[i + j*12500] = temp_list
        except:
            print('{} is bad'.format(i))
            bad_neg.append(i + j*12500)
        if i % 500 == 0:
            print('Done with {}'.format(i))
#%% append positive words to a list

# we really dont need to do this, they are ordered in positive/negative already 
# this is just a safety check/ would be good if positive and negative results are mixed for some reason

labels = []
for i in range(25000):
    if i < 12500:
        x = pos_names_train[i]
    else:
        x = pos_names_test[i-12500]
    if pos_text[i] != '':
        temp = x.split('_')[1]
        val = temp.split('.')[0]
        if int(val) < 5:
            lab = 0
        else:
            lab = 1
        labels.append(lab)

for i in range(len(neg_text)):
    if i < 12500:
        x = neg_names_train[i]
    else:
        x = neg_names_test[i-12500]
    if neg_text[i] != '':
        temp = x.split('_')[1]
        val = temp.split('.')[0]
        if int(val) < 5:
            lab = 0
        else:
            lab = 1
        labels.append(lab)


labels = pd.DataFrame(labels,columns=['Labels'])
labels.to_csv(base_path + '/labels.csv')

#%%
# figure out length of observations

len_pos = [len(x) for x in pos_text]
len_neg = [len(x) for x in neg_text]

# plot histogram of document lengths
plt.hist(len_pos,bins = 20)
plt.show()

plt.hist(len_neg,bins = 20)
plt.show()

# count the number of documents of each length
input_dist = {}
for x in len_pos:
    try:
        input_dist.update({str(x): input_dist[str(x)] + 1})
    except:
        input_dist.update({str(x): 1})
for x in len_neg:
    try:
        input_dist.update({str(x): input_dist[str(x)] + 1})
    except:
        input_dist.update({str(x): 1})

# turn the dictionary into a dataframe, sort by length
input_dist = pd.DataFrame(np.array([list(input_dist.keys()),list(input_dist.values())]).T,columns = ['len','count']).astype(int)
input_dist.sort_values(by='len',inplace=True)

# calculate cumulative sum and percentage
input_dist['cum_sum'] = np.cumsum(input_dist['count'])
input_dist['cum_perc'] = [x/50000 for x in input_dist['cum_sum']]

# 512 cutoff gives us 92% of observations

#input_dist.to_csv(base_path + '/initial_dist.csv',index=False)
#%% create a list of all words

words = []
for x in range(len(pos_text)):
    if pos_text[x] != '':
        words.extend(pos_text[x])

for x in range(len(neg_text)):
    if neg_text[x] != '':
        words.extend(pos_text[x])
        
for x in range(len(words)):
    if type(words[x]) != 'str':
        words[x] = str(words[x])
        

#%% find the unique set of words and how many of each word there is
vocab = {}
vocab_count = {}

k = 0
for i in range(len(words)):
    # if the word is in the dictionary iterate by 1 else add to dict.
    try:
        vocab_count[words[i]] += 1
    except:
        vocab_count.update({words[i]:1})
        vocab.update({words[i]:k})
        k += 1

# calculate how many observations occur x times
vocab_dist = {}
for i,j in vocab_count.items():
    # if number of occurrences not in dist add, else iter by 1
    try:
        vocab_dist[j] += 1
    except:
        vocab_dist.update({j:1})

# calculate cumulative percentages

vocab_Df = pd.DataFrame([vocab_dist.keys(),vocab_dist.values()]).T
vocab_Df.columns = ['num_occ','num_words']
vocab_Df.sort_values('num_occ',inplace=True)
vocab_Df['cumSum'] = np.cumsum(vocab_Df['num_words'])

# reverse cum sum and calculate cum perceent. 
vocab_Df['cumSum'] = [len(vocab) - x for x in vocab_Df['cumSum']]
vocab_Df['cumPerc'] = [x/len(vocab) for x in vocab_Df['cumSum']]

#%%
# plot cumulative percentage on a log scale
plt.plot(vocab_Df['num_occ'],vocab_Df['cumPerc'])
plt.yscale('log')
plt.title('survival graph of words based on number of times occurred')
plt.ylabel('cumulative percentage above')
plt.xlabel('number of words which occurred atleast this many times')
plt.show()

# save vocab distribution
vocab_Df.to_csv(base_path + '/vocab_Dist.csv')

#%%
boundary = 300 # this boundary was chosen from the vocab distribution
# total of words is  3514

# make a dictionary of approved words that are above the boundary
words_high = {}
k = 0
for i,j in vocab_count.items():
    if j >= boundary:
        words_high.update({i:k})
        k += 1

numWords = len(words_high)
print(numWords)



#%%
def filterText(text):
    '''
    Objective: Filter text so that only approved words remain
               The remaining words are assigned a unique integer for their representation
    
    Input: text - input sentence to be filtered
    Output: processed_text - text which only has common words (> 300 occurrences)
    '''
    processed_text = []
    for x in range(len(text)):
        if text[x] in words_high:
            processed_text.append(words_high[text[x]])
    return(processed_text)

#%% # filter out words which do not occur atleast 300 times

# filter our positives and negatives
pos_text_post = ['']*len(pos_text)
neg_text_post = ['']*len(neg_text)

for i in range(len(pos_text)):
     pos_text_post[i] = filterText(pos_text[i])
     
for i in range(len(neg_text)):
     neg_text_post[i] = filterText(neg_text[i])


# this determines how many W2V pairs there are with a window of 3
pos_count = [max(3*len(x)-6,1) for x in pos_text_post]
neg_count = [max(3*len(x)-6,1) for x in neg_text_post]
    
total_count = sum(pos_count) + sum(neg_count) # total number of w2v Pairs
print('Number of training Pairs: {}'.format(total_count))

# this number was 41480526 --> 36771226

#%% Determine how much text is below a certain length threhold (we chose 500)

maxLen = max(max([len(x) for x in pos_text_post]),max([len(x) for x in neg_text_post])) # maximum length

# lists of lengths of each observation
len_pos = [len(x) for x in pos_text_post] 
len_neg = [len(x) for x in neg_text_post]

plt.hist(len_pos,bins = 20)
plt.show()

plt.hist(len_neg,bins = 20)
plt.show()

# create a distribution of lengths
filt_dist = {}
for x in len_pos:
    try:
        filt_dist.update({str(x): filt_dist[str(x)] + 1})
    except:
        filt_dist.update({str(x): 1})
for x in len_neg:
    try:
        filt_dist.update({str(x): filt_dist[str(x)] + 1})
    except:
        filt_dist.update({str(x): 1})

# transofrm the length distribution to a data frame and calculate cumulative percentage
filt_dist = pd.DataFrame(np.array([list(filt_dist.keys()),list(filt_dist.values())]).T,columns = ['len','count'])
filt_dist['len'] = [int(x) for x in filt_dist['len']]
filt_dist['count'] = [int(x) for x in filt_dist['count']]
filt_dist.sort_values(by='len',inplace=True)
filt_dist['cum_sum'] = np.cumsum(filt_dist['count'])

filt_dist['cum_perc'] = [x/50000 for x in filt_dist['cum_sum']]

# 95% of observations fall below 500

input_dist.to_csv(base_path + '/filtered_dist.csv',index=False)



#%% Transform each document back into a comma delimited string and then turn the list of strings into a dataframe to save
pos_csv = ['']*len(pos_text_post)
neg_csv = ['']*len(neg_text_post)

for entry in range(len(pos_text_post)):
    row_str = ''
    k = 0
    for word in pos_text_post[entry]:
        if k == 0:
            row_str += str(word)
            k = 1
        else:
            row_str += ','+str(word)
    pos_csv[entry] = row_str

for entry in range(len(neg_text_post)):
    row_str = ''
    k = 0
    for word in neg_text_post[entry]:
        if k == 0:
            row_str += str(word)
            k = 1
        else:
            row_str += ','+str(word)
    neg_csv[entry] = row_str

pos_csv = pd.DataFrame(pos_csv)
neg_csv = pd.DataFrame(neg_csv)


# these positive and negative reviews is whats used in later training.
pos_csv.to_csv(base_path + '/positiveReviews.csv',index=False)
neg_csv.to_csv(base_path + '/negativeReviews.csv',index=False)
#%% Generate negative samples for each word

# set seed
np.random.seed(10)
# number of words in all documents remaining
numObs = sum([len(x) for x in pos_text_post]) + sum([len(x) for x in neg_text_post])

# WP is the word pairs from our documents, NS is the negative samples for each word. up to 5 samples
WP = np.zeros((total_count*2,2)) # because we have both positive and negative pairs, this is the symmetric case
NS = np.zeros((numObs*5,2)) # initial number was wrong, need to adjust for future

# k k2 and c just keep track of where we are in a iteration

k = 0
k2 = 0
c = 0

# for each document in our set
for text in pos_text_post:
    # for each word in the document
    for x in range(len(text)):
        # rnadomly generate 5 negative samples and add them
        randomWords = np.random.choice(list(range(numWords)),5)
        for i in randomWords:
            NS[k2,:] = [text[x],i]
            k2 += 1
            
        # then add all applicable word pairs within the window of 3
        if x < len(text)-3:
            WP[k,:] = [text[x],text[x+3]]
            WP[k+1,:] = [text[x+3],text[x]]
            k += 2
        if x < len(text)-2:
            WP[k,:] = [text[x],text[x+2]]
            WP[k+1,:] = [text[x+2],text[x]]
            k += 2
        if x < len(text)-1:
            WP[k,:] = [text[x],text[x+1]]
            WP[k+1,:] = [text[x+1],text[x]]
            k += 2
    # keep track of how many documents processed and print checkpoints
    c += 1
    if c%500==0:
        print('done with {}/50'.format(c//500 +1))

for text in neg_text_post:
    for x in range(len(text)):
        randomWords = np.random.choice(list(range(numWords)),5)
        for i in randomWords:
            NS[k2,:] = [text[x],i]
            k2 += 1
        if x < len(text)-3:
            WP[k,:] = [text[x],text[x+3]]
            WP[k+1,:] = [text[x+3],text[x]]
            k += 2
        if x < len(text)-2:
            WP[k,:] = [text[x],text[x+2]]
            WP[k+1,:] = [text[x+2],text[x]]
            k += 2
        if x < len(text)-1:
            WP[k,:] = [text[x],text[x+1]]
            WP[k+1,:] = [text[x+1],text[x]]
            k += 2
    c += 1
    if c%500==0:
        print('done with {}/50'.format(c//500 +1))


# save word pairs and negative samples
WP_df = pd.DataFrame(WP,columns=['context','target'])
NS_df = pd.DataFrame(NS,columns=['context','target'])

WP_df.to_csv(base_path + '/WordPairs.csv',index=False)
NS_df.to_csv(base_path + '/NegativeSamples.csv',index=False)

#%% get indices of documents
# basically determine for each document which indicies are applicable

WP_ind = np.zeros((total_count*2,2)) # because we have both positive and negative pairs, this is the symmetric case
NS_ind = np.zeros((numObs*5,2))

obs = 0
k = 0
k2 = 0
c = 0
for text in pos_text_post:
    for x in range(len(text)):
        for i in range(5):
            NS_ind[k2,:] = [obs,k2]
            k2 += 1
        if x < len(text) -3:
            WP_ind[k,:] = [obs,k]
            WP_ind[k+1,:] = [obs,k+1]
            k += 2
        if x < len(text) -2:
            WP_ind[k,:] = [obs,k]
            WP_ind[k+1,:] = [obs,k+1]
            k += 2
        if x < len(text) -1:
            WP_ind[k,:] = [obs,k]
            WP_ind[k+1,:] = [obs,k+1]
            k += 2
    obs += 1
    c += 1
    if c%500==0:
        print('done with {}/50'.format(c//500 +1))

for text in neg_text_post:
    for x in range(len(text)):
        for i in range(5):
            NS_ind[k2,:] = [obs,k2]
            k2 += 1
        if x < len(text) -3:
            WP_ind[k,:] = [obs,k]
            WP_ind[k+1,:] = [obs,k+1]
            k += 2
        if x < len(text) -2:
            WP_ind[k,:] = [obs,k]
            WP_ind[k+1,:] = [obs,k+1]
            k += 2
        if x < len(text) -1:
            WP_ind[k,:] = [obs,k]
            WP_ind[k+1,:] = [obs,k+1]
            k += 2
    obs += 1
    c += 1
    if c%500==0:
        print('done with {}/50'.format(c//500 +1))


# transform indicies to a dataframe and save
WP_ind_df = pd.DataFrame(WP_ind,columns=['obs','row'])
NS_ind_df = pd.DataFrame(NS_ind,columns=['obs','row'])

WP_ind_df = WP_ind_df.astype('int')
NS_ind_df = NS_ind_df.astype('int')

WP_ind_df.to_csv(base_path + '/WordPairs_ind.csv',index=False)
NS_ind_df.to_csv(base_path + '/NegativeSamples_ind.csv',index=False)


#%%
# generate the splits for each of our CV's, this is still determined by the previously set seed.
CVS_pos = []
CVS_neg = []

# create a set of 1-25000 and randomly sample from this half the values
# that half will be the training set for 1 CV the other will be the test set
# do this 20 times and store in a matrix
sample = set(range(25000))
for cv in range(20):
    pos =  set(random.sample(sample, 12500))
    pos_ind = [0]*25000
    for i in pos:
        pos_ind[i] = 1
    
    neg = set(random.sample(sample,12500))
    neg_ind = [0]*25000
    for i in neg:
        neg_ind[i] = 1
    
    CVS_pos.append(pos_ind)
    CVS_neg.append(neg_ind)

CVS_pos = pd.DataFrame(CVS_pos)
CVS_neg = pd.DataFrame(CVS_neg)

CVS_pos.to_csv(base_path + '/pos_cvs.csv',index=False)
CVS_neg.to_csv(base_path + '/neg_cvs.csv',index=False)

# this gives us our CV filters This is where we stopped as far as code counts