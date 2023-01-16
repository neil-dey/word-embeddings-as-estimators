from random import random, seed
from collections import Counter
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import os
import sys
from gensim.models.word2vec import Word2Vec
from svd_zipf_mv import svd_missing_values
from compute_copula import compute_copula_parameters, generate_copula
from scipy.sparse.linalg import svds

if len(sys.argv) != 2:
    print("Arguments not given correctly")
    exit()
mode = sys.argv[1]
if mode not in ["pmi", "w2v", "em_mvsvd", "dd_mvsvd", "sppmi"]:
    print("Argument 1 not recognized")
    exit()

# Seed value used to get the saved covariance we use later; basically we just arbitrarily
# chose a Gaussian covariance of -0.1 and tried seeds until one of them was close enough
# This was done *before* the compute_copula script was created; there's no longer a need to do this
SEED_VALUE = 6
seed(SEED_VALUE)
np.random.seed(SEED_VALUE)

vocabulary_size = 500

#print("Reading in all words")
"""
First, read in all the words and find how often they occur
"""
word_counts = Counter()
SCRIPT_DIR = os.path.dirname(__file__)
with open(os.path.join(SCRIPT_DIR, "../Brown/corpus.txt")) as f:
    for line in f.readlines():
        words = line.split()
        for word in words:
            word_counts[word] += 1

"""
Discard all words that occurred infrequently (setting MIN_FREQ_LIMIT = 2 removes about half the words)
"""
MIN_FREQ_LIMIT = 2
for word in word_counts:
    if word_counts[word] <= MIN_FREQ_LIMIT:
        word_counts[word] = 0
word_counts = +word_counts

"""
Calculate the empirical probabilities of the top 10 remaining words
"""
total = sum(word_counts.values())

"""
Fit a Zipfian distribution and print the top 10 probabilities
"""
corpus_unique_words = len(word_counts)

"""
Determine the X words (words that appear first in a bigram) and Y words (those that appear second in a bigram)
as well as the rankings of each
"""
x_word_counts = Counter()
with open(os.path.join(SCRIPT_DIR, "../Brown/corpus.txt")) as f:
    for line in f.readlines():
        words = line.split()
        previous_word = None
        for word in words:
            if previous_word not in word_counts:
                previous_word = word
                continue
            x_word_counts[previous_word] += 1
            previous_word = word
x_word_to_index = dict()
index = 0
for (k, v) in word_counts.most_common():
    x_word_to_index[k] = index
    index += 1


y_word_counts = Counter()
with open(os.path.join(SCRIPT_DIR, "../Brown/corpus.txt")) as f:
    for line in f.readlines():
        words = line.split()
        previous_word = None
        for word in words:
            if word not in word_counts:
                continue
            if not previous_word:
                previous_word = word
                continue
            y_word_counts[word] += 1

y_word_to_index = dict()
index = 0
for (k, v) in y_word_counts.most_common():
    y_word_to_index[k] = index
    index += 1

"""
Generate a matrix of bigram frequencies
"""
empirical_pmf = np.zeros(shape=(corpus_unique_words, corpus_unique_words))
with open(os.path.join(SCRIPT_DIR, "../Brown/corpus.txt")) as f:
    for line in f.readlines():
        words = line.split()
        previous_word = None
        for word in words:
            if word not in word_counts:
                continue
            if not previous_word:
                previous_word = word
                continue
            empirical_pmf[x_word_to_index[previous_word]][y_word_to_index[word]] += 1
            previous_word = word

#print("Sampling subet of words")
"""
We sample a subset of the words to keep
"""
while True:
    SAMPLE_SIZE = vocabulary_size
    total = sum(word_counts.values())
    words_sample = np.random.choice(list(word_counts.keys()), size = SAMPLE_SIZE, replace = False)

    """
    Remove columns and rows of non-sampled
    """
    try:
        sampled_empirical_pmf = np.copy(empirical_pmf)
        sampled_empirical_pmf = np.delete(sampled_empirical_pmf, list(set(range(len(sampled_empirical_pmf))).difference(set([x_word_to_index[w] for w in words_sample]))), axis = 0)
        sampled_empirical_pmf = np.delete(sampled_empirical_pmf, list(set(range(len(sampled_empirical_pmf[0]))).difference(set([y_word_to_index[w] for w in words_sample]))), axis = 1)
        break
    except KeyError:
        # We got a bit unlucky with our sample of words and one of the sampled words either never ended up
        # as a first word, or it was never a second word. Just try again
        pass


"""
Resort the rows and columns so that we still word frequencies ordered least to greatest
"""
# Sorts rows
sampled_empirical_pmf = sampled_empirical_pmf[np.argsort(sampled_empirical_pmf.sum(axis=1))[::-1]]
# Sorts columns
sampled_empirical_pmf = sampled_empirical_pmf[:, np.argsort(sampled_empirical_pmf.sum(axis=0))[::-1]]

# Sort the sampled words by their original frequencies
index_to_x_word = sorted(words_sample, key=lambda w: x_word_to_index[w])
index_to_y_word = sorted(words_sample, key=lambda w: y_word_to_index[w])

x_word_to_index = dict()
y_word_to_index = dict()
# Now record the new x/y indices of each word
for (i, word) in enumerate(index_to_x_word):
    x_word_to_index[word] = np.argsort(sampled_empirical_pmf.sum(axis=1))[::-1][i]
for (i, word) in enumerate(index_to_y_word):
    y_word_to_index[word] = np.argsort(sampled_empirical_pmf.sum(axis=0))[::-1][i]


#print("Creating empirical pmf")
total = sampled_empirical_pmf.sum()
sampled_empirical_pmf  = sampled_empirical_pmf/total

# Create a dense transition probability matrix of words: Pr(w_t | w_{t-1})
ZIPF_CONSTANT = 1
#print("Creating TPM")
cooccurrence_matrix = np.load(os.path.join(SCRIPT_DIR, "./joint_zipf_pmf_seed6covar_" + str(vocabulary_size) + ".npy"))

generative_tpm = cooccurrence_matrix/cooccurrence_matrix.sum(axis=1, keepdims=1)

# Figure out the true probabilities Pr(w) of each word by calculating the stationary distribution of the tpm
#print("Calculating stationary distribution")
eigenvalues, left_eigenvectors = scipy.linalg.eig(generative_tpm, left=True, right=False)
index = 0
for w in eigenvalues:
    if np.isclose(w, 1.0):
        break
    index += 1
stationary_distribution = np.real(left_eigenvectors[:,index]/sum(left_eigenvectors[:,index]))
if not all([prob > 0 for prob in stationary_distribution]):
    print("Stationary distribution was no good :(")
    exit()

# Calculate the true PMI
# TODO: This should later take into account asymmetry of rankings between X words and Y words
#print("Generating true PMI")
true_PMI = np.zeros(shape=(vocabulary_size, vocabulary_size))
for i in range(vocabulary_size):
    for j in range(vocabulary_size):
        true_PMI[i][j] = np.log(cooccurrence_matrix[i][j]) - np.log(stationary_distribution[i]) - np.log(stationary_distribution[j])



corpus_sizes = [int(np.exp(n)) for n in range(15)][3:]
d = 100
k = 10
corpuses_considered = []
frobenius_norms = []
props_missing = []
num_reps = 0
CORPUS_REPS = 20
corpus_index = 0

while True: #for corpus_size in corpus_sizes:
    if corpus_index >= len(corpus_sizes):
        break
    corpus_size = corpus_sizes[corpus_index]
    frob_diff = 0
    non_missing = 0

    if num_reps == 0:
        print("Corpus of size", corpus_size)
    if num_reps >= CORPUS_REPS:
        num_reps = 0
        corpus_index += 1
        continue
    num_reps += 1

    # Create a corpus
    corpus = [index_to_x_word[np.random.choice(range(vocabulary_size), p = stationary_distribution)]]
    word_pair_counts = Counter()
    word_counts = Counter()
    word_counts[corpus[0]] += 1
    context_counts = Counter()
    context_counts[corpus[0]] += 1
    for i in range(corpus_size - 1):
        prev_word = corpus[i]
        new_word = index_to_y_word[np.random.choice(range(vocabulary_size), p = generative_tpm[x_word_to_index[prev_word]])]
        corpus.append(new_word)
        word_pair_counts[(corpus[i], new_word)] += 1
        word_pair_counts[(new_word, corpus[i])] += 1
        word_counts[new_word] += 1
        context_counts[new_word] += 2
    context_counts[corpus[-1]] -= 1

    # Print the corpus as a readable paragraph
    #print(" ".join(corpus))

    if mode == "pmi":
        print("Calculating Emperical PMI from corpus of size", corpus_size)
        PMI = np.zeros(shape=(vocabulary_size, vocabulary_size))
        D = sum(word_pair_counts.values())
        for i in range(vocabulary_size):
            for j in range(vocabulary_size):
                w = index_to_x_word[i]
                c = index_to_y_word[j]
                if word_pair_counts[(w, c)] == 0:
                    PMI[i][j] = -np.inf
                else:
                    PMI[i][j] = np.log(word_pair_counts[(w,c)] * D /(word_counts[w] * context_counts[c]))

        print("Calculating Frobenius norm")
        frob = 0
        non_missing = 0
        for i in range(vocabulary_size):
            for j in range(vocabulary_size):
                if np.isfinite(PMI[i][j]):
                    non_missing += 1
                    frob += (true_PMI[i][j] - PMI[i][j])**2
        frob = (frob/non_missing)**0.5
        frobenius_norms.append(frob)
        props_missing.append(1 - non_missing/vocabulary_size**2)
        print(set(zip(corpus_sizes, frobenius_norms, props_missing)))

    elif mode == "w2v":
        corpuses_considered.append(corpus_size)
        w2vmodel = Word2Vec(sentences=[corpus], vector_size = d, window = 2, min_count = 1, workers = 3, sg = 1, hs = 0, negative = k, ns_exponent = 1, epochs=100, sample=0)
        W = w2vmodel.wv
        C = w2vmodel.syn1neg
        for i in range(vocabulary_size):
            for j in range(vocabulary_size):
                try:
                    frob_diff += (W.get_vector(index_to_x_word[i]).dot(C[w2vmodel.wv.key_to_index[index_to_y_word[j]]]) - (true_PMI[i][j] - np.log(k)))**2
                    non_missing += 1
                except KeyError:
                    pass

        frob = (frob_diff/non_missing)**0.5
        frobenius_norms.append(frob)
        props_missing.append(1 - non_missing/vocabulary_size**2)
        print(list(zip(corpuses_considered, frobenius_norms, props_missing)))
        #print("RMSE: ", (frob_diff/non_missing)**0.5, "Missing:", 1- non_missing/vocabulary_size**2)

    elif mode == "em_mvsvd" or mode == "dd_mvsvd":
        """
        1. Estimate the Zipf parameter (For now, assume Zipf(1))
        2. Estimate the covariance
        3. Generate a copula distribution as an estimate for the true co-occurance
        4. Use this to generate a estimated dense PMI
        5. Plug this into missing values SVD for Goodness of Fit
        """
        empirical_vocab_size = len(set(corpus))
        # Don't go on if the dimensionality is less than the number of unique words
        """
        if empirical_vocab_size <= d:
            continue
        else:
            """
        corpuses_considered.append(corpus_size)

        zipf_param = 1.0
        print("Computing Correlation")
        correlation = compute_copula_parameters(corpus)
        print("Computing Copula")
        dense_cooccurance = generate_copula(empirical_vocab_size, correlation)

        print("Creating empirical PMI")
        PMI = np.zeros(shape=(vocabulary_size, vocabulary_size))
        D = sum(word_pair_counts.values())
        for i in range(vocabulary_size):
            for j in range(vocabulary_size):
                w = index_to_x_word[i]
                c = index_to_y_word[j]
                PMI[i][j] = np.log(word_pair_counts[(w,c)]) + np.log(D) - np.log(word_counts[w]) - np.log(context_counts[c]) - np.log(k)

        print("Estimating dense PMI")
        est_PMI = np.zeros(shape=(empirical_vocab_size, empirical_vocab_size))

        denom = sum([1/(i+1)**ZIPF_CONSTANT for i in range(empirical_vocab_size)])
        est_stationary_dist = [1/(i+1)**ZIPF_CONSTANT/denom for i in range(vocabulary_size)]

        for i in range(empirical_vocab_size):
            for j in range(empirical_vocab_size):
                est_PMI[i][j] = np.log(dense_cooccurance[i][j]) - np.log(est_stationary_dist[i]) - np.log(est_stationary_dist[j]) - np.log(k)

        print("Performing SVD")
        if mode == "em_mvsvd":
            alpha = 0
        else:
            alpha = 1
        U, S, Vt = svd_missing_values(PMI, 100, est_PMI, alpha)
        PMI = U @ np.diag(S) @ Vt

        print("Calculating Frobenius norm")
        index_to_empirical_word = [word for (word, count) in word_counts.most_common()]
        frob = 0
        non_missing = 0
        for i in range(vocabulary_size):
            for j in range(vocabulary_size):
                w = index_to_x_word[i]
                c = index_to_y_word[j]
                if word_counts[w] != 0 and word_counts[c] != 0:
                    non_missing += 1
                    frob += (true_PMI[i][j] - np.log(k) - PMI[index_to_empirical_word.index(w)][index_to_empirical_word.index(c)])**2
        frob = (frob/non_missing)**0.5
        frobenius_norms.append(frob)
        props_missing.append(1 - non_missing/vocabulary_size**2)
        print(list(zip(corpuses_considered, frobenius_norms, props_missing)))
        print()

    elif mode == "sppmi":
        corpuses_considered.append(corpus_size)
        print("Calculating Emperical PMI from corpus of size", corpus_size)
        PMI = np.zeros(shape=(vocabulary_size, vocabulary_size))
        D = sum(word_pair_counts.values())
        for i in range(vocabulary_size):
            for j in range(vocabulary_size):
                w = index_to_x_word[i]
                c = index_to_y_word[j]
                PMI[i][j] = np.log(word_pair_counts[(w,c)]) + np.log(D) - np.log(word_counts[w]) - np.log(context_counts[c])
                if PMI[i][j] < 0 or not np.isfinite(PMI[i][j]):
                    PMI[i][j] = 0

        U, S, Vt = svds(PMI, k=d)
        PMI = U @ np.diag(S) @ Vt

        #print("Calculating Frobenius norm")
        frob = 0
        for i in range(vocabulary_size):
            for j in range(vocabulary_size):
                frob += (true_PMI[i][j] - PMI[i][j])**2
        frob = frob**0.5
        frobenius_norms.append(frob)
        print(list(zip(corpuses_considered, frobenius_norms)))


if mode == "pmi":
    print()
    print(set(zip(corpus_sizes, frobenius_norms, props_missing)))
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.scatter(corpus_sizes, frobenius_norms, c='b')
    ax2.scatter(corpus_sizes, props_missing, c='r')
    plt.xlabel("Corpus Size")
    ax1.set_ylabel("Root Mean Sqaured Error")
    ax2.set_ylabel("Proportion Missing")
    ax2.set(ylim=(0, 1))
    #plt.ylabel("Frobenius Norm")
    plt.xscale("log")
    plt.savefig("Generative_trend_text_generated_vocab" + str(vocabulary_size) + "_zipf" + str(ZIPF_CONSTANT) + "_logscale.png")
