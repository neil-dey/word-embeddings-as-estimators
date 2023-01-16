from collections import Counter
import numpy as np
from scipy.stats import norm
from scipy.stats import mvn

def compute_copula_parameters(corpus, ZIPF_CONSTANT = 1.0):
    word_counts = Counter()
    for word in corpus:
        word_counts[word] += 1

    """
    Calculate the empirical probabilities
    """
    total = sum(word_counts.values())
    emperical_dist = sorted([v/total for (k, v) in word_counts.items()])[::-1]


    """
    Fit a Zipfian distribution
    """
    vocabulary_size = len(word_counts)
    denom = sum([1/(i+1)**ZIPF_CONSTANT for i in range(vocabulary_size)])
    finite_zipf_pmf = [1/(i+1)**ZIPF_CONSTANT/denom for i in range(vocabulary_size)]


    """
    Determine the X words (words that appear first in a bigram) and Y words (those that appear second in a bigram)
    as well as the rankings of each
    """
    #print()
    #print("Ranking frequencies of X and Y words")
    x_word_counts = Counter()
    y_word_counts = Counter()
    previous_word = None
    for word in corpus:
        if not previous_word:
            previous_word = word
            continue
        x_word_counts[previous_word] += 1
        y_word_counts[word] += 1
        previous_word = word
    #print("Creating index to word dictionaries")
    x_word_to_index = dict()
    index = 0
    for (k, v) in word_counts.most_common():
        x_word_to_index[k] = index
        index += 1

    y_word_to_index = dict()
    index = 0
    for (k, v) in y_word_counts.most_common():
        y_word_to_index[k] = index
        index += 1

    #print("x_count_size", len(word_counts))
    #print("y_count_size", len(y_word_counts))

    """
    Generate a matrix of bigram frequencies
    """
    empirical_pmf = np.zeros(shape=(vocabulary_size, vocabulary_size))
    #print("Creating empirical frequency matrix")
    previous_word = None
    for word in corpus:
        if word not in word_counts:
            continue
        if not previous_word:
            previous_word = word
            continue
        empirical_pmf[x_word_to_index[previous_word]][y_word_to_index[word]] += 1
        previous_word = word
    total = empirical_pmf.sum()
    #print("Creating empirical pmf")
    empirical_pmf  = empirical_pmf/total


    """
    Calculate the marginal distribution of first/second word rankings and their means
    """
    #print("Calculating marginals")
    mu_x = 0
    mu_y = 0
    x_marginals = empirical_pmf.sum(axis=0)
    #print(x_marginals[0:10])
    y_marginals = empirical_pmf.sum(axis=1)
    #print(y_marginals[0:10])
    for i in range(len(empirical_pmf)):
        mu_x += (i+1)*x_marginals[i]
        mu_y += (i+1)*y_marginals[i]

    #print(mu_x)
    #print(mu_y)

    """
    Compute the covariance of first word and second word rankings
    """
    #print("Computing covariance")
    covariance = 0
    x_var = 0
    y_var = 0
    for i in range(len(empirical_pmf)):
        for j in range(len(empirical_pmf[0])):
            covariance += empirical_pmf[i][j] * (i+1 - mu_x) * (j+1 - mu_y)
            x_var += empirical_pmf[i][j] * (i+1 - mu_x)**2
            y_var +=  empirical_pmf[i][j] * (j+1 - mu_y)**2

    return covariance/(x_var * y_var)**0.5


def harmonic(n, m):
    return sum([1/(i+1)**m for i in range(n)])


def generate_copula(vocabulary_size, CORRELATION, ZIPF_CONSTANT = 1.0):
    denom = sum([1/(i+1)**ZIPF_CONSTANT for i in range(vocabulary_size)])
    finite_zipf_pmf = [1/(i+1)**ZIPF_CONSTANT/denom for i in range(vocabulary_size)]

    xis = [sum([finite_zipf_pmf[i] for i in range(j)]) for j in range(1, vocabulary_size)] + [1]
    uis = xis

    est_variance = harmonic(vocabulary_size, ZIPF_CONSTANT - 2)/harmonic(vocabulary_size, ZIPF_CONSTANT) - harmonic(vocabulary_size, ZIPF_CONSTANT-1)**2/harmonic(vocabulary_size, ZIPF_CONSTANT)**2

    bivariate_table = np.zeros((vocabulary_size, vocabulary_size))
    #print("Computing copula")
    for i in range(vocabulary_size):
        for j in range(vocabulary_size):
            bivariate_table[i][j] = mvn.mvnun([-1*est_variance**0.5*5, -1*est_variance**0.5*5], [norm.ppf(uis[i]), norm.ppf(uis[j])], [0, 0], [[1, CORRELATION], [CORRELATION, 1]])[0]

    #print("Computing pmf table")
    joint_zipf_pmf = np.copy(bivariate_table)
    for i in range(vocabulary_size):
        for j in range(vocabulary_size):
            if i==0 and j==0:
                continue
            elif i==0:
                joint_zipf_pmf[i][j] -= bivariate_table[i][j-1]
            elif j==0:
                joint_zipf_pmf[i][j] -= bivariate_table[i-1][j]
            else:
                joint_zipf_pmf[i][j] = bivariate_table[i][j] - bivariate_table[i-1][j] - bivariate_table[i][j-1] + bivariate_table[i-1][j-1]

    #print()
    #print(joint_zipf_pmf)
    return joint_zipf_pmf
