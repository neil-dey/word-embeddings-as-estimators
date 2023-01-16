from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
import os

word_counts = Counter()

def harmonic(n, m):
    return sum([1/(i+1)**m for i in range(n)])
def zipf_pdf(vocabulary_size, zipf_param):
    denom = harmonic(vocabulary_size, zipf_param)
    finite_zipf_pmf = [1/(i+1)**zipf_param/denom for i in range(vocabulary_size)]
    return finite_zipf_pmf

SCRIPT_DIR = os.path.dirname(__file__)
with open(os.path.join(SCRIPT_DIR, "./corpus.txt")) as f:
    for line in f.readlines():
        words = line.split()
        for word in words:
            word_counts[word] += 1

MIN_FREQ_LIMIT = 10
for word in word_counts:
    if word_counts[word] < MIN_FREQ_LIMIT:
        word_counts[word] = 0
word_counts = +word_counts

ranks = [i+1 for i in range(len(word_counts.values()))]
frequencies = sorted(list(word_counts.values()))[::-1]
plt.scatter(ranks, frequencies, marker="+", label="Observed Frequency") # Empirical frequencies
plt.plot(ranks, np.array(zipf_pdf(len(word_counts), 1.0)) * sum(word_counts.values()), color='black', label="Expected Frequency")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Rank")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("zipf_fit.png")
