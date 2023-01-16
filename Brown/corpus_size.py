from collections import Counter
import numpy as np
import sys
import os

infile = "corpus.txt"

vocab_size = 500
"""
Get a dictionary of how often each pair of words appears
"""
word_pair_counts = Counter()
WINDOW_SIZE = 2

unique_words = set()

with open(infile, "r") as corpus:
    for chapter in corpus:
        words_in_chapter = chapter.split()
        for i in range(len(words_in_chapter)):
            current_word = words_in_chapter[i]
            unique_words.add(current_word)
            window = words_in_chapter[max(i-WINDOW_SIZE, 0):min(i+WINDOW_SIZE + 1, len(words_in_chapter))]
            for word in window:
                word_pair_counts[(current_word, word)] += 1
            word_pair_counts[(current_word, current_word)] -= 1
        if len(unique_words) >= vocab_size:
            word_counts = Counter()
            context_counts = Counter()
            for w, c in +word_pair_counts:
                word_counts[w] += word_pair_counts[(w, c)]
                context_counts[c] += word_pair_counts[(w, c)]

            """
            Create the PMI matrix M(w, c) = PMI(w, c) = log[#(w, c)/(#w * #c)]
            """
            words = list(word_counts)
            word_to_index = dict()
            D = sum(word_pair_counts[(w, c)] for w, c in word_pair_counts) # Total number of word-context pairs

            num_zero = 0
            PMI_size = len(words)**2
            for i in range(len(words)):
                for j in range(len(words)):
                    w = words[i]
                    word_to_index[w] = i
                    c = words[j]
                    PMI = np.log(word_pair_counts[(w, c)]*D/(word_counts[w]*context_counts[c]))
                    if PMI <= 0:
                        num_zero += 1

            expon = np.log(PMI_size) - np.log(PMI_size - num_zero)
            print(vocab_size, num_zero/PMI_size, "= 1 - 10^-" + str(expon/np.log(10)))
            vocab_size += 500
