#!/bin/bash

# Code to replicate figure displaying Zipfian fit of Brown Corpus
python3 Brown/zipf_fit.py

########################################################################
# Code to replicate section 5
# Note that all commands in loops can actually be performed in parallel.
########################################################################
# Run on an Intel Core i7-7600 CPU @ 2.80 GHz, 16GB RAM
# W2V was timed to take ~2 hours, EM-MVSVD ~10 hours, DD-MVSVD ~18 hours, and SPPMI ~2 hours
for method in pmi w2v em_mvsvd dd_mvsvd sppmi
do
  python3 generative_model/text_generation.py $method
done

########################################################################
# Code to replicate section 6
# Note that all commands in loops can actually be performed in parallel.
########################################################################

# Runtimes for this section are provided below:
# Initial_Filtering  2hr 12 min
# PMI_Calc  40 min
# PMI_est 1hr 43 min
# Calc_W2V  2hr 56min
# Calc_EM  25 min
# Calc_DD  35 min
# Embed_Matrices  3 min
# Calc_LSTM 1hr 15 min [2 epochs]
#
# A significant portion of time related to Initial_Filtering can be
# attributed to calculating the negative samples for W2V
#
# PMI_calc calculates the PMI and SPPMI matrices
# PMI_est calculates the Zipf and estimated PMI
#
# Total W2V time = ~1 hr + 3 hr = ~4hr
# total DD/EM time ~2/3 hr + 5/3 hr + .5 hr = ~3hr

SCRIPT_DIR=$(dirname -- "$0")
code_dir=$SCRIPT_DIR/IMDB_final

imdb_dir=/mnt/d/Word2Vec/IMDB # Replace argument with location of the IMDB dataset. The 'test' and 'train' folders should be subdirectories in this location.

# Creates additional necessary subdirectories for storing results
mkdir $imdb_dir/Results
mkdir $imdb_dir/Matrices
mkdir $imdb_dir/Embeddings
mkdir $imdb_dir/LSTM

python3 $code_dir/Initial_Filtering.py $imdb_dir

# Requires no more than 3 hours and 16GB per iteration
for i in {1..20}
do
  python3 $code_dir/PMI_calc.py $i $imdb_dir
done

# Requires no more than 3 hours and 16GB per iteration
for i in {1..60}
do
  python3 $code_dir/PMI_est.py $i $imdb_dir
done

# Requires no more than 5 hours and 16GB per iteration
for i in {1..60}
do
  python3 $code_dir/Calc_EM.py $i $imdb_dir
done

# Requires no more than 5 hours and 16GB per iteration
for i in {1..60}
do
  python3 $code_dir/Calc_DD.py $i $imdb_dir
done

# Requires no more than 5 hours and 16GB per iteration
for i in {1..60}
do
  python3 $code_dir/Calc_W2V.py $i $imdb_dir
done

# Requires no more than 5 hours and 16GB per iteration
for i in {1..60}
do
  python3 $code_dir/Embed_Matrices.py $i $imdb_dir
done

# Requires no more than 2 hours and 16GB per iteration
for i in {1..300}
do
  python3 $code_dir/Calc_LSTM.py $i $imdb_dir
done

# Requires no more than 30 minutes and 4GB
python3 $code_dir/Avg_Results.py $imdb_dir
