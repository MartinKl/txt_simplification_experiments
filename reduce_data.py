import numpy as np
import pickle
from nltk import pos_tag

with open('bin_data/uniwiki/dict_reduced.bin', 'rb') as f:
    d = pickle.load(f)

normal = np.load('bin_data/uniwiki34/normal.npy')
simple = np.load('bin_data/uniwiki34/simple.npy')



