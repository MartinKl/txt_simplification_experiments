import numpy as np
import os
import pickle

source_dir = 'bin_data/wiki/'
target_dir = 'bin_data/uniwiki/'

with open(os.path.join(source_dir, 'normal_dict.bin'), 'rb') as f:
    normal_d = pickle.load(f)
with open(os.path.join(source_dir, 'simple_dict.bin'), 'rb') as f:
    simple_d = pickle.load(f)
data = np.load(os.path.join(source_dir, 'paired_array_normal_simple.npy'))

common_d =