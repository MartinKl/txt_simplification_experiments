import numpy as np
import os
import pickle

TARGET_DIR = 'bin_data/uniwiki_minimal/'
if not os.path.exists(TARGET_DIR):
    os.mkdir(TARGET_DIR)

with open('bin_data/uniwiki/data_minimal.bin', 'rb') as f:
    data = pickle.load(f)

max_len = 34

data_normal = []
weights_normal = []
data_simple = []
weights_simple = []
for normal_p, simple_p in zip(data['normal'], data['simple']):
    data_normal.append(np.pad(normal_p, pad_width=(0, max_len - len(normal_p)), mode='constant', constant_values=(0,)))
    data_simple.append(np.pad(simple_p, pad_width=(0, max_len - len(simple_p)), mode='constant', constant_values=(0,)))
    weights_normal.append(np.pad(np.ones(len(normal_p), dtype=np.int32),
                                 pad_width=(0, max_len - len(normal_p)),
                                 mode='constant',
                                 constant_values=(0,)))
    weights_simple.append(np.pad(np.ones(len(simple_p), dtype=np.int32),
                                 pad_width=(0, max_len - len(simple_p)),
                                 mode='constant',
                                 constant_values=(0,)))

a_normal = np.array(data_normal).astype(np.int32)
a_simple = np.array(data_simple).astype(np.int32)
w_normal = np.array(weights_normal).astype(np.int32)
w_simple = np.array(weights_simple).astype(np.int32)
np.save(os.path.join(TARGET_DIR, 'normal.npy'), a_normal)
np.save(os.path.join(TARGET_DIR, 'simple.npy'), a_simple)
np.save(os.path.join(TARGET_DIR, 'w_normal.npy'), w_normal)
np.save(os.path.join(TARGET_DIR, 'w_simple.npy'), w_simple)

print(a_normal.shape, a_normal.dtype, a_normal[0])
print(w_normal.shape, w_normal.dtype, w_normal[0])
print(a_simple.shape, a_simple.dtype, a_simple[0])
print(w_simple.shape, w_simple.dtype, w_simple[0])
