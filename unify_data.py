from collections import Counter, defaultdict
from nltk import pos_tag
from itertools import chain
import numpy as np
import os
import pickle

SOURCE_DIR = 'bin_data/wiki/'
TARGET_DIR = 'bin_data/uniwiki/'
SIMPLE = 'simple'
NORMAL = 'normal'
# only have sentences with a length <= 33 (sentence length frequency >= 1000, still enough data afterwards)
# vocabulary in reduced form only sticks to closed-class words
LEN_THRESHOLD = 33
W_F_THRESHOLD = 100

with open(os.path.join(SOURCE_DIR, 'normal_dict.pkl'), 'rb') as f:
    normal_d = pickle.load(f)
with open(os.path.join(SOURCE_DIR, 'simple_dict.pkl'), 'rb') as f:
    simple_d = pickle.load(f)
normal, simple = np.load(os.path.join(SOURCE_DIR, 'paired_array_normal_simple.npy'))

common_d = {'': 0}
common_d.update({w: i + 1 for i, w in enumerate(set(normal_d.values()).intersection(set(simple_d.values())))})

indices = [i for i, paragraph in enumerate(normal)
           if len(paragraph) <= LEN_THRESHOLD and len(simple[i]) <= LEN_THRESHOLD]
data = {SIMPLE: [], NORMAL: []}
data_reduced = {SIMPLE: [], NORMAL: []}

# standard data set
for i in indices:
    par_n = [normal_d[ix] for ix in normal[i]]
    par_s = [simple_d[ix] for ix in simple[i]]
    n_standard = []
    s_standard = []
    for w in par_n:
        if w not in common_d:
            common_d[w] = len(common_d)
        n_standard.append(common_d[w])
    for w in par_s:
        if w not in common_d:
            common_d[w] = len(common_d)
        s_standard.append(common_d[w])
    data[NORMAL].append(n_standard)
    data[SIMPLE].append(s_standard)
if not os.path.exists(TARGET_DIR):
    os.mkdir(TARGET_DIR)
with open(os.path.join(TARGET_DIR, 'dict.bin'), 'wb') as f:
    pickle.dump(common_d, f)
with open(os.path.join(TARGET_DIR, 'data.bin'), 'wb') as f:
    pickle.dump(data, f)
print('Done with data, starting reduction ...')
# reduced data set
decrypter = {v: k for k, v in common_d.items()}
all_w = (decrypter[ix] for ix in chain(*data[NORMAL], *data[SIMPLE]))
freq_data = Counter(all_w)
tagset_info = defaultdict(int)
with open('data/tagset.tsv') as f:
    tagset_info_raw = f.readlines()
for line in tagset_info_raw:
    l = line.strip().split('\t')
    tag = l[0].strip()
    decision = l[2].strip()
    tagset_info[tag] = int(decision)
vocab = {'': 0}
for par_n, par_s in zip(data[NORMAL], data[SIMPLE]):
    readable_n = [decrypter[ix] for ix in par_n]
    readable_s = [decrypter[ix] for ix in par_s]
    pos_n = pos_tag(readable_n)
    pos_s = pos_tag(readable_s)
    reduced = [[], []]
    for i in range(2):
        for w, tag in (pos_n, pos_s)[i]:
            if tagset_info[tag] or freq_data[w] >= W_F_THRESHOLD:
                representation = w
            else:
                representation = tag
            if representation not in vocab:
                vocab[representation] = len(vocab)
            reduced[i].append(vocab[representation])
    data_reduced[NORMAL].append(reduced[0])
    data_reduced[SIMPLE].append(reduced[1])
print(len(vocab))
with open(os.path.join(TARGET_DIR, 'dict_reduced.bin'), 'wb') as f:
    pickle.dump(vocab, f)
with open(os.path.join(TARGET_DIR, 'data_reduced.bin'), 'wb') as f:
    pickle.dump(data_reduced, f)
