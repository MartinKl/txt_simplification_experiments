from collections import defaultdict
from functools import partial
import os
import pickle
import time

DATA_PATH = 'data/wiki/'
DATA_OUT_DIR = 'bin_data/wiki/'
if not os.path.exists(DATA_OUT_DIR):
    os.mkdir(DATA_OUT_DIR)
SEP = '\t'

data = defaultdict(partial(defaultdict, dict))
vocabs = {}
for f_name in os.listdir(DATA_PATH):
    path = os.path.join(DATA_PATH, f_name)
    with open(path) as f:
        lines = f.readlines()
    name = f_name.split('.')[0]
    vocabs[name] = set()
    for line in lines:
        article_id, paragraph_id, text = line.strip().split(SEP)
        data[article_id][paragraph_id][name] = text
        vocabs[name].update(set(text.strip().split(' ')))

timestamp = time.time()
with open(os.path.join(DATA_OUT_DIR, 'data-{timestamp}.pkl'.format(timestamp=timestamp)), 'wb') as f:
    pickle.dump(data, f)

with open(os.path.join(DATA_OUT_DIR, 'vocabs-{timestamp}.pkl'.format(timestamp=timestamp)), 'wb') as f:
    pickle.dump(vocabs, f)
