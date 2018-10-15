from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import os
import pickle

AE_DIR = 'ae_eval_out'
DSC_DIR = 'd_eval_out'
NORMAL_NAME = 'normal.npy'
SIMPLE_NAME = 'simple.npy'

normal_data = np.load('bin_data/uniwiki_minimal/normal.npy')
simple_data = np.load('bin_data/uniwiki_minimal/simple.npy')
with open('bin_data/uniwiki/dict_minimal.bin', 'rb') as f:
    w_dict = {v: k for k, v in pickle.load(f).items()}

# evaluate AE-model results:
## evaluate predictions with bleu score
normal = np.load(NORMAL_NAME)
simple = np.load(SIMPLE_NAME)

print(normal.shape)