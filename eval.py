from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import os

AE_DIR = 'ae_eval_out'
DSC_DIR = 'd_eval_out'
NORMAL_NAME = 'normal.npy'
SIMPLE_NAME = 'simple.npy'

# evaluate AE-model results:
## evaluate predictions with bleu score
normal_pred