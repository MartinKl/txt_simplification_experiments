from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import os

AE_DIR = 'ae_eval_out'
DSC_DIR = 'd_eval_out'
PRED_N_NAME = 'normal_logits'


# evaluate AE-model results:
