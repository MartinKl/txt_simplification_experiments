from bokeh.charts import HeatMap, output_file, Line
from bokeh.plotting import show, figure
from lib.data import DataCollection
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import os
import pickle

AE_DIR = 'ae_eval_out'
DSC_DIR = 'd_eval_out'
NORMAL_NAME = 'normal.npy'
SIMPLE_NAME = 'simple.npy'

data = DataCollection(np.load('bin_data/uniwiki_minimal/normal.npy'),
                      np.load('bin_data/uniwiki_minimal/simple.npy'),
                      np.load('bin_data/uniwiki_minimal/w_normal.npy'),
                      np.load('bin_data/uniwiki_minimal/w_simple.npy'))
with open('bin_data/uniwiki/dict_minimal.bin', 'rb') as f:
    w_dict = {v: k for k, v in pickle.load(f).items()}
w_dict[1] = ''  # replace eos with nothing, makes the life a little easier
ix_to_human_readable = {}
for index, batch in data['test'].batches(batch_size=1):
    normal_target, simple_target, _, _ = batch.reshape(4, 34)
    ix_to_human_readable[index] = ([w_dict[ix] for ix in normal_target if w_dict[ix]],
                                   [w_dict[ix] for ix in simple_target if w_dict[ix]])
# evaluate D-model results:
## evaluate discriminator
d_outs_0, d_outs_1 = np.load(os.path.join(DSC_DIR, 'd_outs.npy')).reshape(2, 134 * 32)
output_file('discriminator_out.html')
x = np.arange(d_outs_0.shape[0])
p = figure()
p.line(x[:2000], d_outs_0[:2000], color='blue')
p.line(x[2000:], d_outs_1[2000:], color='red')
#show(p)

## evaluate z
z_normal = np.load(os.path.join(DSC_DIR, 'z_' + NORMAL_NAME))[0].reshape(134 * 32, 64)
z_simple = np.load(os.path.join(DSC_DIR, 'z_' + SIMPLE_NAME))[0].reshape(134 * 32, 64)
print(z_normal.flatten())

output_file('heat_map_dsc.html')
sq_diff = ((z_normal - z_simple) ** 2).flatten()
y = np.array([list(range(64))] * 4288).flatten()
x = np.array([[i] * 64 for i in range(4288)]).flatten()
show(HeatMap({'values': sq_diff, 'z[i]': x, 'example': y},
             x='z[i]',
             y='example',
             values='values',
             stat=None,
             title='squared difference of z along the test set (implicit model)'))
output_file('normal_z_heat_map_dsc.html')
show(HeatMap({'values': z_normal.flatten(), 'z[i]': x, 'example': y},
             x='z[i]',
             y='example',
             values='values',
             stat=None,
             title='normal z-values along the test set (implicit model)'))
output_file('simple_z_heat_map_dsc.html')
show(HeatMap({'values': z_simple.flatten(), 'z[i]': x, 'example': y},
             x='z[i]',
             y='example',
             values='values',
             stat=None,
             title='simple z-values along the test set (implicit model)'))


## evaluate predictions with bleu score
normal = np.load(os.path.join(DSC_DIR, NORMAL_NAME))[0].swapaxes(1, 2).reshape(134 * 32, 34)
simple = np.load(os.path.join(DSC_DIR, SIMPLE_NAME))[0].swapaxes(1, 2).reshape(134 * 32, 34)

transfer_scores = []
for index in range(len(ix_to_human_readable)):
    n_ix_seq = normal[index]
    eos_ix = np.where(n_ix_seq == 1)[0]
    read_until = eos_ix[0] if np.any(eos_ix) else 34
    n_w_seq = [w_dict[ix] for ix in n_ix_seq[:read_until]]

    s_ix_seq = simple[index]
    eos_ix = np.where(s_ix_seq == 1)[0]
    read_until = eos_ix[0] if np.any(eos_ix) else 34
    s_w_seq = [w_dict[ix] for ix in s_ix_seq[:read_until]]

    normal_seq, simple_seq = ix_to_human_readable[index]
    simplification_score = sentence_bleu([normal_seq, simple_seq], s_w_seq)
    reversed_score = sentence_bleu([simple_seq, normal_seq], n_w_seq)
    transfer_scores.append((simplification_score, reversed_score))
mean_score_simplification, mean_score_reverse = np.array(transfer_scores).mean(axis=0)
print('DSC - mean bleu score simplification:', mean_score_simplification)
print('DSC - mean bleu score reverse transfer:', mean_score_reverse)

# evaluate AE-model results:
## evaluate z
z_normal = np.load(os.path.join(AE_DIR, 'z_' + NORMAL_NAME)).reshape(134 * 32, 64)
z_simple = np.load(os.path.join(AE_DIR, 'z_' + SIMPLE_NAME)).reshape(134 * 32, 64)

output_file('heat_map_ae.html')
sq_diff = ((z_normal - z_simple) ** 2).flatten()
show(HeatMap({'values': sq_diff, 'z[i]': x, 'example': y},
             x='z[i]',
             y='example',
             values='values',
             stat=None,
             title='squared difference of z along the test set (explicit model)'))
output_file('normal_z_heat_map_ae.html')
show(HeatMap({'values': z_normal.flatten(), 'z[i]': x, 'example': y},
             x='z[i]',
             y='example',
             values='values',
             stat=None,
             title='normal z-values along the test set (explicit model)'))
output_file('simple_z_heat_map_ae.html')
show(HeatMap({'values': z_simple.flatten(), 'z[i]': x, 'example': y},
             x='z[i]',
             y='example',
             values='values',
             stat=None,
             title='simple z-values along the test set (explicit model)'))

## evaluate predictions with bleu score
normal = np.load(os.path.join(AE_DIR, NORMAL_NAME)).swapaxes(2, 3).reshape(134 * 32, 34)
simple = np.load(os.path.join(AE_DIR, SIMPLE_NAME)).swapaxes(2, 3).reshape(134 * 32, 34)

transfer_scores = []
for index in range(len(ix_to_human_readable)):
    n_ix_seq = normal[index]
    eos_ix = np.where(n_ix_seq == 1)[0]
    read_until = eos_ix[0] if eos_ix else 34
    n_w_seq = [w_dict[ix] for ix in n_ix_seq[:read_until]]

    s_ix_seq = simple[index]
    eos_ix = np.where(s_ix_seq == 1)[0]
    read_until = eos_ix[0] if eos_ix else 34
    s_w_seq = [w_dict[ix] for ix in s_ix_seq[:read_until]]

    normal_seq, simple_seq = ix_to_human_readable[index]
    simplification_score = sentence_bleu([normal_seq, simple_seq], s_w_seq)
    reversed_score = sentence_bleu([simple_seq, normal_seq], n_w_seq)
    transfer_scores.append((simplification_score, reversed_score))
mean_score_simplification, mean_score_reverse = np.array(transfer_scores).mean(axis=0)
print('mean bleu score simplification:', mean_score_simplification)
print('mean bleu score reverse transfer:', mean_score_reverse)
