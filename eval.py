from argparse import ArgumentParser
from lib.data import DataCollection
from lib.model import AE, DiscriminatorModel, ModelParameters, TrainingParameters
import numpy as np
import os
import pickle

parser = ArgumentParser()
parser.add_argument('directory', type=str, help='training and log dir')
parser.add_argument('model_type', type=str)
parser.add_argument('out_dir', type=str)
args = parser.parse_args()

normal = np.load('bin_data/uniwiki_minimal/normal.npy')
normal_w = np.load('bin_data/uniwiki_minimal/w_normal.npy')
simple = np.load('bin_data/uniwiki_minimal/simple.npy')
simple_w = np.load('bin_data/uniwiki_minimal/w_simple.npy')

train_set = (0, int(.8 * normal.shape[0]))
valid_set = (train_set[1] + 1, int(.9 * normal.shape[0]))

model_params = ModelParameters(sequence_length=normal.shape[1],
                               vocabulary_size=max(normal.max(), simple.max()) + 1,
                               embedding_dim=16,
                               hidden_dim=64)
training_params = TrainingParameters(os.path.abspath(args.directory))
print(training_params)
data = DataCollection(normal, simple, normal_w, simple_w)
model_type = eval(args.model_type)
print('Model type:', model_type.__name__)
predictions_normal = ([], [])
predictions_simple = ([], [])
z_normal = ([], [])
z_simple = ([], [])
d_vals = ([], [])
run_again = False
kwargs = {}
if model_type == DiscriminatorModel:
    run_again = True
with model_type(training_params=training_params,
                model_params=model_params,
                clean_environment=True,
                load=True,
                auto_save=False) as model:
    for r in range(1 + run_again):
        for batch_index, batch in data['test'].batches(batch_size=training_params.batch_size):
            if not batch_index % 50:
                print('Current batch index:', batch_index)
            if batch.shape[1] != 32:
                print('end!')
                break
            values = [np.array(v) for v in model.predict(*batch, **kwargs)]
            predictions_normal[r].append(values[0])
            predictions_simple[r].append(values[1])
            z_normal[r].append(values[2])
            z_simple[r].append(values[3])
            if run_again:
                d_vals[r].append(values[4])
        if run_again:
            print('running again with theta=1')
            kwargs = dict(tc=1.)

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)
np.save(os.path.join(args.out_dir, 'normal_logits.npy'), np.array(predictions_normal))
np.save(os.path.join(args.out_dir, 'simple_logits.npy'), np.array(predictions_simple))
np.save(os.path.join(args.out_dir, 'z_normal.npy'), np.array(z_normal))
np.save(os.path.join(args.out_dir, 'z_simple.npy'), np.array(z_simple))
np.save(os.path.join(args.out_dir, 'd_outs.npy'), np.array(d_outs))
