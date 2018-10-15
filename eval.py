from argparse import ArgumentParser
from lib.data import DataCollection
from lib.model import AE, SimpleAE, ModelParameters, TrainingParameters
import numpy as np
import os

parser = ArgumentParser()
parser.add_argument('directory', type=str, help='training and log dir')
parser.add_argument('--lr', type=float, default=.1, help='learning rate')
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--report', type=int, default=100)
parser.add_argument('--log', type=int, default=50)
parser.add_argument('--epochs', type=int, default=200)
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
training_params = TrainingParameters(os.path.abspath(args.directory),
                                     batch_size=args.bs,
                                     learning_rate=args.lr)
print(training_params)
data = DataCollection(normal, simple, normal_w, simple_w)
with AE(training_params=training_params, model_params=model_params, load=True, auto_save=False) as model:
    for batch_index, batch in data['test'].batches(batch_size=training_params.batch_size):
        values = model.predict(*batch)
        print(values)
        break
