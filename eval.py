from argparse import ArgumentParser
from lib.data import DataCollection
from lib.model import AE, DiscriminatorModel, ModelParameters, TrainingParameters
import numpy as np
import os

parser = ArgumentParser()
parser.add_argument('directory', type=str, help='training and log dir')
parser.add_argument('model_type', type=str)
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
with model_type(training_params=training_params,
                model_params=model_params,
                clean_environment=True,
                load=True,
                auto_save=False) as model:
    for batch_index, batch in data['test'].batches(batch_size=training_params.batch_size):
        values = model.predict(*batch)
        print(*[np.array(v).shape for v in values], sep=os.linesep)
        break
