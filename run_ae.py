from lib.data import DataCollection
from lib.model import AE, ModelParameters, TrainingParameters
import numpy as np
import os

normal = np.load('bin_data/uniwiki34/normal.npy')
normal_w = np.load('bin_data/uniwiki34/w_normal.npy')
simple = np.load('bin_data/uniwiki34/simple.npy')
simple_w = np.load('bin_data/uniwiki34/w_simple.npy')

train_set = (0, int(.8 * normal.shape[0]))
valid_set = (train_set[1] + 1, int(.9 * normal.shape[0]))

model_params = ModelParameters(sequence_length=normal.shape[1],
                               vocabulary_size=max(normal.max(), simple.max()) + 1,
                               embedding_dim=16,
                               hidden_dim=64)
training_params = TrainingParameters('test_new_model_delete_please')
data = DataCollection(normal, simple, normal_w, simple_w)
with AE(training_params=training_params, model_params=model_params) as model:
    model.loop(training_data=data['train'],
               validation_data=data['valid'],
               report_every=400,
               log_every=100,
               continue_callback=lambda m: m.age < 100,
               callback_args=[model])