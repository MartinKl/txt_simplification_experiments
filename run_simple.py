from lib.model import AE, ModelParameters, TrainingParameters
import numpy as np
import os

normal = np.load('bin_data/uniwiki34/normal.npy')#.clip(max=v-1)
normal_w = np.load('bin_data/uniwiki34/w_normal.npy')
simple = np.load('bin_data/uniwiki34/simple.npy')#.clip(max=v-1)
simple_w = np.load('bin_data/uniwiki34/w_simple.npy')

train_set = (0, int(.8 * normal.shape[0]))
valid_set = (train_set[1] + 1, int(.9 * normal.shape[0]))

model_params = ModelParameters(34, 8, 8, 16)
training_params = TrainingParameters('test_new_model_delete_please')

with AE(model_params, training_params) as model:
    print('huhu')
