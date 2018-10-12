from lib.data import DataCollection
from lib.model import AE, ModelParameters, TrainingParameters
import logging
import numpy as np
import os
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

V = 5
TEST_DIR = '_test'
if not os.path.exists(TEST_DIR):
    os.mkdir(TEST_DIR)
else:
    logger.warn('test directory already exists!')

normal = np.load('bin_data/uniwiki34/normal.npy').clip(max=V-1)
normal_w = np.load('bin_data/uniwiki34/w_normal.npy')
simple = np.load('bin_data/uniwiki34/simple.npy').clip(max=V-1)
simple_w = np.load('bin_data/uniwiki34/w_simple.npy')

data = DataCollection(normal, simple, normal_w, simple_w)

train_set = (0, int(.8 * normal.shape[0]))
valid_set = (train_set[1] + 1, int(.9 * normal.shape[0]))

model_params = ModelParameters(34, V, 8, 16)
training_params = TrainingParameters(os.path.join(TEST_DIR, 'test'))
test = 'TRAIN'
try:
    logger.info('Training test ...')
    with AE(model_params, training_params) as model:
        model.loop(data['train'], data['valid'], steps=1, continue_callback=lambda: False)
        logger.info('Training and validation test successful!')
        model.save()
        logger.info('Saving successful!')
    test = 'LOAD'
    logger.info('Loading test ...')
    with AE(model_params, training_params) as model:
        model.load(TEST_DIR)
        logger.info('Loading successful!')
    logger.info('All tests successful!')
except Exception as e:
    logger.error('{} occured in test {}'.format(type(e), test))
finally:
    shutil.rmtree(TEST_DIR)
