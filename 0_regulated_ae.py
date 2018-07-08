import gc
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d
from tensorflow.contrib.rnn import MultiRNNCell, LSTMCell, LSTMStateTuple
from tensorflow.contrib.seq2seq import sequence_loss
import numpy as np
import os
import pickle

with open('bin_data/uniwiki/dict_reduced.bin', 'rb') as f:
    vocab = pickle.load(f)
eos_ix = len(vocab)
v = len(vocab) + 1
EMB_D = 16
D = 8
LR = .1
CLIP_NORM = 5.
BS = 1
LEN = 34

# model definition
print('building model ...')
x_normal = tf.placeholder(tf.int32, shape=(None, LEN,), name='normal')
w_normal = tf.placeholder(tf.float32, shape=(None, LEN,), name='nweights')
x_simple = tf.placeholder(tf.int32, shape=(None, LEN,), name='simple')
w_simple = tf.placeholder(tf.float32, shape=(None, LEN,), name='sweights')

initial_encoder_state = [
    LSTMStateTuple(tf.zeros((BS, D,), dtype=tf.float32), tf.zeros((BS, D,), dtype=tf.float32)),
    LSTMStateTuple(tf.zeros((BS, D,), dtype=tf.float32), tf.zeros((BS, D,), dtype=tf.float32))
]

with tf.variable_scope('embedding') as scope:
    embedded_n = tf.contrib.layers.embed_sequence(x_normal, vocab_size=v, embed_dim=EMB_D, scope=scope)
    embedded_s = tf.contrib.layers.embed_sequence(x_simple, reuse=True, scope=scope)

with tf.variable_scope('encoder'):
    encoder = MultiRNNCell([LSTMCell(D), LSTMCell(D)])

with tf.variable_scope('encode_n') as scope:
    z_normal_arr = []
    z, state = encoder(embedded_n[:, 0], initial_encoder_state)
    z_normal_arr.append(z)
with tf.variable_scope(scope, reuse=True):
    for i in range(1, LEN):
        z, state = encoder(embedded_n[:, i], state)
        z_normal_arr.append(z)

with tf.variable_scope('encode_s') as scope:
    z_simple_arr = []
    z, state = encoder(embedded_s[:, 0], initial_encoder_state)
    z_simple_arr.append(z)
with tf.variable_scope(scope, reuse=True):
    for i in range(1, LEN):
        z, state = encoder(embedded_s[:, i], state)
        z_simple_arr.append(z)

z_n = tf.transpose(tf.convert_to_tensor(z_normal_arr))
z_s = tf.transpose(tf.convert_to_tensor(z_simple_arr))
with tf.variable_scope('reduction_layer/0') as scope:
    z_normal = conv2d(z_n, LEN // 2, (32,))
    z_simple = conv2d(z_s, LEN // 2, (32,), reuse=True, scope=scope)
with tf.variable_scope('reduction_layer/1') as scope:
    z_normal = conv2d(z_normal, LEN // 4, (32,))
    z_simple = conv2d(z_simple, LEN // 4, (32,), reuse=True, scope=scope)
with tf.variable_scope('reduction_layer/pool') as scope:
    z_normal = max_pool2d(tf.transpose(z_normal[None], (0, 1, 3, 2)), kernel_size=(1, LEN // 4), stride=1)
    z_simple = max_pool2d(tf.transpose(z_simple[None], (0, 1, 3, 2)), kernel_size=(1, LEN // 4), stride=1)
with tf.variable_scope('reduction/reshape'):
    z_normal = tf.reshape(z_normal, (BS, D))
    z_simple = tf.reshape(z_simple, (BS, D))

initial_decoder_state = [
    LSTMStateTuple(tf.zeros((BS, D,), dtype=tf.float32), tf.zeros((BS, D,), dtype=tf.float32)),
    LSTMStateTuple(tf.zeros((BS, D,), dtype=tf.float32), tf.zeros((BS, D,), dtype=tf.float32)),
    LSTMStateTuple(tf.zeros((BS, v,), dtype=tf.float32), tf.zeros((BS, v,), dtype=tf.float32))
]

with tf.variable_scope('decode_n') as scope:
    decoder_n = MultiRNNCell([LSTMCell(D), LSTMCell(D), LSTMCell(v)])
    state = initial_decoder_state
    logits_normal = []
    logits, state = decoder_n(z_normal, state)
    logits_normal.append(logits)
with tf.variable_scope(scope, reuse=True):
    for _ in range(1, LEN):
        logits, state = decoder_n(z_normal, state)
        logits_normal.append(logits)
    err_r_normal = tf.reduce_sum(sequence_loss(tf.transpose(tf.convert_to_tensor(logits_normal), (1, 0, 2)),
                                               x_normal,
                                               w_normal))

with tf.variable_scope('decode_s') as scope:
    decoder_s = MultiRNNCell([LSTMCell(D), LSTMCell(D), LSTMCell(v)])
    state = initial_decoder_state
    logits_simple = []
    logits, state = decoder_s(z_simple, state)
    logits_simple.append(logits)
with tf.variable_scope(scope, reuse=True):
    for _ in range(1, LEN):
        logits, state = decoder_s(z_simple, state)
        logits_simple.append(logits)
    err_r_simple = tf.reduce_sum(sequence_loss(tf.transpose(tf.convert_to_tensor(logits_simple), (1, 0, 2)),
                                               x_simple,
                                               w_simple))

err_z = tf.reduce_sum(tf.squared_difference(z_simple, z_normal), name='repr_error')
err = err_z + err_r_normal + err_r_simple

params = tf.trainable_variables()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LR)
gradients = tf.gradients(err, params)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=CLIP_NORM)
update = optimizer.apply_gradients(grads_and_vars=zip(clipped_gradients, params))

# load data
print('loading data ...')
normal = np.load('bin_data/uniwiki34/normal.npy')
normal_w = np.load('bin_data/uniwiki34/w_normal.npy')
simple = np.load('bin_data/uniwiki34/simple.npy')
simple_w = np.load('bin_data/uniwiki34/w_simple.npy')

# training
with tf.Session() as session:
    print('Inititalizing ...')
    session.run(tf.global_variables_initializer())
    print('Starting training ...')
    epoch = 0
    while True:
        epoch += 1
        print('Starting epoch', epoch, '...')
        for batch_index in range(0, normal.shape[0], BS):
            if not batch_index / BS % 5:
                print('Feeding batch', 1 + batch_index // BS)
            result = session.run(update, feed_dict={
                'normal:0': normal[batch_index:batch_index + BS],
                'simple:0': simple[batch_index:batch_index + BS],
                'nweights:0': normal_w[batch_index:batch_index + BS],
                'sweights:0': simple_w[batch_index:batch_index + BS]
            })
