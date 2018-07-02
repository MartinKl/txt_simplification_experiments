from functools import partial
import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell, LSTMCell, LSTMStateTuple
from tensorflow.contrib.seq2seq import sequence_loss
import numpy as np
import os

V = 1000
EMB_D = 64
LR = .1
CLIP_NORM = 5.
BS = 1
LEN = 40


x_normal = tf.placeholder(tf.int32, shape=(None, LEN,), name='normal')
x_simple = tf.placeholder(tf.int32, shape=(None, LEN,), name='simple')

initial_encoder_state = [
    (tf.zeros((BS, 96,), dtype=np.float32), tf.zeros((BS, 96,), dtype=np.float32)),
    (tf.zeros((BS, 96,), dtype=np.float32), tf.zeros((BS, 96,), dtype=np.float32))
]

with tf.variable_scope('embedding') as scope:
    embedding_n = tf.contrib.layers.embed_sequence(x_normal, vocab_size=V, embed_dim=EMB_D)
    print('1:', tf.global_variables())
    embedding_s = tf.contrib.layers.embed_sequence(x_simple, vocab_size=V, embed_dim=EMB_D, reuse=True, scope=scope)
    print('2:', tf.global_variables())

with tf.variable_scope('encoder'):
    encoder = MultiRNNCell([LSTMCell(96), LSTMCell(96)])
with tf.variable_scope('encode_n'):
    z_normal, _ = encoder(tf.to_float(x_normal, name='normal_to_float'), initial_encoder_state)
with tf.variable_scope('encode_s'):
    z_simple, _ = encoder(tf.to_float(x_simple, name='simple_to_float'), initial_encoder_state)

initial_decoder_state = [
    LSTMStateTuple(tf.zeros((BS, 96,), dtype=tf.float32), tf.zeros((BS, 96,), dtype=tf.float32)),
    LSTMStateTuple(tf.zeros((BS, 96,), dtype=tf.float32), tf.zeros((BS, 96,), dtype=tf.float32)),
    LSTMStateTuple(tf.zeros((BS, V,), dtype=tf.float32), tf.zeros((BS, V,), dtype=tf.float32))
]

with tf.variable_scope('decode_n') as scope:
    decoder_n = MultiRNNCell([LSTMCell(96), LSTMCell(96), LSTMCell(V)])
    state = initial_decoder_state
    logits_normal = []
    logits, state = decoder_n(z_normal, state)
    logits_normal.append(logits)
with tf.variable_scope(scope, reuse=True):
    for _ in range(1, LEN):
        logits, state = decoder_n(z_normal, state)
        logits_normal.append(logits)
    err_r_normal = sequence_loss(logits=tf.convert_to_tensor(logits_normal),
                                 targets=x_normal,
                                 weights=tf.ones_like(x_normal, dtype=tf.float32))

with tf.variable_scope('decode_s') as scope:
    decoder_s = MultiRNNCell([LSTMCell(96), LSTMCell(96), LSTMCell(V)])
    state = initial_decoder_state
    logits_simple = []
    logits, state = decoder_s(z_simple, state)
    logits_simple.append(logits)
with tf.variable_scope(scope, reuse=True):
    for _ in range(1, LEN):
        logits, state = decoder_s(z_simple, state)
        logits_simple.append(logits)
    err_r_simple = sequence_loss(logits=tf.convert_to_tensor(logits_simple),
                                 targets=x_simple,
                                 weights=tf.ones_like(x_simple, dtype=tf.float32))

err_z = tf.reduce_sum(tf.squared_difference(z_simple, z_normal), name='repr_error')
err = err_r_normal + err_r_simple + err_z

params = tf.trainable_variables()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=LR)
gradients = tf.gradients(err, params)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=CLIP_NORM)
update = optimizer.apply_gradients(grads_and_vars=zip(clipped_gradients, params))
