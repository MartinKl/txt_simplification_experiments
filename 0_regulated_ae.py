from functools import partial
import tensorflow as tf
from tensorflow.contrib.rnn import MultiRNNCell, LSTMCell, LSTMStateTuple
from tensorflow.contrib.seq2seq import sequence_loss
import numpy as np
import os

V = 1000
EMB_D = 64
D = 96
LR = .1
CLIP_NORM = 5.
BS = 1
LEN = 40


x_normal = tf.placeholder(tf.int32, shape=(None, LEN,), name='normal')
x_simple = tf.placeholder(tf.int32, shape=(None, LEN,), name='simple')

initial_encoder_state = [
    LSTMStateTuple(tf.zeros((BS, D,), dtype=tf.float32), tf.zeros((BS, D,), dtype=tf.float32)),
    LSTMStateTuple(tf.zeros((BS, D,), dtype=tf.float32), tf.zeros((BS, D,), dtype=tf.float32))
]

with tf.variable_scope('embedding') as scope:
    embedded_n = tf.contrib.layers.embed_sequence(x_normal, vocab_size=V, embed_dim=EMB_D, scope=scope)
    embedded_s = tf.contrib.layers.embed_sequence(x_simple, reuse=True, scope=scope)

with tf.variable_scope('encoder'):
    encoder = MultiRNNCell([LSTMCell(D), LSTMCell(D)])

with tf.variable_scope('encode_n') as scope:
    z_normal_arr = tf.TensorArray(dtype=tf.float32, size=LEN)
    z, state = encoder(embedded_n[:, 0], initial_encoder_state)
    z_normal_arr.write(0, z)
with tf.variable_scope(scope, reuse=True):
    for i in range(1, LEN):
        z, state = encoder(embedded_n[:, i], state)
        z_normal_arr.write(i, z)

with tf.variable_scope('encode_s') as scope:
    z_simple_arr = tf.TensorArray(dtype=tf.float32, size=LEN)
    z, state = encoder(embedded_s[:, 0], initial_encoder_state)
    z_simple_arr.write(0, z)
with tf.variable_scope(scope, reuse=True):
    for i in range(1, LEN):
        z, state = encoder(embedded_s[:, i], state)
        z_simple_arr.write(i, z)

exit()

initial_decoder_state = [
    LSTMStateTuple(tf.zeros((BS, D,), dtype=tf.float32), tf.zeros((BS, D,), dtype=tf.float32)),
    LSTMStateTuple(tf.zeros((BS, D,), dtype=tf.float32), tf.zeros((BS, D,), dtype=tf.float32)),
    LSTMStateTuple(tf.zeros((BS, V,), dtype=tf.float32), tf.zeros((BS, V,), dtype=tf.float32))
]

with tf.variable_scope('decode_n') as scope:
    decoder_n = MultiRNNCell([LSTMCell(D), LSTMCell(D), LSTMCell(V)])
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
    decoder_s = MultiRNNCell([LSTMCell(D), LSTMCell(D), LSTMCell(V)])
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
