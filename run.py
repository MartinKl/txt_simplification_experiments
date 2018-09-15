from argparse import ArgumentParser
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, fully_connected, max_pool2d, softmax, stack
from tensorflow.contrib.rnn import MultiRNNCell, LSTMCell, LSTMStateTuple
from tensorflow.contrib.seq2seq import sequence_loss
import numpy as np
import os
import pickle

AE = 'ae'
VAE = 'vae'
DSC1 = 'd'
DSC2 = '2d'
OPTIONS = (AE, VAE, DSC1, DSC2)

parser = ArgumentParser()
parser.add_argument('restriction', choices=OPTIONS)
args = parser.parse_args()
r = args.restriction

EMB_D = 16
D = 64
LR = .1
CLIP_NORM = 5.
BS = 1
LEN = 34
PRE_TRAIN_ADVANTAGE = 600

with open('bin_data/uniwiki/dict_reduced.bin', 'rb') as f:
    vocab = pickle.load(f)
eos_ix = len(vocab)
v = len(vocab) + 1
## DEBUG
#v = 10
## END OF DEBUG

# model definition
print('building model ...')
x_normal = tf.placeholder(tf.int32, shape=(None, LEN,), name='normal')
w_normal = tf.placeholder(tf.float32, shape=(None, LEN,), name='nweights')
x_simple = tf.placeholder(tf.int32, shape=(None, LEN,), name='simple')
w_simple = tf.placeholder(tf.float32, shape=(None, LEN,), name='sweights')
global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')

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
with tf.variable_scope('reduction/0') as scope:
    z_normal = conv2d(z_n, LEN // 2, (32,))
    z_simple = conv2d(z_s, LEN // 2, (32,), reuse=True, scope=scope)
with tf.variable_scope('reduction/1') as scope:
    z_normal = conv2d(z_normal, LEN // 4, (32,))
    z_simple = conv2d(z_simple, LEN // 4, (32,), reuse=True, scope=scope)
if r == VAE:
    with tf.variable_scope('reduction/split'):
        raise NotImplementedError
else:
    with tf.variable_scope('reduction/pool'):
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

idle = tf.no_op('idle')
update = idle
d_update = idle
d_loss = idle
g_update = idle
g_loss = idle
r_update = idle
r_loss = idle
losses = []
loss_labels = []
print('training loss ...')
if r == AE:
    print('.building direct representation loss ...')
    err_z = tf.reduce_sum(tf.squared_difference(z_simple, z_normal), name='repr_error')
    err_r = tf.add(err_r_normal, err_r_simple, name='rec_err')
    err = tf.add(err_z, err_r, name='err')
    params = tf.trainable_variables()
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LR)
    gradients = tf.gradients(err, params)
    clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=CLIP_NORM)
    update = optimizer.apply_gradients(grads_and_vars=zip(clipped_gradients, params), global_step=global_step)
    losses.extend([err, err_r, err_z])
    loss_labels.extend(['Overall error', 'reconstruction error', 'representation error'])
elif r == VAE:
    print('.building vae loss ...')
    err = tf.constant(0)
    raise NotImplementedError
elif r == DSC1:
    d_dims = [D // (2 ** e) for e in range(1, int(np.log2(D // 2)))] + [1]
    print('.building single discriminator loss with dims {dims} ...'.format(dims=tuple(d_dims)))
    with tf.variable_scope('discriminator') as scope:
        # the discriminator's output probability is chosen as probability that the input was a normal, i. e. complex
        # expression. Thus, the probability can also be interpreted as complexity score with higher values expressing
        # a higher level of complexity
        normal_is_normal = tf.sigmoid(stack(z_normal,
                                         layer=fully_connected,
                                         stack_args=d_dims,
                                         scope=scope.name))
        simple_is_normal = tf.sigmoid(stack(z_simple,
                                         layer=fully_connected,
                                         stack_args=d_dims,
                                         scope=scope.name,
                                         reuse=True))
        confusion_normal = 1. - normal_is_normal
        confusion_simple = simple_is_normal
        d_loss = confusion_normal #+ confusion_simple
        g_loss = 2. - confusion_normal - confusion_simple
        # optimize discriminator
        discriminator_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
        optimizer = tf.train.GradientDescentOptimizer(LR)
        d_gradients = tf.gradients(d_loss, discriminator_params)
        d_clipped_gradients, _ = tf.clip_by_global_norm(d_gradients, clip_norm=CLIP_NORM)
        d_update = optimizer.apply_gradients(grads_and_vars=zip(d_clipped_gradients, discriminator_params))
        # optimize encoder network
        g_params = list(set(tf.global_variables()).difference(discriminator_params))
        g_gradients = tf.gradients(g_loss, g_params)
        g_clipped_gradients, _ = tf.clip_by_global_norm(g_gradients, clip_norm=CLIP_NORM)
        g_update = optimizer.apply_gradients(grads_and_vars=zip(g_clipped_gradients, g_params))
        # optimize reconstruction function as well
        r_loss = err_r_simple + err_r_normal
        r_params = tf.global_variables()
        r_gradients = tf.gradients(r_loss, r_params)
        r_clipped_gradients, _ = tf.clip_by_global_norm(r_gradients, clip_norm=CLIP_NORM)
        r_update = optimizer.apply_gradients(grads_and_vars=zip(r_clipped_gradients, r_params))
        losses.extend([r_loss,
                       g_loss,
                       d_loss])
        loss_labels.extend(['Reconstruction error',
                            'Generator loss',
                            'Discriminator loss'])
elif r == DSC2:
    print('.building dual discriminator loss ...')
    with tf.variable_scope('discriminators/n'):
        pass
    with tf.variable_scope('discriminators/s'):
        pass
    err = tf.constant(0)
    raise NotImplementedError

# load data
print('loading data ...')
normal = np.load('bin_data/uniwiki34/normal.npy').clip(max=v-1)
normal_w = np.load('bin_data/uniwiki34/w_normal.npy')
simple = np.load('bin_data/uniwiki34/simple.npy').clip(max=v-1)
simple_w = np.load('bin_data/uniwiki34/w_simple.npy')

train_set = (0, int(.8 * normal.shape[0]))
valid_set = (train_set[1] + 1, int(.9 * normal.shape[0]))

# training
config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
with tf.Session(config=config) as session:
    print('Inititalizing ...')
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list=tf.global_variables())
    print('Starting training ...')
    train_dir = '_'.join(('train', r))
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    epoch = 0
    if r == DSC1:
        # discriminator pre-training
        print('Pretraining encoding and reconstruction')
        for index in range(PRE_TRAIN_ADVANTAGE):
            r_loss_val = session.run([r_loss, r_update], feed_dict={'normal:0': normal[index:index + 1],
                                               'simple:0': simple[index:index + 1],
                                               'nweights:0': normal_w[index:index + 1],
                                               'sweights:0': simple_w[index:index + 1]})
            print('r loss in pre-tr.', r_loss_val)
        print('Pretraining discriminator with', PRE_TRAIN_ADVANTAGE, 'examples ...')
        pre_train_step = 0
        pre_train_loss = np.array([1., 1.])
        for index in range(100 * PRE_TRAIN_ADVANTAGE):
            *loss_value, _ = session.run([d_loss, g_loss, d_update],
                                         feed_dict={'normal:0': normal[index:index + 1],
                                                    'simple:0': simple[index:index + 1],
                                                    'nweights:0': normal_w[index:index + 1],
                                                    'sweights:0': simple_w[index:index + 1]})
            if not index % 100:
                print('d_loss:', loss_value[0], 'g_loss:', loss_value[1])
            #pre_train_step += 1
            #pre_train_loss *= np.array(loss_value).flatten()
        print('Discriminator loss after pre-training:', pre_train_loss ** (1 / pre_train_step))
    while True:
        if r == AE:
            print('Saving ...')
            saver.save(session, train_dir)
            epoch += 1
            print('Starting epoch', epoch, '...')
            train_error = np.array([1.] * len(losses))
            step = 0
            for batch_index in range(*train_set, BS):
                *loss_values, _ = session.run(losses + [update],
                                              feed_dict={'normal:0': normal[batch_index:batch_index + BS],
                                                         'simple:0': simple[batch_index:batch_index + BS],
                                                         'nweights:0': normal_w[batch_index:batch_index + BS],
                                                         'sweights:0': simple_w[batch_index:batch_index + BS]})
                train_error *= np.array(loss_values).flatten()
                step += 1
                # DEBUG
                ## END OF DEBUG
            print('Current train. loss:',
                  *[': '.join((name, str(value))) for name, value in zip(loss_labels, train_error ** (1 / step))],
                  sep='\t')
            # validation part
            valid_step = 0
            valid_error = np.array([1.] * len(losses))
            for batch_index in range(*valid_set, BS):
                loss_values = session.run(losses,
                                          feed_dict={'normal:0': normal[batch_index:batch_index + BS],
                                                     'simple:0': simple[batch_index:batch_index + BS],
                                                     'nweights:0': normal_w[batch_index:batch_index + BS],
                                                     'sweights:0': simple_w[batch_index:batch_index + BS]})
                valid_error *= np.array(loss_values).flatten()
                valid_step += 1
            print('Current valid. loss:',
                  *[': '.join((name, str(value))) for name, value in zip(loss_labels, valid_error ** (1 / valid_step))],
                  sep='\t')
        elif r == DSC1:
            print('Saving ...')
            saver.save(session, train_dir)
            epoch += 1
            print('Starting epoch', epoch, '...')
            train_error = np.array([1.] * 3)
            step = 0
            for batch_index in range(*train_set, BS):
                base_feed = {'normal:0': normal[batch_index:batch_index + BS],
                             'simple:0': simple[batch_index:batch_index + BS],
                             'nweights:0': normal_w[batch_index:batch_index + BS],
                             'sweights:0': simple_w[batch_index:batch_index + BS]}
                g_loss_value, _ = session.run([g_loss, g_update], feed_dict=base_feed)
                d_loss_value, _ = session.run([d_loss, d_update], feed_dict=base_feed)
                r_loss_val, _ = session.run([r_loss, r_update], feed_dict=base_feed)
                train_error *= np.array([g_loss_value, d_loss_value, r_loss_val]).flatten()
                step += 1
            print('Current train. loss:',
                  *[': '.join((name, str(value))) for name, value in zip(loss_labels, train_error ** (1 / step))],
                  sep='\t')
            # validation part
            valid_step = 0
            valid_error = np.array([1.] * len(losses))
            for batch_index in range(*valid_set, BS):
                loss_values = session.run(losses,
                                          feed_dict={'normal:0': normal[batch_index:batch_index + BS],
                                                     'simple:0': simple[batch_index:batch_index + BS],
                                                     'nweights:0': normal_w[batch_index:batch_index + BS],
                                                     'sweights:0': simple_w[batch_index:batch_index + BS]})
                valid_error *= np.array(loss_values).flatten()
                valid_step += 1
            print('Current valid. loss:',
                  *[': '.join((name, str(value))) for name, value in zip(loss_labels, valid_error ** (1 / valid_step))],
                  sep='\t')
