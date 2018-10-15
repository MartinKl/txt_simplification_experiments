import logging
import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, fully_connected, max_pool2d, softmax, stack
from tensorflow.contrib.rnn import MultiRNNCell, LSTMCell, LSTMStateTuple
from tensorflow.contrib.seq2seq import sequence_loss
tf.reset_default_graph()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelParameters(object):
    def __init__(self,
                 sequence_length,
                 vocabulary_size,
                 embedding_dim,
                 hidden_dim):
        self._emb = embedding_dim
        self._n = sequence_length
        self._v = vocabulary_size
        self._h = hidden_dim

    @property
    def v(self):
        return self._v

    @property
    def emb(self):
        return self._emb

    @property
    def h(self):
        return self._h

    @property
    def n(self):
        return self._n

    def __str__(self):
        return ':'.join((type(self).__name__,
                         'V', str(self.v),
                         'EMB', str(self.emb),
                         'H', str(self.h),
                         'N', str(self.n)))


class TrainingParameters(object):
    def __init__(self,
                 checkpoint_path,
                 batch_size=32,
                 learning_rate=.1,
                 clip_norm=5.,
                 optimizer=tf.train.GradientDescentOptimizer,
                 **opt_kwargs):
        self._path = checkpoint_path
        self._bs = batch_size
        self._lr = learning_rate
        self._clip_norm = clip_norm
        self._opt = optimizer(learning_rate=learning_rate, **opt_kwargs)

    @property
    def batch_size(self):
        return self._bs

    @property
    def learning_rate(self):
        return self._lr

    @property
    def clip_norm(self):
        return self._clip_norm

    @property
    def optimizer(self):
        return self._opt

    @property
    def path(self):
        return self._path

    def __str__(self):
        return ':'.join((type(self).__name__,
                         'BS', str(self.batch_size),
                         'LR', str(self.learning_rate),
                         'PATH', str(self.path)))


class SimplificationModel(object):
    def __init__(self,
                 model_params,
                 training_params,
                 clean_environment=True,
                 auto_save=True,
                 overwrite=False,
                 load=False,
                 log=True):
        self._auto_save = auto_save
        self._path = training_params.path
        self._session = None
        self._saver = None
        if clean_environment:
            tf.reset_default_graph()
        self.x_normal = tf.placeholder(tf.int32, shape=(None, model_params.n,), name='normal')
        self.w_normal = tf.placeholder(tf.float32, shape=(None, model_params.n,), name='nweights')
        self.x_simple = tf.placeholder(tf.int32, shape=(None, model_params.n,), name='simple')
        self.w_simple = tf.placeholder(tf.float32, shape=(None, model_params.n,), name='sweights')
        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        self._epoch = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name='epoch')
        self._count_epoch = tf.assign(self._epoch, tf.add(self._epoch, 1))
        b = training_params.batch_size
        d = model_params.h
        v = model_params.v
        self._initial_encoder_state = [
            LSTMStateTuple(tf.zeros((b, d,), dtype=tf.float32), tf.zeros((b, d,), dtype=tf.float32)),
            LSTMStateTuple(tf.zeros((b, d,), dtype=tf.float32), tf.zeros((b, d,), dtype=tf.float32))
        ]
        self._initial_decoder_state = initial_decoder_state = [
            LSTMStateTuple(tf.zeros((b, d,), dtype=tf.float32), tf.zeros((b, d,), dtype=tf.float32)),
            LSTMStateTuple(tf.zeros((b, d,), dtype=tf.float32), tf.zeros((b, d,), dtype=tf.float32)),
            LSTMStateTuple(tf.zeros((b, v,), dtype=tf.float32), tf.zeros((b, v,), dtype=tf.float32))
        ]
        self._update = None
        self._losses = []
        self._training_params = training_params
        self._model_params = model_params
        self._build()
        self._variables = tf.global_variables()  # check if more sophisticated sub setting is possible
        if log:
            self._train_writer = None
            self._valid_writer = None
            self._summary = None
            self._log_step = 0
            self._log()
        self._has_summary = log
        self._active = False
        self._load = load

    def _build(self):
        raise NotImplementedError('Abstract class cannot be instantiated.')

    def _log(self):
        self._train_writer = tf.summary.FileWriter(logdir=os.path.join(self._training_params.path, 'train'))
        self._valid_writer = tf.summary.FileWriter(logdir=os.path.join(self._training_params.path, 'valid'))
        for loss in self._losses:
            tf.summary.scalar(loss.name, loss)
        self._summary = tf.summary.merge_all()

    def _step(self, x, y=None, weights_x=None, weights_y=None):
        raise NotImplementedError

    def load(self, directory):
        loader = tf.train.Saver(var_list=self._variables)
        checkpoint = tf.train.get_checkpoint_state(directory)
        loader.restore(sess=self._session, save_path=checkpoint.model_checkpoint_path)

    def save(self):
        if self._saver is None:
            self._saver = tf.train.Saver(var_list=self._variables)
        self._saver.save(sess=self._session, save_path=os.path.join(self._path, type(self).__name__))
        logging.info('Model state saved in ' + self._path)

    def loop(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def active(self):
        return self._active

    @property
    def age(self):
        return self._epoch.eval(session=self._session)

    def __enter__(self):
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self._session = tf.Session(config=config)
        self._session.__enter__()
        if self._load:
            self.load(self._path)
        else:
            self._session.run(tf.global_variables_initializer())
        self._active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._auto_save:
            self.save()
        self._active = False
        self._session.__exit__(exc_type, exc_val, exc_tb)


class AE(SimplificationModel):
    def _build(self):
        with tf.variable_scope('embedding') as scope:
            embedded_n = tf.contrib.layers.embed_sequence(self.x_normal,
                                                          vocab_size=self._model_params.v,
                                                          embed_dim=self._model_params.emb,
                                                          scope=scope)
            embedded_s = tf.contrib.layers.embed_sequence(self.x_simple,
                                                          reuse=True,
                                                          scope=scope)
        with tf.variable_scope('encoder'):
            encoder = MultiRNNCell([LSTMCell(self._model_params.h), LSTMCell(self._model_params.h)])
        with tf.variable_scope('encode_n') as scope:
            z_normal_arr = []
            z, state = encoder(embedded_n[:, 0], self._initial_encoder_state)
            z_normal_arr.append(z)
        with tf.variable_scope(scope, reuse=True):
            for i in range(1, self._model_params.n):
                z, state = encoder(embedded_n[:, i], state)
                z_normal_arr.append(z)
        with tf.variable_scope('encode_s') as scope:
            z_simple_arr = []
            z, state = encoder(embedded_s[:, 0], self._initial_encoder_state)
            z_simple_arr.append(z)
        with tf.variable_scope(scope, reuse=True):
            for i in range(1, self._model_params.n):
                z, state = encoder(embedded_s[:, i], state)
                z_simple_arr.append(z)
        z_n = tf.transpose(tf.convert_to_tensor(z_normal_arr))
        z_s = tf.transpose(tf.convert_to_tensor(z_simple_arr))
        with tf.variable_scope('reduction/0') as scope:
            z_normal = conv2d(z_n, self._model_params.n // 2, (32,))
            z_simple = conv2d(z_s, self._model_params.n // 2, (32,), reuse=tf.AUTO_REUSE, scope=scope)
        with tf.variable_scope('reduction/1') as scope:
            z_normal = conv2d(z_normal, self._model_params.n // 4, (32,))
            z_simple = conv2d(z_simple, self._model_params.n // 4, (32,), reuse=tf.AUTO_REUSE, scope=scope)
        with tf.variable_scope('reduction/pool'):
            z_normal = max_pool2d(tf.transpose(z_normal[None], (0, 1, 3, 2)),
                                  kernel_size=(1, self._model_params.n // 4),
                                  stride=1)
            z_simple = max_pool2d(tf.transpose(z_simple[None], (0, 1, 3, 2)),
                                  kernel_size=(1, self._model_params.n // 4),
                                  stride=1)
        with tf.variable_scope('reduction/reshape'):
            z_normal = tf.reshape(z_normal, (self._training_params.batch_size, self._model_params.h))
            z_simple = tf.reshape(z_simple, (self._training_params.batch_size, self._model_params.h))
        with tf.variable_scope('decode_n') as scope:
            decoder_n = MultiRNNCell([
                LSTMCell(self._model_params.h),
                LSTMCell(self._model_params.h),
                LSTMCell(self._model_params.v)
            ])
            state = self._initial_decoder_state
            logits_normal = []
            logits, state = decoder_n(z_normal, state)
            logits_normal.append(logits)
        with tf.variable_scope(scope, reuse=True):
            for _ in range(1, self._model_params.n):
                logits, state = decoder_n(z_normal, state)
                logits_normal.append(logits)
            err_r_normal = tf.reduce_sum(sequence_loss(tf.transpose(tf.convert_to_tensor(logits_normal), (1, 0, 2)),
                                                       self.x_normal,
                                                       self.w_normal))
        with tf.variable_scope('decode_s') as scope:
            decoder_s = MultiRNNCell([
                LSTMCell(self._model_params.h),
                LSTMCell(self._model_params.h),
                LSTMCell(self._model_params.v)
            ])
            state = self._initial_decoder_state
            logits_simple = []
            logits, state = decoder_s(z_simple, state)
            logits_simple.append(logits)
        with tf.variable_scope(scope, reuse=True):
            for _ in range(1, self._model_params.n):
                logits, state = decoder_s(z_simple, state)
                logits_simple.append(logits)
            err_r_simple = tf.reduce_sum(sequence_loss(tf.transpose(tf.convert_to_tensor(logits_simple), (1, 0, 2)),
                                                       self.x_simple,
                                                       self.w_simple))
        # losses
        err_z = tf.reduce_sum(tf.squared_difference(z_simple, z_normal), name='repr_error')
        err_r = tf.add(err_r_normal, err_r_simple, name='rec_err')
        err = tf.add(err_z, err_r, name='err')
        self._losses = [err, err_r, err_z]
        params = tf.trainable_variables()
        self._update = tf.train.AdamOptimizer(learning_rate=self._training_params.learning_rate).minimize(err)

    def _step(self, x_n, x_s=None, weights_x_n=None, weights_x_s=None, forward_only=False):
        if not self.active:
            raise ValueError
        variables = self._losses + ([] if forward_only else [self._update])
        if self._has_summary:
            variables.append(self._summary)
        values = self._session.run(variables,
                                   feed_dict={'normal:0': x_n,
                                              'simple:0': x_s,
                                              'nweights:0': weights_x_n,
                                              'sweights:0': weights_x_s})
        return np.array(values[:len(self._losses)]).flatten(), values[-1]

    def loop(self, training_data, validation_data, continue_callback=lambda: True, callback_args=(), **kwargs):
        while continue_callback(*callback_args):
            if not self.active:
                logger.warn('Model inactive, cancelling loop ...')
                return
            self._session.run([self._count_epoch])
            logger.info('Starting epoch ' + str(self._epoch.eval(session=self._session)))
            self._consume(data=training_data, **kwargs)
            self._consume(data=validation_data, forward_only=True, select=2000, **kwargs)
            if self._auto_save:
                self.save()
        logger.info('Finished training after {} epochs'.format(self._epoch.eval(session=self._session)))

    def _consume(self,
                 data,
                 forward_only=False,
                 steps=None,
                 report_every=1000,
                 log_every=100,
                 save_every=sys.maxsize,
                 **kwargs):
        max_i = sys.maxsize if steps is None else steps
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Starting data consumption ...')
        if forward_only:
            logger.info('- validating / testing')
        else:
            logger.info('- training')
        if steps is not None:
            logger.warn('- max steps is ' + str(max_i))
        error = 1.
        for batch_index, batch in data.batches(batch_size=self._training_params.batch_size, **kwargs):
            err_values, tail = self._step(*batch, forward_only=forward_only)
            error *= err_values
            if self._has_summary and not batch_index % log_every:
                if forward_only:
                    self._valid_writer.add_summary(tail, self._log_step)
                else:
                    self._train_writer.add_summary(tail, self._log_step)
                self._log_step += 1
            if batch_index and not batch_index % report_every:
                logger.info(
                    'Step {}: current {} error is {}, accumulated error is {}'\
                        .format(batch_index, 'VALID' if forward_only else 'TRAIN', error ** (1 / report_every), error)
                )
                error = 1.
            if self._auto_save and batch_index and not batch_index % save_every:
                self.save()
            if batch_index >= max_i:
                break
        logger.info('Epoch finished')


class SimpleAE(AE):
    def _build(self):
        with tf.variable_scope('embedding') as scope:
            embedded_n = tf.contrib.layers.embed_sequence(self.x_normal,
                                                          vocab_size=self._model_params.v,
                                                          embed_dim=self._model_params.emb,
                                                          scope=scope)
            embedded_s = tf.contrib.layers.embed_sequence(self.x_simple,
                                                          reuse=True,
                                                          scope=scope)
        with tf.variable_scope('encoder'):
            encoder = MultiRNNCell([LSTMCell(self._model_params.h), LSTMCell(self._model_params.h)])
        with tf.variable_scope('encode_n') as scope:
            z, state = encoder(embedded_n[:, 0], self._initial_encoder_state)
        with tf.variable_scope(scope, reuse=True):
            for i in range(1, self._model_params.n):
                z, state = encoder(embedded_n[:, i], state)
            z_normal = z
        with tf.variable_scope('encode_s') as scope:
            z, state = encoder(embedded_s[:, 0], self._initial_encoder_state)
        with tf.variable_scope(scope, reuse=True):
            for i in range(1, self._model_params.n):
                z, state = encoder(embedded_s[:, i], state)
            z_simple = z
        with tf.variable_scope('decode_n') as scope:
            decoder_n = MultiRNNCell([
                LSTMCell(self._model_params.h),
                LSTMCell(self._model_params.h),
                LSTMCell(self._model_params.v)
            ])
            state = self._initial_decoder_state
            logits_normal = []
            logits, state = decoder_n(z_normal, state)
            logits_normal.append(logits)
        with tf.variable_scope(scope, reuse=True):
            for _ in range(1, self._model_params.n):
                logits, state = decoder_n(z_normal, state)
                logits_normal.append(logits)
            err_r_normal = tf.reduce_sum(sequence_loss(tf.transpose(tf.convert_to_tensor(logits_normal), (1, 0, 2)),
                                                       self.x_normal,
                                                       self.w_normal))
        with tf.variable_scope('decode_s') as scope:
            decoder_s = MultiRNNCell([
                LSTMCell(self._model_params.h),
                LSTMCell(self._model_params.h),
                LSTMCell(self._model_params.v)
            ])
            state = self._initial_decoder_state
            logits_simple = []
            logits, state = decoder_s(z_simple, state)
            logits_simple.append(logits)
        with tf.variable_scope(scope, reuse=True):
            for _ in range(1, self._model_params.n):
                logits, state = decoder_s(z_simple, state)
                logits_simple.append(logits)
            err_r_simple = tf.reduce_sum(sequence_loss(tf.transpose(tf.convert_to_tensor(logits_simple), (1, 0, 2)),
                                                       self.x_simple,
                                                       self.w_simple))
        # losses
        err_z = tf.reduce_sum(tf.squared_difference(z_simple, z_normal), name='repr_error')
        err_r = tf.add(err_r_normal, err_r_simple, name='rec_err')
        err = tf.add(err_z, err_r, name='err')
        self._losses = [err, err_r, err_z]
        self._update = tf.train.AdamOptimizer(learning_rate=self._training_params.learning_rate).minimize(err)


class DiscriminatorModel(SimplificationModel):
    def __init__(self, *args, **kwargs):
        self._dsc_update = None
        self._enc_update = None
        super().__init__(*args, **kwargs)

    def _log(self):
        self._train_writer = tf.summary.FileWriter(logdir=os.path.join(self._training_params.path, 'train'))
        self._valid_writer = tf.summary.FileWriter(logdir=os.path.join(self._training_params.path, 'valid'))
        for loss in self._losses:
            tf.summary.scalar(loss.name, loss)
        self._summary = tf.summary.merge_all()

    def _build(self):
        with tf.variable_scope('embedding') as embedding_scope:
            embedded_n = tf.contrib.layers.embed_sequence(self.x_normal,
                                                          vocab_size=self._model_params.v,
                                                          embed_dim=self._model_params.emb,
                                                          scope=embedding_scope)
            embedded_s = tf.contrib.layers.embed_sequence(self.x_simple,
                                                          reuse=True,
                                                          scope=embedding_scope)
        with tf.variable_scope('encoder') as encoder_scope:
            encoder = MultiRNNCell([LSTMCell(self._model_params.h), LSTMCell(self._model_params.h)])
        with tf.variable_scope('encode/n') as encoding_scope_n:
            z_normal_arr = []
            z, state = encoder(embedded_n[:, 0], self._initial_encoder_state)
            z_normal_arr.append(z)
        with tf.variable_scope(encoding_scope_n, reuse=True):
            for i in range(1, self._model_params.n):
                z, state = encoder(embedded_n[:, i], state)
                z_normal_arr.append(z)
        with tf.variable_scope('encode/s') as encoding_scope_s:
            z_simple_arr = []
            z, state = encoder(embedded_s[:, 0], self._initial_encoder_state)
            z_simple_arr.append(z)
        with tf.variable_scope(encoding_scope_s, reuse=True):
            for i in range(1, self._model_params.n):
                z, state = encoder(embedded_s[:, i], state)
                z_simple_arr.append(z)
        z_n = tf.transpose(tf.convert_to_tensor(z_normal_arr))
        z_s = tf.transpose(tf.convert_to_tensor(z_simple_arr))
        with tf.variable_scope('reduction/0') as scope:
            z_normal = conv2d(z_n, self._model_params.n // 2, (32,))
            z_simple = conv2d(z_s, self._model_params.n // 2, (32,), reuse=tf.AUTO_REUSE, scope=scope)
        with tf.variable_scope('reduction/1') as scope:
            z_normal = conv2d(z_normal, self._model_params.n // 4, (32,))
            z_simple = conv2d(z_simple, self._model_params.n // 4, (32,), reuse=tf.AUTO_REUSE, scope=scope)
        with tf.variable_scope('reduction/pool'):
            z_normal = max_pool2d(tf.transpose(z_normal[None], (0, 1, 3, 2)),
                                  kernel_size=(1, self._model_params.n // 4),
                                  stride=1)
            z_simple = max_pool2d(tf.transpose(z_simple[None], (0, 1, 3, 2)),
                                  kernel_size=(1, self._model_params.n // 4),
                                  stride=1)
        with tf.variable_scope('reduction/reshape'):
            z_normal = tf.reshape(z_normal, (self._training_params.batch_size, self._model_params.h))
            z_simple = tf.reshape(z_simple, (self._training_params.batch_size, self._model_params.h))

        # decode
        with tf.variable_scope('decode_n') as scope:
            decoder_n = MultiRNNCell([
                LSTMCell(self._model_params.h),
                LSTMCell(self._model_params.h),
                LSTMCell(self._model_params.v)
            ])
            state = self._initial_decoder_state
            logits_normal = []
            logits, state = decoder_n(z_normal, state)
            logits_normal.append(logits)
        with tf.variable_scope(scope, reuse=True):
            for _ in range(1, self._model_params.n):
                logits, state = decoder_n(z_normal, state)
                logits_normal.append(logits)
            decoder_loss_normal = tf.reduce_sum(sequence_loss(tf.transpose(tf.convert_to_tensor(logits_normal),
                                                                           (1, 0, 2)),
                                                self.x_normal,
                                                self.w_normal))
        with tf.variable_scope('decode_s') as scope:
            decoder_s = MultiRNNCell([
                LSTMCell(self._model_params.h),
                LSTMCell(self._model_params.h),
                LSTMCell(self._model_params.v)
            ])
            state = self._initial_decoder_state
            logits_simple = []
            logits, state = decoder_s(z_simple, state)
            logits_simple.append(logits)
        with tf.variable_scope(scope, reuse=True):
            for _ in range(1, self._model_params.n):
                logits, state = decoder_s(z_simple, state)
                logits_simple.append(logits)
            decoder_loss_simple = tf.reduce_sum(sequence_loss(tf.transpose(tf.convert_to_tensor(logits_simple),
                                                                           (1, 0, 2)),
                                                self.x_simple,
                                                self.w_simple))
        # losses
        ae_loss = tf.add(decoder_loss_normal, decoder_loss_simple, name='seq2seq_loss')
        dsc_dims = [self._model_params.h // (2 ** e) for e in range(1, int(np.log2(self._model_params.h // 2)))] + [1]
        with tf.variable_scope('discriminator') as dsc_scope:
            # the discriminator's output probability is chosen as probability that the input was a normal, i. e. complex
            # expression. Thus, the probability can also be interpreted as complexity score with higher values
            # expressing a higher level of complexity
            from_normal = tf.sigmoid(stack(z_normal,
                                           layer=fully_connected,
                                           stack_args=dsc_dims))
            from_simple = tf.sigmoid(stack(z_simple,
                                           layer=fully_connected,
                                           stack_args=dsc_dims,
                                           reuse=tf.AUTO_REUSE))
            err_dsc = tf.add(tf.squared_difference(1., from_normal), tf.squared_difference(0., from_simple),
                             name='discriminator_loss')
        err_g = tf.negative(tf.log(err_dsc))
        # optimization
        ## optimizers
        dsc_optimizer = tf.train.AdamOptimizer(learning_rate=self._training_params.learning_rate)
        ae_optimizer = tf.train.AdamOptimizer(learning_rate=self._training_params.learning_rate)  # shared optimizer for alignment of z's and auto-encoding (potential pitfall, but separate not better)
        ## updates
        dsc_vars = tf.global_variables(scope=dsc_scope.name)
        self._dsc_update = dsc_optimizer.minimize(err_dsc, var_list=dsc_vars)
        self._enc_update = ae_optimizer.minimize(err_g)
        enc_scopes = [embedding_scope, encoder_scope.name, 'encoding', 'reduction']
        enc_vars = []
        for scope_name in enc_scopes:
            enc_vars.extend(tf.global_variables(scope=scope_name))
        self._enc_update = ae_optimizer.minimize(err_g, var_list=enc_vars)
        ae_vars = [var for var in tf.global_variables() if var not in set(dsc_vars)]
        self._update = ae_optimizer.minimize(ae_optimizer, var_list=ae_vars)
        self._losses = [ae_loss, err_dsc]

    def _step(self, x_n, x_s=None, weights_x_n=None, weights_x_s=None, forward_only=False):
        if not self.active:
            raise ValueError
        variables = self._losses + ([] if forward_only else [self._update])
        if self._has_summary:
            variables.append(self._summary)
        values = self._session.run(variables,
                                   feed_dict={'normal:0': x_n,
                                              'simple:0': x_s,
                                              'nweights:0': weights_x_n,
                                              'sweights:0': weights_x_s})
        return np.array(values[:len(self._losses)]).flatten(), values[-1]

    def _pre_train_dsc(self, data, examples=1000):
        # first pre-train the ae a little, to make sure, encoding of z is a little more sophisticated than just
        # max-values or 0-vectors
        for _, batch in data.batches(select=50):
            x_n, x_s, weights_x_n, weights_x_s = batch
            self._session.run([self._update],
                              feed_dict={'normal:0': x_n,
                                         'simple:0': x_s,
                                         'nweights:0': weights_x_n,
                                         'sweights:0': weights_x_s})
        n = examples // self._training_params.batch_size
        loss_values = []
        loss = self._losses[1]
        for _, batch in data.batches(select=n):
            x_n, x_s, weights_x_n, weights_x_s = batch
            value, _ = self._session.run([loss, self._dsc_update],
                                         feed_dict={'normal:0': x_n,
                                                    'simple:0': x_s,
                                                    'nweights:0': weights_x_n,
                                                    'sweights:0': weights_x_s})
            loss_values.append(value)
        return loss_values[:3], loss_values[-3:]

    def loop(self, training_data, validation_data, continue_callback=lambda: True, callback_args=(), **kwargs):
        warn_message = 'Model inactive, cancelling loop ...'
        if not self.active:
            logger.warn(warn_message)
            return
        before, after = self._pre_train_dsc(training_data)
        logger.info('Pre-training loss values: {} ... {}'.format(before, after))
        if sum(before) <= sum(after):
            logger.warn('Pre-training was useless ...')
        while continue_callback(*callback_args):
            if not self.active:
                logger.warn(warn_message)
                return
            self._session.run([self._count_epoch])
            logger.info('Starting epoch ' + str(self._epoch.eval(session=self._session)))
            self._consume(data=training_data, **kwargs)
            self._consume(data=validation_data, forward_only=True, select=200, **kwargs)
            if self._auto_save:
                self.save()
        logger.info('Finished training after {} epochs'.format(self._epoch.eval(session=self._session)))

    def _consume(self,
                 data,
                 forward_only=False,
                 steps=None,
                 report_every=1000,
                 log_every=100,
                 save_every=sys.maxsize,
                 **kwargs):
        max_i = sys.maxsize if steps is None else steps
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug('Starting data consumption ...')
        if forward_only:
            logger.info('- validating / testing')
        else:
            logger.info('- training')
        if steps is not None:
            logger.warn('- max steps is ' + str(max_i))
        error = 1.
        for batch_index, batch in data.batches(batch_size=self._training_params.batch_size, **kwargs):
            err_values, tail = self._step(*batch, forward_only=forward_only)
            error *= err_values
            if self._has_summary and not batch_index % log_every:
                if forward_only:
                    self._valid_writer.add_summary(tail, self._log_step)
                else:
                    self._train_writer.add_summary(tail, self._log_step)
                self._log_step += 1
            if batch_index and not batch_index % report_every:
                logger.info(
                    'Step {}: current {} error is {}, accumulated error is {}'\
                        .format(batch_index, 'VALID' if forward_only else 'TRAIN', error ** (1 / report_every), error)
                )
                error = 1.
            if self._auto_save and batch_index and not batch_index % save_every:
                self.save()
            if batch_index >= max_i:
                break
        logger.info('Epoch finished')
