import tensorflow as tf


class WaveNet:
    def __init__(self,
                 name='wavenet', dilations=[2 ** i for i in [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8]],
                 reuse=False, predict_future=True,
                 input_channels=512, output_channels=512,
                 residual_channels=128, dilation_channels=128, skip_channels=128,
                 tanh_skip=False,
                 ):
        self.name = name
        self.reuse = reuse
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.dilations = dilations
        self.predict_future = predict_future
        self.trainable_variables = []
        self.tanh_skip = tanh_skip

    def create_network(self, input_batch: tf.Tensor) -> tf.Tensor:
        batch_num = input_batch.get_shape().as_list()[0]
        current_layer = input_batch
        with tf.variable_scope(self.name, reuse=self.reuse):
            with tf.variable_scope('causal_layer'):
                W, b = self._create_conv_vars(
                    ksize=1, in_chan=self.input_channels, out_chan=self.residual_channels)
                current_layer = tf.nn.conv1d(
                    current_layer, W, stride=1, padding='SAME',
                )
                current_layer = tf.nn.bias_add(current_layer, b)
                del W, b

            skipped_layers = []
            output_length = 0
            with tf.variable_scope('dialted_stack'):
                for i, dilation in enumerate(self.dilations):
                    with tf.variable_scope('conv{}'.format(i)):
                        W_filter, b_filter = self._create_conv_vars(
                            ksize=2, in_chan=self.residual_channels, out_chan=self.dilation_channels,
                            filter='filter', bias='filter_bias',
                        )
                        W_gate, b_gate = self._create_conv_vars(
                            ksize=2, in_chan=self.residual_channels, out_chan=self.dilation_channels,
                            filter='gate', bias='gate_bias',
                        )
                        W_skip, b_skip = self._create_conv_vars(
                            ksize=1, in_chan=self.dilation_channels, out_chan=self.skip_channels,
                            filter='skip', bias='skip_bias',
                        )
                        W_dense, b_dense = self._create_conv_vars(
                            ksize=1, in_chan=self.dilation_channels, out_chan=self.residual_channels,
                            filter='dense', bias='dense_bias',
                        )

                        conv_filter = tf.nn.bias_add(tf.nn.convolution(
                            current_layer, W_filter, padding='VALID', dilation_rate=[dilation],
                        ), b_filter)
                        conv_gate = tf.nn.bias_add(tf.nn.convolution(
                            current_layer, W_gate, padding='VALID', dilation_rate=[dilation],
                        ), b_gate)
                        dilation_layer = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)
                        dense = tf.nn.bias_add(tf.nn.conv1d(
                            dilation_layer, W_dense, stride=1, padding='SAME'
                        ), b_dense)
                        skip = tf.nn.bias_add(tf.nn.conv1d(
                            dilation_layer, W_skip, stride=1, padding='SAME'
                        ), b_skip)
                        if self.tanh_skip:
                            skip = tf.tanh(skip)

                        dense_output_length = tf.shape(dense)[1]
                        if self.predict_future:
                            input_length = tf.shape(current_layer)[1]
                            sliced_input = tf.slice(
                                current_layer,
                                [0, input_length - dense_output_length, 0],
                                [-1, -1, -1],
                            )
                        else:
                            sliced_input = tf.slice(
                                current_layer,
                                [0, 0, 0],
                                [-1, dense_output_length, -1],
                            )

                        current_layer = tf.add(
                            dense,
                            sliced_input,
                        )
                        skipped_layers.append(skip)
                        output_length = skip.get_shape().as_list()[1]

            with tf.variable_scope('skip_conv'):
                if self.predict_future:
                    total = sum(
                        tf.slice(skip, [0, tf.shape(skip)[1] - output_length, 0], [-1, -1, -1])
                        for skip in skipped_layers
                    )
                else:
                    total = sum(
                        tf.slice(skip, [0, 0, 0], [-1, output_length, -1])
                        for skip in skipped_layers
                    )

            with tf.variable_scope('postprocessing'):
                inp = tf.nn.elu(total)
                w1, b1 = self._create_conv_vars(
                    ksize=1, in_chan=self.skip_channels, out_chan=self.skip_channels,
                    filter='postprosess1', bias='postprosess1_bias'
                )
                conv1 = tf.nn.elu(
                    tf.nn.bias_add(
                        tf.nn.conv1d(inp, w1, stride=1, padding='SAME'),
                        b1,
                    )
                )
                w2, b2 = self._create_conv_vars(
                    ksize=1, in_chan=self.skip_channels, out_chan=self.output_channels,
                    filter='postprosess2', bias='postprosess2_bias'
                )
                output = tf.nn.bias_add(
                    tf.nn.conv1d(conv1, w2, stride=1, padding='SAME'),
                    b2,
                )
                output = tf.reshape(output, [batch_num, output_length, self.output_channels])

        self.reuse = True
        return output

    def _create_conv_vars(self, ksize, in_chan, out_chan, filter='filter', bias='bias'):
        if filter:
            initializer = tf.contrib.layers.xavier_initializer_conv2d()
            W = tf.get_variable(filter, shape=[ksize, in_chan, out_chan], initializer=initializer)
        else:
            W = None
        if bias:
            initializer = tf.constant_initializer(value=0., dtype=tf.float32)
            b = tf.get_variable(bias, shape=[out_chan], initializer=initializer)
        else:
            b = None
        return W, b

    def get_trainable_variables(self):
        return [
            v for v in tf.trainable_variables()
            if v.name.startswith('{}/'.format(self.name))
            ]


class Encoder:
    def encode(self, input_batch_f, size) -> tf.Tensor:
        '''
        :return: input_batch_i 
        '''
        raise NotImplementedError

    def decode(self, input_batch_i, size) -> tf.Tensor:
        '''
        :return: input_batch_f [-1, 1] 
        '''
        raise NotImplementedError


class NormalEncoder(Encoder):
    def encode(self, input_batch_f, size):
        _assert = tf.assert_less_equal(input_batch_f, 1.)
        with tf.control_dependencies([_assert]):
            return tf.cast(tf.floor((input_batch_f + 1) * (size - 1) / 2), tf.int32)

    def decode(self, input_batch_i, size):
        return tf.cast(input_batch_i, tf.float32) * 2 / (size - 1) - 1


class InvULawEncoder(Encoder):
    pass


class Trainer:
    def __init__(self, input_channels, output_channels,
                 reuse=False, encoder=None, name='trainer',
                 ):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.pred_future = WaveNet(
            'CondFuture',
            predict_future=True, reuse=reuse,
            input_channels=input_channels, output_channels=output_channels,
        )
        self.pred_past = WaveNet(
            'CondPast',
            predict_future=False, reuse=reuse,
            input_channels=input_channels, output_channels=output_channels,
        )
        self.px = tf.Variable(tf.zeros([output_channels]), name='px', trainable=False)
        self.name = name
        if encoder is None:
            encoder = NormalEncoder()
        self.encoder = encoder

    def create_network(self, input_batch_f, l2_strength=0.0001, learning_rate=0.001):
        with tf.name_scope('encoder'):
            length = input_batch_f.get_shape().as_list()[1]

            input_batch_i = self.encoder.encode(input_batch_f, self.input_channels)
            inputs = tf.one_hot(
                input_batch_i,
                self.input_channels,
            )
            labels = self.encoder.encode(input_batch_f, self.output_channels)
        with tf.name_scope('px'):
            _px, _ = tf.nn.moments(tf.one_hot(labels, self.output_channels), [0, 1])
            self.op_update_px = self.px.assign(0.99 * self.px + 0.01 * _px)

        output_future = self.pred_future.create_network(inputs)
        output_future_length = output_future.get_shape().as_list()[1]

        output_past = self.pred_past.create_network(inputs)
        output_past_length = output_past.get_shape().as_list()[1]

        _, input_variance = tf.nn.moments(input_batch_f, axes=[0, 1])
        tf.summary.scalar('input_variance', input_variance)
        tf.summary.tensor_summary('px', self.px)
        tf.summary.scalar('px', tf.reduce_sum(self.px))

        with tf.name_scope('CEFuture'):
            label_future_f = tf.slice(input_batch_f, [0, length - output_future_length + 1], [-1, -1])
            label_future = self.encoder.encode(label_future_f, self.output_channels)
            _output_future = tf.slice(output_future, [0, 0, 0], [-1, output_future_length - 1, -1])
            loss_ce_future = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    _output_future,
                    label_future,
                )
            )
        with tf.name_scope('AccFuture'):
            acc_future = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.cast(tf.argmax(_output_future, axis=2), tf.int32),
                        label_future,
                    ),
                    tf.float32,
                )
            )
        with tf.name_scope('L2Future'):
            loss_l2_future = tf.reduce_mean(
                [
                    tf.nn.l2_loss(x) for x in self.pred_future.get_trainable_variables()
                    if 'bias' not in x.name
                    ]
            )
        loss_future = loss_ce_future + l2_strength * loss_l2_future
        self.loss_future = loss_future
        self.acc_future = acc_future

        with tf.name_scope('CEPast'):
            label_past_f = tf.slice(input_batch_f, [0, 0], [-1, output_past_length - 1])
            label_past = self.encoder.encode(label_past_f, self.output_channels)
            _output_past = tf.slice(output_past, [0, 1, 0], [-1, -1, -1])
            loss_ce_past = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    _output_past,
                    label_past,
                )
            )
        with tf.name_scope('AccPast'):
            acc_past = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        tf.cast(tf.argmax(_output_past, axis=2), tf.int32),
                        label_past,
                    ),
                    tf.float32,
                )
            )
        with tf.name_scope('L2Past'):
            loss_l2_past = tf.reduce_mean(
                [
                    tf.nn.l2_loss(x) for x in self.pred_past.get_trainable_variables()
                    if 'bias' not in x.name
                    ]
            )
        loss_past = loss_ce_past + l2_strength * loss_l2_past
        self.loss_past = loss_past
        self.acc_past = acc_past
        tf.summary.scalar('future_loss_ce', loss_ce_future)
        tf.summary.scalar('future_loss_l2', loss_l2_future)
        tf.summary.scalar('future_loss_total', loss_future)
        tf.summary.scalar('future_acc', acc_future)
        tf.summary.scalar('past_loss_ce', loss_ce_past)
        tf.summary.scalar('past_loss_l2', loss_l2_past)
        tf.summary.scalar('past_loss_total', loss_past)
        tf.summary.scalar('past_acc', acc_past)

        self.optimizer_future = tf.train.AdamOptimizer(learning_rate)
        self.optimizer_past = tf.train.AdamOptimizer(learning_rate)
        op_minimize_future_loss = self.optimizer_future.minimize(
            loss_future,
            var_list=self.pred_future.get_trainable_variables(),
        )
        op_minimize_past_loss = self.optimizer_past.minimize(
            loss_past,
            var_list=self.pred_past.get_trainable_variables(),
        )
        self.op_minimize_future_loss = op_minimize_future_loss
        self.op_minimize_past_loss = op_minimize_past_loss

        update_ops = [self.op_minimize_future_loss, self.op_minimize_past_loss, self.op_update_px]
        with tf.control_dependencies(update_ops):
            self.op_train = tf.no_op()

    def get_variables(self):
        return self.pred_future.get_trainable_variables() + \
               self.pred_past.get_trainable_variables() + \
               self.px


def main():
    import numpy as np
    with tf.Session() as sess:
        wavenet = WaveNet(dilations=[2, 4, 8, 16])
        input_batch = tf.placeholder(tf.float32, [10, 1024, 512])
        output = wavenet.create_network(input_batch)
        print(output)

        tf.global_variables_initializer().run()
        res = sess.run(output, feed_dict={input_batch: np.random.uniform(size=[10, 1024, 512])})
        print(res.shape)


if __name__ == '__main__':
    main()
