import os

import soundfile as sf
import tensorflow as tf
import time

from reader import MultipleAudioReader


def create_conv_weight_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_bias_variable(name, shape, initial_value=0.):
    initializer = tf.constant_initializer(value=initial_value, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


def causal_conv(value, filter_, dilation, name='causal_conv', padding='VALID'):
    with tf.name_scope(name):
        output = tf.nn.convolution(value, filter_, padding=padding, dilation_rate=[dilation])
    return output


def quantize(input_batch, channels):
    return tf.cast(
        (input_batch / 2 + 0.5) * (channels - 1),
        tf.int32
    )


def decode(self, input_batch, channels):
    return tf.clip_by_value(tf.cast(input_batch, 'float') * 2 / (channels - 1) - 1, -1, 1)


class Wavenet(object):
    def __init__(self,
            batch_size,
            residual_channels, dilation_channels, skip_channels,
            input_channels, output_channels,
            dilation_num=5,
            use_bias=True,
            name='wavenet',
            filter_width=2,
    ):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.batch_size = batch_size
        self.name = name
        self.filter_width = filter_width

        self.dilations = [filter_width ** i for i in range(dilation_num)]
        self.label_offset = filter_width ** dilation_num - 1

        self.use_bias = use_bias
        self.variables = self._create_variables()

    def _create_causal_layer(self, input_batch):
        W = self.variables['causal_layer']['filter']
        return causal_conv(input_batch, W, 1)

    def _create_dilation_layer(self, input_batch, layer_index, dilation, histogram):
        '''
                |-> [gate]   -|        |-> 1x1 conv -> skip output
        input - |             |-> (*) -|
                |-> [filter] -|        |-> 1x1 conv -|
                |                                    |-> (+) -> dense output
                |------------------------------------|
        '''
        variables = self.variables['dilated_stack'][layer_index]

        weight_filter = variables['filter']
        weight_gate = variables['gate']
        weight_dense = variables['dense']
        weight_skip = variables['skip']
        bias_filter = variables.get('filter_bias')
        bias_gate = variables.get('gate_bias')
        bias_dense = variables.get('dense_bias')
        bias_skip = variables.get('skip_bias')

        conv_filter = causal_conv(input_batch, weight_filter, dilation)
        conv_gate = causal_conv(input_batch, weight_gate, dilation)
        if self.use_bias:
            conv_filter = tf.add(conv_filter, bias_filter)
            conv_gate = tf.add(conv_gate, bias_gate)
        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        dense_tmp = tf.nn.conv1d(
            out, weight_dense,
            stride=1, padding="SAME", name='skip'
        )

        if self.use_bias:
            dense_tmp = tf.add(dense_tmp, bias_dense)

        dense_output_length = tf.shape(dense_tmp)[1]
        input_length = tf.shape(input_batch)[1]

        dense_output = tf.add(
            dense_tmp,
            tf.slice(
                input_batch,
                [0, input_length - dense_output_length, 0],
                [-1, -1, -1],
            ),
        )

        skipped_output = tf.nn.conv1d(
            out, weight_skip,
            stride=1, padding="SAME", name='skip'
        )
        if self.use_bias:
            skipped_output = tf.add(skipped_output, bias_skip)

        return skipped_output, dense_output

    def _create_variables(self):
        variables = {}
        with tf.variable_scope(self.name):
            with tf.variable_scope('causal_convolution'):
                layer = {}
                layer['filter'] = create_conv_weight_variable(
                    name='filter',
                    shape=[1, self.input_channels, self.residual_channels]
                )
                variables['causal_layer'] = layer

            dilated_stack = []
            with tf.variable_scope('dilated_stack'):
                for i, dilation in enumerate(self.dilations):
                    current = {}
                    with tf.variable_scope('layer{}'.format(i)):
                        current['filter'] = create_conv_weight_variable(
                            name='filter',
                            shape=[self.filter_width, self.residual_channels, self.dilation_channels]
                        )
                        current['gate'] = create_conv_weight_variable(
                            name='gate',
                            shape=[self.filter_width, self.residual_channels, self.dilation_channels]
                        )
                        current['skip'] = create_conv_weight_variable(
                            name='skip',
                            shape=[1, self.dilation_channels, self.skip_channels],
                        )
                        current['dense'] = create_conv_weight_variable(
                            name='skip',
                            shape=[1, self.dilation_channels, self.residual_channels],
                        )
                        if self.use_bias:
                            current['filter_bias'] = create_bias_variable(
                                'filter_bias',
                                [self.dilation_channels],
                            )
                            current['gate_bias'] = create_bias_variable(
                                'gate_bias',
                                [self.dilation_channels],
                            )
                            current['dense_bias'] = create_bias_variable(
                                'dense_bias',
                                [self.residual_channels],
                            )
                            current['skip_bias'] = create_bias_variable(
                                'skip_bias',
                                [self.skip_channels],
                            )
                        dilated_stack.append(current)
            variables['dilated_stack'] = dilated_stack

            with tf.variable_scope('postprocessing'):
                current = dict()
                current['postprocess1'] = create_conv_weight_variable(
                    name='postprocess1',
                    shape=[1, self.skip_channels, self.skip_channels],
                )
                current['postprocess2'] = create_conv_weight_variable(
                    name='postprocess2',
                    shape=[1, self.skip_channels, self.output_channels],
                )
                if self.use_bias:
                    current['postprocess1_bias'] = create_bias_variable(
                        name='postprocess1_bias',
                        shape=[self.skip_channels],
                    )
                    current['postprocess2_bias'] = create_bias_variable(
                        name='postprocess2_bias',
                        shape=[self.output_channels],
                    )
                variables['postprocessing'] = current
        return variables

    def create_network(self, input_batch, histogram):
        current_layer = input_batch

        with tf.name_scope('causal_layer'):
            current_layer = self._create_causal_layer(current_layer)

        outputs = []
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    skipped_output, current_layer = self._create_dilation_layer(
                        current_layer, layer_index, dilation, histogram
                    )
                    outputs.append(skipped_output)
                    outputs_length = tf.shape(skipped_output)[1]

        with tf.name_scope('postprocessing'):
            # (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv 
            w1 = self.variables['postprocessing']['postprocess1']
            w2 = self.variables['postprocessing']['postprocess2']
            if self.use_bias:
                b1 = self.variables['postprocessing']['postprocess1_bias']
                b2 = self.variables['postprocessing']['postprocess2_bias']

            if histogram:
                tf.summary.histogram('postprocess1_weight', w1)
                tf.summary.histogram('postprocess2_weight', w2)
                if self.use_bias:
                    tf.summary.histogram('postprocess1_bias', b1)
                    tf.summary.histogram('postprocess2_bias', b2)

            total = sum(
                tf.slice(output, [0, tf.shape(output)[1] - outputs_length, 0], [-1, -1, -1])
                for output in outputs
            )

            transformed1 = tf.nn.relu(total)
            conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
            if self.use_bias:
                conv1 = tf.add(conv1, b1)
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
            if self.use_bias:
                conv2 = tf.add(conv2, b2)
        return conv2


class Declipper(object):
    def __init__(
            self, batch_size,
            input_quantization_channels=512,
            output_quantization_channels=512,
    ):
        self.wavenet = Wavenet(
            batch_size=batch_size,
            residual_channels=64,
            dilation_channels=32,
            skip_channels=128,
            dilation_num=10,
            input_channels=input_quantization_channels,
            output_channels=output_quantization_channels,
            name='declipper',
        )
        self.scale = 1.6
        self.batch_size = batch_size
        self.label_offset = self.wavenet.label_offset
        self.input_quantization_channels = input_quantization_channels
        self.output_quantization_channels = output_quantization_channels

    def encode_one_hot(self, input_batch, channels):
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(
                input_batch,
                depth=channels,
                dtype=tf.float32,
            )
            shape = [self.batch_size, -1, channels]
            encoded = tf.reshape(encoded, shape)
        return encoded

    def generate_input(self, input_batch, is_clip=True, zero_filler=False):
        with tf.name_scope('preprocessing'):
            scale = self.scale
            if is_clip:
                tf.transpose(tf.transpose(input_batch, [1, 2, 0]) * scale, [2, 0, 1])
                scaled_input = tf.transpose(input_batch, [1, 2, 0]) * scale
                input_batch = tf.transpose(
                    tf.clip_by_value(
                        scaled_input,
                        -1, 1) / scale,
                    [2, 0, 1]
                )
                is_clipped = tf.cast(
                    tf.transpose(tf.abs(scaled_input) > 1, [2, 0, 1]),
                    'float32',
                )
            else:
                is_clipped = tf.zeros_like(input_batch)
            encoded = quantize(input_batch, self.input_quantization_channels)
            network_input = self.encode_one_hot(encoded, self.input_quantization_channels)

            if zero_filler:
                network_input = network_input * (
                    (1 - tf.cast(
                        is_clipped,
                        'float'
                    ) * tf.ones((1, self.input_quantization_channels)))
                )
        tf.assert_equal(tf.shape(network_input), tf.shape(input_batch))
        tf.assert_equal(tf.shape(is_clipped), tf.shape(input_batch))
        return network_input, input_batch, is_clipped

    def generate_label(self, input_batch):
        encoded = quantize(input_batch, self.output_quantization_channels)
        one_hot_label = self.encode_one_hot(encoded, self.output_quantization_channels)
        return one_hot_label, input_batch

    def loss(self, input_batch, label, label_float,
            l2_regularization_strength=1e-6,
            mse_regularization_strength=None,
            summary_prefix='declipper_',
            summary_postfix='',
            histogram=False,
            sample_rate=44100,
    ):
        with tf.name_scope('declipper'):
            with tf.name_scope('preprocess'):
                length = tf.shape(input_batch)[1]
                input_batch = tf.slice(
                    input_batch,
                    [0, 0, 0],
                    [-1, length - 1, -1],
                )
                label = tf.slice(
                    label,
                    [0, self.wavenet.label_offset + 1, 0],
                    [-1, -1, -1],
                )
                label_float = tf.slice(
                    label_float,
                    [0, self.wavenet.label_offset + 1, 0],
                    [-1, -1, -1],
                )
            wavenet_output = self.wavenet.create_network(
                input_batch, histogram,
            )

            with tf.name_scope('predict'):
                prediction_idx = tf.argmax(wavenet_output, 2)
                label_idx = tf.argmax(label, 2)
                prediction_float = tf.cast(
                    prediction_idx,
                    "float",
                ) * 2 / (self.output_quantization_channels - 1) - 1

            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(
                    tf.cast(
                        tf.equal(
                            prediction_idx,
                            label_idx,
                        ),
                        "float",
                    )
                )
                tf.summary.scalar('{}accuracy{}'.format(summary_prefix, summary_postfix), accuracy)

            with tf.name_scope('loss'):
                with tf.name_scope('cross_entropy'):
                    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        wavenet_output,
                        label,
                    ))
                    tf.summary.scalar('cross_entropy', cross_entropy_loss)

                with tf.name_scope('MSE'):
                    mse_loss = tf.nn.l2_loss(prediction_float - label_float) \
                               / tf.cast(tf.reduce_prod(tf.shape(prediction_float)), 'float')
                    tf.summary.scalar('{}mse{}'.format(summary_prefix, summary_postfix), mse_loss)

                with tf.name_scope('l2'):
                    l2_loss = tf.add_n([
                        tf.nn.l2_loss(v)
                        for v in self.get_trainable_variables()
                        if 'bias' not in v.name
                        ])
                    tf.summary.scalar('{}l2{}'.format(summary_prefix, summary_postfix), l2_loss)

                total_loss = cross_entropy_loss
                if mse_regularization_strength is not None:
                    total_loss += mse_regularization_strength * mse_loss
                if l2_regularization_strength is not None:
                    total_loss += l2_regularization_strength * l2_loss

        return prediction_float, total_loss

    def get_trainable_variables(self):
        return [
            v for v in tf.trainable_variables()
            if v.name.startswith('declipper/')
            ]


class Discriminator(object):
    def __init__(
            self, batch_size,
            name='discriminator',
            input_quantization_channels=512,
            dilation_num=5,
            dilation_channels=16,
            residual_channels=32,
    ):
        self.batch_size = batch_size
        self.input_quantization_channels = input_quantization_channels
        self.wavenet_output_chan = 64
        self.wavenet = Wavenet(
            batch_size=batch_size,
            residual_channels=residual_channels,
            dilation_channels=dilation_channels,
            skip_channels=self.wavenet_output_chan,
            input_channels=input_quantization_channels,
            output_channels=1,
            dilation_num=dilation_num,
            name=name,
            filter_width=2,
        )
        self.name = name

    def get_trainable_variables(self):
        return [
            v for v in tf.trainable_variables()
            if v.name.startswith('{}/'.format(self.name))
            ]

    def encode_one_hot(self, input_batch, channels):
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(
                input_batch,
                depth=channels,
                dtype=tf.float32,
            )
            shape = [self.batch_size, -1, channels]
            encoded = tf.reshape(encoded, shape)
        return encoded

    def generate_input(self, input_batch):
        with tf.name_scope('preprocessing'):
            encoded = quantize(input_batch, self.input_quantization_channels)
            network_input = self.encode_one_hot(encoded, self.input_quantization_channels)
        return network_input

    def discriminate(self, input_batch, histogram=False):
        with tf.name_scope('discriminator'):
            wavenet_input = self.generate_input(input_batch)
            wavenet_output = self.wavenet.create_network(
                wavenet_input, histogram,
            )
            return wavenet_output


class GAN:
    def __init__(self, sess, batch_size, config,
            declipper_reconstruction_ratio=.8,
            sample_size=2 ** 16,
            logdir='./log',
            train_data_dir='./data', test_data_dir='./test_data'):
        self.sess = sess

        self.declipper = Declipper(batch_size)
        self.clip_discriminator = Discriminator(batch_size, name='clip_discriminator', dilation_num=5)
        self.fake_discriminator = Discriminator(batch_size, name='fake_discriminator', dilation_num=8)
        self.batch_size = batch_size
        self.declipper_reconstruction_ratio = declipper_reconstruction_ratio
        self.writer = tf.train.SummaryWriter(logdir, tf.get_default_graph())

        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir

        self.coord = tf.train.Coordinator()
        self.train_data_reader = MultipleAudioReader(
            train_data_dir, self.coord, 44100,
            audio_pattern='*.ogg', sample_size=sample_size,
        )
        self.create_network()
        self.saver = tf.train.Saver(max_to_keep=None)
        self.threads = tf.train.start_queue_runners(sess=self.sess, coord=self.coord)

        self.clip_discriminator_optimizer = tf.train.AdamOptimizer(
            learning_rate=config.discriminator_learning_rate,
            beta1=config.beta1,
        )
        self.clip_discriminator_optimize = self.clip_discriminator_optimizer.minimize(
            self.clip_discriminator_loss, var_list=self.clip_discriminator.get_trainable_variables())

        self.fake_discriminator_optimizer = tf.train.AdamOptimizer(
            learning_rate=config.discriminator_learning_rate,
            beta1=config.beta1,
        )
        self.fake_discriminator_optimize = self.fake_discriminator_optimizer.minimize(
            self.fake_discriminator_loss, var_list=self.fake_discriminator.get_trainable_variables())

        self.declipper_optimizer = tf.train.AdamOptimizer(
            learning_rate=config.declipper_learning_rate,
            beta1=config.beta1,
        )
        self.declipper_optimize = self.declipper_optimizer.minimize(
            self.declipper_loss, var_list=self.declipper.get_trainable_variables()
        )

        self.summary = tf.merge_all_summaries()

    def create_network(self, sample_rate=44100):
        batch_num = self.batch_size
        input_float = self.train_data_reader.dequeue(self.batch_size)
        length = tf.shape(input_float)[1]
        # float: B x L x 1

        declipper_input_raw, raw_float, raw_mask = self.declipper.generate_input(input_float, is_clip=False)
        declipper_input_clip, clip_float, clip_mask = self.declipper.generate_input(input_float, is_clip=True,
            zero_filler=True)
        declipper_label, declipper_label_float = self.declipper.generate_label(input_float)
        # one hot: B x L x output quantization size

        declipper_prediction_clip, declipper_loss_reconstruction = self.declipper.loss(
            declipper_input_clip,
            declipper_label,
            declipper_label_float,
            summary_prefix='declipper-clip/',
        )

        tf.summary.audio('clipped'.format(), clip_float, sample_rate)
        tf.summary.audio('label'.format(), declipper_label_float, sample_rate)
        tf.summary.audio('predict-clip'.format(), declipper_prediction_clip, sample_rate)

        clip_disciminator_pred_raw = self.clip_discriminator.discriminate(raw_float)
        clip_disciminator_pred_clip = self.clip_discriminator.discriminate(clip_float)
        clip_disciminator_pred_declipped = self.clip_discriminator.discriminate(
            declipper_prediction_clip
        )

        fake_discriminator_pred_raw = self.fake_discriminator.discriminate(raw_float)
        fake_discriminator_pred_declipped = self.fake_discriminator.discriminate(
            declipper_prediction_clip
        )

        clip_loss_raw = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            clip_disciminator_pred_raw,
            tf.zeros_like(clip_disciminator_pred_raw),
        ))
        clip_loss_clip = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            clip_disciminator_pred_clip,
            tf.slice(
                clip_mask,
                [0, length - tf.shape(clip_disciminator_pred_clip)[1], 0],
                [-1, tf.shape(clip_disciminator_pred_clip)[1], -1]
            ),
        ))
        declipper_loss_declip = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            clip_disciminator_pred_declipped,
            tf.zeros_like(clip_disciminator_pred_declipped),
        ))
        clip_discriminator_loss = clip_loss_raw + clip_loss_clip
        clip_acc = tf.reduce_mean(tf.cast(
            (
                2 * tf.sigmoid(clip_disciminator_pred_clip) - 1
            ) * (
                2 * tf.slice(
                    clip_mask,
                    [0, length - tf.shape(clip_disciminator_pred_clip)[1], 0],
                    [-1, tf.shape(clip_disciminator_pred_clip)[1], -1]
                ) - 1
            ) > 0, 'float'))
        clip_rate = tf.reduce_mean(clip_mask)

        fake_loss_raw = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            fake_discriminator_pred_raw,
            tf.ones_like(fake_discriminator_pred_raw)
        ))
        fake_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            fake_discriminator_pred_declipped,
            tf.zeros_like(fake_discriminator_pred_declipped)
        ))
        fake_discriminator_loss = fake_loss_raw + fake_loss_fake

        declipper_loss_deceive = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            fake_discriminator_pred_declipped,
            tf.ones_like(fake_discriminator_pred_declipped)
        ))

        fake_acc = 0.5 * tf.reduce_mean(tf.cast(tf.sigmoid(fake_discriminator_pred_raw) > 0.5, 'float')) \
                   + 0.5 * tf.reduce_mean(tf.cast(tf.sigmoid(fake_discriminator_pred_declipped) < 0.5, 'float'))

        tf.summary.scalar('ClipDiscriminator_loss_raw', clip_loss_raw)
        tf.summary.scalar('ClipDiscriminator_loss_clip', clip_loss_clip)
        tf.summary.scalar('ClipDiscriminator_loss_total', clip_discriminator_loss)
        tf.summary.scalar('ClipDiscriminator_acc', clip_acc)
        tf.summary.scalar('ClipDiscriminator_clip_rate', clip_rate)

        tf.summary.scalar('FakeDiscriminator_loss_raw', fake_loss_raw)
        tf.summary.scalar('FakeDiscriminator_loss_fake', fake_loss_fake)
        tf.summary.scalar('FakeDiscriminator_loss_total', fake_discriminator_loss)
        tf.summary.scalar('FakeDiscriminator_acc', fake_acc)

        declipper_loss = (self.declipper_reconstruction_ratio) * declipper_loss_reconstruction \
                         + (1 - self.declipper_reconstruction_ratio) * (
            declipper_loss_deceive + declipper_loss_declip
        )

        tf.summary.scalar('declipper_loss_recostruction', declipper_loss_reconstruction)
        tf.summary.scalar('declipper_loss_deceive', declipper_loss_deceive)
        tf.summary.scalar('declipper_loss_declip', declipper_loss_declip)
        tf.summary.scalar('declipper_loss_total', declipper_loss)

        self.declipper_reconstruction_loss = declipper_loss_reconstruction
        self.declipper_loss = declipper_loss
        self.clip_discriminator_loss = clip_discriminator_loss
        self.fake_discriminator_loss = fake_discriminator_loss
        self.clipped_input = clip_float
        self.declipper_label_float = declipper_label_float
        self.declipper_prediction_clip = declipper_prediction_clip

    def train(self, config):
        self.train_data_reader.start_threads(self.sess)
        start_iteration = None
        if config.resume:
            start_iteration = self.load_checkpoint()
        if start_iteration is None:
            start_iteration = 0

        for iteration in range(start_iteration, config.num_iter):
            start_time = time.time()
            clip_loss, fake_loss, dec_loss, _, _, _, summary_value = self.sess.run([
                self.clip_discriminator_loss, self.fake_discriminator_loss, self.declipper_loss,
                self.clip_discriminator_optimize, self.fake_discriminator_optimize, self.declipper_optimize,
                self.summary
            ])
            self.writer.add_summary(summary_value, iteration)
            print(
                'iteration: {iter} {sec:.3f}s/iter dec loss: {dec:.4f} clip loss : {clip:.4f} fake loss: {fake:.4f}'.format(
                    iter=iteration,
                    sec=time.time() - start_time,
                    dec=dec_loss,
                    clip=clip_loss,
                    fake=fake_loss,
                ))

            if iteration % 200 == 0:
                label_shape = tf.shape(self.declipper_label_float)
                resized_label = tf.slice(
                    self.declipper_label_float,
                    [0, label_shape[1] - tf.shape(self.declipper_prediction_clip)[1], 0],
                    [-1, -1, -1]
                )
                clipped, label, pred_clip, = self.sess.run([
                    self.clipped_input, resized_label,
                    self.declipper_prediction_clip,
                ])
                self.save_audio('clipped', iteration, clipped, config.sample_dir)
                self.save_audio('label', iteration, label, config.sample_dir)
                self.save_audio('pred_clip', iteration, pred_clip, config.sample_dir)

            if iteration % 1000 == 0:
                self.save(config.checkpoint_dir, iteration)

    def save(self, checkpoint_dir, step):
        model_name = 'gan-declip.model'
        path = os.path.join(checkpoint_dir, model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess, path, global_step=step)

    def save_audio(self, name, step, batch_data, sample_dir):
        for batch_idx, data in enumerate(batch_data):
            filename = '{step:08d}-{batch_idx:03d}-{name}.flac'.format(
                step=step, batch_idx=batch_idx, name=name)
            path = os.path.join(sample_dir, filename)
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)
            sf.write(path, data * 0.9, 44100)

    def load_checkpoint(self):
        ckpt = tf.train.get_checkpoint_state("result/checkpoint")
        if ckpt and ckpt.model_checkpoint_path:
            path = ckpt.model_checkpoint_path
            print('loading checkpoint : {}'.format(path))
            self.saver.restore(self.sess, path)
            iteration = int(path.split('-')[-1])
            return iteration
