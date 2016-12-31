import os

import tensorflow as tf
from model import GAN

flags = tf.app.flags
flags.DEFINE_integer('num_iter', 1000000, '')
flags.DEFINE_integer('sample_size', 2 ** 15, '')
flags.DEFINE_integer('batch_size', 1, '')
flags.DEFINE_float('declipper_learning_rate', 0.002, '')
flags.DEFINE_float('discriminator_learning_rate', 0.001, '')
flags.DEFINE_float('beta1', 0.5, '')
flags.DEFINE_string('checkpoint_dir', 'result/checkpoint', '')
flags.DEFINE_string('sample_dir', 'result/sample', '')
flags.DEFINE_string('log_dir', 'result/log', '')
flags.DEFINE_boolean('is_train', True, '')
flags.DEFINE_boolean('histogram', False, '')
flags.DEFINE_boolean('resume', True, '')

FLAGS = flags.FLAGS


def main(_):
    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    with tf.Session() as sess:
        gan = GAN(
            sess,
            config=FLAGS,
            batch_size=FLAGS.batch_size,
            sample_size=FLAGS.sample_size,
            logdir=FLAGS.log_dir,
        )
        tf.initialize_all_variables().run()

        gan.train(FLAGS)
        global gan


if __name__ == '__main__':
    tf.app.run()
