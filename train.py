import os

import tensorflow as tf
from reader import MultipleAudioReader
from model import Trainer


def main():
    with tf.Session() as sess:
        trainer = Trainer(input_channels=512, output_channels=512)
        coord = tf.train.Coordinator()
        reader = MultipleAudioReader('./data', coord, sample_rate=44100, sample_size=2 ** 13)
        batch_num = 1

        input_batch_f = reader.dequeue_many(batch_num)
        trainer.create_network(input_batch_f)
        op_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('result/log', graph=tf.get_default_graph())
        tf.global_variables_initializer().run()
        
        reader.start_threads(sess, 8)
            
        for iteration in range(10000000):
            loss_future, loss_past, _, summary = sess.run(
                [
                    trainer.loss_future,
                    trainer.loss_past,
                    trainer.op_train,
                    op_summary,
                ]
            )
            print(
                'iter:{iter:08d} loss future:{future:04f} past:{past:04f}'.format(
                    iter=iteration,
                    future=loss_future,
                    past=loss_past,
                )
            )
            writer.add_summary(summary, iteration)
            if iteration % 10000 == 0:
                saver = tf.train.Saver()
                if not os.path.exists('result/snapshot/'):
                    os.makedirs('result/snapshot/', exist_ok=True)
                saver.save(sess, 'result/snapshot/{:08d}.data'.format(iteration))
        pass


if __name__ == '__main__':
    main()
