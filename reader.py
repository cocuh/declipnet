import os
import fnmatch
import random
import itertools as it
import threading

import soundfile as sf
import numpy as np
import tensorflow as tf


def find_files(directory, patterns):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for pattern in patterns:
            for filename in fnmatch.filter(filenames, pattern):
                files.append(os.path.join(root, filename))
    return files


class MultipleAudioReader:
    def __init__(self,
                 audio_dir: str,
                 coord: tf.train.Coordinator,
                 sample_rate: int,
                 audio_patterns=None,
                 sample_size=2 ** 16,
                 queue_size=32,
                 ):
        if audio_patterns is None:
            audio_patterns = ['*.ogg']
        self.audio_dir = audio_dir
        self.coord = coord
        self.audio_patterns = audio_patterns
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.queue_size = queue_size
        self.sample_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=(sample_size,),
            name='sample',
        )
        self.queue = tf.PaddingFIFOQueue(
            queue_size,
            [tf.float32],
            shapes=[(sample_size,)]
        )
        self.enqueue = self.queue.enqueue([self.sample_placeholder])
        self.threads = []
        if len(find_files(self.audio_dir, self.audio_patterns)) == 0:
            raise ValueError('file not found')

    def dequeue_many(self, num_elements: int):
        output = self.queue.dequeue_many(num_elements)
        return output

    def dequeue(self):
        output = self.queue.dequeue()
        return output

    def start_threads(self, sess: tf.Session, thread_num=2):
        for _ in range(thread_num):
            thread = threading.Thread(target=self._worker, kwargs={'sess': sess})
            thread.daemon = True  # thread will close when parent process quits
            thread.start()
            self.threads.append(thread)
        return self.threads

    def _worker(self, sess: tf.Session):
        should_stop = False
        while not should_stop:
            filepaths = find_files(self.audio_dir, self.audio_patterns)
            random.shuffle(filepaths)
            file_generator = it.chain.from_iterable(
                self._load_sound(filepath)
                for filepath in filepaths
            )
            buffer = np.array([])
            for data, filepath in file_generator:
                print('reading: {}'.format(filepath))
                if self.coord.should_stop():
                    should_stop = True
                    break
                if self.sample_size:
                    buffer = np.append(buffer, data)
                    while len(buffer) > self.sample_size:
                        piece = buffer[:self.sample_size]
                        buffer = buffer[self.sample_size:]
                        if np.abs(piece).max() > 1.:
                            piece = piece / np.abs(piece).max()
                        sess.run(
                            self.enqueue,
                            feed_dict={
                                self.sample_placeholder: piece,
                            }
                        )
                else:
                    sess.run(
                        self.enqueue,
                        feed_dict={
                            self.sample_placeholder: data,
                        }
                    )

    @staticmethod
    def _load_sound(filepath):
        try:
            audio, sr = sf.read(filepath)
            yield audio.T[0], filepath
            yield audio.T[1], filepath
        except:
            pass


def main():
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        sample_size = 2 ** 16
        reader = MultipleAudioReader('./data', coord, 44100, sample_size=sample_size)
        reader.start_threads(sess)
        dequeue = reader.dequeue_many(10)
        for i in range(1000):
            print('loop:{}'.format(i))
            result = sess.run(dequeue)
            print(result.shape)
            assert result.shape[1] == sample_size
            print(result)
        coord.request_stop()


if __name__ == '__main__':
    main()
