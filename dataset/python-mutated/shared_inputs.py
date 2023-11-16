"""Placeholders for non-task-specific model inputs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

class Inputs(object):

    def __init__(self, config):
        if False:
            print('Hello World!')
        self._config = config
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.label_smoothing = tf.placeholder(tf.float32, name='label_smoothing')
        self.lengths = tf.placeholder(tf.int32, shape=[None], name='lengths')
        self.mask = tf.placeholder(tf.float32, [None, None], name='mask')
        self.words = tf.placeholder(tf.int32, shape=[None, None], name='words')
        self.chars = tf.placeholder(tf.int32, shape=[None, None, None], name='chars')

    def create_feed_dict(self, mb, is_training):
        if False:
            return 10
        cvt = mb.task_name == 'unlabeled'
        return {self.keep_prob: 1.0 if not is_training else self._config.unlabeled_keep_prob if cvt else self._config.labeled_keep_prob, self.label_smoothing: self._config.label_smoothing if is_training and (not cvt) else 0.0, self.lengths: mb.lengths, self.words: mb.words, self.chars: mb.chars, self.mask: mb.mask.astype('float32')}