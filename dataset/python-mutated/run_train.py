"""Script for training an RL agent using the UVF algorithm.

To run locally: See scripts/local_train.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import gin.tf
import train
flags = tf.app.flags
FLAGS = flags.FLAGS

def main(_):
    if False:
        i = 10
        return i + 15
    tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.config_file:
        for config_file in FLAGS.config_file:
            gin.parse_config_file(config_file)
    if FLAGS.params:
        gin.parse_config(FLAGS.params)
    assert FLAGS.train_dir, "Flag 'train_dir' must be set."
    return train.train_uvf(FLAGS.train_dir)
if __name__ == '__main__':
    tf.app.run()