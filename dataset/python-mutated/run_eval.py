"""Script for evaluating a UVF agent.

To run locally: See scripts/local_eval.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import gin.tf
import eval as eval_
flags = tf.app.flags
FLAGS = flags.FLAGS

def main(_):
    if False:
        i = 10
        return i + 15
    tf.logging.set_verbosity(tf.logging.INFO)
    assert FLAGS.checkpoint_dir, "Flag 'checkpoint_dir' must be set."
    assert FLAGS.eval_dir, "Flag 'eval_dir' must be set."
    if FLAGS.config_file:
        for config_file in FLAGS.config_file:
            gin.parse_config_file(config_file)
    if FLAGS.params:
        gin.parse_config(FLAGS.params)
    eval_.evaluate(FLAGS.checkpoint_dir, FLAGS.eval_dir)
if __name__ == '__main__':
    tf.app.run()