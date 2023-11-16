"""A binary to evaluate Inception on the flowers data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from inception import inception_eval
from inception.flowers_data import FlowersData
FLAGS = tf.app.flags.FLAGS

def main(unused_argv=None):
    if False:
        i = 10
        return i + 15
    dataset = FlowersData(subset=FLAGS.subset)
    assert dataset.data_files()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    inception_eval.evaluate(dataset)
if __name__ == '__main__':
    tf.app.run()