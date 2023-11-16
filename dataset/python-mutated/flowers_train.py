"""A binary to train Inception on the flowers data set.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from inception import inception_train
from inception.flowers_data import FlowersData
FLAGS = tf.app.flags.FLAGS

def main(_):
    if False:
        return 10
    dataset = FlowersData(subset=FLAGS.subset)
    assert dataset.data_files()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    inception_train.train(dataset)
if __name__ == '__main__':
    tf.app.run()