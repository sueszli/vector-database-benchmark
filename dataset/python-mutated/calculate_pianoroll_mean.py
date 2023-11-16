"""Script to calculate the mean of a pianoroll dataset.

Given a pianoroll pickle file, this script loads the dataset and
calculates the mean of the training set. Then it updates the pickle file
so that the key "train_mean" points to the mean vector.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pickle
import numpy as np
import tensorflow as tf
from datasets import sparse_pianoroll_to_dense
tf.app.flags.DEFINE_string('in_file', None, 'Filename of the pickled pianoroll dataset to load.')
tf.app.flags.DEFINE_string('out_file', None, 'Name of the output pickle file. Defaults to in_file, updating the input pickle file.')
tf.app.flags.mark_flag_as_required('in_file')
FLAGS = tf.app.flags.FLAGS
MIN_NOTE = 21
MAX_NOTE = 108
NUM_NOTES = MAX_NOTE - MIN_NOTE + 1

def main(unused_argv):
    if False:
        return 10
    if FLAGS.out_file is None:
        FLAGS.out_file = FLAGS.in_file
    with tf.gfile.Open(FLAGS.in_file, 'r') as f:
        pianorolls = pickle.load(f)
    dense_pianorolls = [sparse_pianoroll_to_dense(p, MIN_NOTE, NUM_NOTES)[0] for p in pianorolls['train']]
    concatenated = np.concatenate(dense_pianorolls, axis=0)
    mean = np.mean(concatenated, axis=0)
    pianorolls['train_mean'] = mean
    pickle.dump(pianorolls, open(FLAGS.out_file, 'wb'))
if __name__ == '__main__':
    tf.app.run()