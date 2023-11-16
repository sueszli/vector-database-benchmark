"""Pretrains a recurrent language model.

Computational time:
  2 days to train 100000 steps on 1 layer 1024 hidden units LSTM,
  256 embeddings, 400 truncated BP, 256 minibatch and on single GPU (Pascal
  Titan X, cuDNNv5).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import graphs
import train_utils
FLAGS = tf.app.flags.FLAGS

def main(_):
    if False:
        i = 10
        return i + 15
    'Trains Language Model.'
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
        model = graphs.get_model()
        (train_op, loss, global_step) = model.language_model_training()
        train_utils.run_training(train_op, loss, global_step)
if __name__ == '__main__':
    tf.app.run()