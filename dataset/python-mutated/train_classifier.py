"""Trains LSTM text classification model.

Model trains with adversarial or virtual adversarial training.

Computational time:
  1.8 hours to train 10000 steps without adversarial or virtual adversarial
    training, on 1 layer 1024 hidden units LSTM, 256 embeddings, 400 truncated
    BP, 64 minibatch and on single GPU (Pascal Titan X, cuDNNv5).

  4 hours to train 10000 steps with adversarial or virtual adversarial
    training, with above condition.

To initialize embedding and LSTM cell weights from a pretrained model, set
FLAGS.pretrained_model_dir to the pretrained model's checkpoint directory.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import graphs
import train_utils
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('pretrained_model_dir', None, 'Directory path to pretrained model to restore from')

def main(_):
    if False:
        print('Hello World!')
    'Trains LSTM classification model.'
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
        model = graphs.get_model()
        (train_op, loss, global_step) = model.classifier_training()
        train_utils.run_training(train_op, loss, global_step, variables_to_restore=model.pretrained_variables, pretrained_model_dir=FLAGS.pretrained_model_dir)
if __name__ == '__main__':
    tf.app.run()