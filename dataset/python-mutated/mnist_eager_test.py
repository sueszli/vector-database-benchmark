from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest
import tensorflow as tf
from tensorflow.python import eager as tfe
from official.r1.mnist import mnist
from official.r1.mnist import mnist_eager
from official.utils.misc import keras_utils

def device():
    if False:
        print('Hello World!')
    return '/device:GPU:0' if tfe.context.num_gpus() else '/device:CPU:0'

def data_format():
    if False:
        return 10
    return 'channels_first' if tfe.context.num_gpus() else 'channels_last'

def random_dataset():
    if False:
        i = 10
        return i + 15
    batch_size = 64
    images = tf.random_normal([batch_size, 784])
    labels = tf.random_uniform([batch_size], minval=0, maxval=10, dtype=tf.int32)
    return tf.data.Dataset.from_tensors((images, labels))

def train(defun=False):
    if False:
        i = 10
        return i + 15
    model = mnist.create_model(data_format())
    if defun:
        model.call = tf.function(model.call)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    dataset = random_dataset()
    with tf.device(device()):
        mnist_eager.train(model, optimizer, dataset, step_counter=tf.train.get_or_create_global_step())

def evaluate(defun=False):
    if False:
        for i in range(10):
            print('nop')
    model = mnist.create_model(data_format())
    dataset = random_dataset()
    if defun:
        model.call = tf.function(model.call)
    with tf.device(device()):
        mnist_eager.test(model, dataset)

class MNISTTest(tf.test.TestCase):
    """Run tests for MNIST eager loop.

  MNIST eager uses contrib and will not work with TF 2.0.  All tests are
  disabled if using TF 2.0.
  """

    def setUp(self):
        if False:
            return 10
        if not keras_utils.is_v2_0():
            tf.compat.v1.enable_v2_behavior()
        super(MNISTTest, self).setUp()

    @unittest.skipIf(keras_utils.is_v2_0(), 'TF 1.0 only test.')
    def test_train(self):
        if False:
            while True:
                i = 10
        train(defun=False)

    @unittest.skipIf(keras_utils.is_v2_0(), 'TF 1.0 only test.')
    def test_evaluate(self):
        if False:
            print('Hello World!')
        evaluate(defun=False)

    @unittest.skipIf(keras_utils.is_v2_0(), 'TF 1.0 only test.')
    def test_train_with_defun(self):
        if False:
            i = 10
            return i + 15
        train(defun=True)

    @unittest.skipIf(keras_utils.is_v2_0(), 'TF 1.0 only test.')
    def test_evaluate_with_defun(self):
        if False:
            while True:
                i = 10
        evaluate(defun=True)
if __name__ == '__main__':
    tf.test.main()