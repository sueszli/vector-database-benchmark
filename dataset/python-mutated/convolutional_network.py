""" Convolutional Neural Network.

Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

This example is using TensorFlow layers API, see 'convolutional_network_raw' 
example for a raw implementation with variables.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=False)
import tensorflow as tf
learning_rate = 0.001
num_steps = 2000
batch_size = 128
num_input = 784
num_classes = 10
dropout = 0.25

def conv_net(x_dict, n_classes, dropout, reuse, is_training):
    if False:
        print('Hello World!')
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = x_dict['images']
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, 1024)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        out = tf.layers.dense(fc1, n_classes)
    return out

def model_fn(features, labels, mode):
    if False:
        return 10
    logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=True)
    logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=False)
    pred_classes = tf.argmax(logits_test, axis=1)
    pred_probas = tf.nn.softmax(logits_test)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
    estim_specs = tf.estimator.EstimatorSpec(mode=mode, predictions=pred_classes, loss=loss_op, train_op=train_op, eval_metric_ops={'accuracy': acc_op})
    return estim_specs
model = tf.estimator.Estimator(model_fn)
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': mnist.train.images}, y=mnist.train.labels, batch_size=batch_size, num_epochs=None, shuffle=True)
model.train(input_fn, steps=num_steps)
input_fn = tf.estimator.inputs.numpy_input_fn(x={'images': mnist.test.images}, y=mnist.test.labels, batch_size=batch_size, shuffle=False)
e = model.evaluate(input_fn)
print('Testing Accuracy:', e['accuracy'])