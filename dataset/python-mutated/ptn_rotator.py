"""Creates rotator network model.

This model performs the out-of-plane rotations given input image and action.
The action is either no-op, rotate clockwise or rotate counter-clockwise.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def bilinear(input_x, input_y, output_size):
    if False:
        return 10
    'Define the bilinear transformation layer.'
    shape_x = input_x.get_shape().as_list()
    shape_y = input_y.get_shape().as_list()
    weights_initializer = tf.truncated_normal_initializer(stddev=0.02, seed=1)
    biases_initializer = tf.constant_initializer(0.0)
    matrix = tf.get_variable('Matrix', [shape_x[1], shape_y[1], output_size], tf.float32, initializer=weights_initializer)
    bias = tf.get_variable('Bias', [output_size], initializer=biases_initializer)
    tf.contrib.framework.add_model_variable(matrix)
    tf.contrib.framework.add_model_variable(bias)
    h0 = tf.matmul(input_x, tf.reshape(matrix, [shape_x[1], shape_y[1] * output_size]))
    h0 = tf.reshape(h0, [-1, shape_y[1], output_size])
    h1 = tf.tile(tf.reshape(input_y, [-1, shape_y[1], 1]), [1, 1, output_size])
    h1 = tf.multiply(h0, h1)
    return tf.reduce_sum(h1, 1) + bias

def model(poses, actions, params, is_training):
    if False:
        for i in range(10):
            print('nop')
    'Model for performing rotation.'
    del is_training
    return bilinear(poses, actions, params.z_dim)