"""Contains a model definition for AlexNet.

This work was first described in:
  ImageNet Classification with Deep Convolutional Neural Networks
  Alex Krizhevsky, Ilya Sutskever and Geoffrey E. Hinton

and later refined in:
  One weird trick for parallelizing convolutional neural networks
  Alex Krizhevsky, 2014

Here we provide the implementation proposed in "One weird trick" and not
"ImageNet Classification", as per the paper, the LRN layers have been removed.

Usage:
  with slim.arg_scope(alexnet.alexnet_v2_arg_scope()):
    outputs, end_points = alexnet.alexnet_v2(inputs)

@@alexnet_v2
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def alexnet_v1_arg_scope(weight_decay=0.0005):
    if False:
        return 10
    with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu, biases_initializer=tf.constant_initializer(0.1), weights_regularizer=slim.l2_regularizer(weight_decay), data_format='NCHW'):
        with slim.arg_scope([slim.conv2d], padding='SAME', data_format='NCHW'):
            with slim.arg_scope([slim.max_pool2d], padding='VALID', data_format='NCHW') as arg_sc:
                return arg_sc

def alexnet_v1(inputs, num_classes=1000, is_training=True, dropout_keep_prob=0.5, spatial_squeeze=False, scope='alexnet_v2'):
    if False:
        while True:
            i = 10
    'AlexNet version 2.\n\n  Described in: http://arxiv.org/pdf/1404.5997v2.pdf\n  Parameters from:\n  github.com/akrizhevsky/cuda-convnet2/blob/master/layers/\n  layers-imagenet-1gpu.cfg\n\n  Note: All the fully_connected layers have been transformed to conv2d layers.\n        To use in classification mode, resize input to 224x224. To use in fully\n        convolutional mode, set spatial_squeeze to false.\n        The LRN layers have been removed and change the initializers from\n        random_normal_initializer to xavier_initializer.\n\n  Args:\n    inputs: a tensor of size [batch_size, height, width, channels].\n    num_classes: number of predicted classes.\n    is_training: whether or not the model is being trained.\n    dropout_keep_prob: the probability that activations are kept in the dropout\n      layers during training.\n    spatial_squeeze: whether or not should squeeze the spatial dimensions of the\n      outputs. Useful to remove unnecessary dimensions for classification.\n    scope: Optional scope for the variables.\n\n  Returns:\n    the last op containing the log predictions and end_points dict.\n  '
    with tf.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], outputs_collections=[end_points_collection], data_format='NCHW'):
            net = slim.conv2d(inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool1')
            net = slim.conv2d(net, 192, [5, 5], scope='conv2')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = slim.conv2d(net, 384, [3, 3], scope='conv3')
            net = slim.conv2d(net, 384, [3, 3], scope='conv4')
            net = slim.conv2d(net, 256, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool5')
            with slim.arg_scope([slim.conv2d], weights_initializer=trunc_normal(0.005), biases_initializer=tf.constant_initializer(0.1), data_format='NCHW'):
                net = slim.flatten(net, [-1, 6 * 6 * 256])
                net = slim.legacy_fully_connected(net, 4096, activation_fn=tf.nn.relu)
                net = slim.legacy_fully_connected(net, 4096, activation_fn=tf.nn.relu)
                net = slim.legacy_fully_connected(net, num_classes, activation_fn=tf.nn.relu)
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if spatial_squeeze:
                net = tf.squeeze(net, [2, 3], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
            return (net, end_points)