"""Contains a variant of the CIFAR-10 model definition."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
slim = contrib_slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)

def cifarnet(images, num_classes=10, is_training=False, dropout_keep_prob=0.5, prediction_fn=slim.softmax, scope='CifarNet'):
    if False:
        return 10
    "Creates a variant of the CifarNet model.\n\n  Note that since the output is a set of 'logits', the values fall in the\n  interval of (-infinity, infinity). Consequently, to convert the outputs to a\n  probability distribution over the characters, one will need to convert them\n  using the softmax function:\n\n        logits = cifarnet.cifarnet(images, is_training=False)\n        probabilities = tf.nn.softmax(logits)\n        predictions = tf.argmax(logits, 1)\n\n  Args:\n    images: A batch of `Tensors` of size [batch_size, height, width, channels].\n    num_classes: the number of classes in the dataset. If 0 or None, the logits\n      layer is omitted and the input features to the logits layer are returned\n      instead.\n    is_training: specifies whether or not we're currently training the model.\n      This variable will determine the behaviour of the dropout layer.\n    dropout_keep_prob: the percentage of activation values that are retained.\n    prediction_fn: a function to get predictions out of logits.\n    scope: Optional variable_scope.\n\n  Returns:\n    net: a 2D Tensor with the logits (pre-softmax activations) if num_classes\n      is a non-zero integer, or the input to the logits layer if num_classes\n      is 0 or None.\n    end_points: a dictionary from components of the network to the corresponding\n      activation.\n  "
    end_points = {}
    with tf.variable_scope(scope, 'CifarNet', [images]):
        net = slim.conv2d(images, 64, [5, 5], scope='conv1')
        end_points['conv1'] = net
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
        end_points['pool1'] = net
        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        end_points['conv2'] = net
        net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
        end_points['pool2'] = net
        net = slim.flatten(net)
        end_points['Flatten'] = net
        net = slim.fully_connected(net, 384, scope='fc3')
        end_points['fc3'] = net
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout3')
        net = slim.fully_connected(net, 192, scope='fc4')
        end_points['fc4'] = net
        if not num_classes:
            return (net, end_points)
        logits = slim.fully_connected(net, num_classes, biases_initializer=tf.zeros_initializer(), weights_initializer=trunc_normal(1 / 192.0), weights_regularizer=None, activation_fn=None, scope='logits')
        end_points['Logits'] = logits
        end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return (logits, end_points)
cifarnet.default_image_size = 32

def cifarnet_arg_scope(weight_decay=0.004):
    if False:
        return 10
    'Defines the default cifarnet argument scope.\n\n  Args:\n    weight_decay: The weight decay to use for regularizing the model.\n\n  Returns:\n    An `arg_scope` to use for the inception v3 model.\n  '
    with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(stddev=0.05), activation_fn=tf.nn.relu):
        with slim.arg_scope([slim.fully_connected], biases_initializer=tf.constant_initializer(0.1), weights_initializer=trunc_normal(0.04), weights_regularizer=slim.l2_regularizer(weight_decay), activation_fn=tf.nn.relu) as sc:
            return sc