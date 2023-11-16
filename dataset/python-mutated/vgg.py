"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(inputs)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(inputs)

@@vgg_a
@@vgg_16
@@vgg_19
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
slim = contrib_slim

def vgg_arg_scope(weight_decay=0.0005):
    if False:
        print('Hello World!')
    'Defines the VGG arg scope.\n\n  Args:\n    weight_decay: The l2 regularization coefficient.\n\n  Returns:\n    An arg_scope.\n  '
    with slim.arg_scope([slim.conv2d, slim.fully_connected], activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(weight_decay), biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
            return arg_sc

def vgg_a(inputs, num_classes=1000, is_training=True, dropout_keep_prob=0.5, spatial_squeeze=True, reuse=None, scope='vgg_a', fc_conv_padding='VALID', global_pool=False):
    if False:
        i = 10
        return i + 15
    "Oxford Net VGG 11-Layers version A Example.\n\n  Note: All the fully_connected layers have been transformed to conv2d layers.\n        To use in classification mode, resize input to 224x224.\n\n  Args:\n    inputs: a tensor of size [batch_size, height, width, channels].\n    num_classes: number of predicted classes. If 0 or None, the logits layer is\n      omitted and the input features to the logits layer are returned instead.\n    is_training: whether or not the model is being trained.\n    dropout_keep_prob: the probability that activations are kept in the dropout\n      layers during training.\n    spatial_squeeze: whether or not should squeeze the spatial dimensions of the\n      outputs. Useful to remove unnecessary dimensions for classification.\n    reuse: whether or not the network and its variables should be reused. To be\n      able to reuse 'scope' must be given.\n    scope: Optional scope for the variables.\n    fc_conv_padding: the type of padding to use for the fully connected layer\n      that is implemented as a convolutional layer. Use 'SAME' padding if you\n      are applying the network in a fully convolutional manner and want to\n      get a prediction map downsampled by a factor of 32 as an output.\n      Otherwise, the output prediction map will be (input / 32) - 6 in case of\n      'VALID' padding.\n    global_pool: Optional boolean flag. If True, the input to the classification\n      layer is avgpooled to size 1x1, for any input size. (This is not part\n      of the original VGG architecture.)\n\n  Returns:\n    net: the output of the logits layer (if num_classes is a non-zero integer),\n      or the input to the logits layer (if num_classes is 0 or None).\n    end_points: a dict of tensors with intermediate activations.\n  "
    with tf.variable_scope(scope, 'vgg_a', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.max_pool2d], outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 1, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if global_pool:
                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                end_points['global_pool'] = net
            if num_classes:
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
            return (net, end_points)
vgg_a.default_image_size = 224

def vgg_16(inputs, num_classes=1000, is_training=True, dropout_keep_prob=0.5, spatial_squeeze=True, reuse=None, scope='vgg_16', fc_conv_padding='VALID', global_pool=False):
    if False:
        print('Hello World!')
    "Oxford Net VGG 16-Layers version D Example.\n\n  Note: All the fully_connected layers have been transformed to conv2d layers.\n        To use in classification mode, resize input to 224x224.\n\n  Args:\n    inputs: a tensor of size [batch_size, height, width, channels].\n    num_classes: number of predicted classes. If 0 or None, the logits layer is\n      omitted and the input features to the logits layer are returned instead.\n    is_training: whether or not the model is being trained.\n    dropout_keep_prob: the probability that activations are kept in the dropout\n      layers during training.\n    spatial_squeeze: whether or not should squeeze the spatial dimensions of the\n      outputs. Useful to remove unnecessary dimensions for classification.\n    reuse: whether or not the network and its variables should be reused. To be\n      able to reuse 'scope' must be given.\n    scope: Optional scope for the variables.\n    fc_conv_padding: the type of padding to use for the fully connected layer\n      that is implemented as a convolutional layer. Use 'SAME' padding if you\n      are applying the network in a fully convolutional manner and want to\n      get a prediction map downsampled by a factor of 32 as an output.\n      Otherwise, the output prediction map will be (input / 32) - 6 in case of\n      'VALID' padding.\n    global_pool: Optional boolean flag. If True, the input to the classification\n      layer is avgpooled to size 1x1, for any input size. (This is not part\n      of the original VGG architecture.)\n\n  Returns:\n    net: the output of the logits layer (if num_classes is a non-zero integer),\n      or the input to the logits layer (if num_classes is 0 or None).\n    end_points: a dict of tensors with intermediate activations.\n  "
    with tf.variable_scope(scope, 'vgg_16', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d], outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if global_pool:
                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                end_points['global_pool'] = net
            if num_classes:
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
            return (net, end_points)
vgg_16.default_image_size = 224

def vgg_19(inputs, num_classes=1000, is_training=True, dropout_keep_prob=0.5, spatial_squeeze=True, reuse=None, scope='vgg_19', fc_conv_padding='VALID', global_pool=False):
    if False:
        for i in range(10):
            print('nop')
    "Oxford Net VGG 19-Layers version E Example.\n\n  Note: All the fully_connected layers have been transformed to conv2d layers.\n        To use in classification mode, resize input to 224x224.\n\n  Args:\n    inputs: a tensor of size [batch_size, height, width, channels].\n    num_classes: number of predicted classes. If 0 or None, the logits layer is\n      omitted and the input features to the logits layer are returned instead.\n    is_training: whether or not the model is being trained.\n    dropout_keep_prob: the probability that activations are kept in the dropout\n      layers during training.\n    spatial_squeeze: whether or not should squeeze the spatial dimensions of the\n      outputs. Useful to remove unnecessary dimensions for classification.\n    reuse: whether or not the network and its variables should be reused. To be\n      able to reuse 'scope' must be given.\n    scope: Optional scope for the variables.\n    fc_conv_padding: the type of padding to use for the fully connected layer\n      that is implemented as a convolutional layer. Use 'SAME' padding if you\n      are applying the network in a fully convolutional manner and want to\n      get a prediction map downsampled by a factor of 32 as an output.\n      Otherwise, the output prediction map will be (input / 32) - 6 in case of\n      'VALID' padding.\n    global_pool: Optional boolean flag. If True, the input to the classification\n      layer is avgpooled to size 1x1, for any input size. (This is not part\n      of the original VGG architecture.)\n\n  Returns:\n    net: the output of the logits layer (if num_classes is a non-zero integer),\n      or the non-dropped-out input to the logits layer (if num_classes is 0 or\n      None).\n    end_points: a dict of tensors with intermediate activations.\n  "
    with tf.variable_scope(scope, 'vgg_19', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d], outputs_collections=end_points_collection):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)
            if global_pool:
                net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                end_points['global_pool'] = net
            if num_classes:
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='dropout7')
                net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='fc8')
                if spatial_squeeze:
                    net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
                end_points[sc.name + '/fc8'] = net
            return (net, end_points)
vgg_19.default_image_size = 224
vgg_d = vgg_16
vgg_e = vgg_19