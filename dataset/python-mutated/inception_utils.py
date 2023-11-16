"""Contains common code shared by all inception models.

Usage of arg scope:
  with slim.arg_scope(inception_arg_scope()):
    logits, end_points = inception.inception_v3(images, num_classes,
                                                is_training=is_training)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
slim = contrib_slim

def inception_arg_scope(weight_decay=4e-05, use_batch_norm=True, batch_norm_decay=0.9997, batch_norm_epsilon=0.001, activation_fn=tf.nn.relu, batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS, batch_norm_scale=False):
    if False:
        for i in range(10):
            print('nop')
    'Defines the default arg scope for inception models.\n\n  Args:\n    weight_decay: The weight decay to use for regularizing the model.\n    use_batch_norm: "If `True`, batch_norm is applied after each convolution.\n    batch_norm_decay: Decay for batch norm moving average.\n    batch_norm_epsilon: Small float added to variance to avoid dividing by zero\n      in batch norm.\n    activation_fn: Activation function for conv2d.\n    batch_norm_updates_collections: Collection for the update ops for\n      batch norm.\n    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the\n      activations in the batch normalization layer.\n\n  Returns:\n    An `arg_scope` to use for the inception models.\n  '
    batch_norm_params = {'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon, 'updates_collections': batch_norm_updates_collections, 'fused': None, 'scale': batch_norm_scale}
    if use_batch_norm:
        normalizer_fn = slim.batch_norm
        normalizer_params = batch_norm_params
    else:
        normalizer_fn = None
        normalizer_params = {}
    with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope([slim.conv2d], weights_initializer=slim.variance_scaling_initializer(), activation_fn=activation_fn, normalizer_fn=normalizer_fn, normalizer_params=normalizer_params) as sc:
            return sc