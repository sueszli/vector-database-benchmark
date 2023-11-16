"""Image embedding ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
slim = tf.contrib.slim

def inception_v3(images, trainable=True, is_training=True, weight_decay=4e-05, stddev=0.1, dropout_keep_prob=0.8, use_batch_norm=True, batch_norm_params=None, add_summaries=True, scope='InceptionV3'):
    if False:
        for i in range(10):
            print('nop')
    'Builds an Inception V3 subgraph for image embeddings.\n\n  Args:\n    images: A float32 Tensor of shape [batch, height, width, channels].\n    trainable: Whether the inception submodel should be trainable or not.\n    is_training: Boolean indicating training mode or not.\n    weight_decay: Coefficient for weight regularization.\n    stddev: The standard deviation of the trunctated normal weight initializer.\n    dropout_keep_prob: Dropout keep probability.\n    use_batch_norm: Whether to use batch normalization.\n    batch_norm_params: Parameters for batch normalization. See\n      tf.contrib.layers.batch_norm for details.\n    add_summaries: Whether to add activation summaries.\n    scope: Optional Variable scope.\n\n  Returns:\n    end_points: A dictionary of activations from inception_v3 layers.\n  '
    is_inception_model_training = trainable and is_training
    if use_batch_norm:
        if not batch_norm_params:
            batch_norm_params = {'is_training': is_inception_model_training, 'trainable': trainable, 'decay': 0.9997, 'epsilon': 0.001, 'variables_collections': {'beta': None, 'gamma': None, 'moving_mean': ['moving_vars'], 'moving_variance': ['moving_vars']}}
    else:
        batch_norm_params = None
    if trainable:
        weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        weights_regularizer = None
    with tf.variable_scope(scope, 'InceptionV3', [images]) as scope:
        with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=weights_regularizer, trainable=trainable):
            with slim.arg_scope([slim.conv2d], weights_initializer=tf.truncated_normal_initializer(stddev=stddev), activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params):
                (net, end_points) = inception_v3_base(images, scope=scope)
                with tf.variable_scope('logits'):
                    shape = net.get_shape()
                    net = slim.avg_pool2d(net, shape[1:3], padding='VALID', scope='pool')
                    net = slim.dropout(net, keep_prob=dropout_keep_prob, is_training=is_inception_model_training, scope='dropout')
                    net = slim.flatten(net, scope='flatten')
    if add_summaries:
        for v in end_points.values():
            tf.contrib.layers.summaries.summarize_activation(v)
    return net