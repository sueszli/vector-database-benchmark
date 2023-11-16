"""Training/Pretraining encoder as used in PTN (NIPS16)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim

def _preprocess(images):
    if False:
        return 10
    return images * 2 - 1

def model(images, params, is_training):
    if False:
        return 10
    'Model encoding the images into view-invariant embedding.'
    del is_training
    image_size = images.get_shape().as_list()[1]
    f_dim = params.f_dim
    fc_dim = params.fc_dim
    z_dim = params.z_dim
    outputs = dict()
    images = _preprocess(images)
    with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1)):
        h0 = slim.conv2d(images, f_dim, [5, 5], stride=2, activation_fn=tf.nn.relu)
        h1 = slim.conv2d(h0, f_dim * 2, [5, 5], stride=2, activation_fn=tf.nn.relu)
        h2 = slim.conv2d(h1, f_dim * 4, [5, 5], stride=2, activation_fn=tf.nn.relu)
        s8 = image_size // 8
        h2 = tf.reshape(h2, [-1, s8 * s8 * f_dim * 4])
        h3 = slim.fully_connected(h2, fc_dim, activation_fn=tf.nn.relu)
        h4 = slim.fully_connected(h3, fc_dim, activation_fn=tf.nn.relu)
        outputs['ids'] = slim.fully_connected(h4, z_dim, activation_fn=tf.nn.relu)
        outputs['poses'] = slim.fully_connected(h4, z_dim, activation_fn=tf.nn.relu)
    return outputs