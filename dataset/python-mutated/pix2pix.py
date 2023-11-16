"""Implementation of the Image-to-Image Translation model.

This network represents a port of the following work:

  Image-to-Image Translation with Conditional Adversarial Networks
  Phillip Isola, Jun-Yan Zhu, Tinghui Zhou and Alexei A. Efros
  Arxiv, 2017
  https://phillipi.github.io/pix2pix/

A reference implementation written in Lua can be found at:
https://github.com/phillipi/pix2pix/blob/master/models.lua
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import functools
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers
layers = contrib_layers

def pix2pix_arg_scope():
    if False:
        return 10
    'Returns a default argument scope for isola_net.\n\n  Returns:\n    An arg scope.\n  '
    instance_norm_params = {'center': True, 'scale': True, 'epsilon': 1e-05}
    with contrib_framework.arg_scope([layers.conv2d, layers.conv2d_transpose], normalizer_fn=layers.instance_norm, normalizer_params=instance_norm_params, weights_initializer=tf.random_normal_initializer(0, 0.02)) as sc:
        return sc

def upsample(net, num_outputs, kernel_size, method='nn_upsample_conv'):
    if False:
        print('Hello World!')
    'Upsamples the given inputs.\n\n  Args:\n    net: A `Tensor` of size [batch_size, height, width, filters].\n    num_outputs: The number of output filters.\n    kernel_size: A list of 2 scalars or a 1x2 `Tensor` indicating the scale,\n      relative to the inputs, of the output dimensions. For example, if kernel\n      size is [2, 3], then the output height and width will be twice and three\n      times the input size.\n    method: The upsampling method.\n\n  Returns:\n    An `Tensor` which was upsampled using the specified method.\n\n  Raises:\n    ValueError: if `method` is not recognized.\n  '
    net_shape = tf.shape(net)
    height = net_shape[1]
    width = net_shape[2]
    if method == 'nn_upsample_conv':
        net = tf.image.resize_nearest_neighbor(net, [kernel_size[0] * height, kernel_size[1] * width])
        net = layers.conv2d(net, num_outputs, [4, 4], activation_fn=None)
    elif method == 'conv2d_transpose':
        net = layers.conv2d_transpose(net, num_outputs, [4, 4], stride=kernel_size, activation_fn=None)
    else:
        raise ValueError('Unknown method: [%s]' % method)
    return net

class Block(collections.namedtuple('Block', ['num_filters', 'decoder_keep_prob'])):
    """Represents a single block of encoder and decoder processing.

  The Image-to-Image translation paper works a bit differently than the original
  U-Net model. In particular, each block represents a single operation in the
  encoder which is concatenated with the corresponding decoder representation.
  A dropout layer follows the concatenation and convolution of the concatenated
  features.
  """
    pass

def _default_generator_blocks():
    if False:
        i = 10
        return i + 15
    'Returns the default generator block definitions.\n\n  Returns:\n    A list of generator blocks.\n  '
    return [Block(64, 0.5), Block(128, 0.5), Block(256, 0.5), Block(512, 0), Block(512, 0), Block(512, 0), Block(512, 0)]

def pix2pix_generator(net, num_outputs, blocks=None, upsample_method='nn_upsample_conv', is_training=False):
    if False:
        for i in range(10):
            print('nop')
    "Defines the network architecture.\n\n  Args:\n    net: A `Tensor` of size [batch, height, width, channels]. Note that the\n      generator currently requires square inputs (e.g. height=width).\n    num_outputs: The number of (per-pixel) outputs.\n    blocks: A list of generator blocks or `None` to use the default generator\n      definition.\n    upsample_method: The method of upsampling images, one of 'nn_upsample_conv'\n      or 'conv2d_transpose'\n    is_training: Whether or not we're in training or testing mode.\n\n  Returns:\n    A `Tensor` representing the model output and a dictionary of model end\n      points.\n\n  Raises:\n    ValueError: if the input heights do not match their widths.\n  "
    end_points = {}
    blocks = blocks or _default_generator_blocks()
    input_size = net.get_shape().as_list()
    input_size[3] = num_outputs
    upsample_fn = functools.partial(upsample, method=upsample_method)
    encoder_activations = []
    with tf.variable_scope('encoder'):
        with contrib_framework.arg_scope([layers.conv2d], kernel_size=[4, 4], stride=2, activation_fn=tf.nn.leaky_relu):
            for (block_id, block) in enumerate(blocks):
                if block_id == 0:
                    net = layers.conv2d(net, block.num_filters, normalizer_fn=None)
                elif block_id < len(blocks) - 1:
                    net = layers.conv2d(net, block.num_filters)
                else:
                    net = layers.conv2d(net, block.num_filters, activation_fn=None, normalizer_fn=None)
                encoder_activations.append(net)
                end_points['encoder%d' % block_id] = net
    reversed_blocks = list(blocks)
    reversed_blocks.reverse()
    with tf.variable_scope('decoder'):
        with contrib_framework.arg_scope([layers.dropout], is_training=True):
            for (block_id, block) in enumerate(reversed_blocks):
                if block_id > 0:
                    net = tf.concat([net, encoder_activations[-block_id - 1]], axis=3)
                net = tf.nn.relu(net)
                net = upsample_fn(net, block.num_filters, [2, 2])
                if block.decoder_keep_prob > 0:
                    net = layers.dropout(net, keep_prob=block.decoder_keep_prob)
                end_points['decoder%d' % block_id] = net
    with tf.variable_scope('output'):
        logits = layers.conv2d(net, num_outputs, [4, 4], activation_fn=None, normalizer_fn=None)
        logits = tf.reshape(logits, input_size)
        end_points['logits'] = logits
        end_points['predictions'] = tf.tanh(logits)
    return (logits, end_points)

def pix2pix_discriminator(net, num_filters, padding=2, pad_mode='REFLECT', activation_fn=tf.nn.leaky_relu, is_training=False):
    if False:
        i = 10
        return i + 15
    'Creates the Image2Image Translation Discriminator.\n\n  Args:\n    net: A `Tensor` of size [batch_size, height, width, channels] representing\n      the input.\n    num_filters: A list of the filters in the discriminator. The length of the\n      list determines the number of layers in the discriminator.\n    padding: Amount of reflection padding applied before each convolution.\n    pad_mode: mode for tf.pad, one of "CONSTANT", "REFLECT", or "SYMMETRIC".\n    activation_fn: activation fn for layers.conv2d.\n    is_training: Whether or not the model is training or testing.\n\n  Returns:\n    A logits `Tensor` of size [batch_size, N, N, 1] where N is the number of\n    \'patches\' we\'re attempting to discriminate and a dictionary of model end\n    points.\n  '
    del is_training
    end_points = {}
    num_layers = len(num_filters)

    def padded(net, scope):
        if False:
            while True:
                i = 10
        if padding:
            with tf.variable_scope(scope):
                spatial_pad = tf.constant([[0, 0], [padding, padding], [padding, padding], [0, 0]], dtype=tf.int32)
                return tf.pad(net, spatial_pad, pad_mode)
        else:
            return net
    with contrib_framework.arg_scope([layers.conv2d], kernel_size=[4, 4], stride=2, padding='valid', activation_fn=activation_fn):
        net = layers.conv2d(padded(net, 'conv0'), num_filters[0], normalizer_fn=None, scope='conv0')
        end_points['conv0'] = net
        for i in range(1, num_layers - 1):
            net = layers.conv2d(padded(net, 'conv%d' % i), num_filters[i], scope='conv%d' % i)
            end_points['conv%d' % i] = net
        net = layers.conv2d(padded(net, 'conv%d' % (num_layers - 1)), num_filters[-1], stride=1, scope='conv%d' % (num_layers - 1))
        end_points['conv%d' % (num_layers - 1)] = net
        logits = layers.conv2d(padded(net, 'conv%d' % num_layers), 1, stride=1, activation_fn=None, normalizer_fn=None, scope='conv%d' % num_layers)
        end_points['logits'] = logits
        end_points['predictions'] = tf.sigmoid(logits)
    return (logits, end_points)