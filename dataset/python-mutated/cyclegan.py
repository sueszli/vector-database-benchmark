"""Defines the CycleGAN generator and discriminator networks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import util as contrib_util
layers = contrib_layers

def cyclegan_arg_scope(instance_norm_center=True, instance_norm_scale=True, instance_norm_epsilon=0.001, weights_init_stddev=0.02, weight_decay=0.0):
    if False:
        i = 10
        return i + 15
    'Returns a default argument scope for all generators and discriminators.\n\n  Args:\n    instance_norm_center: Whether instance normalization applies centering.\n    instance_norm_scale: Whether instance normalization applies scaling.\n    instance_norm_epsilon: Small float added to the variance in the instance\n      normalization to avoid dividing by zero.\n    weights_init_stddev: Standard deviation of the random values to initialize\n      the convolution kernels with.\n    weight_decay: Magnitude of weight decay applied to all convolution kernel\n      variables of the generator.\n\n  Returns:\n    An arg-scope.\n  '
    instance_norm_params = {'center': instance_norm_center, 'scale': instance_norm_scale, 'epsilon': instance_norm_epsilon}
    weights_regularizer = None
    if weight_decay and weight_decay > 0.0:
        weights_regularizer = layers.l2_regularizer(weight_decay)
    with contrib_framework.arg_scope([layers.conv2d], normalizer_fn=layers.instance_norm, normalizer_params=instance_norm_params, weights_initializer=tf.random_normal_initializer(0, weights_init_stddev), weights_regularizer=weights_regularizer) as sc:
        return sc

def cyclegan_upsample(net, num_outputs, stride, method='conv2d_transpose', pad_mode='REFLECT', align_corners=False):
    if False:
        while True:
            i = 10
    'Upsamples the given inputs.\n\n  Args:\n    net: A Tensor of size [batch_size, height, width, filters].\n    num_outputs: The number of output filters.\n    stride: A list of 2 scalars or a 1x2 Tensor indicating the scale,\n      relative to the inputs, of the output dimensions. For example, if kernel\n      size is [2, 3], then the output height and width will be twice and three\n      times the input size.\n    method: The upsampling method: \'nn_upsample_conv\', \'bilinear_upsample_conv\',\n      or \'conv2d_transpose\'.\n    pad_mode: mode for tf.pad, one of "CONSTANT", "REFLECT", or "SYMMETRIC".\n    align_corners: option for method, \'bilinear_upsample_conv\'. If true, the\n      centers of the 4 corner pixels of the input and output tensors are\n      aligned, preserving the values at the corner pixels.\n\n  Returns:\n    A Tensor which was upsampled using the specified method.\n\n  Raises:\n    ValueError: if `method` is not recognized.\n  '
    with tf.variable_scope('upconv'):
        net_shape = tf.shape(net)
        height = net_shape[1]
        width = net_shape[2]
        spatial_pad_1 = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])
        if method == 'nn_upsample_conv':
            net = tf.image.resize_nearest_neighbor(net, [stride[0] * height, stride[1] * width])
            net = tf.pad(net, spatial_pad_1, pad_mode)
            net = layers.conv2d(net, num_outputs, kernel_size=[3, 3], padding='valid')
        elif method == 'bilinear_upsample_conv':
            net = tf.image.resize_bilinear(net, [stride[0] * height, stride[1] * width], align_corners=align_corners)
            net = tf.pad(net, spatial_pad_1, pad_mode)
            net = layers.conv2d(net, num_outputs, kernel_size=[3, 3], padding='valid')
        elif method == 'conv2d_transpose':
            net = layers.conv2d_transpose(net, num_outputs, kernel_size=[3, 3], stride=stride, padding='valid')
            net = net[:, 1:, 1:, :]
        else:
            raise ValueError('Unknown method: [%s]' % method)
        return net

def _dynamic_or_static_shape(tensor):
    if False:
        while True:
            i = 10
    shape = tf.shape(tensor)
    static_shape = contrib_util.constant_value(shape)
    return static_shape if static_shape is not None else shape

def cyclegan_generator_resnet(images, arg_scope_fn=cyclegan_arg_scope, num_resnet_blocks=6, num_filters=64, upsample_fn=cyclegan_upsample, kernel_size=3, tanh_linear_slope=0.0, is_training=False):
    if False:
        return 10
    "Defines the cyclegan resnet network architecture.\n\n  As closely as possible following\n  https://github.com/junyanz/CycleGAN/blob/master/models/architectures.lua#L232\n\n  FYI: This network requires input height and width to be divisible by 4 in\n  order to generate an output with shape equal to input shape. Assertions will\n  catch this if input dimensions are known at graph construction time, but\n  there's no protection if unknown at graph construction time (you'll see an\n  error).\n\n  Args:\n    images: Input image tensor of shape [batch_size, h, w, 3].\n    arg_scope_fn: Function to create the global arg_scope for the network.\n    num_resnet_blocks: Number of ResNet blocks in the middle of the generator.\n    num_filters: Number of filters of the first hidden layer.\n    upsample_fn: Upsampling function for the decoder part of the generator.\n    kernel_size: Size w or list/tuple [h, w] of the filter kernels for all inner\n      layers.\n    tanh_linear_slope: Slope of the linear function to add to the tanh over the\n      logits.\n    is_training: Whether the network is created in training mode or inference\n      only mode. Not actually needed, just for compliance with other generator\n      network functions.\n\n  Returns:\n    A `Tensor` representing the model output and a dictionary of model end\n      points.\n\n  Raises:\n    ValueError: If the input height or width is known at graph construction time\n      and not a multiple of 4.\n  "
    del is_training
    end_points = {}
    input_size = images.shape.as_list()
    (height, width) = (input_size[1], input_size[2])
    if height and height % 4 != 0:
        raise ValueError('The input height must be a multiple of 4.')
    if width and width % 4 != 0:
        raise ValueError('The input width must be a multiple of 4.')
    num_outputs = input_size[3]
    if not isinstance(kernel_size, (list, tuple)):
        kernel_size = [kernel_size, kernel_size]
    kernel_height = kernel_size[0]
    kernel_width = kernel_size[1]
    pad_top = (kernel_height - 1) // 2
    pad_bottom = kernel_height // 2
    pad_left = (kernel_width - 1) // 2
    pad_right = kernel_width // 2
    paddings = np.array([[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], dtype=np.int32)
    spatial_pad_3 = np.array([[0, 0], [3, 3], [3, 3], [0, 0]])
    with contrib_framework.arg_scope(arg_scope_fn()):
        with tf.variable_scope('input'):
            net = tf.pad(images, spatial_pad_3, 'REFLECT')
            net = layers.conv2d(net, num_filters, kernel_size=[7, 7], padding='VALID')
            end_points['encoder_0'] = net
        with tf.variable_scope('encoder'):
            with contrib_framework.arg_scope([layers.conv2d], kernel_size=kernel_size, stride=2, activation_fn=tf.nn.relu, padding='VALID'):
                net = tf.pad(net, paddings, 'REFLECT')
                net = layers.conv2d(net, num_filters * 2)
                end_points['encoder_1'] = net
                net = tf.pad(net, paddings, 'REFLECT')
                net = layers.conv2d(net, num_filters * 4)
                end_points['encoder_2'] = net
        with tf.variable_scope('residual_blocks'):
            with contrib_framework.arg_scope([layers.conv2d], kernel_size=kernel_size, stride=1, activation_fn=tf.nn.relu, padding='VALID'):
                for block_id in xrange(num_resnet_blocks):
                    with tf.variable_scope('block_{}'.format(block_id)):
                        res_net = tf.pad(net, paddings, 'REFLECT')
                        res_net = layers.conv2d(res_net, num_filters * 4)
                        res_net = tf.pad(res_net, paddings, 'REFLECT')
                        res_net = layers.conv2d(res_net, num_filters * 4, activation_fn=None)
                        net += res_net
                        end_points['resnet_block_%d' % block_id] = net
        with tf.variable_scope('decoder'):
            with contrib_framework.arg_scope([layers.conv2d], kernel_size=kernel_size, stride=1, activation_fn=tf.nn.relu):
                with tf.variable_scope('decoder1'):
                    net = upsample_fn(net, num_outputs=num_filters * 2, stride=[2, 2])
                end_points['decoder1'] = net
                with tf.variable_scope('decoder2'):
                    net = upsample_fn(net, num_outputs=num_filters, stride=[2, 2])
                end_points['decoder2'] = net
        with tf.variable_scope('output'):
            net = tf.pad(net, spatial_pad_3, 'REFLECT')
            logits = layers.conv2d(net, num_outputs, [7, 7], activation_fn=None, normalizer_fn=None, padding='valid')
            logits = tf.reshape(logits, _dynamic_or_static_shape(images))
            end_points['logits'] = logits
            end_points['predictions'] = tf.tanh(logits) + logits * tanh_linear_slope
    return (end_points['predictions'], end_points)