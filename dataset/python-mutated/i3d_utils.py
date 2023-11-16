"""Utilities for building I3D network models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers
add_arg_scope = contrib_framework.add_arg_scope
layers = contrib_layers

def center_initializer():
    if False:
        while True:
            i = 10
    'Centering Initializer for I3D.\n\n  This initializer allows identity mapping for temporal convolution at the\n  initialization, which is critical for a desired convergence behavior\n  for training a seprable I3D model.\n\n  The centering behavior of this initializer requires an odd-sized kernel,\n  typically set to 3.\n\n  Returns:\n    A weight initializer op used in temporal convolutional layers.\n\n  Raises:\n    ValueError: Input tensor data type has to be tf.float32.\n    ValueError: If input tensor is not a 5-D tensor.\n    ValueError: If input and output channel dimensions are different.\n    ValueError: If spatial kernel sizes are not 1.\n    ValueError: If temporal kernel size is even.\n  '

    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if False:
            return 10
        'Initializer op.'
        if dtype != tf.float32 and dtype != tf.bfloat16:
            raise ValueError('Input tensor data type has to be tf.float32 or tf.bfloat16.')
        if len(shape) != 5:
            raise ValueError('Input tensor has to be 5-D.')
        if shape[3] != shape[4]:
            raise ValueError('Input and output channel dimensions must be the same.')
        if shape[1] != 1 or shape[2] != 1:
            raise ValueError('Spatial kernel sizes must be 1 (pointwise conv).')
        if shape[0] % 2 == 0:
            raise ValueError('Temporal kernel size has to be odd.')
        center_pos = int(shape[0] / 2)
        init_mat = np.zeros([shape[0], shape[1], shape[2], shape[3], shape[4]], dtype=np.float32)
        for i in range(0, shape[3]):
            init_mat[center_pos, 0, 0, i, i] = 1.0
        init_op = tf.constant(init_mat, dtype=dtype)
        return init_op
    return _initializer

@add_arg_scope
def conv3d_spatiotemporal(inputs, num_outputs, kernel_size, stride=1, padding='SAME', activation_fn=None, normalizer_fn=None, normalizer_params=None, weights_regularizer=None, separable=False, data_format='NDHWC', scope=''):
    if False:
        while True:
            i = 10
    'A wrapper for conv3d to model spatiotemporal representations.\n\n  This allows switching between original 3D convolution and separable 3D\n  convolutions for spatial and temporal features respectively. On Kinetics,\n  seprable 3D convolutions yields better classification performance.\n\n  Args:\n    inputs: a 5-D tensor  `[batch_size, depth, height, width, channels]`.\n    num_outputs: integer, the number of output filters.\n    kernel_size: a list of length 3\n      `[kernel_depth, kernel_height, kernel_width]` of the filters. Can be an\n      int if all values are the same.\n    stride: a list of length 3 `[stride_depth, stride_height, stride_width]`.\n      Can be an int if all strides are the same.\n    padding: one of `VALID` or `SAME`.\n    activation_fn: activation function.\n    normalizer_fn: normalization function to use instead of `biases`.\n    normalizer_params: dictionary of normalization function parameters.\n    weights_regularizer: Optional regularizer for the weights.\n    separable: If `True`, use separable spatiotemporal convolutions.\n    data_format: An optional string from: "NDHWC", "NCDHW". Defaults to "NDHWC".\n      The data format of the input and output data. With the default format\n      "NDHWC", the data is stored in the order of: [batch, in_depth, in_height,\n      in_width, in_channels]. Alternatively, the format could be "NCDHW", the\n      data storage order is:\n      [batch, in_channels, in_depth, in_height, in_width].\n    scope: scope for `variable_scope`.\n\n  Returns:\n    A tensor representing the output of the (separable) conv3d operation.\n\n  '
    assert len(kernel_size) == 3
    if separable and kernel_size[0] != 1:
        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
        temporal_kernel_size = [kernel_size[0], 1, 1]
        if isinstance(stride, list) and len(stride) == 3:
            spatial_stride = [1, stride[1], stride[2]]
            temporal_stride = [stride[0], 1, 1]
        else:
            spatial_stride = [1, stride, stride]
            temporal_stride = [stride, 1, 1]
        net = layers.conv3d(inputs, num_outputs, spatial_kernel_size, stride=spatial_stride, padding=padding, activation_fn=activation_fn, normalizer_fn=normalizer_fn, normalizer_params=normalizer_params, weights_regularizer=weights_regularizer, data_format=data_format, scope=scope)
        net = layers.conv3d(net, num_outputs, temporal_kernel_size, stride=temporal_stride, padding=padding, scope=scope + '/temporal', activation_fn=activation_fn, normalizer_fn=None, data_format=data_format, weights_initializer=center_initializer())
        return net
    else:
        return layers.conv3d(inputs, num_outputs, kernel_size, stride=stride, padding=padding, activation_fn=activation_fn, normalizer_fn=normalizer_fn, normalizer_params=normalizer_params, weights_regularizer=weights_regularizer, data_format=data_format, scope=scope)

@add_arg_scope
def inception_block_v1_3d(inputs, num_outputs_0_0a, num_outputs_1_0a, num_outputs_1_0b, num_outputs_2_0a, num_outputs_2_0b, num_outputs_3_0b, temporal_kernel_size=3, self_gating_fn=None, data_format='NDHWC', scope=''):
    if False:
        return 10
    'A 3D Inception v1 block.\n\n  This allows use of separable 3D convolutions and self-gating, as\n  described in:\n  Saining Xie, Chen Sun, Jonathan Huang, Zhuowen Tu and Kevin Murphy,\n    Rethinking Spatiotemporal Feature Learning For Video Understanding.\n    https://arxiv.org/abs/1712.04851.\n\n  Args:\n    inputs: a 5-D tensor  `[batch_size, depth, height, width, channels]`.\n    num_outputs_0_0a: integer, the number of output filters for Branch 0,\n      operation Conv2d_0a_1x1.\n    num_outputs_1_0a: integer, the number of output filters for Branch 1,\n      operation Conv2d_0a_1x1.\n    num_outputs_1_0b: integer, the number of output filters for Branch 1,\n      operation Conv2d_0b_3x3.\n    num_outputs_2_0a: integer, the number of output filters for Branch 2,\n      operation Conv2d_0a_1x1.\n    num_outputs_2_0b: integer, the number of output filters for Branch 2,\n      operation Conv2d_0b_3x3.\n    num_outputs_3_0b: integer, the number of output filters for Branch 3,\n      operation Conv2d_0b_1x1.\n    temporal_kernel_size: integer, the size of the temporal convolutional\n      filters in the conv3d_spatiotemporal blocks.\n    self_gating_fn: function which optionally performs self-gating.\n      Must have two arguments, `inputs` and `scope`, and return one output\n      tensor the same size as `inputs`. If `None`, no self-gating is\n      applied.\n    data_format: An optional string from: "NDHWC", "NCDHW". Defaults to "NDHWC".\n      The data format of the input and output data. With the default format\n      "NDHWC", the data is stored in the order of: [batch, in_depth, in_height,\n      in_width, in_channels]. Alternatively, the format could be "NCDHW", the\n      data storage order is:\n      [batch, in_channels, in_depth, in_height, in_width].\n    scope: scope for `variable_scope`.\n\n  Returns:\n    A 5-D tensor `[batch_size, depth, height, width, out_channels]`, where\n    `out_channels = num_outputs_0_0a + num_outputs_1_0b + num_outputs_2_0b\n    + num_outputs_3_0b`.\n\n  '
    use_gating = self_gating_fn is not None
    with tf.variable_scope(scope):
        with tf.variable_scope('Branch_0'):
            branch_0 = layers.conv3d(inputs, num_outputs_0_0a, [1, 1, 1], scope='Conv2d_0a_1x1')
            if use_gating:
                branch_0 = self_gating_fn(branch_0, scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = layers.conv3d(inputs, num_outputs_1_0a, [1, 1, 1], scope='Conv2d_0a_1x1')
            branch_1 = conv3d_spatiotemporal(branch_1, num_outputs_1_0b, [temporal_kernel_size, 3, 3], scope='Conv2d_0b_3x3')
            if use_gating:
                branch_1 = self_gating_fn(branch_1, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            branch_2 = layers.conv3d(inputs, num_outputs_2_0a, [1, 1, 1], scope='Conv2d_0a_1x1')
            branch_2 = conv3d_spatiotemporal(branch_2, num_outputs_2_0b, [temporal_kernel_size, 3, 3], scope='Conv2d_0b_3x3')
            if use_gating:
                branch_2 = self_gating_fn(branch_2, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_3'):
            branch_3 = layers.max_pool3d(inputs, [3, 3, 3], scope='MaxPool_0a_3x3')
            branch_3 = layers.conv3d(branch_3, num_outputs_3_0b, [1, 1, 1], scope='Conv2d_0b_1x1')
            if use_gating:
                branch_3 = self_gating_fn(branch_3, scope='Conv2d_0b_1x1')
        index_c = data_format.index('C')
        assert 1 <= index_c <= 4, 'Cannot identify channel dimension.'
        output = tf.concat([branch_0, branch_1, branch_2, branch_3], index_c)
    return output

def reduced_kernel_size_3d(input_tensor, kernel_size):
    if False:
        i = 10
        return i + 15
    'Define kernel size which is automatically reduced for small input.\n\n  If the shape of the input images is unknown at graph construction time this\n  function assumes that the input images are large enough.\n\n  Args:\n    input_tensor: input tensor of size\n      [batch_size, time, height, width, channels].\n    kernel_size: desired kernel size of length 3, corresponding to time,\n      height and width.\n\n  Returns:\n    a tensor with the kernel size.\n  '
    assert len(kernel_size) == 3
    shape = input_tensor.get_shape().as_list()
    assert len(shape) == 5
    if None in shape[1:4]:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1]), min(shape[3], kernel_size[2])]
    return kernel_size_out