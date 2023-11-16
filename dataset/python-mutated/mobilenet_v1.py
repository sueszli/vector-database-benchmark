"""MobileNet v1.

MobileNet is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and different
head (for example: embeddings, localization and classification).

As described in https://arxiv.org/abs/1704.04861.

  MobileNets: Efficient Convolutional Neural Networks for
    Mobile Vision Applications
  Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang,
    Tobias Weyand, Marco Andreetto, Hartwig Adam

100% Mobilenet V1 (base) with input size 224x224:

See mobilenet_v1()

Layer                                                     params           macs
--------------------------------------------------------------------------------
MobilenetV1/Conv2d_0/Conv2D:                                 864      10,838,016
MobilenetV1/Conv2d_1_depthwise/depthwise:                    288       3,612,672
MobilenetV1/Conv2d_1_pointwise/Conv2D:                     2,048      25,690,112
MobilenetV1/Conv2d_2_depthwise/depthwise:                    576       1,806,336
MobilenetV1/Conv2d_2_pointwise/Conv2D:                     8,192      25,690,112
MobilenetV1/Conv2d_3_depthwise/depthwise:                  1,152       3,612,672
MobilenetV1/Conv2d_3_pointwise/Conv2D:                    16,384      51,380,224
MobilenetV1/Conv2d_4_depthwise/depthwise:                  1,152         903,168
MobilenetV1/Conv2d_4_pointwise/Conv2D:                    32,768      25,690,112
MobilenetV1/Conv2d_5_depthwise/depthwise:                  2,304       1,806,336
MobilenetV1/Conv2d_5_pointwise/Conv2D:                    65,536      51,380,224
MobilenetV1/Conv2d_6_depthwise/depthwise:                  2,304         451,584
MobilenetV1/Conv2d_6_pointwise/Conv2D:                   131,072      25,690,112
MobilenetV1/Conv2d_7_depthwise/depthwise:                  4,608         903,168
MobilenetV1/Conv2d_7_pointwise/Conv2D:                   262,144      51,380,224
MobilenetV1/Conv2d_8_depthwise/depthwise:                  4,608         903,168
MobilenetV1/Conv2d_8_pointwise/Conv2D:                   262,144      51,380,224
MobilenetV1/Conv2d_9_depthwise/depthwise:                  4,608         903,168
MobilenetV1/Conv2d_9_pointwise/Conv2D:                   262,144      51,380,224
MobilenetV1/Conv2d_10_depthwise/depthwise:                 4,608         903,168
MobilenetV1/Conv2d_10_pointwise/Conv2D:                  262,144      51,380,224
MobilenetV1/Conv2d_11_depthwise/depthwise:                 4,608         903,168
MobilenetV1/Conv2d_11_pointwise/Conv2D:                  262,144      51,380,224
MobilenetV1/Conv2d_12_depthwise/depthwise:                 4,608         225,792
MobilenetV1/Conv2d_12_pointwise/Conv2D:                  524,288      25,690,112
MobilenetV1/Conv2d_13_depthwise/depthwise:                 9,216         451,584
MobilenetV1/Conv2d_13_pointwise/Conv2D:                1,048,576      51,380,224
--------------------------------------------------------------------------------
Total:                                                 3,185,088     567,716,352


75% Mobilenet V1 (base) with input size 128x128:

See mobilenet_v1_075()

Layer                                                     params           macs
--------------------------------------------------------------------------------
MobilenetV1/Conv2d_0/Conv2D:                                 648       2,654,208
MobilenetV1/Conv2d_1_depthwise/depthwise:                    216         884,736
MobilenetV1/Conv2d_1_pointwise/Conv2D:                     1,152       4,718,592
MobilenetV1/Conv2d_2_depthwise/depthwise:                    432         442,368
MobilenetV1/Conv2d_2_pointwise/Conv2D:                     4,608       4,718,592
MobilenetV1/Conv2d_3_depthwise/depthwise:                    864         884,736
MobilenetV1/Conv2d_3_pointwise/Conv2D:                     9,216       9,437,184
MobilenetV1/Conv2d_4_depthwise/depthwise:                    864         221,184
MobilenetV1/Conv2d_4_pointwise/Conv2D:                    18,432       4,718,592
MobilenetV1/Conv2d_5_depthwise/depthwise:                  1,728         442,368
MobilenetV1/Conv2d_5_pointwise/Conv2D:                    36,864       9,437,184
MobilenetV1/Conv2d_6_depthwise/depthwise:                  1,728         110,592
MobilenetV1/Conv2d_6_pointwise/Conv2D:                    73,728       4,718,592
MobilenetV1/Conv2d_7_depthwise/depthwise:                  3,456         221,184
MobilenetV1/Conv2d_7_pointwise/Conv2D:                   147,456       9,437,184
MobilenetV1/Conv2d_8_depthwise/depthwise:                  3,456         221,184
MobilenetV1/Conv2d_8_pointwise/Conv2D:                   147,456       9,437,184
MobilenetV1/Conv2d_9_depthwise/depthwise:                  3,456         221,184
MobilenetV1/Conv2d_9_pointwise/Conv2D:                   147,456       9,437,184
MobilenetV1/Conv2d_10_depthwise/depthwise:                 3,456         221,184
MobilenetV1/Conv2d_10_pointwise/Conv2D:                  147,456       9,437,184
MobilenetV1/Conv2d_11_depthwise/depthwise:                 3,456         221,184
MobilenetV1/Conv2d_11_pointwise/Conv2D:                  147,456       9,437,184
MobilenetV1/Conv2d_12_depthwise/depthwise:                 3,456          55,296
MobilenetV1/Conv2d_12_pointwise/Conv2D:                  294,912       4,718,592
MobilenetV1/Conv2d_13_depthwise/depthwise:                 6,912         110,592
MobilenetV1/Conv2d_13_pointwise/Conv2D:                  589,824       9,437,184
--------------------------------------------------------------------------------
Total:                                                 1,800,144     106,002,432

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import namedtuple
import functools
import tensorflow as tf
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import slim as contrib_slim
slim = contrib_slim
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])
MOBILENETV1_CONV_DEFS = [Conv(kernel=[3, 3], stride=2, depth=32), DepthSepConv(kernel=[3, 3], stride=1, depth=64), DepthSepConv(kernel=[3, 3], stride=2, depth=128), DepthSepConv(kernel=[3, 3], stride=1, depth=128), DepthSepConv(kernel=[3, 3], stride=2, depth=256), DepthSepConv(kernel=[3, 3], stride=1, depth=256), DepthSepConv(kernel=[3, 3], stride=2, depth=512), DepthSepConv(kernel=[3, 3], stride=1, depth=512), DepthSepConv(kernel=[3, 3], stride=1, depth=512), DepthSepConv(kernel=[3, 3], stride=1, depth=512), DepthSepConv(kernel=[3, 3], stride=1, depth=512), DepthSepConv(kernel=[3, 3], stride=1, depth=512), DepthSepConv(kernel=[3, 3], stride=2, depth=1024), DepthSepConv(kernel=[3, 3], stride=1, depth=1024)]

def _fixed_padding(inputs, kernel_size, rate=1):
    if False:
        print('Hello World!')
    "Pads the input along the spatial dimensions independently of input size.\n\n  Pads the input such that if it was used in a convolution with 'VALID' padding,\n  the output would have the same dimensions as if the unpadded input was used\n  in a convolution with 'SAME' padding.\n\n  Args:\n    inputs: A tensor of size [batch, height_in, width_in, channels].\n    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.\n    rate: An integer, rate for atrous convolution.\n\n  Returns:\n    output: A tensor of size [batch, height_out, width_out, channels] with the\n      input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).\n  "
    kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1), kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
    pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
    pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
    pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]], [pad_beg[1], pad_end[1]], [0, 0]])
    return padded_inputs

def mobilenet_v1_base(inputs, final_endpoint='Conv2d_13_pointwise', min_depth=8, depth_multiplier=1.0, conv_defs=None, output_stride=None, use_explicit_padding=False, scope=None):
    if False:
        return 10
    "Mobilenet v1.\n\n  Constructs a Mobilenet v1 network from inputs to the given final endpoint.\n\n  Args:\n    inputs: a tensor of shape [batch_size, height, width, channels].\n    final_endpoint: specifies the endpoint to construct the network up to. It\n      can be one of ['Conv2d_0', 'Conv2d_1_pointwise', 'Conv2d_2_pointwise',\n      'Conv2d_3_pointwise', 'Conv2d_4_pointwise', 'Conv2d_5'_pointwise,\n      'Conv2d_6_pointwise', 'Conv2d_7_pointwise', 'Conv2d_8_pointwise',\n      'Conv2d_9_pointwise', 'Conv2d_10_pointwise', 'Conv2d_11_pointwise',\n      'Conv2d_12_pointwise', 'Conv2d_13_pointwise'].\n    min_depth: Minimum depth value (number of channels) for all convolution ops.\n      Enforced when depth_multiplier < 1, and not an active constraint when\n      depth_multiplier >= 1.\n    depth_multiplier: Float multiplier for the depth (number of channels)\n      for all convolution ops. The value must be greater than zero. Typical\n      usage will be to set this value in (0, 1) to reduce the number of\n      parameters or computation cost of the model.\n    conv_defs: A list of ConvDef namedtuples specifying the net architecture.\n    output_stride: An integer that specifies the requested ratio of input to\n      output spatial resolution. If not None, then we invoke atrous convolution\n      if necessary to prevent the network from reducing the spatial resolution\n      of the activation maps. Allowed values are 8 (accurate fully convolutional\n      mode), 16 (fast fully convolutional mode), 32 (classification mode).\n    use_explicit_padding: Use 'VALID' padding for convolutions, but prepad\n      inputs so that the output dimensions are the same as if 'SAME' padding\n      were used.\n    scope: Optional variable_scope.\n\n  Returns:\n    tensor_out: output tensor corresponding to the final_endpoint.\n    end_points: a set of activations for external use, for example summaries or\n                losses.\n\n  Raises:\n    ValueError: if final_endpoint is not set to one of the predefined values,\n                or depth_multiplier <= 0, or the target output_stride is not\n                allowed.\n  "
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    end_points = {}
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    if conv_defs is None:
        conv_defs = MOBILENETV1_CONV_DEFS
    if output_stride is not None and output_stride not in [8, 16, 32]:
        raise ValueError('Only allowed output_stride values are 8, 16, 32.')
    padding = 'SAME'
    if use_explicit_padding:
        padding = 'VALID'
    with tf.variable_scope(scope, 'MobilenetV1', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding=padding):
            current_stride = 1
            rate = 1
            net = inputs
            for (i, conv_def) in enumerate(conv_defs):
                end_point_base = 'Conv2d_%d' % i
                if output_stride is not None and current_stride == output_stride:
                    layer_stride = 1
                    layer_rate = rate
                    rate *= conv_def.stride
                else:
                    layer_stride = conv_def.stride
                    layer_rate = 1
                    current_stride *= conv_def.stride
                if isinstance(conv_def, Conv):
                    end_point = end_point_base
                    if use_explicit_padding:
                        net = _fixed_padding(net, conv_def.kernel)
                    net = slim.conv2d(net, depth(conv_def.depth), conv_def.kernel, stride=conv_def.stride, scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return (net, end_points)
                elif isinstance(conv_def, DepthSepConv):
                    end_point = end_point_base + '_depthwise'
                    if use_explicit_padding:
                        net = _fixed_padding(net, conv_def.kernel, layer_rate)
                    net = slim.separable_conv2d(net, None, conv_def.kernel, depth_multiplier=1, stride=layer_stride, rate=layer_rate, scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return (net, end_points)
                    end_point = end_point_base + '_pointwise'
                    net = slim.conv2d(net, depth(conv_def.depth), [1, 1], stride=1, scope=end_point)
                    end_points[end_point] = net
                    if end_point == final_endpoint:
                        return (net, end_points)
                else:
                    raise ValueError('Unknown convolution type %s for layer %d' % (conv_def.ltype, i))
    raise ValueError('Unknown final endpoint %s' % final_endpoint)

def mobilenet_v1(inputs, num_classes=1000, dropout_keep_prob=0.999, is_training=True, min_depth=8, depth_multiplier=1.0, conv_defs=None, prediction_fn=contrib_layers.softmax, spatial_squeeze=True, reuse=None, scope='MobilenetV1', global_pool=False):
    if False:
        print('Hello World!')
    "Mobilenet v1 model for classification.\n\n  Args:\n    inputs: a tensor of shape [batch_size, height, width, channels].\n    num_classes: number of predicted classes. If 0 or None, the logits layer\n      is omitted and the input features to the logits layer (before dropout)\n      are returned instead.\n    dropout_keep_prob: the percentage of activation values that are retained.\n    is_training: whether is training or not.\n    min_depth: Minimum depth value (number of channels) for all convolution ops.\n      Enforced when depth_multiplier < 1, and not an active constraint when\n      depth_multiplier >= 1.\n    depth_multiplier: Float multiplier for the depth (number of channels)\n      for all convolution ops. The value must be greater than zero. Typical\n      usage will be to set this value in (0, 1) to reduce the number of\n      parameters or computation cost of the model.\n    conv_defs: A list of ConvDef namedtuples specifying the net architecture.\n    prediction_fn: a function to get predictions out of logits.\n    spatial_squeeze: if True, logits is of shape is [B, C], if false logits is\n        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.\n    reuse: whether or not the network and its variables should be reused. To be\n      able to reuse 'scope' must be given.\n    scope: Optional variable_scope.\n    global_pool: Optional boolean flag to control the avgpooling before the\n      logits layer. If false or unset, pooling is done with a fixed window\n      that reduces default-sized inputs to 1x1, while larger inputs lead to\n      larger outputs. If true, any input size is pooled down to 1x1.\n\n  Returns:\n    net: a 2D Tensor with the logits (pre-softmax activations) if num_classes\n      is a non-zero integer, or the non-dropped-out input to the logits layer\n      if num_classes is 0 or None.\n    end_points: a dictionary from components of the network to the corresponding\n      activation.\n\n  Raises:\n    ValueError: Input rank is invalid.\n  "
    input_shape = inputs.get_shape().as_list()
    if len(input_shape) != 4:
        raise ValueError('Invalid input tensor rank, expected 4, was: %d' % len(input_shape))
    with tf.variable_scope(scope, 'MobilenetV1', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            (net, end_points) = mobilenet_v1_base(inputs, scope=scope, min_depth=min_depth, depth_multiplier=depth_multiplier, conv_defs=conv_defs)
            with tf.variable_scope('Logits'):
                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    end_points['global_pool'] = net
                else:
                    kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
                    net = slim.avg_pool2d(net, kernel_size, padding='VALID', scope='AvgPool_1a')
                    end_points['AvgPool_1a'] = net
                if not num_classes:
                    return (net, end_points)
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            end_points['Logits'] = logits
            if prediction_fn:
                end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return (logits, end_points)
mobilenet_v1.default_image_size = 224

def wrapped_partial(func, *args, **kwargs):
    if False:
        while True:
            i = 10
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func
mobilenet_v1_075 = wrapped_partial(mobilenet_v1, depth_multiplier=0.75)
mobilenet_v1_050 = wrapped_partial(mobilenet_v1, depth_multiplier=0.5)
mobilenet_v1_025 = wrapped_partial(mobilenet_v1, depth_multiplier=0.25)

def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    if False:
        print('Hello World!')
    'Define kernel size which is automatically reduced for small input.\n\n  If the shape of the input images is unknown at graph construction time this\n  function assumes that the input images are large enough.\n\n  Args:\n    input_tensor: input tensor of size [batch_size, height, width, channels].\n    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]\n\n  Returns:\n    a tensor with the kernel size.\n  '
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])]
    return kernel_size_out

def mobilenet_v1_arg_scope(is_training=True, weight_decay=4e-05, stddev=0.09, regularize_depthwise=False, batch_norm_decay=0.9997, batch_norm_epsilon=0.001, batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS, normalizer_fn=slim.batch_norm):
    if False:
        for i in range(10):
            print('nop')
    "Defines the default MobilenetV1 arg scope.\n\n  Args:\n    is_training: Whether or not we're training the model. If this is set to\n      None, the parameter is not added to the batch_norm arg_scope.\n    weight_decay: The weight decay to use for regularizing the model.\n    stddev: The standard deviation of the trunctated normal weight initializer.\n    regularize_depthwise: Whether or not apply regularization on depthwise.\n    batch_norm_decay: Decay for batch norm moving average.\n    batch_norm_epsilon: Small float added to variance to avoid dividing by zero\n      in batch norm.\n    batch_norm_updates_collections: Collection for the update ops for\n      batch norm.\n    normalizer_fn: Normalization function to apply after convolution.\n\n  Returns:\n    An `arg_scope` to use for the mobilenet v1 model.\n  "
    batch_norm_params = {'center': True, 'scale': True, 'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon, 'updates_collections': batch_norm_updates_collections}
    if is_training is not None:
        batch_norm_params['is_training'] = is_training
    weights_init = tf.truncated_normal_initializer(stddev=stddev)
    regularizer = contrib_layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None
    with slim.arg_scope([slim.conv2d, slim.separable_conv2d], weights_initializer=weights_init, activation_fn=tf.nn.relu6, normalizer_fn=normalizer_fn):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
                with slim.arg_scope([slim.separable_conv2d], weights_regularizer=depthwise_regularizer) as sc:
                    return sc