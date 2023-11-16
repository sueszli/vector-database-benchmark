"""Contains the definition for inception v2 classification network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from nets import inception_utils
slim = contrib_slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def inception_v2_base(inputs, final_endpoint='Mixed_5c', min_depth=16, depth_multiplier=1.0, use_separable_conv=True, data_format='NHWC', include_root_block=True, scope=None):
    if False:
        return 10
    "Inception v2 (6a2).\n\n  Constructs an Inception v2 network from inputs to the given final endpoint.\n  This method can construct the network up to the layer inception(5b) as\n  described in http://arxiv.org/abs/1502.03167.\n\n  Args:\n    inputs: a tensor of shape [batch_size, height, width, channels].\n    final_endpoint: specifies the endpoint to construct the network up to. It\n      can be one of ['Conv2d_1a_7x7', 'MaxPool_2a_3x3', 'Conv2d_2b_1x1',\n      'Conv2d_2c_3x3', 'MaxPool_3a_3x3', 'Mixed_3b', 'Mixed_3c', 'Mixed_4a',\n      'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_5a', 'Mixed_5b',\n      'Mixed_5c']. If include_root_block is False, ['Conv2d_1a_7x7',\n      'MaxPool_2a_3x3', 'Conv2d_2b_1x1', 'Conv2d_2c_3x3', 'MaxPool_3a_3x3'] will\n      not be available.\n    min_depth: Minimum depth value (number of channels) for all convolution ops.\n      Enforced when depth_multiplier < 1, and not an active constraint when\n      depth_multiplier >= 1.\n    depth_multiplier: Float multiplier for the depth (number of channels)\n      for all convolution ops. The value must be greater than zero. Typical\n      usage will be to set this value in (0, 1) to reduce the number of\n      parameters or computation cost of the model.\n    use_separable_conv: Use a separable convolution for the first layer\n      Conv2d_1a_7x7. If this is False, use a normal convolution instead.\n    data_format: Data format of the activations ('NHWC' or 'NCHW').\n    include_root_block: If True, include the convolution and max-pooling layers\n      before the inception modules. If False, excludes those layers.\n    scope: Optional variable_scope.\n\n  Returns:\n    tensor_out: output tensor corresponding to the final_endpoint.\n    end_points: a set of activations for external use, for example summaries or\n                losses.\n\n  Raises:\n    ValueError: if final_endpoint is not set to one of the predefined values,\n                or depth_multiplier <= 0\n  "
    end_points = {}
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    if data_format != 'NHWC' and data_format != 'NCHW':
        raise ValueError('data_format must be either NHWC or NCHW.')
    if data_format == 'NCHW' and use_separable_conv:
        raise ValueError('separable convolution only supports NHWC layout. NCHW data format can only be used when use_separable_conv is False.')
    concat_dim = 3 if data_format == 'NHWC' else 1
    with tf.variable_scope(scope, 'InceptionV2', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME', data_format=data_format):
            net = inputs
            if include_root_block:
                end_point = 'Conv2d_1a_7x7'
                if use_separable_conv:
                    depthwise_multiplier = min(int(depth(64) / 3), 8)
                    net = slim.separable_conv2d(inputs, depth(64), [7, 7], depth_multiplier=depthwise_multiplier, stride=2, padding='SAME', weights_initializer=trunc_normal(1.0), scope=end_point)
                else:
                    net = slim.conv2d(inputs, depth(64), [7, 7], stride=2, weights_initializer=trunc_normal(1.0), scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return (net, end_points)
                end_point = 'MaxPool_2a_3x3'
                net = slim.max_pool2d(net, [3, 3], scope=end_point, stride=2)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return (net, end_points)
                end_point = 'Conv2d_2b_1x1'
                net = slim.conv2d(net, depth(64), [1, 1], scope=end_point, weights_initializer=trunc_normal(0.1))
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return (net, end_points)
                end_point = 'Conv2d_2c_3x3'
                net = slim.conv2d(net, depth(192), [3, 3], scope=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return (net, end_points)
                end_point = 'MaxPool_3a_3x3'
                net = slim.max_pool2d(net, [3, 3], scope=end_point, stride=2)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return (net, end_points)
            end_point = 'Mixed_3b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(64), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(64), [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(64), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(32), [1, 1], weights_initializer=trunc_normal(0.1), scope='Conv2d_0b_1x1')
                net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return (net, end_points)
            end_point = 'Mixed_3c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(64), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(64), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(64), [1, 1], weights_initializer=trunc_normal(0.1), scope='Conv2d_0b_1x1')
                net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return (net, end_points)
            end_point = 'Mixed_4a'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(128), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, depth(160), [3, 3], stride=2, scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(64), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], stride=2, scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_1a_3x3')
                net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2])
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return (net, end_points)
            end_point = 'Mixed_4b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(224), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(64), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(96), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(128), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(128), [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(128), [1, 1], weights_initializer=trunc_normal(0.1), scope='Conv2d_0b_1x1')
                net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return (net, end_points)
            end_point = 'Mixed_4c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(96), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(128), [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(96), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(128), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(128), [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(128), [1, 1], weights_initializer=trunc_normal(0.1), scope='Conv2d_0b_1x1')
                net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return (net, end_points)
            end_point = 'Mixed_4d'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(128), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(160), [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(128), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(160), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(160), [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(96), [1, 1], weights_initializer=trunc_normal(0.1), scope='Conv2d_0b_1x1')
                net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return (net, end_points)
            end_point = 'Mixed_4e'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(96), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(128), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(192), [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(160), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(192), [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(96), [1, 1], weights_initializer=trunc_normal(0.1), scope='Conv2d_0b_1x1')
                net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return (net, end_points)
            end_point = 'Mixed_5a'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(128), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, depth(192), [3, 3], stride=2, scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(192), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(256), [3, 3], scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, depth(256), [3, 3], stride=2, scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, scope='MaxPool_1a_3x3')
                net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2])
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return (net, end_points)
            end_point = 'Mixed_5b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(192), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(320), [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(160), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(224), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(224), [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(128), [1, 1], weights_initializer=trunc_normal(0.1), scope='Conv2d_0b_1x1')
                net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return (net, end_points)
            end_point = 'Mixed_5c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(352), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(192), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(320), [3, 3], scope='Conv2d_0b_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(192), [1, 1], weights_initializer=trunc_normal(0.09), scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(224), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(224), [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(128), [1, 1], weights_initializer=trunc_normal(0.1), scope='Conv2d_0b_1x1')
                net = tf.concat(axis=concat_dim, values=[branch_0, branch_1, branch_2, branch_3])
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return (net, end_points)
        raise ValueError('Unknown final endpoint %s' % final_endpoint)

def inception_v2(inputs, num_classes=1000, is_training=True, dropout_keep_prob=0.8, min_depth=16, depth_multiplier=1.0, prediction_fn=slim.softmax, spatial_squeeze=True, reuse=None, scope='InceptionV2', global_pool=False):
    if False:
        return 10
    "Inception v2 model for classification.\n\n  Constructs an Inception v2 network for classification as described in\n  http://arxiv.org/abs/1502.03167.\n\n  The default image size used to train this network is 224x224.\n\n  Args:\n    inputs: a tensor of shape [batch_size, height, width, channels].\n    num_classes: number of predicted classes. If 0 or None, the logits layer\n      is omitted and the input features to the logits layer (before dropout)\n      are returned instead.\n    is_training: whether is training or not.\n    dropout_keep_prob: the percentage of activation values that are retained.\n    min_depth: Minimum depth value (number of channels) for all convolution ops.\n      Enforced when depth_multiplier < 1, and not an active constraint when\n      depth_multiplier >= 1.\n    depth_multiplier: Float multiplier for the depth (number of channels)\n      for all convolution ops. The value must be greater than zero. Typical\n      usage will be to set this value in (0, 1) to reduce the number of\n      parameters or computation cost of the model.\n    prediction_fn: a function to get predictions out of logits.\n    spatial_squeeze: if True, logits is of shape [B, C], if false logits is of\n        shape [B, 1, 1, C], where B is batch_size and C is number of classes.\n    reuse: whether or not the network and its variables should be reused. To be\n      able to reuse 'scope' must be given.\n    scope: Optional variable_scope.\n    global_pool: Optional boolean flag to control the avgpooling before the\n      logits layer. If false or unset, pooling is done with a fixed window\n      that reduces default-sized inputs to 1x1, while larger inputs lead to\n      larger outputs. If true, any input size is pooled down to 1x1.\n\n  Returns:\n    net: a Tensor with the logits (pre-softmax activations) if num_classes\n      is a non-zero integer, or the non-dropped-out input to the logits layer\n      if num_classes is 0 or None.\n    end_points: a dictionary from components of the network to the corresponding\n      activation.\n\n  Raises:\n    ValueError: if final_endpoint is not set to one of the predefined values,\n                or depth_multiplier <= 0\n  "
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    with tf.variable_scope(scope, 'InceptionV2', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            (net, end_points) = inception_v2_base(inputs, scope=scope, min_depth=min_depth, depth_multiplier=depth_multiplier)
            with tf.variable_scope('Logits'):
                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
                    end_points['global_pool'] = net
                else:
                    kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
                    net = slim.avg_pool2d(net, kernel_size, padding='VALID', scope='AvgPool_1a_{}x{}'.format(*kernel_size))
                    end_points['AvgPool_1a'] = net
                if not num_classes:
                    return (net, end_points)
                net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
                end_points['PreLogits'] = net
                logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Conv2d_1c_1x1')
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
    return (logits, end_points)
inception_v2.default_image_size = 224

def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    if False:
        for i in range(10):
            print('nop')
    'Define kernel size which is automatically reduced for small input.\n\n  If the shape of the input images is unknown at graph construction time this\n  function assumes that the input images are is large enough.\n\n  Args:\n    input_tensor: input tensor of size [batch_size, height, width, channels].\n    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]\n\n  Returns:\n    a tensor with the kernel size.\n\n  TODO(jrru): Make this function work with unknown shapes. Theoretically, this\n  can be done with the code below. Problems are two-fold: (1) If the shape was\n  known, it will be lost. (2) inception.slim.ops._two_element_tuple cannot\n  handle tensors that define the kernel size.\n      shape = tf.shape(input_tensor)\n      return = tf.stack([tf.minimum(shape[1], kernel_size[0]),\n                         tf.minimum(shape[2], kernel_size[1])])\n\n  '
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])]
    return kernel_size_out
inception_v2_arg_scope = inception_utils.inception_arg_scope