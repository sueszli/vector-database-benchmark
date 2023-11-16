"""Contains the definition for inception v3 classification network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from nets import inception_utils
slim = contrib_slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)

def inception_v3_base(inputs, final_endpoint='Mixed_7c', min_depth=16, depth_multiplier=1.0, scope=None):
    if False:
        while True:
            i = 10
    "Inception model from http://arxiv.org/abs/1512.00567.\n\n  Constructs an Inception v3 network from inputs to the given final endpoint.\n  This method can construct the network up to the final inception block\n  Mixed_7c.\n\n  Note that the names of the layers in the paper do not correspond to the names\n  of the endpoints registered by this function although they build the same\n  network.\n\n  Here is a mapping from the old_names to the new names:\n  Old name          | New name\n  =======================================\n  conv0             | Conv2d_1a_3x3\n  conv1             | Conv2d_2a_3x3\n  conv2             | Conv2d_2b_3x3\n  pool1             | MaxPool_3a_3x3\n  conv3             | Conv2d_3b_1x1\n  conv4             | Conv2d_4a_3x3\n  pool2             | MaxPool_5a_3x3\n  mixed_35x35x256a  | Mixed_5b\n  mixed_35x35x288a  | Mixed_5c\n  mixed_35x35x288b  | Mixed_5d\n  mixed_17x17x768a  | Mixed_6a\n  mixed_17x17x768b  | Mixed_6b\n  mixed_17x17x768c  | Mixed_6c\n  mixed_17x17x768d  | Mixed_6d\n  mixed_17x17x768e  | Mixed_6e\n  mixed_8x8x1280a   | Mixed_7a\n  mixed_8x8x2048a   | Mixed_7b\n  mixed_8x8x2048b   | Mixed_7c\n\n  Args:\n    inputs: a tensor of size [batch_size, height, width, channels].\n    final_endpoint: specifies the endpoint to construct the network up to. It\n      can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',\n      'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',\n      'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',\n      'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c'].\n    min_depth: Minimum depth value (number of channels) for all convolution ops.\n      Enforced when depth_multiplier < 1, and not an active constraint when\n      depth_multiplier >= 1.\n    depth_multiplier: Float multiplier for the depth (number of channels)\n      for all convolution ops. The value must be greater than zero. Typical\n      usage will be to set this value in (0, 1) to reduce the number of\n      parameters or computation cost of the model.\n    scope: Optional variable_scope.\n\n  Returns:\n    tensor_out: output tensor corresponding to the final_endpoint.\n    end_points: a set of activations for external use, for example summaries or\n                losses.\n\n  Raises:\n    ValueError: if final_endpoint is not set to one of the predefined values,\n                or depth_multiplier <= 0\n  "
    end_points = {}
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    with tf.variable_scope(scope, 'InceptionV3', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='VALID'):
            end_point = 'Conv2d_1a_3x3'
            net = slim.conv2d(inputs, depth(32), [3, 3], stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return (net, end_points)
            end_point = 'Conv2d_2a_3x3'
            net = slim.conv2d(net, depth(32), [3, 3], scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return (net, end_points)
            end_point = 'Conv2d_2b_3x3'
            net = slim.conv2d(net, depth(64), [3, 3], padding='SAME', scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return (net, end_points)
            end_point = 'MaxPool_3a_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return (net, end_points)
            end_point = 'Conv2d_3b_1x1'
            net = slim.conv2d(net, depth(80), [1, 1], scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return (net, end_points)
            end_point = 'Conv2d_4a_3x3'
            net = slim.conv2d(net, depth(192), [3, 3], scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return (net, end_points)
            end_point = 'MaxPool_5a_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
            end_points[end_point] = net
            if end_point == final_endpoint:
                return (net, end_points)
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
            end_point = 'Mixed_5b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(32), [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return (net, end_points)
            end_point = 'Mixed_5c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(64), [5, 5], scope='Conv_1_0c_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return (net, end_points)
            end_point = 'Mixed_5d'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(64), [5, 5], scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = slim.conv2d(branch_2, depth(96), [3, 3], scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(64), [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return (net, end_points)
            end_point = 'Mixed_6a'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(384), [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], scope='Conv2d_0b_3x3')
                    branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_1x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return (net, end_points)
            end_point = 'Mixed_6b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(128), [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(128), [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(128), [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, depth(128), [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return (net, end_points)
            end_point = 'Mixed_6c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return (net, end_points)
            end_point = 'Mixed_6d'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(160), [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(160), [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, depth(160), [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return (net, end_points)
            end_point = 'Mixed_6e'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [7, 1], scope='Conv2d_0b_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0c_1x7')
                    branch_2 = slim.conv2d(branch_2, depth(192), [7, 1], scope='Conv2d_0d_7x1')
                    branch_2 = slim.conv2d(branch_2, depth(192), [1, 7], scope='Conv2d_0e_1x7')
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return (net, end_points)
            end_point = 'Mixed_7a'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_0 = slim.conv2d(branch_0, depth(320), [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = slim.conv2d(branch_1, depth(192), [1, 7], scope='Conv2d_0b_1x7')
                    branch_1 = slim.conv2d(branch_1, depth(192), [7, 1], scope='Conv2d_0c_7x1')
                    branch_1 = slim.conv2d(branch_1, depth(192), [3, 3], stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return (net, end_points)
            end_point = 'Mixed_7b'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat(axis=3, values=[slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'), slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')])
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat(axis=3, values=[slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'), slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return (net, end_points)
            end_point = 'Mixed_7c'
            with tf.variable_scope(end_point):
                with tf.variable_scope('Branch_0'):
                    branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
                    branch_1 = tf.concat(axis=3, values=[slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'), slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')])
                with tf.variable_scope('Branch_2'):
                    branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
                    branch_2 = slim.conv2d(branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
                    branch_2 = tf.concat(axis=3, values=[slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'), slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
                with tf.variable_scope('Branch_3'):
                    branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
                    branch_3 = slim.conv2d(branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
                net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
            end_points[end_point] = net
            if end_point == final_endpoint:
                return (net, end_points)
        raise ValueError('Unknown final endpoint %s' % final_endpoint)

def inception_v3(inputs, num_classes=1000, is_training=True, dropout_keep_prob=0.8, min_depth=16, depth_multiplier=1.0, prediction_fn=slim.softmax, spatial_squeeze=True, reuse=None, create_aux_logits=True, scope='InceptionV3', global_pool=False):
    if False:
        print('Hello World!')
    'Inception model from http://arxiv.org/abs/1512.00567.\n\n  "Rethinking the Inception Architecture for Computer Vision"\n\n  Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,\n  Zbigniew Wojna.\n\n  With the default arguments this method constructs the exact model defined in\n  the paper. However, one can experiment with variations of the inception_v3\n  network by changing arguments dropout_keep_prob, min_depth and\n  depth_multiplier.\n\n  The default image size used to train this network is 299x299.\n\n  Args:\n    inputs: a tensor of size [batch_size, height, width, channels].\n    num_classes: number of predicted classes. If 0 or None, the logits layer\n      is omitted and the input features to the logits layer (before dropout)\n      are returned instead.\n    is_training: whether is training or not.\n    dropout_keep_prob: the percentage of activation values that are retained.\n    min_depth: Minimum depth value (number of channels) for all convolution ops.\n      Enforced when depth_multiplier < 1, and not an active constraint when\n      depth_multiplier >= 1.\n    depth_multiplier: Float multiplier for the depth (number of channels)\n      for all convolution ops. The value must be greater than zero. Typical\n      usage will be to set this value in (0, 1) to reduce the number of\n      parameters or computation cost of the model.\n    prediction_fn: a function to get predictions out of logits.\n    spatial_squeeze: if True, logits is of shape [B, C], if false logits is of\n        shape [B, 1, 1, C], where B is batch_size and C is number of classes.\n    reuse: whether or not the network and its variables should be reused. To be\n      able to reuse \'scope\' must be given.\n    create_aux_logits: Whether to create the auxiliary logits.\n    scope: Optional variable_scope.\n    global_pool: Optional boolean flag to control the avgpooling before the\n      logits layer. If false or unset, pooling is done with a fixed window\n      that reduces default-sized inputs to 1x1, while larger inputs lead to\n      larger outputs. If true, any input size is pooled down to 1x1.\n\n  Returns:\n    net: a Tensor with the logits (pre-softmax activations) if num_classes\n      is a non-zero integer, or the non-dropped-out input to the logits layer\n      if num_classes is 0 or None.\n    end_points: a dictionary from components of the network to the corresponding\n      activation.\n\n  Raises:\n    ValueError: if \'depth_multiplier\' is less than or equal to zero.\n  '
    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    with tf.variable_scope(scope, 'InceptionV3', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            (net, end_points) = inception_v3_base(inputs, scope=scope, min_depth=min_depth, depth_multiplier=depth_multiplier)
            if create_aux_logits and num_classes:
                with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                    aux_logits = end_points['Mixed_6e']
                    with tf.variable_scope('AuxLogits'):
                        aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID', scope='AvgPool_1a_5x5')
                        aux_logits = slim.conv2d(aux_logits, depth(128), [1, 1], scope='Conv2d_1b_1x1')
                        kernel_size = _reduced_kernel_size_for_small_input(aux_logits, [5, 5])
                        aux_logits = slim.conv2d(aux_logits, depth(768), kernel_size, weights_initializer=trunc_normal(0.01), padding='VALID', scope='Conv2d_2a_{}x{}'.format(*kernel_size))
                        aux_logits = slim.conv2d(aux_logits, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, weights_initializer=trunc_normal(0.001), scope='Conv2d_2b_1x1')
                        if spatial_squeeze:
                            aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
                        end_points['AuxLogits'] = aux_logits
            with tf.variable_scope('Logits'):
                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='GlobalPool')
                    end_points['global_pool'] = net
                else:
                    kernel_size = _reduced_kernel_size_for_small_input(net, [8, 8])
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
inception_v3.default_image_size = 299

def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    if False:
        while True:
            i = 10
    'Define kernel size which is automatically reduced for small input.\n\n  If the shape of the input images is unknown at graph construction time this\n  function assumes that the input images are is large enough.\n\n  Args:\n    input_tensor: input tensor of size [batch_size, height, width, channels].\n    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]\n\n  Returns:\n    a tensor with the kernel size.\n\n  TODO(jrru): Make this function work with unknown shapes. Theoretically, this\n  can be done with the code below. Problems are two-fold: (1) If the shape was\n  known, it will be lost. (2) inception.slim.ops._two_element_tuple cannot\n  handle tensors that define the kernel size.\n      shape = tf.shape(input_tensor)\n      return = tf.stack([tf.minimum(shape[1], kernel_size[0]),\n                         tf.minimum(shape[2], kernel_size[1])])\n\n  '
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]), min(shape[2], kernel_size[1])]
    return kernel_size_out
inception_v3_arg_scope = inception_utils.inception_arg_scope