"""Builds the Shake-Shake Model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import custom_ops as ops
import tensorflow as tf

def _shake_shake_skip_connection(x, output_filters, stride):
    if False:
        i = 10
        return i + 15
    'Adds a residual connection to the filter x for the shake-shake model.'
    curr_filters = int(x.shape[3])
    if curr_filters == output_filters:
        return x
    stride_spec = ops.stride_arr(stride, stride)
    path1 = tf.nn.avg_pool(x, [1, 1, 1, 1], stride_spec, 'VALID', data_format='NHWC')
    path1 = ops.conv2d(path1, int(output_filters / 2), 1, scope='path1_conv')
    pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
    path2 = tf.pad(x, pad_arr)[:, 1:, 1:, :]
    concat_axis = 3
    path2 = tf.nn.avg_pool(path2, [1, 1, 1, 1], stride_spec, 'VALID', data_format='NHWC')
    path2 = ops.conv2d(path2, int(output_filters / 2), 1, scope='path2_conv')
    final_path = tf.concat(values=[path1, path2], axis=concat_axis)
    final_path = ops.batch_norm(final_path, scope='final_path_bn')
    return final_path

def _shake_shake_branch(x, output_filters, stride, rand_forward, rand_backward, is_training):
    if False:
        for i in range(10):
            print('nop')
    'Building a 2 branching convnet.'
    x = tf.nn.relu(x)
    x = ops.conv2d(x, output_filters, 3, stride=stride, scope='conv1')
    x = ops.batch_norm(x, scope='bn1')
    x = tf.nn.relu(x)
    x = ops.conv2d(x, output_filters, 3, scope='conv2')
    x = ops.batch_norm(x, scope='bn2')
    if is_training:
        x = x * rand_backward + tf.stop_gradient(x * rand_forward - x * rand_backward)
    else:
        x *= 1.0 / 2
    return x

def _shake_shake_block(x, output_filters, stride, is_training):
    if False:
        return 10
    'Builds a full shake-shake sub layer.'
    batch_size = tf.shape(x)[0]
    rand_forward = [tf.random_uniform([batch_size, 1, 1, 1], minval=0, maxval=1, dtype=tf.float32) for _ in range(2)]
    rand_backward = [tf.random_uniform([batch_size, 1, 1, 1], minval=0, maxval=1, dtype=tf.float32) for _ in range(2)]
    total_forward = tf.add_n(rand_forward)
    total_backward = tf.add_n(rand_backward)
    rand_forward = [samp / total_forward for samp in rand_forward]
    rand_backward = [samp / total_backward for samp in rand_backward]
    zipped_rand = zip(rand_forward, rand_backward)
    branches = []
    for (branch, (r_forward, r_backward)) in enumerate(zipped_rand):
        with tf.variable_scope('branch_{}'.format(branch)):
            b = _shake_shake_branch(x, output_filters, stride, r_forward, r_backward, is_training)
            branches.append(b)
    res = _shake_shake_skip_connection(x, output_filters, stride)
    return res + tf.add_n(branches)

def _shake_shake_layer(x, output_filters, num_blocks, stride, is_training):
    if False:
        for i in range(10):
            print('nop')
    'Builds many sub layers into one full layer.'
    for block_num in range(num_blocks):
        curr_stride = stride if block_num == 0 else 1
        with tf.variable_scope('layer_{}'.format(block_num)):
            x = _shake_shake_block(x, output_filters, curr_stride, is_training)
    return x

def build_shake_shake_model(images, num_classes, hparams, is_training):
    if False:
        for i in range(10):
            print('nop')
    'Builds the Shake-Shake model.\n\n  Build the Shake-Shake model from https://arxiv.org/abs/1705.07485.\n\n  Args:\n    images: Tensor of images that will be fed into the Wide ResNet Model.\n    num_classes: Number of classed that the model needs to predict.\n    hparams: tf.HParams object that contains additional hparams needed to\n      construct the model. In this case it is the `shake_shake_widen_factor`\n      that is used to determine how many filters the model has.\n    is_training: Is the model training or not.\n\n  Returns:\n    The logits of the Shake-Shake model.\n  '
    depth = 26
    k = hparams.shake_shake_widen_factor
    n = int((depth - 2) / 6)
    x = images
    x = ops.conv2d(x, 16, 3, scope='init_conv')
    x = ops.batch_norm(x, scope='init_bn')
    with tf.variable_scope('L1'):
        x = _shake_shake_layer(x, 16 * k, n, 1, is_training)
    with tf.variable_scope('L2'):
        x = _shake_shake_layer(x, 32 * k, n, 2, is_training)
    with tf.variable_scope('L3'):
        x = _shake_shake_layer(x, 64 * k, n, 2, is_training)
    x = tf.nn.relu(x)
    x = ops.global_avg_pool(x)
    logits = ops.fc(x, num_classes)
    return logits