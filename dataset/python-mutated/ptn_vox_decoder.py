"""Training decoder as used in PTN (NIPS16)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim

@tf.contrib.framework.add_arg_scope
def conv3d_transpose(inputs, num_outputs, kernel_size, stride=1, padding='SAME', activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(), biases_initializer=tf.zeros_initializer(), reuse=None, trainable=True, scope=None):
    if False:
        i = 10
        return i + 15
    "Wrapper for conv3d_transpose layer.\n\n  This function wraps the tf.conv3d_transpose with basic non-linearity.\n  Tt creates a variable called `weights`, representing the kernel,\n  that is convoled with the input. A second varibale called `biases'\n  is added to the result of operation.\n  "
    with tf.variable_scope(scope, 'Conv3d_transpose', [inputs], reuse=reuse):
        dtype = inputs.dtype.base_dtype
        (kernel_d, kernel_h, kernel_w) = kernel_size[0:3]
        num_filters_in = inputs.get_shape()[4]
        weights_shape = [kernel_d, kernel_h, kernel_w, num_outputs, num_filters_in]
        weights = tf.get_variable('weights', shape=weights_shape, dtype=dtype, initializer=weights_initializer, trainable=trainable)
        tf.contrib.framework.add_model_variable(weights)
        input_shape = inputs.get_shape().as_list()
        batch_size = input_shape[0]
        depth = input_shape[1]
        height = input_shape[2]
        width = input_shape[3]

        def get_deconv_dim(dim_size, stride_size):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(dim_size, tf.Tensor):
                dim_size = tf.multiply(dim_size, stride_size)
            elif dim_size is not None:
                dim_size *= stride_size
            return dim_size
        out_depth = get_deconv_dim(depth, stride)
        out_height = get_deconv_dim(height, stride)
        out_width = get_deconv_dim(width, stride)
        out_shape = [batch_size, out_depth, out_height, out_width, num_outputs]
        outputs = tf.nn.conv3d_transpose(inputs, weights, out_shape, [1, stride, stride, stride, 1], padding=padding)
        outputs.set_shape(out_shape)
        if biases_initializer is not None:
            biases = tf.get_variable('biases', shape=[num_outputs], dtype=dtype, initializer=biases_initializer, trainable=trainable)
            tf.contrib.framework.add_model_variable(biases)
            outputs = tf.nn.bias_add(outputs, biases)
        if activation_fn:
            outputs = activation_fn(outputs)
        return outputs

def model(identities, params, is_training):
    if False:
        print('Hello World!')
    'Model transforming embedding to voxels.'
    del is_training
    f_dim = params.f_dim
    with slim.arg_scope([slim.fully_connected, conv3d_transpose], weights_initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1)):
        h0 = slim.fully_connected(identities, 4 * 4 * 4 * f_dim * 8, activation_fn=tf.nn.relu)
        h1 = tf.reshape(h0, [-1, 4, 4, 4, f_dim * 8])
        h1 = conv3d_transpose(h1, f_dim * 4, [4, 4, 4], stride=2, activation_fn=tf.nn.relu)
        h2 = conv3d_transpose(h1, int(f_dim * 3 / 2), [5, 5, 5], stride=2, activation_fn=tf.nn.relu)
        h3 = conv3d_transpose(h2, 1, [6, 6, 6], stride=2, activation_fn=tf.nn.sigmoid)
    return h3