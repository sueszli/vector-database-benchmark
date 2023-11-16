"""Convolutional LSTM implementation."""
import tensorflow as tf
from tensorflow.contrib.slim import add_arg_scope
from tensorflow.contrib.slim import layers

def init_state(inputs, state_shape, state_initializer=tf.zeros_initializer(), dtype=tf.float32):
    if False:
        print('Hello World!')
    'Helper function to create an initial state given inputs.\n\n  Args:\n    inputs: input Tensor, at least 2D, the first dimension being batch_size\n    state_shape: the shape of the state.\n    state_initializer: Initializer(shape, dtype) for state Tensor.\n    dtype: Optional dtype, needed when inputs is None.\n  Returns:\n     A tensors representing the initial state.\n  '
    if inputs is not None:
        inferred_batch_size = inputs.get_shape().with_rank_at_least(1)[0]
        dtype = inputs.dtype
    else:
        inferred_batch_size = 0
    initial_state = state_initializer([inferred_batch_size] + state_shape, dtype=dtype)
    return initial_state

@add_arg_scope
def basic_conv_lstm_cell(inputs, state, num_channels, filter_size=5, forget_bias=1.0, scope=None, reuse=None):
    if False:
        return 10
    'Basic LSTM recurrent network cell, with 2D convolution connctions.\n\n  We add forget_bias (default: 1) to the biases of the forget gate in order to\n  reduce the scale of forgetting in the beginning of the training.\n\n  It does not allow cell clipping, a projection layer, and does not\n  use peep-hole connections: it is the basic baseline.\n\n  Args:\n    inputs: input Tensor, 4D, batch x height x width x channels.\n    state: state Tensor, 4D, batch x height x width x channels.\n    num_channels: the number of output channels in the layer.\n    filter_size: the shape of the each convolution filter.\n    forget_bias: the initial value of the forget biases.\n    scope: Optional scope for variable_scope.\n    reuse: whether or not the layer and the variables should be reused.\n\n  Returns:\n     a tuple of tensors representing output and the new state.\n  '
    spatial_size = inputs.get_shape()[1:3]
    if state is None:
        state = init_state(inputs, list(spatial_size) + [2 * num_channels])
    with tf.variable_scope(scope, 'BasicConvLstmCell', [inputs, state], reuse=reuse):
        inputs.get_shape().assert_has_rank(4)
        state.get_shape().assert_has_rank(4)
        (c, h) = tf.split(axis=3, num_or_size_splits=2, value=state)
        inputs_h = tf.concat(axis=3, values=[inputs, h])
        i_j_f_o = layers.conv2d(inputs_h, 4 * num_channels, [filter_size, filter_size], stride=1, activation_fn=None, scope='Gates')
        (i, j, f, o) = tf.split(axis=3, num_or_size_splits=4, value=i_j_f_o)
        new_c = c * tf.sigmoid(f + forget_bias) + tf.sigmoid(i) * tf.tanh(j)
        new_h = tf.tanh(new_c) * tf.sigmoid(o)
        return (new_h, tf.concat(axis=3, values=[new_c, new_h]))