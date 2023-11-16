import numpy as np
import tensorflow as tf

def ortho_init(scale=1.0):
    if False:
        return 10
    '\n    Orthogonal initialization for the policy weights\n\n    :param scale: (float) Scaling factor for the weights.\n    :return: (function) an initialization function for the weights\n    '

    def _ortho_init(shape, *_, **_kwargs):
        if False:
            while True:
                i = 10
        'Intialize weights as Orthogonal matrix.\n\n        Orthogonal matrix initialization [1]_. For n-dimensional shapes where\n        n > 2, the n-1 trailing axes are flattened. For convolutional layers, this\n        corresponds to the fan-in, so this makes the initialization usable for\n        both dense and convolutional layers.\n\n        References\n        ----------\n        .. [1] Saxe, Andrew M., James L. McClelland, and Surya Ganguli.\n               "Exact solutions to the nonlinear dynamics of learning in deep\n               linear\n        '
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4:
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        gaussian_noise = np.random.normal(0.0, 1.0, flat_shape)
        (u, _, v) = np.linalg.svd(gaussian_noise, full_matrices=False)
        weights = u if u.shape == flat_shape else v
        weights = weights.reshape(shape)
        return (scale * weights[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init

def mlp(input_tensor, layers, activ_fn=tf.nn.relu, layer_norm=False):
    if False:
        while True:
            i = 10
    '\n    Create a multi-layer fully connected neural network.\n\n    :param input_tensor: (tf.placeholder)\n    :param layers: ([int]) Network architecture\n    :param activ_fn: (tf.function) Activation function\n    :param layer_norm: (bool) Whether to apply layer normalization or not\n    :return: (tf.Tensor)\n    '
    output = input_tensor
    for (i, layer_size) in enumerate(layers):
        output = tf.layers.dense(output, layer_size, name='fc' + str(i))
        if layer_norm:
            output = tf.contrib.layers.layer_norm(output, center=True, scale=True)
        output = activ_fn(output)
    return output

def conv(input_tensor, scope, *, n_filters, filter_size, stride, pad='VALID', init_scale=1.0, data_format='NHWC', one_dim_bias=False):
    if False:
        return 10
    "\n    Creates a 2d convolutional layer for TensorFlow\n\n    :param input_tensor: (TensorFlow Tensor) The input tensor for the convolution\n    :param scope: (str) The TensorFlow variable scope\n    :param n_filters: (int) The number of filters\n    :param filter_size:  (Union[int, [int], tuple<int, int>]) The filter size for the squared kernel matrix,\n    or the height and width of kernel filter if the input is a list or tuple\n    :param stride: (int) The stride of the convolution\n    :param pad: (str) The padding type ('VALID' or 'SAME')\n    :param init_scale: (int) The initialization scale\n    :param data_format: (str) The data format for the convolution weights\n    :param one_dim_bias: (bool) If the bias should be one dimentional or not\n    :return: (TensorFlow Tensor) 2d convolutional layer\n    "
    if isinstance(filter_size, list) or isinstance(filter_size, tuple):
        assert len(filter_size) == 2, 'Filter size must have 2 elements (height, width), {} were given'.format(len(filter_size))
        filter_height = filter_size[0]
        filter_width = filter_size[1]
    else:
        filter_height = filter_size
        filter_width = filter_size
    if data_format == 'NHWC':
        channel_ax = 3
        strides = [1, stride, stride, 1]
        bshape = [1, 1, 1, n_filters]
    elif data_format == 'NCHW':
        channel_ax = 1
        strides = [1, 1, stride, stride]
        bshape = [1, n_filters, 1, 1]
    else:
        raise NotImplementedError
    bias_var_shape = [n_filters] if one_dim_bias else [1, n_filters, 1, 1]
    n_input = input_tensor.get_shape()[channel_ax].value
    wshape = [filter_height, filter_width, n_input, n_filters]
    with tf.variable_scope(scope):
        weight = tf.get_variable('w', wshape, initializer=ortho_init(init_scale))
        bias = tf.get_variable('b', bias_var_shape, initializer=tf.constant_initializer(0.0))
        if not one_dim_bias and data_format == 'NHWC':
            bias = tf.reshape(bias, bshape)
        return bias + tf.nn.conv2d(input_tensor, weight, strides=strides, padding=pad, data_format=data_format)

def linear(input_tensor, scope, n_hidden, *, init_scale=1.0, init_bias=0.0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Creates a fully connected layer for TensorFlow\n\n    :param input_tensor: (TensorFlow Tensor) The input tensor for the fully connected layer\n    :param scope: (str) The TensorFlow variable scope\n    :param n_hidden: (int) The number of hidden neurons\n    :param init_scale: (int) The initialization scale\n    :param init_bias: (int) The initialization offset bias\n    :return: (TensorFlow Tensor) fully connected layer\n    '
    with tf.variable_scope(scope):
        n_input = input_tensor.get_shape()[1].value
        weight = tf.get_variable('w', [n_input, n_hidden], initializer=ortho_init(init_scale))
        bias = tf.get_variable('b', [n_hidden], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(input_tensor, weight) + bias

def lstm(input_tensor, mask_tensor, cell_state_hidden, scope, n_hidden, init_scale=1.0, layer_norm=False):
    if False:
        print('Hello World!')
    '\n    Creates an Long Short Term Memory (LSTM) cell for TensorFlow\n\n    :param input_tensor: (TensorFlow Tensor) The input tensor for the LSTM cell\n    :param mask_tensor: (TensorFlow Tensor) The mask tensor for the LSTM cell\n    :param cell_state_hidden: (TensorFlow Tensor) The state tensor for the LSTM cell\n    :param scope: (str) The TensorFlow variable scope\n    :param n_hidden: (int) The number of hidden neurons\n    :param init_scale: (int) The initialization scale\n    :param layer_norm: (bool) Whether to apply Layer Normalization or not\n    :return: (TensorFlow Tensor) LSTM cell\n    '
    (_, n_input) = [v.value for v in input_tensor[0].get_shape()]
    with tf.variable_scope(scope):
        weight_x = tf.get_variable('wx', [n_input, n_hidden * 4], initializer=ortho_init(init_scale))
        weight_h = tf.get_variable('wh', [n_hidden, n_hidden * 4], initializer=ortho_init(init_scale))
        bias = tf.get_variable('b', [n_hidden * 4], initializer=tf.constant_initializer(0.0))
        if layer_norm:
            gain_x = tf.get_variable('gx', [n_hidden * 4], initializer=tf.constant_initializer(1.0))
            bias_x = tf.get_variable('bx', [n_hidden * 4], initializer=tf.constant_initializer(0.0))
            gain_h = tf.get_variable('gh', [n_hidden * 4], initializer=tf.constant_initializer(1.0))
            bias_h = tf.get_variable('bh', [n_hidden * 4], initializer=tf.constant_initializer(0.0))
            gain_c = tf.get_variable('gc', [n_hidden], initializer=tf.constant_initializer(1.0))
            bias_c = tf.get_variable('bc', [n_hidden], initializer=tf.constant_initializer(0.0))
    (cell_state, hidden) = tf.split(axis=1, num_or_size_splits=2, value=cell_state_hidden)
    for (idx, (_input, mask)) in enumerate(zip(input_tensor, mask_tensor)):
        cell_state = cell_state * (1 - mask)
        hidden = hidden * (1 - mask)
        if layer_norm:
            gates = _ln(tf.matmul(_input, weight_x), gain_x, bias_x) + _ln(tf.matmul(hidden, weight_h), gain_h, bias_h) + bias
        else:
            gates = tf.matmul(_input, weight_x) + tf.matmul(hidden, weight_h) + bias
        (in_gate, forget_gate, out_gate, cell_candidate) = tf.split(axis=1, num_or_size_splits=4, value=gates)
        in_gate = tf.nn.sigmoid(in_gate)
        forget_gate = tf.nn.sigmoid(forget_gate)
        out_gate = tf.nn.sigmoid(out_gate)
        cell_candidate = tf.tanh(cell_candidate)
        cell_state = forget_gate * cell_state + in_gate * cell_candidate
        if layer_norm:
            hidden = out_gate * tf.tanh(_ln(cell_state, gain_c, bias_c))
        else:
            hidden = out_gate * tf.tanh(cell_state)
        input_tensor[idx] = hidden
    cell_state_hidden = tf.concat(axis=1, values=[cell_state, hidden])
    return (input_tensor, cell_state_hidden)

def _ln(input_tensor, gain, bias, epsilon=1e-05, axes=None):
    if False:
        return 10
    '\n    Apply layer normalisation.\n\n    :param input_tensor: (TensorFlow Tensor) The input tensor for the Layer normalization\n    :param gain: (TensorFlow Tensor) The scale tensor for the Layer normalization\n    :param bias: (TensorFlow Tensor) The bias tensor for the Layer normalization\n    :param epsilon: (float) The epsilon value for floating point calculations\n    :param axes: (tuple, list or int) The axes to apply the mean and variance calculation\n    :return: (TensorFlow Tensor) a normalizing layer\n    '
    if axes is None:
        axes = [1]
    (mean, variance) = tf.nn.moments(input_tensor, axes=axes, keep_dims=True)
    input_tensor = (input_tensor - mean) / tf.sqrt(variance + epsilon)
    input_tensor = input_tensor * gain + bias
    return input_tensor

def lnlstm(input_tensor, mask_tensor, cell_state, scope, n_hidden, init_scale=1.0):
    if False:
        i = 10
        return i + 15
    '\n    Creates a LSTM with Layer Normalization (lnlstm) cell for TensorFlow\n\n    :param input_tensor: (TensorFlow Tensor) The input tensor for the LSTM cell\n    :param mask_tensor: (TensorFlow Tensor) The mask tensor for the LSTM cell\n    :param cell_state: (TensorFlow Tensor) The state tensor for the LSTM cell\n    :param scope: (str) The TensorFlow variable scope\n    :param n_hidden: (int) The number of hidden neurons\n    :param init_scale: (int) The initialization scale\n    :return: (TensorFlow Tensor) lnlstm cell\n    '
    return lstm(input_tensor, mask_tensor, cell_state, scope, n_hidden, init_scale, layer_norm=True)

def conv_to_fc(input_tensor):
    if False:
        while True:
            i = 10
    '\n    Reshapes a Tensor from a convolutional network to a Tensor for a fully connected network\n\n    :param input_tensor: (TensorFlow Tensor) The convolutional input tensor\n    :return: (TensorFlow Tensor) The fully connected output tensor\n    '
    n_hidden = np.prod([v.value for v in input_tensor.get_shape()[1:]])
    input_tensor = tf.reshape(input_tensor, [-1, n_hidden])
    return input_tensor