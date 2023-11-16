"""Augment slim.conv2d with optional Weight Standardization (WS).

WS is a normalization method to accelerate micro-batch training. When used with
Group Normalization and trained with 1 image/GPU, WS is able to match or
outperform the performances of BN trained with large batch sizes.
[1] Siyuan Qiao, Huiyu Wang, Chenxi Liu, Wei Shen, Alan Yuille
    Weight Standardization. arXiv:1903.10520
[2] Lei Huang, Xianglong Liu, Yang Liu, Bo Lang, Dacheng Tao
    Centered Weight Normalization in Accelerating Training of Deep Neural
    Networks. ICCV 2017
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import utils

class Conv2D(tf.keras.layers.Conv2D, tf.layers.Layer):
    """2D convolution layer (e.g. spatial convolution over images).

  This layer creates a convolution kernel that is convolved
  (actually cross-correlated) with the layer input to produce a tensor of
  outputs. If `use_bias` is True (and a `bias_initializer` is provided),
  a bias vector is created and added to the outputs. Finally, if
  `activation` is not `None`, it is applied to the outputs as well.
  """

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', data_format='channels_last', dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer=None, bias_initializer=tf.zeros_initializer(), kernel_regularizer=None, bias_regularizer=None, use_weight_standardization=False, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, trainable=True, name=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Constructs the 2D convolution layer.\n\n    Args:\n      filters: Integer, the dimensionality of the output space (i.e. the number\n        of filters in the convolution).\n      kernel_size: An integer or tuple/list of 2 integers, specifying the height\n        and width of the 2D convolution window. Can be a single integer to\n        specify the same value for all spatial dimensions.\n      strides: An integer or tuple/list of 2 integers, specifying the strides of\n        the convolution along the height and width. Can be a single integer to\n        specify the same value for all spatial dimensions. Specifying any stride\n        value != 1 is incompatible with specifying any `dilation_rate` value !=\n        1.\n      padding: One of `"valid"` or `"same"` (case-insensitive).\n      data_format: A string, one of `channels_last` (default) or\n        `channels_first`. The ordering of the dimensions in the inputs.\n        `channels_last` corresponds to inputs with shape `(batch, height, width,\n        channels)` while `channels_first` corresponds to inputs with shape\n        `(batch, channels, height, width)`.\n      dilation_rate: An integer or tuple/list of 2 integers, specifying the\n        dilation rate to use for dilated convolution. Can be a single integer to\n        specify the same value for all spatial dimensions. Currently, specifying\n        any `dilation_rate` value != 1 is incompatible with specifying any\n        stride value != 1.\n      activation: Activation function. Set it to None to maintain a linear\n        activation.\n      use_bias: Boolean, whether the layer uses a bias.\n      kernel_initializer: An initializer for the convolution kernel.\n      bias_initializer: An initializer for the bias vector. If None, the default\n        initializer will be used.\n      kernel_regularizer: Optional regularizer for the convolution kernel.\n      bias_regularizer: Optional regularizer for the bias vector.\n      use_weight_standardization: Boolean, whether the layer uses weight\n        standardization.\n      activity_regularizer: Optional regularizer function for the output.\n      kernel_constraint: Optional projection function to be applied to the\n        kernel after being updated by an `Optimizer` (e.g. used to implement\n        norm constraints or value constraints for layer weights). The function\n        must take as input the unprojected variable and must return the\n        projected variable (which must have the same shape). Constraints are not\n        safe to use when doing asynchronous distributed training.\n      bias_constraint: Optional projection function to be applied to the bias\n        after being updated by an `Optimizer`.\n      trainable: Boolean, if `True` also add variables to the graph collection\n        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).\n      name: A string, the name of the layer.\n      **kwargs: Arbitrary keyword arguments passed to tf.keras.layers.Conv2D\n    '
        super(Conv2D, self).__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, trainable=trainable, name=name, **kwargs)
        self.use_weight_standardization = use_weight_standardization

    def call(self, inputs):
        if False:
            i = 10
            return i + 15
        if self.use_weight_standardization:
            (mean, var) = tf.nn.moments(self.kernel, [0, 1, 2], keep_dims=True)
            kernel = (self.kernel - mean) / tf.sqrt(var + 1e-05)
            outputs = self._convolution_op(inputs, kernel)
        else:
            outputs = self._convolution_op(inputs, self.kernel)
        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    bias = tf.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = tf.nn.bias_add(outputs, self.bias, data_format='NCHW')
            else:
                outputs = tf.nn.bias_add(outputs, self.bias, data_format='NHWC')
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

@contrib_framework.add_arg_scope
def conv2d(inputs, num_outputs, kernel_size, stride=1, padding='SAME', data_format=None, rate=1, activation_fn=tf.nn.relu, normalizer_fn=None, normalizer_params=None, weights_initializer=contrib_layers.xavier_initializer(), weights_regularizer=None, biases_initializer=tf.zeros_initializer(), biases_regularizer=None, use_weight_standardization=False, reuse=None, variables_collections=None, outputs_collections=None, trainable=True, scope=None):
    if False:
        i = 10
        return i + 15
    'Adds a 2D convolution followed by an optional batch_norm layer.\n\n  `convolution` creates a variable called `weights`, representing the\n  convolutional kernel, that is convolved (actually cross-correlated) with the\n  `inputs` to produce a `Tensor` of activations. If a `normalizer_fn` is\n  provided (such as `batch_norm`), it is then applied. Otherwise, if\n  `normalizer_fn` is None and a `biases_initializer` is provided then a `biases`\n  variable would be created and added the activations. Finally, if\n  `activation_fn` is not `None`, it is applied to the activations as well.\n\n  Performs atrous convolution with input stride/dilation rate equal to `rate`\n  if a value > 1 for any dimension of `rate` is specified.  In this case\n  `stride` values != 1 are not supported.\n\n  Args:\n    inputs: A Tensor of rank N+2 of shape `[batch_size] + input_spatial_shape +\n      [in_channels]` if data_format does not start with "NC" (default), or\n      `[batch_size, in_channels] + input_spatial_shape` if data_format starts\n      with "NC".\n    num_outputs: Integer, the number of output filters.\n    kernel_size: A sequence of N positive integers specifying the spatial\n      dimensions of the filters.  Can be a single integer to specify the same\n      value for all spatial dimensions.\n    stride: A sequence of N positive integers specifying the stride at which to\n      compute output.  Can be a single integer to specify the same value for all\n      spatial dimensions.  Specifying any `stride` value != 1 is incompatible\n      with specifying any `rate` value != 1.\n    padding: One of `"VALID"` or `"SAME"`.\n    data_format: A string or None.  Specifies whether the channel dimension of\n      the `input` and output is the last dimension (default, or if `data_format`\n      does not start with "NC"), or the second dimension (if `data_format`\n      starts with "NC").  For N=1, the valid values are "NWC" (default) and\n      "NCW".  For N=2, the valid values are "NHWC" (default) and "NCHW". For\n      N=3, the valid values are "NDHWC" (default) and "NCDHW".\n    rate: A sequence of N positive integers specifying the dilation rate to use\n      for atrous convolution.  Can be a single integer to specify the same value\n      for all spatial dimensions.  Specifying any `rate` value != 1 is\n      incompatible with specifying any `stride` value != 1.\n    activation_fn: Activation function. The default value is a ReLU function.\n      Explicitly set it to None to skip it and maintain a linear activation.\n    normalizer_fn: Normalization function to use instead of `biases`. If\n      `normalizer_fn` is provided then `biases_initializer` and\n      `biases_regularizer` are ignored and `biases` are not created nor added.\n      default set to None for no normalizer function\n    normalizer_params: Normalization function parameters.\n    weights_initializer: An initializer for the weights.\n    weights_regularizer: Optional regularizer for the weights.\n    biases_initializer: An initializer for the biases. If None skip biases.\n    biases_regularizer: Optional regularizer for the biases.\n    use_weight_standardization: Boolean, whether the layer uses weight\n      standardization.\n    reuse: Whether or not the layer and its variables should be reused. To be\n      able to reuse the layer scope must be given.\n    variables_collections: Optional list of collections for all the variables or\n      a dictionary containing a different list of collection per variable.\n    outputs_collections: Collection to add the outputs.\n    trainable: If `True` also add variables to the graph collection\n      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).\n    scope: Optional scope for `variable_scope`.\n\n  Returns:\n    A tensor representing the output of the operation.\n\n  Raises:\n    ValueError: If `data_format` is invalid.\n    ValueError: Both \'rate\' and `stride` are not uniformly 1.\n  '
    if data_format not in [None, 'NWC', 'NCW', 'NHWC', 'NCHW', 'NDHWC', 'NCDHW']:
        raise ValueError('Invalid data_format: %r' % (data_format,))
    layer_variable_getter = layers._build_variable_getter({'bias': 'biases', 'kernel': 'weights'})
    with tf.variable_scope(scope, 'Conv', [inputs], reuse=reuse, custom_getter=layer_variable_getter) as sc:
        inputs = tf.convert_to_tensor(inputs)
        input_rank = inputs.get_shape().ndims
        if input_rank != 4:
            raise ValueError('Convolution expects input with rank %d, got %d' % (4, input_rank))
        data_format = 'channels_first' if data_format and data_format.startswith('NC') else 'channels_last'
        layer = Conv2D(filters=num_outputs, kernel_size=kernel_size, strides=stride, padding=padding, data_format=data_format, dilation_rate=rate, activation=None, use_bias=not normalizer_fn and biases_initializer, kernel_initializer=weights_initializer, bias_initializer=biases_initializer, kernel_regularizer=weights_regularizer, bias_regularizer=biases_regularizer, use_weight_standardization=use_weight_standardization, activity_regularizer=None, trainable=trainable, name=sc.name, dtype=inputs.dtype.base_dtype, _scope=sc, _reuse=reuse)
        outputs = layer.apply(inputs)
        layers._add_variable_to_collections(layer.kernel, variables_collections, 'weights')
        if layer.use_bias:
            layers._add_variable_to_collections(layer.bias, variables_collections, 'biases')
        if normalizer_fn is not None:
            normalizer_params = normalizer_params or {}
            outputs = normalizer_fn(outputs, **normalizer_params)
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return utils.collect_named_outputs(outputs_collections, sc.name, outputs)

def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    if False:
        while True:
            i = 10
    "Strided 2-D convolution with 'SAME' padding.\n\n  When stride > 1, then we do explicit zero-padding, followed by conv2d with\n  'VALID' padding.\n\n  Note that\n\n     net = conv2d_same(inputs, num_outputs, 3, stride=stride)\n\n  is equivalent to\n\n     net = conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')\n     net = subsample(net, factor=stride)\n\n  whereas\n\n     net = conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')\n\n  is different when the input's height or width is even, which is why we add the\n  current function. For more details, see ResnetUtilsTest.testConv2DSameEven().\n\n  Args:\n    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].\n    num_outputs: An integer, the number of output filters.\n    kernel_size: An int with the kernel_size of the filters.\n    stride: An integer, the output stride.\n    rate: An integer, rate for atrous convolution.\n    scope: Scope.\n\n  Returns:\n    output: A 4-D tensor of size [batch, height_out, width_out, channels] with\n      the convolution output.\n  "
    if stride == 1:
        return conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate, padding='SAME', scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return conv2d(inputs, num_outputs, kernel_size, stride=stride, rate=rate, padding='VALID', scope=scope)