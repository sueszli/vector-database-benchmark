"""Contains definitions for Residual Networks.

Residual networks ('v1' ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-05
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES

def batch_norm(inputs, training, data_format):
    if False:
        for i in range(10):
            print('nop')
    'Performs a batch normalization using a standard set of parameters.'
    return tf.compat.v1.layers.batch_normalization(inputs=inputs, axis=1 if data_format == 'channels_first' else 3, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=training, fused=True)

def fixed_padding(inputs, kernel_size, data_format):
    if False:
        print('Hello World!')
    "Pads the input along the spatial dimensions independently of input size.\n\n  Args:\n    inputs: A tensor of size [batch, channels, height_in, width_in] or\n      [batch, height_in, width_in, channels] depending on data_format.\n    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.\n                 Should be a positive integer.\n    data_format: The input format ('channels_last' or 'channels_first').\n\n  Returns:\n    A tensor with the same format as the input with the data either intact\n    (if kernel_size == 1) or padded (if kernel_size > 1).\n  "
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    if data_format == 'channels_first':
        padded_inputs = tf.pad(tensor=inputs, paddings=[[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(tensor=inputs, paddings=[[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
    return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
    if False:
        while True:
            i = 10
    'Strided 2-D convolution with explicit padding.'
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    return tf.compat.v1.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding='SAME' if strides == 1 else 'VALID', use_bias=False, kernel_initializer=tf.compat.v1.variance_scaling_initializer(), data_format=data_format)

def _building_block_v1(inputs, filters, training, projection_shortcut, strides, data_format):
    if False:
        while True:
            i = 10
    "A single block for ResNet v1, without a bottleneck.\n\n  Convolution then batch normalization then ReLU as described by:\n    Deep Residual Learning for Image Recognition\n    https://arxiv.org/pdf/1512.03385.pdf\n    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.\n\n  Args:\n    inputs: A tensor of size [batch, channels, height_in, width_in] or\n      [batch, height_in, width_in, channels] depending on data_format.\n    filters: The number of filters for the convolutions.\n    training: A Boolean for whether the model is in training or inference\n      mode. Needed for batch normalization.\n    projection_shortcut: The function to use for projection shortcuts\n      (typically a 1x1 convolution when downsampling the input).\n    strides: The block's stride. If greater than 1, this block will ultimately\n      downsample the input.\n    data_format: The input format ('channels_last' or 'channels_first').\n\n  Returns:\n    The output tensor of the block; shape should match inputs.\n  "
    shortcut = inputs
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training, data_format=data_format)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)
    return inputs

def _building_block_v2(inputs, filters, training, projection_shortcut, strides, data_format):
    if False:
        for i in range(10):
            print('nop')
    "A single block for ResNet v2, without a bottleneck.\n\n  Batch normalization then ReLu then convolution as described by:\n    Identity Mappings in Deep Residual Networks\n    https://arxiv.org/pdf/1603.05027.pdf\n    by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.\n\n  Args:\n    inputs: A tensor of size [batch, channels, height_in, width_in] or\n      [batch, height_in, width_in, channels] depending on data_format.\n    filters: The number of filters for the convolutions.\n    training: A Boolean for whether the model is in training or inference\n      mode. Needed for batch normalization.\n    projection_shortcut: The function to use for projection shortcuts\n      (typically a 1x1 convolution when downsampling the input).\n    strides: The block's stride. If greater than 1, this block will ultimately\n      downsample the input.\n    data_format: The input format ('channels_last' or 'channels_first').\n\n  Returns:\n    The output tensor of the block; shape should match inputs.\n  "
    shortcut = inputs
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=1, data_format=data_format)
    return inputs + shortcut

def _bottleneck_block_v1(inputs, filters, training, projection_shortcut, strides, data_format):
    if False:
        return 10
    'A single block for ResNet v1, with a bottleneck.\n\n  Similar to _building_block_v1(), except using the "bottleneck" blocks\n  described in:\n    Convolution then batch normalization then ReLU as described by:\n      Deep Residual Learning for Image Recognition\n      https://arxiv.org/pdf/1512.03385.pdf\n      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.\n\n  Args:\n    inputs: A tensor of size [batch, channels, height_in, width_in] or\n      [batch, height_in, width_in, channels] depending on data_format.\n    filters: The number of filters for the convolutions.\n    training: A Boolean for whether the model is in training or inference\n      mode. Needed for batch normalization.\n    projection_shortcut: The function to use for projection shortcuts\n      (typically a 1x1 convolution when downsampling the input).\n    strides: The block\'s stride. If greater than 1, this block will ultimately\n      downsample the input.\n    data_format: The input format (\'channels_last\' or \'channels_first\').\n\n  Returns:\n    The output tensor of the block; shape should match inputs.\n  '
    shortcut = inputs
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
        shortcut = batch_norm(inputs=shortcut, training=training, data_format=data_format)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=4 * filters, kernel_size=1, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)
    return inputs

def _bottleneck_block_v2(inputs, filters, training, projection_shortcut, strides, data_format):
    if False:
        while True:
            i = 10
    'A single block for ResNet v2, with a bottleneck.\n\n  Similar to _building_block_v2(), except using the "bottleneck" blocks\n  described in:\n    Convolution then batch normalization then ReLU as described by:\n      Deep Residual Learning for Image Recognition\n      https://arxiv.org/pdf/1512.03385.pdf\n      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.\n\n  Adapted to the ordering conventions of:\n    Batch normalization then ReLu then convolution as described by:\n      Identity Mappings in Deep Residual Networks\n      https://arxiv.org/pdf/1603.05027.pdf\n      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.\n\n  Args:\n    inputs: A tensor of size [batch, channels, height_in, width_in] or\n      [batch, height_in, width_in, channels] depending on data_format.\n    filters: The number of filters for the convolutions.\n    training: A Boolean for whether the model is in training or inference\n      mode. Needed for batch normalization.\n    projection_shortcut: The function to use for projection shortcuts\n      (typically a 1x1 convolution when downsampling the input).\n    strides: The block\'s stride. If greater than 1, this block will ultimately\n      downsample the input.\n    data_format: The input format (\'channels_last\' or \'channels_first\').\n\n  Returns:\n    The output tensor of the block; shape should match inputs.\n  '
    shortcut = inputs
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    if projection_shortcut is not None:
        shortcut = projection_shortcut(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=1, strides=1, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides, data_format=data_format)
    inputs = batch_norm(inputs, training, data_format)
    inputs = tf.nn.relu(inputs)
    inputs = conv2d_fixed_padding(inputs=inputs, filters=4 * filters, kernel_size=1, strides=1, data_format=data_format)
    return inputs + shortcut

def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides, training, name, data_format):
    if False:
        for i in range(10):
            print('nop')
    "Creates one layer of blocks for the ResNet model.\n\n  Args:\n    inputs: A tensor of size [batch, channels, height_in, width_in] or\n      [batch, height_in, width_in, channels] depending on data_format.\n    filters: The number of filters for the first convolution of the layer.\n    bottleneck: Is the block created a bottleneck block.\n    block_fn: The block to use within the model, either `building_block` or\n      `bottleneck_block`.\n    blocks: The number of blocks contained in the layer.\n    strides: The stride to use for the first convolution of the layer. If\n      greater than 1, this layer will ultimately downsample the input.\n    training: Either True or False, whether we are currently training the\n      model. Needed for batch norm.\n    name: A string name for the tensor output of the block layer.\n    data_format: The input format ('channels_last' or 'channels_first').\n\n  Returns:\n    The output tensor of the block layer.\n  "
    filters_out = filters * 4 if bottleneck else filters

    def projection_shortcut(inputs):
        if False:
            for i in range(10):
                print('nop')
        return conv2d_fixed_padding(inputs=inputs, filters=filters_out, kernel_size=1, strides=strides, data_format=data_format)
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides, data_format)
    for _ in range(1, blocks):
        inputs = block_fn(inputs, filters, training, None, 1, data_format)
    return tf.identity(inputs, name)

class Model(object):
    """Base class for building the Resnet Model."""

    def __init__(self, resnet_size, bottleneck, num_classes, num_filters, kernel_size, conv_stride, first_pool_size, first_pool_stride, block_sizes, block_strides, resnet_version=DEFAULT_VERSION, data_format=None, dtype=DEFAULT_DTYPE):
        if False:
            for i in range(10):
                print('nop')
        "Creates a model for classifying an image.\n\n    Args:\n      resnet_size: A single integer for the size of the ResNet model.\n      bottleneck: Use regular blocks or bottleneck blocks.\n      num_classes: The number of classes used as labels.\n      num_filters: The number of filters to use for the first block layer\n        of the model. This number is then doubled for each subsequent block\n        layer.\n      kernel_size: The kernel size to use for convolution.\n      conv_stride: stride size for the initial convolutional layer\n      first_pool_size: Pool size to be used for the first pooling layer.\n        If none, the first pooling layer is skipped.\n      first_pool_stride: stride size for the first pooling layer. Not used\n        if first_pool_size is None.\n      block_sizes: A list containing n values, where n is the number of sets of\n        block layers desired. Each value should be the number of blocks in the\n        i-th set.\n      block_strides: List of integers representing the desired stride size for\n        each of the sets of block layers. Should be same length as block_sizes.\n      resnet_version: Integer representing which version of the ResNet network\n        to use. See README for details. Valid values: [1, 2]\n      data_format: Input format ('channels_last', 'channels_first', or None).\n        If set to None, the format is dependent on whether a GPU is available.\n      dtype: The TensorFlow dtype to use for calculations. If not specified\n        tf.float32 is used.\n\n    Raises:\n      ValueError: if invalid version is selected.\n    "
        self.resnet_size = resnet_size
        if not data_format:
            data_format = 'channels_first' if tf.test.is_built_with_cuda() else 'channels_last'
        self.resnet_version = resnet_version
        if resnet_version not in (1, 2):
            raise ValueError('Resnet version should be 1 or 2. See README for citations.')
        self.bottleneck = bottleneck
        if bottleneck:
            if resnet_version == 1:
                self.block_fn = _bottleneck_block_v1
            else:
                self.block_fn = _bottleneck_block_v2
        elif resnet_version == 1:
            self.block_fn = _building_block_v1
        else:
            self.block_fn = _building_block_v2
        if dtype not in ALLOWED_TYPES:
            raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))
        self.data_format = data_format
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv_stride = conv_stride
        self.first_pool_size = first_pool_size
        self.first_pool_stride = first_pool_stride
        self.block_sizes = block_sizes
        self.block_strides = block_strides
        self.dtype = dtype
        self.pre_activation = resnet_version == 2

    def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE, *args, **kwargs):
        if False:
            return 10
        'Creates variables in fp32, then casts to fp16 if necessary.\n\n    This function is a custom getter. A custom getter is a function with the\n    same signature as tf.get_variable, except it has an additional getter\n    parameter. Custom getters can be passed as the `custom_getter` parameter of\n    tf.variable_scope. Then, tf.get_variable will call the custom getter,\n    instead of directly getting a variable itself. This can be used to change\n    the types of variables that are retrieved with tf.get_variable.\n    The `getter` parameter is the underlying variable getter, that would have\n    been called if no custom getter was used. Custom getters typically get a\n    variable with `getter`, then modify it in some way.\n\n    This custom getter will create an fp32 variable. If a low precision\n    (e.g. float16) variable was requested it will then cast the variable to the\n    requested dtype. The reason we do not directly create variables in low\n    precision dtypes is that applying small gradients to such variables may\n    cause the variable not to change.\n\n    Args:\n      getter: The underlying variable getter, that has the same signature as\n        tf.get_variable and returns a variable.\n      name: The name of the variable to get.\n      shape: The shape of the variable to get.\n      dtype: The dtype of the variable to get. Note that if this is a low\n        precision dtype, the variable will be created as a tf.float32 variable,\n        then cast to the appropriate dtype\n      *args: Additional arguments to pass unmodified to getter.\n      **kwargs: Additional keyword arguments to pass unmodified to getter.\n\n    Returns:\n      A variable which is cast to fp16 if necessary.\n    '
        if dtype in CASTABLE_TYPES:
            var = getter(name, shape, tf.float32, *args, **kwargs)
            return tf.cast(var, dtype=dtype, name=name + '_cast')
        else:
            return getter(name, shape, dtype, *args, **kwargs)

    def _model_variable_scope(self):
        if False:
            while True:
                i = 10
        'Returns a variable scope that the model should be created under.\n\n    If self.dtype is a castable type, model variable will be created in fp32\n    then cast to self.dtype before being used.\n\n    Returns:\n      A variable scope for the model.\n    '
        return tf.compat.v1.variable_scope('resnet_model', custom_getter=self._custom_dtype_getter)

    def __call__(self, inputs, training):
        if False:
            i = 10
            return i + 15
        'Add operations to classify a batch of input images.\n\n    Args:\n      inputs: A Tensor representing a batch of input images.\n      training: A boolean. Set to True to add operations required only when\n        training the classifier.\n\n    Returns:\n      A logits Tensor with shape [<batch_size>, self.num_classes].\n    '
        with self._model_variable_scope():
            if self.data_format == 'channels_first':
                inputs = tf.transpose(a=inputs, perm=[0, 3, 1, 2])
            inputs = conv2d_fixed_padding(inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size, strides=self.conv_stride, data_format=self.data_format)
            inputs = tf.identity(inputs, 'initial_conv')
            if self.resnet_version == 1:
                inputs = batch_norm(inputs, training, self.data_format)
                inputs = tf.nn.relu(inputs)
            if self.first_pool_size:
                inputs = tf.compat.v1.layers.max_pooling2d(inputs=inputs, pool_size=self.first_pool_size, strides=self.first_pool_stride, padding='SAME', data_format=self.data_format)
                inputs = tf.identity(inputs, 'initial_max_pool')
            for (i, num_blocks) in enumerate(self.block_sizes):
                num_filters = self.num_filters * 2 ** i
                inputs = block_layer(inputs=inputs, filters=num_filters, bottleneck=self.bottleneck, block_fn=self.block_fn, blocks=num_blocks, strides=self.block_strides[i], training=training, name='block_layer{}'.format(i + 1), data_format=self.data_format)
            if self.pre_activation:
                inputs = batch_norm(inputs, training, self.data_format)
                inputs = tf.nn.relu(inputs)
            axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
            inputs = tf.reduce_mean(input_tensor=inputs, axis=axes, keepdims=True)
            inputs = tf.identity(inputs, 'final_reduce_mean')
            inputs = tf.squeeze(inputs, axes)
            inputs = tf.compat.v1.layers.dense(inputs=inputs, units=self.num_classes)
            inputs = tf.identity(inputs, 'final_dense')
            return inputs