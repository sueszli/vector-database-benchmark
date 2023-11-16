"""Functions to generate a list of feature maps based on image features.

Provides several feature map generators that can be used to build object
detection feature extractors.

Object detection feature extractors usually are built by stacking two components
- A base feature extractor such as Inception V3 and a feature map generator.
Feature map generators build on the base feature extractors and produce a list
of final feature maps.
"""
import collections
import functools
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from object_detection.utils import ops
from object_detection.utils import shape_utils
slim = contrib_slim
ACTIVATION_BOUND = 6.0

def get_depth_fn(depth_multiplier, min_depth):
    if False:
        print('Hello World!')
    'Builds a callable to compute depth (output channels) of conv filters.\n\n  Args:\n    depth_multiplier: a multiplier for the nominal depth.\n    min_depth: a lower bound on the depth of filters.\n\n  Returns:\n    A callable that takes in a nominal depth and returns the depth to use.\n  '

    def multiply_depth(depth):
        if False:
            while True:
                i = 10
        new_depth = int(depth * depth_multiplier)
        return max(new_depth, min_depth)
    return multiply_depth

def create_conv_block(use_depthwise, kernel_size, padding, stride, layer_name, conv_hyperparams, is_training, freeze_batchnorm, depth):
    if False:
        while True:
            i = 10
    "Create Keras layers for depthwise & non-depthwise convolutions.\n\n  Args:\n    use_depthwise: Whether to use depthwise separable conv instead of regular\n      conv.\n    kernel_size: A list of length 2: [kernel_height, kernel_width] of the\n      filters. Can be an int if both values are the same.\n    padding: One of 'VALID' or 'SAME'.\n    stride: A list of length 2: [stride_height, stride_width], specifying the\n      convolution stride. Can be an int if both strides are the same.\n    layer_name: String. The name of the layer.\n    conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object\n      containing hyperparameters for convolution ops.\n    is_training: Indicates whether the feature generator is in training mode.\n    freeze_batchnorm: Bool. Whether to freeze batch norm parameters during\n      training or not. When training with a small batch size (e.g. 1), it is\n      desirable to freeze batch norm update and use pretrained batch norm\n      params.\n    depth: Depth of output feature maps.\n\n  Returns:\n    A list of conv layers.\n  "
    layers = []
    if use_depthwise:
        kwargs = conv_hyperparams.params()
        kwargs['depthwise_regularizer'] = kwargs['kernel_regularizer']
        kwargs['depthwise_initializer'] = kwargs['kernel_initializer']
        layers.append(tf.keras.layers.SeparableConv2D(depth, [kernel_size, kernel_size], depth_multiplier=1, padding=padding, strides=stride, name=layer_name + '_depthwise_conv', **kwargs))
    else:
        layers.append(tf.keras.layers.Conv2D(depth, [kernel_size, kernel_size], padding=padding, strides=stride, name=layer_name + '_conv', **conv_hyperparams.params()))
    layers.append(conv_hyperparams.build_batch_norm(training=is_training and (not freeze_batchnorm), name=layer_name + '_batchnorm'))
    layers.append(conv_hyperparams.build_activation_layer(name=layer_name))
    return layers

class KerasMultiResolutionFeatureMaps(tf.keras.Model):
    """Generates multi resolution feature maps from input image features.

  A Keras model that generates multi-scale feature maps for detection as in the
  SSD papers by Liu et al: https://arxiv.org/pdf/1512.02325v2.pdf, See Sec 2.1.

  More specifically, when called on inputs it performs the following two tasks:
  1) If a layer name is provided in the configuration, returns that layer as a
     feature map.
  2) If a layer name is left as an empty string, constructs a new feature map
     based on the spatial shape and depth configuration. Note that the current
     implementation only supports generating new layers using convolution of
     stride 2 resulting in a spatial resolution reduction by a factor of 2.
     By default convolution kernel size is set to 3, and it can be customized
     by caller.

  An example of the configuration for Inception V3:
  {
    'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', ''],
    'layer_depth': [-1, -1, -1, 512, 256, 128]
  }

  When this feature generator object is called on input image_features:
    Args:
      image_features: A dictionary of handles to activation tensors from the
        base feature extractor.

    Returns:
      feature_maps: an OrderedDict mapping keys (feature map names) to
        tensors where each tensor has shape [batch, height_i, width_i, depth_i].
  """

    def __init__(self, feature_map_layout, depth_multiplier, min_depth, insert_1x1_conv, is_training, conv_hyperparams, freeze_batchnorm, name=None):
        if False:
            i = 10
            return i + 15
        "Constructor.\n\n    Args:\n      feature_map_layout: Dictionary of specifications for the feature map\n        layouts in the following format (Inception V2/V3 respectively):\n        {\n          'from_layer': ['Mixed_3c', 'Mixed_4c', 'Mixed_5c', '', '', ''],\n          'layer_depth': [-1, -1, -1, 512, 256, 128]\n        }\n        or\n        {\n          'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', ''],\n          'layer_depth': [-1, -1, -1, 512, 256, 128]\n        }\n        If 'from_layer' is specified, the specified feature map is directly used\n        as a box predictor layer, and the layer_depth is directly infered from\n        the feature map (instead of using the provided 'layer_depth' parameter).\n        In this case, our convention is to set 'layer_depth' to -1 for clarity.\n        Otherwise, if 'from_layer' is an empty string, then the box predictor\n        layer will be built from the previous layer using convolution\n        operations. Note that the current implementation only supports\n        generating new layers using convolutions of stride 2 (resulting in a\n        spatial resolution reduction by a factor of 2), and will be extended to\n        a more flexible design. Convolution kernel size is set to 3 by default,\n        and can be customized by 'conv_kernel_size' parameter (similarily,\n        'conv_kernel_size' should be set to -1 if 'from_layer' is specified).\n        The created convolution operation will be a normal 2D convolution by\n        default, and a depthwise convolution followed by 1x1 convolution if\n        'use_depthwise' is set to True.\n      depth_multiplier: Depth multiplier for convolutional layers.\n      min_depth: Minimum depth for convolutional layers.\n      insert_1x1_conv: A boolean indicating whether an additional 1x1\n        convolution should be inserted before shrinking the feature map.\n      is_training: Indicates whether the feature generator is in training mode.\n      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object\n        containing hyperparameters for convolution ops.\n      freeze_batchnorm: Bool. Whether to freeze batch norm parameters during\n        training or not. When training with a small batch size (e.g. 1), it is\n        desirable to freeze batch norm update and use pretrained batch norm\n        params.\n      name: A string name scope to assign to the model. If 'None', Keras\n        will auto-generate one from the class name.\n    "
        super(KerasMultiResolutionFeatureMaps, self).__init__(name=name)
        self.feature_map_layout = feature_map_layout
        self.convolutions = []
        depth_fn = get_depth_fn(depth_multiplier, min_depth)
        base_from_layer = ''
        use_explicit_padding = False
        if 'use_explicit_padding' in feature_map_layout:
            use_explicit_padding = feature_map_layout['use_explicit_padding']
        use_depthwise = False
        if 'use_depthwise' in feature_map_layout:
            use_depthwise = feature_map_layout['use_depthwise']
        for (index, from_layer) in enumerate(feature_map_layout['from_layer']):
            net = []
            layer_depth = feature_map_layout['layer_depth'][index]
            conv_kernel_size = 3
            if 'conv_kernel_size' in feature_map_layout:
                conv_kernel_size = feature_map_layout['conv_kernel_size'][index]
            if from_layer:
                base_from_layer = from_layer
            else:
                if insert_1x1_conv:
                    layer_name = '{}_1_Conv2d_{}_1x1_{}'.format(base_from_layer, index, depth_fn(layer_depth / 2))
                    net.append(tf.keras.layers.Conv2D(depth_fn(layer_depth / 2), [1, 1], padding='SAME', strides=1, name=layer_name + '_conv', **conv_hyperparams.params()))
                    net.append(conv_hyperparams.build_batch_norm(training=is_training and (not freeze_batchnorm), name=layer_name + '_batchnorm'))
                    net.append(conv_hyperparams.build_activation_layer(name=layer_name))
                layer_name = '{}_2_Conv2d_{}_{}x{}_s2_{}'.format(base_from_layer, index, conv_kernel_size, conv_kernel_size, depth_fn(layer_depth))
                stride = 2
                padding = 'SAME'
                if use_explicit_padding:
                    padding = 'VALID'

                    def fixed_padding(features, kernel_size=conv_kernel_size):
                        if False:
                            print('Hello World!')
                        return ops.fixed_padding(features, kernel_size)
                    net.append(tf.keras.layers.Lambda(fixed_padding))
                if use_depthwise:
                    net.append(tf.keras.layers.DepthwiseConv2D([conv_kernel_size, conv_kernel_size], depth_multiplier=1, padding=padding, strides=stride, name=layer_name + '_depthwise_conv', **conv_hyperparams.params()))
                    net.append(conv_hyperparams.build_batch_norm(training=is_training and (not freeze_batchnorm), name=layer_name + '_depthwise_batchnorm'))
                    net.append(conv_hyperparams.build_activation_layer(name=layer_name + '_depthwise'))
                    net.append(tf.keras.layers.Conv2D(depth_fn(layer_depth), [1, 1], padding='SAME', strides=1, name=layer_name + '_conv', **conv_hyperparams.params()))
                    net.append(conv_hyperparams.build_batch_norm(training=is_training and (not freeze_batchnorm), name=layer_name + '_batchnorm'))
                    net.append(conv_hyperparams.build_activation_layer(name=layer_name))
                else:
                    net.append(tf.keras.layers.Conv2D(depth_fn(layer_depth), [conv_kernel_size, conv_kernel_size], padding=padding, strides=stride, name=layer_name + '_conv', **conv_hyperparams.params()))
                    net.append(conv_hyperparams.build_batch_norm(training=is_training and (not freeze_batchnorm), name=layer_name + '_batchnorm'))
                    net.append(conv_hyperparams.build_activation_layer(name=layer_name))
            self.convolutions.append(net)

    def call(self, image_features):
        if False:
            i = 10
            return i + 15
        'Generate the multi-resolution feature maps.\n\n    Executed when calling the `.__call__` method on input.\n\n    Args:\n      image_features: A dictionary of handles to activation tensors from the\n        base feature extractor.\n\n    Returns:\n      feature_maps: an OrderedDict mapping keys (feature map names) to\n        tensors where each tensor has shape [batch, height_i, width_i, depth_i].\n    '
        feature_maps = []
        feature_map_keys = []
        for (index, from_layer) in enumerate(self.feature_map_layout['from_layer']):
            if from_layer:
                feature_map = image_features[from_layer]
                feature_map_keys.append(from_layer)
            else:
                feature_map = feature_maps[-1]
                for layer in self.convolutions[index]:
                    feature_map = layer(feature_map)
                layer_name = self.convolutions[index][-1].name
                feature_map_keys.append(layer_name)
            feature_maps.append(feature_map)
        return collections.OrderedDict([(x, y) for (x, y) in zip(feature_map_keys, feature_maps)])

def multi_resolution_feature_maps(feature_map_layout, depth_multiplier, min_depth, insert_1x1_conv, image_features, pool_residual=False):
    if False:
        i = 10
        return i + 15
    "Generates multi resolution feature maps from input image features.\n\n  Generates multi-scale feature maps for detection as in the SSD papers by\n  Liu et al: https://arxiv.org/pdf/1512.02325v2.pdf, See Sec 2.1.\n\n  More specifically, it performs the following two tasks:\n  1) If a layer name is provided in the configuration, returns that layer as a\n     feature map.\n  2) If a layer name is left as an empty string, constructs a new feature map\n     based on the spatial shape and depth configuration. Note that the current\n     implementation only supports generating new layers using convolution of\n     stride 2 resulting in a spatial resolution reduction by a factor of 2.\n     By default convolution kernel size is set to 3, and it can be customized\n     by caller.\n\n  An example of the configuration for Inception V3:\n  {\n    'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', ''],\n    'layer_depth': [-1, -1, -1, 512, 256, 128]\n  }\n\n  Args:\n    feature_map_layout: Dictionary of specifications for the feature map\n      layouts in the following format (Inception V2/V3 respectively):\n      {\n        'from_layer': ['Mixed_3c', 'Mixed_4c', 'Mixed_5c', '', '', ''],\n        'layer_depth': [-1, -1, -1, 512, 256, 128]\n      }\n      or\n      {\n        'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', ''],\n        'layer_depth': [-1, -1, -1, 512, 256, 128]\n      }\n      If 'from_layer' is specified, the specified feature map is directly used\n      as a box predictor layer, and the layer_depth is directly infered from the\n      feature map (instead of using the provided 'layer_depth' parameter). In\n      this case, our convention is to set 'layer_depth' to -1 for clarity.\n      Otherwise, if 'from_layer' is an empty string, then the box predictor\n      layer will be built from the previous layer using convolution operations.\n      Note that the current implementation only supports generating new layers\n      using convolutions of stride 2 (resulting in a spatial resolution\n      reduction by a factor of 2), and will be extended to a more flexible\n      design. Convolution kernel size is set to 3 by default, and can be\n      customized by 'conv_kernel_size' parameter (similarily, 'conv_kernel_size'\n      should be set to -1 if 'from_layer' is specified). The created convolution\n      operation will be a normal 2D convolution by default, and a depthwise\n      convolution followed by 1x1 convolution if 'use_depthwise' is set to True.\n    depth_multiplier: Depth multiplier for convolutional layers.\n    min_depth: Minimum depth for convolutional layers.\n    insert_1x1_conv: A boolean indicating whether an additional 1x1 convolution\n      should be inserted before shrinking the feature map.\n    image_features: A dictionary of handles to activation tensors from the\n      base feature extractor.\n    pool_residual: Whether to add an average pooling layer followed by a\n      residual connection between subsequent feature maps when the channel\n      depth match. For example, with option 'layer_depth': [-1, 512, 256, 256],\n      a pooling and residual layer is added between the third and forth feature\n      map. This option is better used with Weight Shared Convolution Box\n      Predictor when all feature maps have the same channel depth to encourage\n      more consistent features across multi-scale feature maps.\n\n  Returns:\n    feature_maps: an OrderedDict mapping keys (feature map names) to\n      tensors where each tensor has shape [batch, height_i, width_i, depth_i].\n\n  Raises:\n    ValueError: if the number entries in 'from_layer' and\n      'layer_depth' do not match.\n    ValueError: if the generated layer does not have the same resolution\n      as specified.\n  "
    depth_fn = get_depth_fn(depth_multiplier, min_depth)
    feature_map_keys = []
    feature_maps = []
    base_from_layer = ''
    use_explicit_padding = False
    if 'use_explicit_padding' in feature_map_layout:
        use_explicit_padding = feature_map_layout['use_explicit_padding']
    use_depthwise = False
    if 'use_depthwise' in feature_map_layout:
        use_depthwise = feature_map_layout['use_depthwise']
    for (index, from_layer) in enumerate(feature_map_layout['from_layer']):
        layer_depth = feature_map_layout['layer_depth'][index]
        conv_kernel_size = 3
        if 'conv_kernel_size' in feature_map_layout:
            conv_kernel_size = feature_map_layout['conv_kernel_size'][index]
        if from_layer:
            feature_map = image_features[from_layer]
            base_from_layer = from_layer
            feature_map_keys.append(from_layer)
        else:
            pre_layer = feature_maps[-1]
            pre_layer_depth = pre_layer.get_shape().as_list()[3]
            intermediate_layer = pre_layer
            if insert_1x1_conv:
                layer_name = '{}_1_Conv2d_{}_1x1_{}'.format(base_from_layer, index, depth_fn(layer_depth / 2))
                intermediate_layer = slim.conv2d(pre_layer, depth_fn(layer_depth / 2), [1, 1], padding='SAME', stride=1, scope=layer_name)
            layer_name = '{}_2_Conv2d_{}_{}x{}_s2_{}'.format(base_from_layer, index, conv_kernel_size, conv_kernel_size, depth_fn(layer_depth))
            stride = 2
            padding = 'SAME'
            if use_explicit_padding:
                padding = 'VALID'
                intermediate_layer = ops.fixed_padding(intermediate_layer, conv_kernel_size)
            if use_depthwise:
                feature_map = slim.separable_conv2d(intermediate_layer, None, [conv_kernel_size, conv_kernel_size], depth_multiplier=1, padding=padding, stride=stride, scope=layer_name + '_depthwise')
                feature_map = slim.conv2d(feature_map, depth_fn(layer_depth), [1, 1], padding='SAME', stride=1, scope=layer_name)
                if pool_residual and pre_layer_depth == depth_fn(layer_depth):
                    feature_map += slim.avg_pool2d(pre_layer, [3, 3], padding='SAME', stride=2, scope=layer_name + '_pool')
            else:
                feature_map = slim.conv2d(intermediate_layer, depth_fn(layer_depth), [conv_kernel_size, conv_kernel_size], padding=padding, stride=stride, scope=layer_name)
            feature_map_keys.append(layer_name)
        feature_maps.append(feature_map)
    return collections.OrderedDict([(x, y) for (x, y) in zip(feature_map_keys, feature_maps)])

class KerasFpnTopDownFeatureMaps(tf.keras.Model):
    """Generates Keras based `top-down` feature maps for Feature Pyramid Networks.

  See https://arxiv.org/abs/1612.03144 for details.
  """

    def __init__(self, num_levels, depth, is_training, conv_hyperparams, freeze_batchnorm, use_depthwise=False, use_explicit_padding=False, use_bounded_activations=False, use_native_resize_op=False, scope=None, name=None):
        if False:
            print('Hello World!')
        "Constructor.\n\n    Args:\n      num_levels: the number of image features.\n      depth: depth of output feature maps.\n      is_training: Indicates whether the feature generator is in training mode.\n      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object\n        containing hyperparameters for convolution ops.\n      freeze_batchnorm: Bool. Whether to freeze batch norm parameters during\n        training or not. When training with a small batch size (e.g. 1), it is\n        desirable to freeze batch norm update and use pretrained batch norm\n        params.\n      use_depthwise: whether to use depthwise separable conv instead of regular\n        conv.\n      use_explicit_padding: whether to use explicit padding.\n      use_bounded_activations: Whether or not to clip activations to range\n        [-ACTIVATION_BOUND, ACTIVATION_BOUND]. Bounded activations better lend\n        themselves to quantized inference.\n      use_native_resize_op: If True, uses tf.image.resize_nearest_neighbor op\n        for the upsampling process instead of reshape and broadcasting\n        implementation.\n      scope: A scope name to wrap this op under.\n      name: A string name scope to assign to the model. If 'None', Keras\n        will auto-generate one from the class name.\n    "
        super(KerasFpnTopDownFeatureMaps, self).__init__(name=name)
        self.scope = scope if scope else 'top_down'
        self.top_layers = []
        self.residual_blocks = []
        self.top_down_blocks = []
        self.reshape_blocks = []
        self.conv_layers = []
        padding = 'VALID' if use_explicit_padding else 'SAME'
        stride = 1
        kernel_size = 3

        def clip_by_value(features):
            if False:
                return 10
            return tf.clip_by_value(features, -ACTIVATION_BOUND, ACTIVATION_BOUND)
        self.top_layers.append(tf.keras.layers.Conv2D(depth, [1, 1], strides=stride, padding=padding, name='projection_%d' % num_levels, **conv_hyperparams.params(use_bias=True)))
        if use_bounded_activations:
            self.top_layers.append(tf.keras.layers.Lambda(clip_by_value, name='clip_by_value'))
        for level in reversed(range(num_levels - 1)):
            residual_net = []
            top_down_net = []
            reshaped_residual = []
            conv_net = []
            residual_net.append(tf.keras.layers.Conv2D(depth, [1, 1], padding=padding, strides=1, name='projection_%d' % (level + 1), **conv_hyperparams.params(use_bias=True)))
            if use_bounded_activations:
                residual_net.append(tf.keras.layers.Lambda(clip_by_value, name='clip_by_value'))
            if use_native_resize_op:

                def resize_nearest_neighbor(image):
                    if False:
                        for i in range(10):
                            print('nop')
                    image_shape = shape_utils.combined_static_and_dynamic_shape(image)
                    return tf.image.resize_nearest_neighbor(image, [image_shape[1] * 2, image_shape[2] * 2])
                top_down_net.append(tf.keras.layers.Lambda(resize_nearest_neighbor, name='nearest_neighbor_upsampling'))
            else:

                def nearest_neighbor_upsampling(image):
                    if False:
                        while True:
                            i = 10
                    return ops.nearest_neighbor_upsampling(image, scale=2)
                top_down_net.append(tf.keras.layers.Lambda(nearest_neighbor_upsampling, name='nearest_neighbor_upsampling'))
            if use_explicit_padding:

                def reshape(inputs):
                    if False:
                        for i in range(10):
                            print('nop')
                    residual_shape = tf.shape(inputs[0])
                    return inputs[1][:, :residual_shape[1], :residual_shape[2], :]
                reshaped_residual.append(tf.keras.layers.Lambda(reshape, name='reshape'))
            if use_bounded_activations:
                conv_net.append(tf.keras.layers.Lambda(clip_by_value, name='clip_by_value'))
            if use_explicit_padding:

                def fixed_padding(features, kernel_size=kernel_size):
                    if False:
                        i = 10
                        return i + 15
                    return ops.fixed_padding(features, kernel_size)
                conv_net.append(tf.keras.layers.Lambda(fixed_padding, name='fixed_padding'))
            layer_name = 'smoothing_%d' % (level + 1)
            conv_block = create_conv_block(use_depthwise, kernel_size, padding, stride, layer_name, conv_hyperparams, is_training, freeze_batchnorm, depth)
            conv_net.extend(conv_block)
            self.residual_blocks.append(residual_net)
            self.top_down_blocks.append(top_down_net)
            self.reshape_blocks.append(reshaped_residual)
            self.conv_layers.append(conv_net)

    def call(self, image_features):
        if False:
            print('Hello World!')
        'Generate the multi-resolution feature maps.\n\n    Executed when calling the `.__call__` method on input.\n\n    Args:\n      image_features: list of tuples of (tensor_name, image_feature_tensor).\n        Spatial resolutions of succesive tensors must reduce exactly by a factor\n        of 2.\n\n    Returns:\n      feature_maps: an OrderedDict mapping keys (feature map names) to\n        tensors where each tensor has shape [batch, height_i, width_i, depth_i].\n    '
        output_feature_maps_list = []
        output_feature_map_keys = []
        with tf.name_scope(self.scope):
            top_down = image_features[-1][1]
            for layer in self.top_layers:
                top_down = layer(top_down)
            output_feature_maps_list.append(top_down)
            output_feature_map_keys.append('top_down_%s' % image_features[-1][0])
            num_levels = len(image_features)
            for (index, level) in enumerate(reversed(range(num_levels - 1))):
                residual = image_features[level][1]
                top_down = output_feature_maps_list[-1]
                for layer in self.residual_blocks[index]:
                    residual = layer(residual)
                for layer in self.top_down_blocks[index]:
                    top_down = layer(top_down)
                for layer in self.reshape_blocks[index]:
                    top_down = layer([residual, top_down])
                top_down += residual
                for layer in self.conv_layers[index]:
                    top_down = layer(top_down)
                output_feature_maps_list.append(top_down)
                output_feature_map_keys.append('top_down_%s' % image_features[level][0])
        return collections.OrderedDict(reversed(list(zip(output_feature_map_keys, output_feature_maps_list))))

def fpn_top_down_feature_maps(image_features, depth, use_depthwise=False, use_explicit_padding=False, use_bounded_activations=False, scope=None, use_native_resize_op=False):
    if False:
        return 10
    'Generates `top-down` feature maps for Feature Pyramid Networks.\n\n  See https://arxiv.org/abs/1612.03144 for details.\n\n  Args:\n    image_features: list of tuples of (tensor_name, image_feature_tensor).\n      Spatial resolutions of succesive tensors must reduce exactly by a factor\n      of 2.\n    depth: depth of output feature maps.\n    use_depthwise: whether to use depthwise separable conv instead of regular\n      conv.\n    use_explicit_padding: whether to use explicit padding.\n    use_bounded_activations: Whether or not to clip activations to range\n      [-ACTIVATION_BOUND, ACTIVATION_BOUND]. Bounded activations better lend\n      themselves to quantized inference.\n    scope: A scope name to wrap this op under.\n    use_native_resize_op: If True, uses tf.image.resize_nearest_neighbor op for\n      the upsampling process instead of reshape and broadcasting implementation.\n\n  Returns:\n    feature_maps: an OrderedDict mapping keys (feature map names) to\n      tensors where each tensor has shape [batch, height_i, width_i, depth_i].\n  '
    with tf.name_scope(scope, 'top_down'):
        num_levels = len(image_features)
        output_feature_maps_list = []
        output_feature_map_keys = []
        padding = 'VALID' if use_explicit_padding else 'SAME'
        kernel_size = 3
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding=padding, stride=1):
            top_down = slim.conv2d(image_features[-1][1], depth, [1, 1], activation_fn=None, normalizer_fn=None, scope='projection_%d' % num_levels)
            if use_bounded_activations:
                top_down = tf.clip_by_value(top_down, -ACTIVATION_BOUND, ACTIVATION_BOUND)
            output_feature_maps_list.append(top_down)
            output_feature_map_keys.append('top_down_%s' % image_features[-1][0])
            for level in reversed(range(num_levels - 1)):
                if use_native_resize_op:
                    with tf.name_scope('nearest_neighbor_upsampling'):
                        top_down_shape = shape_utils.combined_static_and_dynamic_shape(top_down)
                        top_down = tf.image.resize_nearest_neighbor(top_down, [top_down_shape[1] * 2, top_down_shape[2] * 2])
                else:
                    top_down = ops.nearest_neighbor_upsampling(top_down, scale=2)
                residual = slim.conv2d(image_features[level][1], depth, [1, 1], activation_fn=None, normalizer_fn=None, scope='projection_%d' % (level + 1))
                if use_bounded_activations:
                    residual = tf.clip_by_value(residual, -ACTIVATION_BOUND, ACTIVATION_BOUND)
                if use_explicit_padding:
                    residual_shape = tf.shape(residual)
                    top_down = top_down[:, :residual_shape[1], :residual_shape[2], :]
                top_down += residual
                if use_bounded_activations:
                    top_down = tf.clip_by_value(top_down, -ACTIVATION_BOUND, ACTIVATION_BOUND)
                if use_depthwise:
                    conv_op = functools.partial(slim.separable_conv2d, depth_multiplier=1)
                else:
                    conv_op = slim.conv2d
                if use_explicit_padding:
                    top_down = ops.fixed_padding(top_down, kernel_size)
                output_feature_maps_list.append(conv_op(top_down, depth, [kernel_size, kernel_size], scope='smoothing_%d' % (level + 1)))
                output_feature_map_keys.append('top_down_%s' % image_features[level][0])
            return collections.OrderedDict(reversed(list(zip(output_feature_map_keys, output_feature_maps_list))))

def pooling_pyramid_feature_maps(base_feature_map_depth, num_layers, image_features, replace_pool_with_conv=False):
    if False:
        for i in range(10):
            print('nop')
    'Generates pooling pyramid feature maps.\n\n  The pooling pyramid feature maps is motivated by\n  multi_resolution_feature_maps. The main difference are that it is simpler and\n  reduces the number of free parameters.\n\n  More specifically:\n   - Instead of using convolutions to shrink the feature map, it uses max\n     pooling, therefore totally gets rid of the parameters in convolution.\n   - By pooling feature from larger map up to a single cell, it generates\n     features in the same feature space.\n   - Instead of independently making box predictions from individual maps, it\n     shares the same classifier across different feature maps, therefore reduces\n     the "mis-calibration" across different scales.\n\n  See go/ppn-detection for more details.\n\n  Args:\n    base_feature_map_depth: Depth of the base feature before the max pooling.\n    num_layers: Number of layers used to make predictions. They are pooled\n      from the base feature.\n    image_features: A dictionary of handles to activation tensors from the\n      feature extractor.\n    replace_pool_with_conv: Whether or not to replace pooling operations with\n      convolutions in the PPN. Default is False.\n\n  Returns:\n    feature_maps: an OrderedDict mapping keys (feature map names) to\n      tensors where each tensor has shape [batch, height_i, width_i, depth_i].\n  Raises:\n    ValueError: image_features does not contain exactly one entry\n  '
    if len(image_features) != 1:
        raise ValueError('image_features should be a dictionary of length 1.')
    image_features = image_features[image_features.keys()[0]]
    feature_map_keys = []
    feature_maps = []
    feature_map_key = 'Base_Conv2d_1x1_%d' % base_feature_map_depth
    if base_feature_map_depth > 0:
        image_features = slim.conv2d(image_features, base_feature_map_depth, [1, 1], padding='SAME', stride=1, scope=feature_map_key)
        image_features = slim.max_pool2d(image_features, [1, 1], padding='SAME', stride=1, scope=feature_map_key)
    feature_map_keys.append(feature_map_key)
    feature_maps.append(image_features)
    feature_map = image_features
    if replace_pool_with_conv:
        with slim.arg_scope([slim.conv2d], padding='SAME', stride=2):
            for i in range(num_layers - 1):
                feature_map_key = 'Conv2d_{}_3x3_s2_{}'.format(i, base_feature_map_depth)
                feature_map = slim.conv2d(feature_map, base_feature_map_depth, [3, 3], scope=feature_map_key)
                feature_map_keys.append(feature_map_key)
                feature_maps.append(feature_map)
    else:
        with slim.arg_scope([slim.max_pool2d], padding='SAME', stride=2):
            for i in range(num_layers - 1):
                feature_map_key = 'MaxPool2d_%d_2x2' % i
                feature_map = slim.max_pool2d(feature_map, [2, 2], padding='SAME', scope=feature_map_key)
                feature_map_keys.append(feature_map_key)
                feature_maps.append(feature_map)
    return collections.OrderedDict([(x, y) for (x, y) in zip(feature_map_keys, feature_maps)])