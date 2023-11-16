"""Extracts features for different models."""
import copy
import functools
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from deeplab.core import nas_network
from deeplab.core import resnet_v1_beta
from deeplab.core import xception
from nets.mobilenet import conv_blocks
from nets.mobilenet import mobilenet
from nets.mobilenet import mobilenet_v2
from nets.mobilenet import mobilenet_v3
slim = contrib_slim
_MOBILENET_V2_FINAL_ENDPOINT = 'layer_18'
_MOBILENET_V3_LARGE_FINAL_ENDPOINT = 'layer_17'
_MOBILENET_V3_SMALL_FINAL_ENDPOINT = 'layer_13'

def _mobilenet_v2(net, depth_multiplier, output_stride, conv_defs=None, divisible_by=None, reuse=None, scope=None, final_endpoint=None):
    if False:
        for i in range(10):
            print('nop')
    "Auxiliary function to add support for 'reuse' to mobilenet_v2.\n\n  Args:\n    net: Input tensor of shape [batch_size, height, width, channels].\n    depth_multiplier: Float multiplier for the depth (number of channels)\n      for all convolution ops. The value must be greater than zero. Typical\n      usage will be to set this value in (0, 1) to reduce the number of\n      parameters or computation cost of the model.\n    output_stride: An integer that specifies the requested ratio of input to\n      output spatial resolution. If not None, then we invoke atrous convolution\n      if necessary to prevent the network from reducing the spatial resolution\n      of the activation maps. Allowed values are 8 (accurate fully convolutional\n      mode), 16 (fast fully convolutional mode), 32 (classification mode).\n    conv_defs: MobileNet con def.\n    divisible_by: None (use default setting) or an integer that ensures all\n      layers # channels will be divisible by this number. Used in MobileNet.\n    reuse: Reuse model variables.\n    scope: Optional variable scope.\n    final_endpoint: The endpoint to construct the network up to.\n\n  Returns:\n    Features extracted by MobileNetv2.\n  "
    if divisible_by is None:
        divisible_by = 8 if depth_multiplier == 1.0 else 1
    if conv_defs is None:
        conv_defs = mobilenet_v2.V2_DEF
    with tf.variable_scope(scope, 'MobilenetV2', [net], reuse=reuse) as scope:
        return mobilenet_v2.mobilenet_base(net, conv_defs=conv_defs, depth_multiplier=depth_multiplier, min_depth=8 if depth_multiplier == 1.0 else 1, divisible_by=divisible_by, final_endpoint=final_endpoint or _MOBILENET_V2_FINAL_ENDPOINT, output_stride=output_stride, scope=scope)

def _mobilenet_v3(net, depth_multiplier, output_stride, conv_defs=None, divisible_by=None, reuse=None, scope=None, final_endpoint=None):
    if False:
        i = 10
        return i + 15
    'Auxiliary function to build mobilenet v3.\n\n  Args:\n    net: Input tensor of shape [batch_size, height, width, channels].\n    depth_multiplier: Float multiplier for the depth (number of channels)\n      for all convolution ops. The value must be greater than zero. Typical\n      usage will be to set this value in (0, 1) to reduce the number of\n      parameters or computation cost of the model.\n    output_stride: An integer that specifies the requested ratio of input to\n      output spatial resolution. If not None, then we invoke atrous convolution\n      if necessary to prevent the network from reducing the spatial resolution\n      of the activation maps. Allowed values are 8 (accurate fully convolutional\n      mode), 16 (fast fully convolutional mode), 32 (classification mode).\n    conv_defs: A list of ConvDef namedtuples specifying the net architecture.\n    divisible_by: None (use default setting) or an integer that ensures all\n      layers # channels will be divisible by this number. Used in MobileNet.\n    reuse: Reuse model variables.\n    scope: Optional variable scope.\n    final_endpoint: The endpoint to construct the network up to.\n\n  Returns:\n    net: The output tensor.\n    end_points: A set of activations for external use.\n\n  Raises:\n    ValueError: If conv_defs or final_endpoint is not specified.\n  '
    del divisible_by
    with tf.variable_scope(scope, 'MobilenetV3', [net], reuse=reuse) as scope:
        if conv_defs is None:
            raise ValueError('conv_defs must be specified for mobilenet v3.')
        if final_endpoint is None:
            raise ValueError('Final endpoint must be specified for mobilenet v3.')
        (net, end_points) = mobilenet_v3.mobilenet_base(net, depth_multiplier=depth_multiplier, conv_defs=conv_defs, output_stride=output_stride, final_endpoint=final_endpoint, scope=scope)
        return (net, end_points)

def mobilenet_v3_large_seg(net, depth_multiplier, output_stride, divisible_by=None, reuse=None, scope=None, final_endpoint=None):
    if False:
        print('Hello World!')
    'Final mobilenet v3 large model for segmentation task.'
    del divisible_by
    del final_endpoint
    conv_defs = copy.deepcopy(mobilenet_v3.V3_LARGE)
    for (layer, expansion) in [(13, 336), (14, 480), (15, 480), (16, None)]:
        conv_defs['spec'][layer].params['num_outputs'] /= 2
        if expansion is not None:
            factor = expansion / conv_defs['spec'][layer - 1].params['num_outputs']
            conv_defs['spec'][layer].params['expansion_size'] = mobilenet_v3.expand_input(factor)
    return _mobilenet_v3(net, depth_multiplier=depth_multiplier, output_stride=output_stride, divisible_by=8, conv_defs=conv_defs, reuse=reuse, scope=scope, final_endpoint=_MOBILENET_V3_LARGE_FINAL_ENDPOINT)

def mobilenet_v3_small_seg(net, depth_multiplier, output_stride, divisible_by=None, reuse=None, scope=None, final_endpoint=None):
    if False:
        return 10
    'Final mobilenet v3 small model for segmentation task.'
    del divisible_by
    del final_endpoint
    conv_defs = copy.deepcopy(mobilenet_v3.V3_SMALL)
    for (layer, expansion) in [(9, 144), (10, 288), (11, 288), (12, None)]:
        conv_defs['spec'][layer].params['num_outputs'] /= 2
        if expansion is not None:
            factor = expansion / conv_defs['spec'][layer - 1].params['num_outputs']
            conv_defs['spec'][layer].params['expansion_size'] = mobilenet_v3.expand_input(factor)
    return _mobilenet_v3(net, depth_multiplier=depth_multiplier, output_stride=output_stride, divisible_by=8, conv_defs=conv_defs, reuse=reuse, scope=scope, final_endpoint=_MOBILENET_V3_SMALL_FINAL_ENDPOINT)
networks_map = {'mobilenet_v2': _mobilenet_v2, 'mobilenet_v3_large_seg': mobilenet_v3_large_seg, 'mobilenet_v3_small_seg': mobilenet_v3_small_seg, 'resnet_v1_18': resnet_v1_beta.resnet_v1_18, 'resnet_v1_18_beta': resnet_v1_beta.resnet_v1_18_beta, 'resnet_v1_50': resnet_v1_beta.resnet_v1_50, 'resnet_v1_50_beta': resnet_v1_beta.resnet_v1_50_beta, 'resnet_v1_101': resnet_v1_beta.resnet_v1_101, 'resnet_v1_101_beta': resnet_v1_beta.resnet_v1_101_beta, 'xception_41': xception.xception_41, 'xception_65': xception.xception_65, 'xception_71': xception.xception_71, 'nas_pnasnet': nas_network.pnasnet, 'nas_hnasnet': nas_network.hnasnet}

def mobilenet_v2_arg_scope(is_training=True, weight_decay=4e-05, stddev=0.09, activation=tf.nn.relu6, bn_decay=0.997, bn_epsilon=None, bn_renorm=None):
    if False:
        print('Hello World!')
    "Defines the default MobilenetV2 arg scope.\n\n  Args:\n    is_training: Whether or not we're training the model. If this is set to None\n      is_training parameter in batch_norm is not set. Please note that this also\n      sets the is_training parameter in dropout to None.\n    weight_decay: The weight decay to use for regularizing the model.\n    stddev: Standard deviation for initialization, if negative uses xavier.\n    activation: If True, a modified activation is used (initialized ~ReLU6).\n    bn_decay: decay for the batch norm moving averages.\n    bn_epsilon: batch normalization epsilon.\n    bn_renorm: whether to use batchnorm renormalization\n\n  Returns:\n    An `arg_scope` to use for the mobilenet v1 model.\n  "
    batch_norm_params = {'center': True, 'scale': True, 'decay': bn_decay}
    if bn_epsilon is not None:
        batch_norm_params['epsilon'] = bn_epsilon
    if is_training is not None:
        batch_norm_params['is_training'] = is_training
    if bn_renorm is not None:
        batch_norm_params['renorm'] = bn_renorm
    dropout_params = {}
    if is_training is not None:
        dropout_params['is_training'] = is_training
    instance_norm_params = {'center': True, 'scale': True, 'epsilon': 0.001}
    if stddev < 0:
        weight_intitializer = slim.initializers.xavier_initializer()
    else:
        weight_intitializer = tf.truncated_normal_initializer(stddev=stddev)
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.separable_conv2d], weights_initializer=weight_intitializer, activation_fn=activation, normalizer_fn=slim.batch_norm), slim.arg_scope([conv_blocks.expanded_conv], normalizer_fn=slim.batch_norm), slim.arg_scope([mobilenet.apply_activation], activation_fn=activation), slim.arg_scope([slim.batch_norm], **batch_norm_params), slim.arg_scope([mobilenet.mobilenet_base, mobilenet.mobilenet], is_training=is_training), slim.arg_scope([slim.dropout], **dropout_params), slim.arg_scope([slim.instance_norm], **instance_norm_params), slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(weight_decay)), slim.arg_scope([slim.separable_conv2d], weights_regularizer=None), slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME') as s:
        return s
arg_scopes_map = {'mobilenet_v2': mobilenet_v2.training_scope, 'mobilenet_v3_large_seg': mobilenet_v2_arg_scope, 'mobilenet_v3_small_seg': mobilenet_v2_arg_scope, 'resnet_v1_18': resnet_v1_beta.resnet_arg_scope, 'resnet_v1_18_beta': resnet_v1_beta.resnet_arg_scope, 'resnet_v1_50': resnet_v1_beta.resnet_arg_scope, 'resnet_v1_50_beta': resnet_v1_beta.resnet_arg_scope, 'resnet_v1_101': resnet_v1_beta.resnet_arg_scope, 'resnet_v1_101_beta': resnet_v1_beta.resnet_arg_scope, 'xception_41': xception.xception_arg_scope, 'xception_65': xception.xception_arg_scope, 'xception_71': xception.xception_arg_scope, 'nas_pnasnet': nas_network.nas_arg_scope, 'nas_hnasnet': nas_network.nas_arg_scope}
DECODER_END_POINTS = 'decoder_end_points'
networks_to_feature_maps = {'mobilenet_v2': {DECODER_END_POINTS: {4: ['layer_4/depthwise_output'], 8: ['layer_7/depthwise_output'], 16: ['layer_14/depthwise_output']}}, 'mobilenet_v3_large_seg': {DECODER_END_POINTS: {4: ['layer_4/depthwise_output'], 8: ['layer_7/depthwise_output'], 16: ['layer_13/depthwise_output']}}, 'mobilenet_v3_small_seg': {DECODER_END_POINTS: {4: ['layer_2/depthwise_output'], 8: ['layer_4/depthwise_output'], 16: ['layer_9/depthwise_output']}}, 'resnet_v1_18': {DECODER_END_POINTS: {4: ['block1/unit_1/lite_bottleneck_v1/conv2'], 8: ['block2/unit_1/lite_bottleneck_v1/conv2'], 16: ['block3/unit_1/lite_bottleneck_v1/conv2']}}, 'resnet_v1_18_beta': {DECODER_END_POINTS: {4: ['block1/unit_1/lite_bottleneck_v1/conv2'], 8: ['block2/unit_1/lite_bottleneck_v1/conv2'], 16: ['block3/unit_1/lite_bottleneck_v1/conv2']}}, 'resnet_v1_50': {DECODER_END_POINTS: {4: ['block1/unit_2/bottleneck_v1/conv3'], 8: ['block2/unit_3/bottleneck_v1/conv3'], 16: ['block3/unit_5/bottleneck_v1/conv3']}}, 'resnet_v1_50_beta': {DECODER_END_POINTS: {4: ['block1/unit_2/bottleneck_v1/conv3'], 8: ['block2/unit_3/bottleneck_v1/conv3'], 16: ['block3/unit_5/bottleneck_v1/conv3']}}, 'resnet_v1_101': {DECODER_END_POINTS: {4: ['block1/unit_2/bottleneck_v1/conv3'], 8: ['block2/unit_3/bottleneck_v1/conv3'], 16: ['block3/unit_22/bottleneck_v1/conv3']}}, 'resnet_v1_101_beta': {DECODER_END_POINTS: {4: ['block1/unit_2/bottleneck_v1/conv3'], 8: ['block2/unit_3/bottleneck_v1/conv3'], 16: ['block3/unit_22/bottleneck_v1/conv3']}}, 'xception_41': {DECODER_END_POINTS: {4: ['entry_flow/block2/unit_1/xception_module/separable_conv2_pointwise'], 8: ['entry_flow/block3/unit_1/xception_module/separable_conv2_pointwise'], 16: ['exit_flow/block1/unit_1/xception_module/separable_conv2_pointwise']}}, 'xception_65': {DECODER_END_POINTS: {4: ['entry_flow/block2/unit_1/xception_module/separable_conv2_pointwise'], 8: ['entry_flow/block3/unit_1/xception_module/separable_conv2_pointwise'], 16: ['exit_flow/block1/unit_1/xception_module/separable_conv2_pointwise']}}, 'xception_71': {DECODER_END_POINTS: {4: ['entry_flow/block3/unit_1/xception_module/separable_conv2_pointwise'], 8: ['entry_flow/block5/unit_1/xception_module/separable_conv2_pointwise'], 16: ['exit_flow/block1/unit_1/xception_module/separable_conv2_pointwise']}}, 'nas_pnasnet': {DECODER_END_POINTS: {4: ['Stem'], 8: ['Cell_3'], 16: ['Cell_7']}}, 'nas_hnasnet': {DECODER_END_POINTS: {4: ['Cell_2'], 8: ['Cell_5'], 16: ['Cell_7']}}}
name_scope = {'mobilenet_v2': 'MobilenetV2', 'mobilenet_v3_large_seg': 'MobilenetV3', 'mobilenet_v3_small_seg': 'MobilenetV3', 'resnet_v1_18': 'resnet_v1_18', 'resnet_v1_18_beta': 'resnet_v1_18', 'resnet_v1_50': 'resnet_v1_50', 'resnet_v1_50_beta': 'resnet_v1_50', 'resnet_v1_101': 'resnet_v1_101', 'resnet_v1_101_beta': 'resnet_v1_101', 'xception_41': 'xception_41', 'xception_65': 'xception_65', 'xception_71': 'xception_71', 'nas_pnasnet': 'pnasnet', 'nas_hnasnet': 'hnasnet'}
_MEAN_RGB = [123.15, 115.9, 103.06]

def _preprocess_subtract_imagenet_mean(inputs, dtype=tf.float32):
    if False:
        print('Hello World!')
    'Subtract Imagenet mean RGB value.'
    mean_rgb = tf.reshape(_MEAN_RGB, [1, 1, 1, 3])
    num_channels = tf.shape(inputs)[-1]
    mean_rgb_extended = tf.concat([mean_rgb, tf.zeros([1, 1, 1, num_channels - 3])], axis=3)
    return tf.cast(inputs - mean_rgb_extended, dtype=dtype)

def _preprocess_zero_mean_unit_range(inputs, dtype=tf.float32):
    if False:
        return 10
    'Map image values from [0, 255] to [-1, 1].'
    preprocessed_inputs = 2.0 / 255.0 * tf.to_float(inputs) - 1.0
    return tf.cast(preprocessed_inputs, dtype=dtype)
_PREPROCESS_FN = {'mobilenet_v2': _preprocess_zero_mean_unit_range, 'mobilenet_v3_large_seg': _preprocess_zero_mean_unit_range, 'mobilenet_v3_small_seg': _preprocess_zero_mean_unit_range, 'resnet_v1_18': _preprocess_subtract_imagenet_mean, 'resnet_v1_18_beta': _preprocess_zero_mean_unit_range, 'resnet_v1_50': _preprocess_subtract_imagenet_mean, 'resnet_v1_50_beta': _preprocess_zero_mean_unit_range, 'resnet_v1_101': _preprocess_subtract_imagenet_mean, 'resnet_v1_101_beta': _preprocess_zero_mean_unit_range, 'xception_41': _preprocess_zero_mean_unit_range, 'xception_65': _preprocess_zero_mean_unit_range, 'xception_71': _preprocess_zero_mean_unit_range, 'nas_pnasnet': _preprocess_zero_mean_unit_range, 'nas_hnasnet': _preprocess_zero_mean_unit_range}

def mean_pixel(model_variant=None):
    if False:
        for i in range(10):
            print('nop')
    'Gets mean pixel value.\n\n  This function returns different mean pixel value, depending on the input\n  model_variant which adopts different preprocessing functions. We currently\n  handle the following preprocessing functions:\n  (1) _preprocess_subtract_imagenet_mean. We simply return mean pixel value.\n  (2) _preprocess_zero_mean_unit_range. We return [127.5, 127.5, 127.5].\n  The return values are used in a way that the padded regions after\n  pre-processing will contain value 0.\n\n  Args:\n    model_variant: Model variant (string) for feature extraction. For\n      backwards compatibility, model_variant=None returns _MEAN_RGB.\n\n  Returns:\n    Mean pixel value.\n  '
    if model_variant in ['resnet_v1_50', 'resnet_v1_101'] or model_variant is None:
        return _MEAN_RGB
    else:
        return [127.5, 127.5, 127.5]

def extract_features(images, output_stride=8, multi_grid=None, depth_multiplier=1.0, divisible_by=None, final_endpoint=None, model_variant=None, weight_decay=0.0001, reuse=None, is_training=False, fine_tune_batch_norm=False, regularize_depthwise=False, preprocess_images=True, preprocessed_images_dtype=tf.float32, num_classes=None, global_pool=False, nas_architecture_options=None, nas_training_hyper_parameters=None, use_bounded_activation=False):
    if False:
        print('Hello World!')
    'Extracts features by the particular model_variant.\n\n  Args:\n    images: A tensor of size [batch, height, width, channels].\n    output_stride: The ratio of input to output spatial resolution.\n    multi_grid: Employ a hierarchy of different atrous rates within network.\n    depth_multiplier: Float multiplier for the depth (number of channels)\n      for all convolution ops used in MobileNet.\n    divisible_by: None (use default setting) or an integer that ensures all\n      layers # channels will be divisible by this number. Used in MobileNet.\n    final_endpoint: The MobileNet endpoint to construct the network up to.\n    model_variant: Model variant for feature extraction.\n    weight_decay: The weight decay for model variables.\n    reuse: Reuse the model variables or not.\n    is_training: Is training or not.\n    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.\n    regularize_depthwise: Whether or not apply L2-norm regularization on the\n      depthwise convolution weights.\n    preprocess_images: Performs preprocessing on images or not. Defaults to\n      True. Set to False if preprocessing will be done by other functions. We\n      supprot two types of preprocessing: (1) Mean pixel substraction and (2)\n      Pixel values normalization to be [-1, 1].\n    preprocessed_images_dtype: The type after the preprocessing function.\n    num_classes: Number of classes for image classification task. Defaults\n      to None for dense prediction tasks.\n    global_pool: Global pooling for image classification task. Defaults to\n      False, since dense prediction tasks do not use this.\n    nas_architecture_options: A dictionary storing NAS architecture options.\n      It is either None or its kerys are:\n      - `nas_stem_output_num_conv_filters`: Number of filters of the NAS stem\n        output tensor.\n      - `nas_use_classification_head`: Boolean, use image classification head.\n    nas_training_hyper_parameters: A dictionary storing hyper-parameters for\n      training nas models. It is either None or its keys are:\n      - `drop_path_keep_prob`: Probability to keep each path in the cell when\n        training.\n      - `total_training_steps`: Total training steps to help drop path\n        probability calculation.\n    use_bounded_activation: Whether or not to use bounded activations. Bounded\n      activations better lend themselves to quantized inference. Currently,\n      bounded activation is only used in xception model.\n\n  Returns:\n    features: A tensor of size [batch, feature_height, feature_width,\n      feature_channels], where feature_height/feature_width are determined\n      by the images height/width and output_stride.\n    end_points: A dictionary from components of the network to the corresponding\n      activation.\n\n  Raises:\n    ValueError: Unrecognized model variant.\n  '
    if 'resnet' in model_variant:
        arg_scope = arg_scopes_map[model_variant](weight_decay=weight_decay, batch_norm_decay=0.95, batch_norm_epsilon=1e-05, batch_norm_scale=True)
        (features, end_points) = get_network(model_variant, preprocess_images, preprocessed_images_dtype, arg_scope)(inputs=images, num_classes=num_classes, is_training=is_training and fine_tune_batch_norm, global_pool=global_pool, output_stride=output_stride, multi_grid=multi_grid, reuse=reuse, scope=name_scope[model_variant])
    elif 'xception' in model_variant:
        arg_scope = arg_scopes_map[model_variant](weight_decay=weight_decay, batch_norm_decay=0.9997, batch_norm_epsilon=0.001, batch_norm_scale=True, regularize_depthwise=regularize_depthwise, use_bounded_activation=use_bounded_activation)
        (features, end_points) = get_network(model_variant, preprocess_images, preprocessed_images_dtype, arg_scope)(inputs=images, num_classes=num_classes, is_training=is_training and fine_tune_batch_norm, global_pool=global_pool, output_stride=output_stride, regularize_depthwise=regularize_depthwise, multi_grid=multi_grid, reuse=reuse, scope=name_scope[model_variant])
    elif 'mobilenet' in model_variant or model_variant.startswith('mnas'):
        arg_scope = arg_scopes_map[model_variant](is_training=is_training and fine_tune_batch_norm, weight_decay=weight_decay)
        (features, end_points) = get_network(model_variant, preprocess_images, preprocessed_images_dtype, arg_scope)(inputs=images, depth_multiplier=depth_multiplier, divisible_by=divisible_by, output_stride=output_stride, reuse=reuse, scope=name_scope[model_variant], final_endpoint=final_endpoint)
    elif model_variant.startswith('nas'):
        arg_scope = arg_scopes_map[model_variant](weight_decay=weight_decay, batch_norm_decay=0.9997, batch_norm_epsilon=0.001)
        (features, end_points) = get_network(model_variant, preprocess_images, preprocessed_images_dtype, arg_scope)(inputs=images, num_classes=num_classes, is_training=is_training and fine_tune_batch_norm, global_pool=global_pool, output_stride=output_stride, nas_architecture_options=nas_architecture_options, nas_training_hyper_parameters=nas_training_hyper_parameters, reuse=reuse, scope=name_scope[model_variant])
    else:
        raise ValueError('Unknown model variant %s.' % model_variant)
    return (features, end_points)

def get_network(network_name, preprocess_images, preprocessed_images_dtype=tf.float32, arg_scope=None):
    if False:
        return 10
    'Gets the network.\n\n  Args:\n    network_name: Network name.\n    preprocess_images: Preprocesses the images or not.\n    preprocessed_images_dtype: The type after the preprocessing function.\n    arg_scope: Optional, arg_scope to build the network. If not provided the\n      default arg_scope of the network would be used.\n\n  Returns:\n    A network function that is used to extract features.\n\n  Raises:\n    ValueError: network is not supported.\n  '
    if network_name not in networks_map:
        raise ValueError('Unsupported network %s.' % network_name)
    arg_scope = arg_scope or arg_scopes_map[network_name]()

    def _identity_function(inputs, dtype=preprocessed_images_dtype):
        if False:
            for i in range(10):
                print('nop')
        return tf.cast(inputs, dtype=dtype)
    if preprocess_images:
        preprocess_function = _PREPROCESS_FN[network_name]
    else:
        preprocess_function = _identity_function
    func = networks_map[network_name]

    @functools.wraps(func)
    def network_fn(inputs, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        with slim.arg_scope(arg_scope):
            return func(preprocess_function(inputs, preprocessed_images_dtype), *args, **kwargs)
    return network_fn