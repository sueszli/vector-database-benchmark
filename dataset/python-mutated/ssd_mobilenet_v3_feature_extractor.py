"""SSDFeatureExtractor for MobileNetV3 features."""
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import context_manager
from object_detection.utils import ops
from object_detection.utils import shape_utils
from nets.mobilenet import mobilenet
from nets.mobilenet import mobilenet_v3
slim = contrib_slim

class SSDMobileNetV3FeatureExtractorBase(ssd_meta_arch.SSDFeatureExtractor):
    """Base class of SSD feature extractor using MobilenetV3 features."""

    def __init__(self, conv_defs, from_layer, is_training, depth_multiplier, min_depth, pad_to_multiple, conv_hyperparams_fn, reuse_weights=None, use_explicit_padding=False, use_depthwise=False, override_base_feature_extractor_hyperparams=False, scope_name='MobilenetV3'):
        if False:
            while True:
                i = 10
        'MobileNetV3 Feature Extractor for SSD Models.\n\n    MobileNet v3. Details found in:\n    https://arxiv.org/abs/1905.02244\n\n    Args:\n      conv_defs: MobileNetV3 conv defs for backbone.\n      from_layer: A cell of two layer names (string) to connect to the 1st and\n        2nd inputs of the SSD head.\n      is_training: whether the network is in training mode.\n      depth_multiplier: float depth multiplier for feature extractor.\n      min_depth: minimum feature extractor depth.\n      pad_to_multiple: the nearest multiple to zero pad the input height and\n        width dimensions to.\n      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d\n        and separable_conv2d ops in the layers that are added on top of the base\n        feature extractor.\n      reuse_weights: Whether to reuse variables. Default is None.\n      use_explicit_padding: Whether to use explicit padding when extracting\n        features. Default is False.\n      use_depthwise: Whether to use depthwise convolutions. Default is False.\n      override_base_feature_extractor_hyperparams: Whether to override\n        hyperparameters of the base feature extractor with the one from\n        `conv_hyperparams_fn`.\n      scope_name: scope name (string) of network variables.\n    '
        super(SSDMobileNetV3FeatureExtractorBase, self).__init__(is_training=is_training, depth_multiplier=depth_multiplier, min_depth=min_depth, pad_to_multiple=pad_to_multiple, conv_hyperparams_fn=conv_hyperparams_fn, reuse_weights=reuse_weights, use_explicit_padding=use_explicit_padding, use_depthwise=use_depthwise, override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams)
        self._conv_defs = conv_defs
        self._from_layer = from_layer
        self._scope_name = scope_name

    def preprocess(self, resized_inputs):
        if False:
            while True:
                i = 10
        'SSD preprocessing.\n\n    Maps pixel values to the range [-1, 1].\n\n    Args:\n      resized_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n\n    Returns:\n      preprocessed_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n    '
        return 2.0 / 255.0 * resized_inputs - 1.0

    def extract_features(self, preprocessed_inputs):
        if False:
            for i in range(10):
                print('nop')
        'Extract features from preprocessed inputs.\n\n    Args:\n      preprocessed_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n\n    Returns:\n      feature_maps: a list of tensors where the ith tensor has shape\n        [batch, height_i, width_i, depth_i]\n    Raises:\n      ValueError if conv_defs is not provided or from_layer does not meet the\n        size requirement.\n    '
        if not self._conv_defs:
            raise ValueError('Must provide backbone conv defs.')
        if len(self._from_layer) != 2:
            raise ValueError('SSD input feature names are not provided.')
        preprocessed_inputs = shape_utils.check_min_image_dim(33, preprocessed_inputs)
        feature_map_layout = {'from_layer': [self._from_layer[0], self._from_layer[1], '', '', '', ''], 'layer_depth': [-1, -1, 512, 256, 256, 128], 'use_depthwise': self._use_depthwise, 'use_explicit_padding': self._use_explicit_padding}
        with tf.variable_scope(self._scope_name, reuse=self._reuse_weights) as scope:
            with slim.arg_scope(mobilenet_v3.training_scope(is_training=None, bn_decay=0.9997)), slim.arg_scope([mobilenet.depth_multiplier], min_depth=self._min_depth):
                with slim.arg_scope(self._conv_hyperparams_fn()) if self._override_base_feature_extractor_hyperparams else context_manager.IdentityContextManager():
                    (_, image_features) = mobilenet_v3.mobilenet_base(ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple), conv_defs=self._conv_defs, final_endpoint=self._from_layer[1], depth_multiplier=self._depth_multiplier, use_explicit_padding=self._use_explicit_padding, scope=scope)
                with slim.arg_scope(self._conv_hyperparams_fn()):
                    feature_maps = feature_map_generators.multi_resolution_feature_maps(feature_map_layout=feature_map_layout, depth_multiplier=self._depth_multiplier, min_depth=self._min_depth, insert_1x1_conv=True, image_features=image_features)
        return feature_maps.values()

class SSDMobileNetV3LargeFeatureExtractor(SSDMobileNetV3FeatureExtractorBase):
    """Mobilenet V3-Large feature extractor."""

    def __init__(self, is_training, depth_multiplier, min_depth, pad_to_multiple, conv_hyperparams_fn, reuse_weights=None, use_explicit_padding=False, use_depthwise=False, override_base_feature_extractor_hyperparams=False, scope_name='MobilenetV3'):
        if False:
            return 10
        super(SSDMobileNetV3LargeFeatureExtractor, self).__init__(conv_defs=mobilenet_v3.V3_LARGE_DETECTION, from_layer=['layer_14/expansion_output', 'layer_17'], is_training=is_training, depth_multiplier=depth_multiplier, min_depth=min_depth, pad_to_multiple=pad_to_multiple, conv_hyperparams_fn=conv_hyperparams_fn, reuse_weights=reuse_weights, use_explicit_padding=use_explicit_padding, use_depthwise=use_depthwise, override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams, scope_name=scope_name)

class SSDMobileNetV3SmallFeatureExtractor(SSDMobileNetV3FeatureExtractorBase):
    """Mobilenet V3-Small feature extractor."""

    def __init__(self, is_training, depth_multiplier, min_depth, pad_to_multiple, conv_hyperparams_fn, reuse_weights=None, use_explicit_padding=False, use_depthwise=False, override_base_feature_extractor_hyperparams=False, scope_name='MobilenetV3'):
        if False:
            while True:
                i = 10
        super(SSDMobileNetV3SmallFeatureExtractor, self).__init__(conv_defs=mobilenet_v3.V3_SMALL_DETECTION, from_layer=['layer_10/expansion_output', 'layer_13'], is_training=is_training, depth_multiplier=depth_multiplier, min_depth=min_depth, pad_to_multiple=pad_to_multiple, conv_hyperparams_fn=conv_hyperparams_fn, reuse_weights=reuse_weights, use_explicit_padding=use_explicit_padding, use_depthwise=use_depthwise, override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams, scope_name=scope_name)