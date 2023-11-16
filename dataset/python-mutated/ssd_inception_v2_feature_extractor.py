"""SSDFeatureExtractor for InceptionV2 features."""
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import ops
from object_detection.utils import shape_utils
from nets import inception_v2
slim = contrib_slim

class SSDInceptionV2FeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
    """SSD Feature Extractor using InceptionV2 features."""

    def __init__(self, is_training, depth_multiplier, min_depth, pad_to_multiple, conv_hyperparams_fn, reuse_weights=None, use_explicit_padding=False, use_depthwise=False, num_layers=6, override_base_feature_extractor_hyperparams=False):
        if False:
            while True:
                i = 10
        'InceptionV2 Feature Extractor for SSD Models.\n\n    Args:\n      is_training: whether the network is in training mode.\n      depth_multiplier: float depth multiplier for feature extractor.\n      min_depth: minimum feature extractor depth.\n      pad_to_multiple: the nearest multiple to zero pad the input height and\n        width dimensions to.\n      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d\n        and separable_conv2d ops in the layers that are added on top of the\n        base feature extractor.\n      reuse_weights: Whether to reuse variables. Default is None.\n      use_explicit_padding: Whether to use explicit padding when extracting\n        features. Default is False.\n      use_depthwise: Whether to use depthwise convolutions. Default is False.\n      num_layers: Number of SSD layers.\n      override_base_feature_extractor_hyperparams: Whether to override\n        hyperparameters of the base feature extractor with the one from\n        `conv_hyperparams_fn`.\n\n    Raises:\n      ValueError: If `override_base_feature_extractor_hyperparams` is False.\n    '
        super(SSDInceptionV2FeatureExtractor, self).__init__(is_training=is_training, depth_multiplier=depth_multiplier, min_depth=min_depth, pad_to_multiple=pad_to_multiple, conv_hyperparams_fn=conv_hyperparams_fn, reuse_weights=reuse_weights, use_explicit_padding=use_explicit_padding, use_depthwise=use_depthwise, num_layers=num_layers, override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams)
        if not self._override_base_feature_extractor_hyperparams:
            raise ValueError('SSD Inception V2 feature extractor always usesscope returned by `conv_hyperparams_fn` for both the base feature extractor and the additional layers added since there is no arg_scope defined for the base feature extractor.')

    def preprocess(self, resized_inputs):
        if False:
            i = 10
            return i + 15
        'SSD preprocessing.\n\n    Maps pixel values to the range [-1, 1].\n\n    Args:\n      resized_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n\n    Returns:\n      preprocessed_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n    '
        return 2.0 / 255.0 * resized_inputs - 1.0

    def extract_features(self, preprocessed_inputs):
        if False:
            i = 10
            return i + 15
        'Extract features from preprocessed inputs.\n\n    Args:\n      preprocessed_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n\n    Returns:\n      feature_maps: a list of tensors where the ith tensor has shape\n        [batch, height_i, width_i, depth_i]\n    '
        preprocessed_inputs = shape_utils.check_min_image_dim(33, preprocessed_inputs)
        feature_map_layout = {'from_layer': ['Mixed_4c', 'Mixed_5c', '', '', '', ''][:self._num_layers], 'layer_depth': [-1, -1, 512, 256, 256, 128][:self._num_layers], 'use_explicit_padding': self._use_explicit_padding, 'use_depthwise': self._use_depthwise}
        with slim.arg_scope(self._conv_hyperparams_fn()):
            with tf.variable_scope('InceptionV2', reuse=self._reuse_weights) as scope:
                (_, image_features) = inception_v2.inception_v2_base(ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple), final_endpoint='Mixed_5c', min_depth=self._min_depth, depth_multiplier=self._depth_multiplier, scope=scope)
                feature_maps = feature_map_generators.multi_resolution_feature_maps(feature_map_layout=feature_map_layout, depth_multiplier=self._depth_multiplier, min_depth=self._min_depth, insert_1x1_conv=True, image_features=image_features)
        return feature_maps.values()