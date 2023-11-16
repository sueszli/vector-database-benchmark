"""Embedded-friendly SSDFeatureExtractor for MobilenetV1 features."""
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import context_manager
from object_detection.utils import ops
from nets import mobilenet_v1
slim = contrib_slim

class EmbeddedSSDMobileNetV1FeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
    """Embedded-friendly SSD Feature Extractor using MobilenetV1 features.

  This feature extractor is similar to SSD MobileNetV1 feature extractor, and
  it fixes input resolution to be 256x256, reduces the number of feature maps
  used for box prediction and ensures convolution kernel to be no larger
  than input tensor in spatial dimensions.

  This feature extractor requires support of the following ops if used in
  embedded devices:
  - Conv
  - DepthwiseConv
  - Relu6

  All conv/depthwiseconv use SAME padding, and no additional spatial padding is
  needed.
  """

    def __init__(self, is_training, depth_multiplier, min_depth, pad_to_multiple, conv_hyperparams_fn, reuse_weights=None, use_explicit_padding=False, use_depthwise=False, override_base_feature_extractor_hyperparams=False):
        if False:
            while True:
                i = 10
        'MobileNetV1 Feature Extractor for Embedded-friendly SSD Models.\n\n    Args:\n      is_training: whether the network is in training mode.\n      depth_multiplier: float depth multiplier for feature extractor.\n      min_depth: minimum feature extractor depth.\n      pad_to_multiple: the nearest multiple to zero pad the input height and\n        width dimensions to. For EmbeddedSSD it must be set to 1.\n      conv_hyperparams_fn: A function to construct tf slim arg_scope for conv2d\n        and separable_conv2d ops in the layers that are added on top of the\n        base feature extractor.\n      reuse_weights: Whether to reuse variables. Default is None.\n      use_explicit_padding: Whether to use explicit padding when extracting\n        features. Default is False.\n      use_depthwise: Whether to use depthwise convolutions. Default is False.\n      override_base_feature_extractor_hyperparams: Whether to override\n        hyperparameters of the base feature extractor with the one from\n        `conv_hyperparams_fn`.\n\n    Raises:\n      ValueError: upon invalid `pad_to_multiple` values.\n    '
        if pad_to_multiple != 1:
            raise ValueError('Embedded-specific SSD only supports `pad_to_multiple` of 1.')
        super(EmbeddedSSDMobileNetV1FeatureExtractor, self).__init__(is_training, depth_multiplier, min_depth, pad_to_multiple, conv_hyperparams_fn, reuse_weights, use_explicit_padding, use_depthwise, override_base_feature_extractor_hyperparams)

    def preprocess(self, resized_inputs):
        if False:
            print('Hello World!')
        'SSD preprocessing.\n\n    Maps pixel values to the range [-1, 1].\n\n    Args:\n      resized_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n\n    Returns:\n      preprocessed_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n    '
        return 2.0 / 255.0 * resized_inputs - 1.0

    def extract_features(self, preprocessed_inputs):
        if False:
            return 10
        'Extract features from preprocessed inputs.\n\n    Args:\n      preprocessed_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n\n    Returns:\n      feature_maps: a list of tensors where the ith tensor has shape\n        [batch, height_i, width_i, depth_i]\n\n    Raises:\n      ValueError: if image height or width are not 256 pixels.\n    '
        image_shape = preprocessed_inputs.get_shape()
        image_shape.assert_has_rank(4)
        image_height = image_shape[1].value
        image_width = image_shape[2].value
        if image_height is None or image_width is None:
            shape_assert = tf.Assert(tf.logical_and(tf.equal(tf.shape(preprocessed_inputs)[1], 256), tf.equal(tf.shape(preprocessed_inputs)[2], 256)), ['image size must be 256 in both height and width.'])
            with tf.control_dependencies([shape_assert]):
                preprocessed_inputs = tf.identity(preprocessed_inputs)
        elif image_height != 256 or image_width != 256:
            raise ValueError('image size must be = 256 in both height and width; image dim = %d,%d' % (image_height, image_width))
        feature_map_layout = {'from_layer': ['Conv2d_11_pointwise', 'Conv2d_13_pointwise', '', '', ''], 'layer_depth': [-1, -1, 512, 256, 256], 'conv_kernel_size': [-1, -1, 3, 3, 2], 'use_explicit_padding': self._use_explicit_padding, 'use_depthwise': self._use_depthwise}
        with tf.variable_scope('MobilenetV1', reuse=self._reuse_weights) as scope:
            with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=None)):
                with slim.arg_scope(self._conv_hyperparams_fn()) if self._override_base_feature_extractor_hyperparams else context_manager.IdentityContextManager():
                    (_, image_features) = mobilenet_v1.mobilenet_v1_base(ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple), final_endpoint='Conv2d_13_pointwise', min_depth=self._min_depth, depth_multiplier=self._depth_multiplier, use_explicit_padding=self._use_explicit_padding, scope=scope)
            with slim.arg_scope(self._conv_hyperparams_fn()):
                feature_maps = feature_map_generators.multi_resolution_feature_maps(feature_map_layout=feature_map_layout, depth_multiplier=self._depth_multiplier, min_depth=self._min_depth, insert_1x1_conv=True, image_features=image_features)
        return feature_maps.values()