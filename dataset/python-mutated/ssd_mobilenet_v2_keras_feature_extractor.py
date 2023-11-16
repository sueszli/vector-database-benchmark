"""SSDFeatureExtractor for MobilenetV2 features."""
import tensorflow as tf
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.models.keras_models import mobilenet_v2
from object_detection.utils import ops
from object_detection.utils import shape_utils

class SSDMobileNetV2KerasFeatureExtractor(ssd_meta_arch.SSDKerasFeatureExtractor):
    """SSD Feature Extractor using MobilenetV2 features."""

    def __init__(self, is_training, depth_multiplier, min_depth, pad_to_multiple, conv_hyperparams, freeze_batchnorm, inplace_batchnorm_update, use_explicit_padding=False, use_depthwise=False, num_layers=6, override_base_feature_extractor_hyperparams=False, name=None):
        if False:
            while True:
                i = 10
        "MobileNetV2 Feature Extractor for SSD Models.\n\n    Mobilenet v2 (experimental), designed by sandler@. More details can be found\n    in //knowledge/cerebra/brain/compression/mobilenet/mobilenet_experimental.py\n\n    Args:\n      is_training: whether the network is in training mode.\n      depth_multiplier: float depth multiplier for feature extractor (Functions\n        as a width multiplier for the mobilenet_v2 network itself).\n      min_depth: minimum feature extractor depth.\n      pad_to_multiple: the nearest multiple to zero pad the input height and\n        width dimensions to.\n      conv_hyperparams: `hyperparams_builder.KerasLayerHyperparams` object\n        containing convolution hyperparameters for the layers added on top of\n        the base feature extractor.\n      freeze_batchnorm: Whether to freeze batch norm parameters during\n        training or not. When training with a small batch size (e.g. 1), it is\n        desirable to freeze batch norm update and use pretrained batch norm\n        params.\n      inplace_batchnorm_update: Whether to update batch norm moving average\n        values inplace. When this is false train op must add a control\n        dependency on tf.graphkeys.UPDATE_OPS collection in order to update\n        batch norm statistics.\n      use_explicit_padding: Whether to use explicit padding when extracting\n        features. Default is False.\n      use_depthwise: Whether to use depthwise convolutions. Default is False.\n      num_layers: Number of SSD layers.\n      override_base_feature_extractor_hyperparams: Whether to override\n        hyperparameters of the base feature extractor with the one from\n        `conv_hyperparams_fn`.\n      name: A string name scope to assign to the model. If 'None', Keras\n        will auto-generate one from the class name.\n    "
        super(SSDMobileNetV2KerasFeatureExtractor, self).__init__(is_training=is_training, depth_multiplier=depth_multiplier, min_depth=min_depth, pad_to_multiple=pad_to_multiple, conv_hyperparams=conv_hyperparams, freeze_batchnorm=freeze_batchnorm, inplace_batchnorm_update=inplace_batchnorm_update, use_explicit_padding=use_explicit_padding, use_depthwise=use_depthwise, num_layers=num_layers, override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams, name=name)
        self._feature_map_layout = {'from_layer': ['layer_15/expansion_output', 'layer_19', '', '', '', ''][:self._num_layers], 'layer_depth': [-1, -1, 512, 256, 256, 128][:self._num_layers], 'use_depthwise': self._use_depthwise, 'use_explicit_padding': self._use_explicit_padding}
        self.mobilenet_v2 = None
        self.feature_map_generator = None

    def build(self, input_shape):
        if False:
            print('Hello World!')
        full_mobilenet_v2 = mobilenet_v2.mobilenet_v2(batchnorm_training=self._is_training and (not self._freeze_batchnorm), conv_hyperparams=self._conv_hyperparams if self._override_base_feature_extractor_hyperparams else None, weights=None, use_explicit_padding=self._use_explicit_padding, alpha=self._depth_multiplier, min_depth=self._min_depth, include_top=False)
        conv2d_11_pointwise = full_mobilenet_v2.get_layer(name='block_13_expand_relu').output
        conv2d_13_pointwise = full_mobilenet_v2.get_layer(name='out_relu').output
        self.mobilenet_v2 = tf.keras.Model(inputs=full_mobilenet_v2.inputs, outputs=[conv2d_11_pointwise, conv2d_13_pointwise])
        self.feature_map_generator = feature_map_generators.KerasMultiResolutionFeatureMaps(feature_map_layout=self._feature_map_layout, depth_multiplier=self._depth_multiplier, min_depth=self._min_depth, insert_1x1_conv=True, is_training=self._is_training, conv_hyperparams=self._conv_hyperparams, freeze_batchnorm=self._freeze_batchnorm, name='FeatureMaps')
        self.built = True

    def preprocess(self, resized_inputs):
        if False:
            i = 10
            return i + 15
        'SSD preprocessing.\n\n    Maps pixel values to the range [-1, 1].\n\n    Args:\n      resized_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n\n    Returns:\n      preprocessed_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n    '
        return 2.0 / 255.0 * resized_inputs - 1.0

    def _extract_features(self, preprocessed_inputs):
        if False:
            i = 10
            return i + 15
        'Extract features from preprocessed inputs.\n\n    Args:\n      preprocessed_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n\n    Returns:\n      feature_maps: a list of tensors where the ith tensor has shape\n        [batch, height_i, width_i, depth_i]\n    '
        preprocessed_inputs = shape_utils.check_min_image_dim(33, preprocessed_inputs)
        image_features = self.mobilenet_v2(ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple))
        feature_maps = self.feature_map_generator({'layer_15/expansion_output': image_features[0], 'layer_19': image_features[1]})
        return feature_maps.values()