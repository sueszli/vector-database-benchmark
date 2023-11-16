"""SSD Keras-based MobilenetV1 FPN Feature Extractor."""
import tensorflow as tf
from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.models.keras_models import mobilenet_v1
from object_detection.models.keras_models import model_utils
from object_detection.utils import ops
from object_detection.utils import shape_utils

def _create_modified_mobilenet_config():
    if False:
        while True:
            i = 10
    conv_def_block_12 = model_utils.ConvDefs(conv_name='conv_pw_12', filters=512)
    conv_def_block_13 = model_utils.ConvDefs(conv_name='conv_pw_13', filters=256)
    return [conv_def_block_12, conv_def_block_13]

class SSDMobileNetV1FpnKerasFeatureExtractor(ssd_meta_arch.SSDKerasFeatureExtractor):
    """SSD Feature Extractor using Keras-based MobilenetV1 FPN features."""

    def __init__(self, is_training, depth_multiplier, min_depth, pad_to_multiple, conv_hyperparams, freeze_batchnorm, inplace_batchnorm_update, fpn_min_level=3, fpn_max_level=7, additional_layer_depth=256, use_explicit_padding=False, use_depthwise=False, use_native_resize_op=False, override_base_feature_extractor_hyperparams=False, name=None):
        if False:
            for i in range(10):
                print('nop')
        "SSD Keras based FPN feature extractor Mobilenet v1 architecture.\n\n    Args:\n      is_training: whether the network is in training mode.\n      depth_multiplier: float depth multiplier for feature extractor.\n      min_depth: minimum feature extractor depth.\n      pad_to_multiple: the nearest multiple to zero pad the input height and\n        width dimensions to.\n      conv_hyperparams: a `hyperparams_builder.KerasLayerHyperparams` object\n        containing convolution hyperparameters for the layers added on top of\n        the base feature extractor.\n      freeze_batchnorm: whether to freeze batch norm parameters during\n        training or not. When training with a small batch size (e.g. 1), it is\n        desirable to freeze batch norm update and use pretrained batch norm\n        params.\n      inplace_batchnorm_update: whether to update batch norm moving average\n        values inplace. When this is false train op must add a control\n        dependency on tf.graphkeys.UPDATE_OPS collection in order to update\n        batch norm statistics.\n      fpn_min_level: the highest resolution feature map to use in FPN. The valid\n        values are {2, 3, 4, 5} which map to MobileNet v1 layers\n        {Conv2d_3_pointwise, Conv2d_5_pointwise, Conv2d_11_pointwise,\n        Conv2d_13_pointwise}, respectively.\n      fpn_max_level: the smallest resolution feature map to construct or use in\n        FPN. FPN constructions uses features maps starting from fpn_min_level\n        upto the fpn_max_level. In the case that there are not enough feature\n        maps in the backbone network, additional feature maps are created by\n        applying stride 2 convolutions until we get the desired number of fpn\n        levels.\n      additional_layer_depth: additional feature map layer channel depth.\n      use_explicit_padding: Whether to use explicit padding when extracting\n        features. Default is False.\n      use_depthwise: whether to use depthwise convolutions. Default is False.\n      use_native_resize_op: Whether to use tf.image.nearest_neighbor_resize\n        to do upsampling in FPN. Default is false.\n      override_base_feature_extractor_hyperparams: Whether to override\n        hyperparameters of the base feature extractor with the one from\n        `conv_hyperparams`.\n      name: a string name scope to assign to the model. If 'None', Keras\n        will auto-generate one from the class name.\n    "
        super(SSDMobileNetV1FpnKerasFeatureExtractor, self).__init__(is_training=is_training, depth_multiplier=depth_multiplier, min_depth=min_depth, pad_to_multiple=pad_to_multiple, conv_hyperparams=conv_hyperparams, freeze_batchnorm=freeze_batchnorm, inplace_batchnorm_update=inplace_batchnorm_update, use_explicit_padding=use_explicit_padding, use_depthwise=use_depthwise, override_base_feature_extractor_hyperparams=override_base_feature_extractor_hyperparams, name=name)
        self._fpn_min_level = fpn_min_level
        self._fpn_max_level = fpn_max_level
        self._additional_layer_depth = additional_layer_depth
        self._conv_defs = None
        if self._use_depthwise:
            self._conv_defs = _create_modified_mobilenet_config()
        self._use_native_resize_op = use_native_resize_op
        self._feature_blocks = ['Conv2d_3_pointwise', 'Conv2d_5_pointwise', 'Conv2d_11_pointwise', 'Conv2d_13_pointwise']
        self._mobilenet_v1 = None
        self._fpn_features_generator = None
        self._coarse_feature_layers = []

    def build(self, input_shape):
        if False:
            i = 10
            return i + 15
        full_mobilenet_v1 = mobilenet_v1.mobilenet_v1(batchnorm_training=self._is_training and (not self._freeze_batchnorm), conv_hyperparams=self._conv_hyperparams if self._override_base_feature_extractor_hyperparams else None, weights=None, use_explicit_padding=self._use_explicit_padding, alpha=self._depth_multiplier, min_depth=self._min_depth, conv_defs=self._conv_defs, include_top=False)
        conv2d_3_pointwise = full_mobilenet_v1.get_layer(name='conv_pw_3_relu').output
        conv2d_5_pointwise = full_mobilenet_v1.get_layer(name='conv_pw_5_relu').output
        conv2d_11_pointwise = full_mobilenet_v1.get_layer(name='conv_pw_11_relu').output
        conv2d_13_pointwise = full_mobilenet_v1.get_layer(name='conv_pw_13_relu').output
        self._mobilenet_v1 = tf.keras.Model(inputs=full_mobilenet_v1.inputs, outputs=[conv2d_3_pointwise, conv2d_5_pointwise, conv2d_11_pointwise, conv2d_13_pointwise])
        self._depth_fn = lambda d: max(int(d * self._depth_multiplier), self._min_depth)
        self._base_fpn_max_level = min(self._fpn_max_level, 5)
        self._num_levels = self._base_fpn_max_level + 1 - self._fpn_min_level
        self._fpn_features_generator = feature_map_generators.KerasFpnTopDownFeatureMaps(num_levels=self._num_levels, depth=self._depth_fn(self._additional_layer_depth), use_depthwise=self._use_depthwise, use_explicit_padding=self._use_explicit_padding, use_native_resize_op=self._use_native_resize_op, is_training=self._is_training, conv_hyperparams=self._conv_hyperparams, freeze_batchnorm=self._freeze_batchnorm, name='FeatureMaps')
        padding = 'VALID' if self._use_explicit_padding else 'SAME'
        kernel_size = 3
        stride = 2
        for i in range(self._base_fpn_max_level + 1, self._fpn_max_level + 1):
            coarse_feature_layers = []
            if self._use_explicit_padding:

                def fixed_padding(features, kernel_size=kernel_size):
                    if False:
                        return 10
                    return ops.fixed_padding(features, kernel_size)
                coarse_feature_layers.append(tf.keras.layers.Lambda(fixed_padding, name='fixed_padding'))
            layer_name = 'bottom_up_Conv2d_{}'.format(i - self._base_fpn_max_level + 13)
            conv_block = feature_map_generators.create_conv_block(self._use_depthwise, kernel_size, padding, stride, layer_name, self._conv_hyperparams, self._is_training, self._freeze_batchnorm, self._depth_fn(self._additional_layer_depth))
            coarse_feature_layers.extend(conv_block)
            self._coarse_feature_layers.append(coarse_feature_layers)
        self.built = True

    def preprocess(self, resized_inputs):
        if False:
            return 10
        'SSD preprocessing.\n\n    Maps pixel values to the range [-1, 1].\n\n    Args:\n      resized_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n\n    Returns:\n      preprocessed_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n    '
        return 2.0 / 255.0 * resized_inputs - 1.0

    def _extract_features(self, preprocessed_inputs):
        if False:
            for i in range(10):
                print('nop')
        'Extract features from preprocessed inputs.\n\n    Args:\n      preprocessed_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n\n    Returns:\n      feature_maps: a list of tensors where the ith tensor has shape\n        [batch, height_i, width_i, depth_i]\n    '
        preprocessed_inputs = shape_utils.check_min_image_dim(33, preprocessed_inputs)
        image_features = self._mobilenet_v1(ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple))
        feature_block_list = []
        for level in range(self._fpn_min_level, self._base_fpn_max_level + 1):
            feature_block_list.append(self._feature_blocks[level - 2])
        feature_start_index = len(self._feature_blocks) - self._num_levels
        fpn_input_image_features = [(key, image_features[feature_start_index + index]) for (index, key) in enumerate(feature_block_list)]
        fpn_features = self._fpn_features_generator(fpn_input_image_features)
        feature_maps = []
        for level in range(self._fpn_min_level, self._base_fpn_max_level + 1):
            feature_maps.append(fpn_features['top_down_{}'.format(self._feature_blocks[level - 2])])
        last_feature_map = fpn_features['top_down_{}'.format(self._feature_blocks[self._base_fpn_max_level - 2])]
        for coarse_feature_layers in self._coarse_feature_layers:
            for layer in coarse_feature_layers:
                last_feature_map = layer(last_feature_map)
            feature_maps.append(last_feature_map)
        return feature_maps