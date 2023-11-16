"""Mobilenet v1 Faster R-CNN implementation."""
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.utils import shape_utils
from nets import mobilenet_v1
slim = contrib_slim

def _get_mobilenet_conv_no_last_stride_defs(conv_depth_ratio_in_percentage):
    if False:
        while True:
            i = 10
    if conv_depth_ratio_in_percentage not in [25, 50, 75, 100]:
        raise ValueError('Only the following ratio percentages are supported: 25, 50, 75, 100')
    conv_depth_ratio_in_percentage = float(conv_depth_ratio_in_percentage) / 100.0
    channels = np.array([32, 64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024], dtype=np.float32)
    channels = (channels * conv_depth_ratio_in_percentage).astype(np.int32)
    return [mobilenet_v1.Conv(kernel=[3, 3], stride=2, depth=channels[0]), mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[1]), mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=2, depth=channels[2]), mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[3]), mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=2, depth=channels[4]), mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[5]), mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=2, depth=channels[6]), mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[7]), mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[8]), mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[9]), mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[10]), mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[11]), mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[12]), mobilenet_v1.DepthSepConv(kernel=[3, 3], stride=1, depth=channels[13])]

class FasterRCNNMobilenetV1FeatureExtractor(faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
    """Faster R-CNN Mobilenet V1 feature extractor implementation."""

    def __init__(self, is_training, first_stage_features_stride, batch_norm_trainable=False, reuse_weights=None, weight_decay=0.0, depth_multiplier=1.0, min_depth=16, skip_last_stride=False, conv_depth_ratio_in_percentage=100):
        if False:
            return 10
        'Constructor.\n\n    Args:\n      is_training: See base class.\n      first_stage_features_stride: See base class.\n      batch_norm_trainable: See base class.\n      reuse_weights: See base class.\n      weight_decay: See base class.\n      depth_multiplier: float depth multiplier for feature extractor.\n      min_depth: minimum feature extractor depth.\n      skip_last_stride: Skip the last stride if True.\n      conv_depth_ratio_in_percentage: Conv depth ratio in percentage. Only\n        applied if skip_last_stride is True.\n\n    Raises:\n      ValueError: If `first_stage_features_stride` is not 8 or 16.\n    '
        if first_stage_features_stride != 8 and first_stage_features_stride != 16:
            raise ValueError('`first_stage_features_stride` must be 8 or 16.')
        self._depth_multiplier = depth_multiplier
        self._min_depth = min_depth
        self._skip_last_stride = skip_last_stride
        self._conv_depth_ratio_in_percentage = conv_depth_ratio_in_percentage
        super(FasterRCNNMobilenetV1FeatureExtractor, self).__init__(is_training, first_stage_features_stride, batch_norm_trainable, reuse_weights, weight_decay)

    def preprocess(self, resized_inputs):
        if False:
            i = 10
            return i + 15
        'Faster R-CNN Mobilenet V1 preprocessing.\n\n    Maps pixel values to the range [-1, 1].\n\n    Args:\n      resized_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n\n    Returns:\n      preprocessed_inputs: a [batch, height, width, channels] float tensor\n        representing a batch of images.\n    '
        return 2.0 / 255.0 * resized_inputs - 1.0

    def _extract_proposal_features(self, preprocessed_inputs, scope):
        if False:
            for i in range(10):
                print('nop')
        'Extracts first stage RPN features.\n\n    Args:\n      preprocessed_inputs: A [batch, height, width, channels] float32 tensor\n        representing a batch of images.\n      scope: A scope name.\n\n    Returns:\n      rpn_feature_map: A tensor with shape [batch, height, width, depth]\n      activations: A dictionary mapping feature extractor tensor names to\n        tensors\n\n    Raises:\n      InvalidArgumentError: If the spatial size of `preprocessed_inputs`\n        (height or width) is less than 33.\n      ValueError: If the created network is missing the required activation.\n    '
        preprocessed_inputs.get_shape().assert_has_rank(4)
        preprocessed_inputs = shape_utils.check_min_image_dim(min_dim=33, image_tensor=preprocessed_inputs)
        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=self._train_batch_norm, weight_decay=self._weight_decay)):
            with tf.variable_scope('MobilenetV1', reuse=self._reuse_weights) as scope:
                params = {}
                if self._skip_last_stride:
                    params['conv_defs'] = _get_mobilenet_conv_no_last_stride_defs(conv_depth_ratio_in_percentage=self._conv_depth_ratio_in_percentage)
                (_, activations) = mobilenet_v1.mobilenet_v1_base(preprocessed_inputs, final_endpoint='Conv2d_11_pointwise', min_depth=self._min_depth, depth_multiplier=self._depth_multiplier, scope=scope, **params)
        return (activations['Conv2d_11_pointwise'], activations)

    def _extract_box_classifier_features(self, proposal_feature_maps, scope):
        if False:
            while True:
                i = 10
        'Extracts second stage box classifier features.\n\n    Args:\n      proposal_feature_maps: A 4-D float tensor with shape\n        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]\n        representing the feature map cropped to each proposal.\n      scope: A scope name (unused).\n\n    Returns:\n      proposal_classifier_features: A 4-D float tensor with shape\n        [batch_size * self.max_num_proposals, height, width, depth]\n        representing box classifier features for each proposal.\n    '
        net = proposal_feature_maps
        conv_depth = 1024
        if self._skip_last_stride:
            conv_depth_ratio = float(self._conv_depth_ratio_in_percentage) / 100.0
            conv_depth = int(float(conv_depth) * conv_depth_ratio)
        depth = lambda d: max(int(d * 1.0), 16)
        with tf.variable_scope('MobilenetV1', reuse=self._reuse_weights):
            with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=self._train_batch_norm, weight_decay=self._weight_decay)):
                with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding='SAME'):
                    net = slim.separable_conv2d(net, depth(conv_depth), [3, 3], depth_multiplier=1, stride=2, scope='Conv2d_12_pointwise')
                    return slim.separable_conv2d(net, depth(conv_depth), [3, 3], depth_multiplier=1, stride=1, scope='Conv2d_13_pointwise')