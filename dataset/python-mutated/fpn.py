"""Feature Pyramid Networks.

Feature Pyramid Networks were proposed in:
[1] Tsung-Yi Lin, Piotr Dollar, Ross Girshick, Kaiming He, Bharath Hariharan,
    , and Serge Belongie
    Feature Pyramid Networks for Object Detection. CVPR 2017.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v2 as tf
from tensorflow.python.keras import backend
from official.vision.detection.modeling.architecture import nn_ops
from official.vision.detection.ops import spatial_transform_ops

class Fpn(object):
    """Feature pyramid networks."""

    def __init__(self, min_level=3, max_level=7, fpn_feat_dims=256, use_separable_conv=False, batch_norm_relu=nn_ops.BatchNormRelu):
        if False:
            for i in range(10):
                print('nop')
        'FPN initialization function.\n\n    Args:\n      min_level: `int` minimum level in FPN output feature maps.\n      max_level: `int` maximum level in FPN output feature maps.\n      fpn_feat_dims: `int` number of filters in FPN layers.\n      use_separable_conv: `bool`, if True use separable convolution for\n        convolution in FPN layers.\n      batch_norm_relu: an operation that includes a batch normalization layer\n        followed by a relu layer(optional).\n    '
        self._min_level = min_level
        self._max_level = max_level
        self._fpn_feat_dims = fpn_feat_dims
        self._batch_norm_relu = batch_norm_relu
        self._batch_norm_relus = {}
        self._lateral_conv2d_op = {}
        self._post_hoc_conv2d_op = {}
        self._coarse_conv2d_op = {}
        for level in range(self._min_level, self._max_level + 1):
            self._batch_norm_relus[level] = batch_norm_relu(relu=False, name='p%d-bn' % level)
            if use_separable_conv:
                self._lateral_conv2d_op[level] = tf.keras.layers.SeparableConv2D(filters=self._fpn_feat_dims, kernel_size=(1, 1), padding='same', depth_multiplier=1, name='l%d' % level)
                self._post_hoc_conv2d_op[level] = tf.keras.layers.SeparableConv2D(filters=self._fpn_feat_dims, strides=(1, 1), kernel_size=(3, 3), padding='same', depth_multiplier=1, name='post_hoc_d%d' % level)
                self._coarse_conv2d_op[level] = tf.keras.layers.SeparableConv2D(filters=self._fpn_feat_dims, strides=(2, 2), kernel_size=(3, 3), padding='same', depth_multiplier=1, name='p%d' % level)
            else:
                self._lateral_conv2d_op[level] = tf.keras.layers.Conv2D(filters=self._fpn_feat_dims, kernel_size=(1, 1), padding='same', name='l%d' % level)
                self._post_hoc_conv2d_op[level] = tf.keras.layers.Conv2D(filters=self._fpn_feat_dims, strides=(1, 1), kernel_size=(3, 3), padding='same', name='post_hoc_d%d' % level)
                self._coarse_conv2d_op[level] = tf.keras.layers.Conv2D(filters=self._fpn_feat_dims, strides=(2, 2), kernel_size=(3, 3), padding='same', name='p%d' % level)

    def __call__(self, multilevel_features, is_training=None):
        if False:
            print('Hello World!')
        'Returns the FPN features for a given multilevel features.\n\n    Args:\n      multilevel_features: a `dict` containing `int` keys for continuous feature\n        levels, e.g., [2, 3, 4, 5]. The values are corresponding features with\n        shape [batch_size, height_l, width_l, num_filters].\n      is_training: `bool` if True, the model is in training mode.\n\n    Returns:\n      a `dict` containing `int` keys for continuous feature levels\n      [min_level, min_level + 1, ..., max_level]. The values are corresponding\n      FPN features with shape [batch_size, height_l, width_l, fpn_feat_dims].\n    '
        input_levels = multilevel_features.keys()
        if min(input_levels) > self._min_level:
            raise ValueError('The minimum backbone level %d should be ' % min(input_levels) + 'less or equal to FPN minimum level %d.:' % self._min_level)
        backbone_max_level = min(max(input_levels), self._max_level)
        with backend.get_graph().as_default(), tf.name_scope('fpn'):
            feats_lateral = {}
            for level in range(self._min_level, backbone_max_level + 1):
                feats_lateral[level] = self._lateral_conv2d_op[level](multilevel_features[level])
            feats = {backbone_max_level: feats_lateral[backbone_max_level]}
            for level in range(backbone_max_level - 1, self._min_level - 1, -1):
                feats[level] = spatial_transform_ops.nearest_upsampling(feats[level + 1], 2) + feats_lateral[level]
            for level in range(self._min_level, backbone_max_level + 1):
                feats[level] = self._post_hoc_conv2d_op[level](feats[level])
            for level in range(backbone_max_level + 1, self._max_level + 1):
                feats_in = feats[level - 1]
                if level > backbone_max_level + 1:
                    feats_in = tf.nn.relu(feats_in)
                feats[level] = self._coarse_conv2d_op[level](feats_in)
            for level in range(self._min_level, self._max_level + 1):
                feats[level] = self._batch_norm_relus[level](feats[level], is_training=is_training)
        return feats