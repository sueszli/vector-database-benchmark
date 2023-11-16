"""Keypoint Head.

Contains Keypoint prediction head classes for different meta architectures.
All the keypoint prediction heads have a predict function that receives the
`features` as the first argument and returns `keypoint_predictions`.
Keypoints could be used to represent the human body joint locations as in
Mask RCNN paper. Or they could be used to represent different part locations of
objects.
"""
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from object_detection.predictors.heads import head
slim = contrib_slim

class MaskRCNNKeypointHead(head.Head):
    """Mask RCNN keypoint prediction head.

  Please refer to Mask RCNN paper:
  https://arxiv.org/abs/1703.06870
  """

    def __init__(self, num_keypoints=17, conv_hyperparams_fn=None, keypoint_heatmap_height=56, keypoint_heatmap_width=56, keypoint_prediction_num_conv_layers=8, keypoint_prediction_conv_depth=512):
        if False:
            i = 10
            return i + 15
        'Constructor.\n\n    Args:\n      num_keypoints: (int scalar) number of keypoints.\n      conv_hyperparams_fn: A function to generate tf-slim arg_scope with\n        hyperparameters for convolution ops.\n      keypoint_heatmap_height: Desired output mask height. The default value\n        is 14.\n      keypoint_heatmap_width: Desired output mask width. The default value\n        is 14.\n      keypoint_prediction_num_conv_layers: Number of convolution layers applied\n        to the image_features in mask prediction branch.\n      keypoint_prediction_conv_depth: The depth for the first conv2d_transpose\n        op applied to the image_features in the mask prediction branch. If set\n        to 0, the depth of the convolution layers will be automatically chosen\n        based on the number of object classes and the number of channels in the\n        image features.\n    '
        super(MaskRCNNKeypointHead, self).__init__()
        self._num_keypoints = num_keypoints
        self._conv_hyperparams_fn = conv_hyperparams_fn
        self._keypoint_heatmap_height = keypoint_heatmap_height
        self._keypoint_heatmap_width = keypoint_heatmap_width
        self._keypoint_prediction_num_conv_layers = keypoint_prediction_num_conv_layers
        self._keypoint_prediction_conv_depth = keypoint_prediction_conv_depth

    def predict(self, features, num_predictions_per_location=1):
        if False:
            i = 10
            return i + 15
        'Performs keypoint prediction.\n\n    Args:\n      features: A float tensor of shape [batch_size, height, width,\n        channels] containing features for a batch of images.\n      num_predictions_per_location: Int containing number of predictions per\n        location.\n\n    Returns:\n      instance_masks: A float tensor of shape\n          [batch_size, 1, num_keypoints, heatmap_height, heatmap_width].\n\n    Raises:\n      ValueError: If num_predictions_per_location is not 1.\n    '
        if num_predictions_per_location != 1:
            raise ValueError('Only num_predictions_per_location=1 is supported')
        with slim.arg_scope(self._conv_hyperparams_fn()):
            net = slim.conv2d(features, self._keypoint_prediction_conv_depth, [3, 3], scope='conv_1')
            for i in range(1, self._keypoint_prediction_num_conv_layers):
                net = slim.conv2d(net, self._keypoint_prediction_conv_depth, [3, 3], scope='conv_%d' % (i + 1))
            net = slim.conv2d_transpose(net, self._num_keypoints, [2, 2], scope='deconv1')
            heatmaps_mask = tf.image.resize_bilinear(net, [self._keypoint_heatmap_height, self._keypoint_heatmap_width], align_corners=True, name='upsample')
            return tf.expand_dims(tf.transpose(heatmaps_mask, perm=[0, 3, 1, 2]), axis=1, name='KeypointPredictor')