"""Inception Resnet v2 Faster R-CNN implementation.

See "Inception-v4, Inception-ResNet and the Impact of Residual Connections on
Learning" by Szegedy et al. (https://arxiv.org/abs/1602.07261)
as well as
"Speed/accuracy trade-offs for modern convolutional object detectors" by
Huang et al. (https://arxiv.org/abs/1611.10012)
"""
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.utils import variables_helper
from nets import inception_resnet_v2
slim = contrib_slim

class FasterRCNNInceptionResnetV2FeatureExtractor(faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
    """Faster R-CNN with Inception Resnet v2 feature extractor implementation."""

    def __init__(self, is_training, first_stage_features_stride, batch_norm_trainable=False, reuse_weights=None, weight_decay=0.0):
        if False:
            i = 10
            return i + 15
        'Constructor.\n\n    Args:\n      is_training: See base class.\n      first_stage_features_stride: See base class.\n      batch_norm_trainable: See base class.\n      reuse_weights: See base class.\n      weight_decay: See base class.\n\n    Raises:\n      ValueError: If `first_stage_features_stride` is not 8 or 16.\n    '
        if first_stage_features_stride != 8 and first_stage_features_stride != 16:
            raise ValueError('`first_stage_features_stride` must be 8 or 16.')
        super(FasterRCNNInceptionResnetV2FeatureExtractor, self).__init__(is_training, first_stage_features_stride, batch_norm_trainable, reuse_weights, weight_decay)

    def preprocess(self, resized_inputs):
        if False:
            print('Hello World!')
        'Faster R-CNN with Inception Resnet v2 preprocessing.\n\n    Maps pixel values to the range [-1, 1].\n\n    Args:\n      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor\n        representing a batch of images with values between 0 and 255.0.\n\n    Returns:\n      preprocessed_inputs: A [batch, height_out, width_out, channels] float32\n        tensor representing a batch of images.\n\n    '
        return 2.0 / 255.0 * resized_inputs - 1.0

    def _extract_proposal_features(self, preprocessed_inputs, scope):
        if False:
            for i in range(10):
                print('nop')
        'Extracts first stage RPN features.\n\n    Extracts features using the first half of the Inception Resnet v2 network.\n    We construct the network in `align_feature_maps=True` mode, which means\n    that all VALID paddings in the network are changed to SAME padding so that\n    the feature maps are aligned.\n\n    Args:\n      preprocessed_inputs: A [batch, height, width, channels] float32 tensor\n        representing a batch of images.\n      scope: A scope name.\n\n    Returns:\n      rpn_feature_map: A tensor with shape [batch, height, width, depth]\n    Raises:\n      InvalidArgumentError: If the spatial size of `preprocessed_inputs`\n        (height or width) is less than 33.\n      ValueError: If the created network is missing the required activation.\n    '
        if len(preprocessed_inputs.get_shape().as_list()) != 4:
            raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a tensor of shape %s' % preprocessed_inputs.get_shape())
        with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope(weight_decay=self._weight_decay)):
            with slim.arg_scope([slim.batch_norm], is_training=self._train_batch_norm):
                with tf.variable_scope('InceptionResnetV2', reuse=self._reuse_weights) as scope:
                    return inception_resnet_v2.inception_resnet_v2_base(preprocessed_inputs, final_endpoint='PreAuxLogits', scope=scope, output_stride=self._first_stage_features_stride, align_feature_maps=True)

    def _extract_box_classifier_features(self, proposal_feature_maps, scope):
        if False:
            print('Hello World!')
        'Extracts second stage box classifier features.\n\n    This function reconstructs the "second half" of the Inception ResNet v2\n    network after the part defined in `_extract_proposal_features`.\n\n    Args:\n      proposal_feature_maps: A 4-D float tensor with shape\n        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]\n        representing the feature map cropped to each proposal.\n      scope: A scope name.\n\n    Returns:\n      proposal_classifier_features: A 4-D float tensor with shape\n        [batch_size * self.max_num_proposals, height, width, depth]\n        representing box classifier features for each proposal.\n    '
        with tf.variable_scope('InceptionResnetV2', reuse=self._reuse_weights):
            with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope(weight_decay=self._weight_decay)):
                with slim.arg_scope([slim.batch_norm], is_training=self._train_batch_norm):
                    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
                        with tf.variable_scope('Mixed_7a'):
                            with tf.variable_scope('Branch_0'):
                                tower_conv = slim.conv2d(proposal_feature_maps, 256, 1, scope='Conv2d_0a_1x1')
                                tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                            with tf.variable_scope('Branch_1'):
                                tower_conv1 = slim.conv2d(proposal_feature_maps, 256, 1, scope='Conv2d_0a_1x1')
                                tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                            with tf.variable_scope('Branch_2'):
                                tower_conv2 = slim.conv2d(proposal_feature_maps, 256, 1, scope='Conv2d_0a_1x1')
                                tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3, scope='Conv2d_0b_3x3')
                                tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3')
                            with tf.variable_scope('Branch_3'):
                                tower_pool = slim.max_pool2d(proposal_feature_maps, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
                            net = tf.concat([tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)
                        net = slim.repeat(net, 9, inception_resnet_v2.block8, scale=0.2)
                        net = inception_resnet_v2.block8(net, activation_fn=None)
                        proposal_classifier_features = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
                return proposal_classifier_features

    def restore_from_classification_checkpoint_fn(self, first_stage_feature_extractor_scope, second_stage_feature_extractor_scope):
        if False:
            print('Hello World!')
        "Returns a map of variables to load from a foreign checkpoint.\n\n    Note that this overrides the default implementation in\n    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor which does not work for\n    InceptionResnetV2 checkpoints.\n\n    TODO(jonathanhuang,rathodv): revisit whether it's possible to force the\n    `Repeat` namescope as created in `_extract_box_classifier_features` to\n    start counting at 2 (e.g. `Repeat_2`) so that the default restore_fn can\n    be used.\n\n    Args:\n      first_stage_feature_extractor_scope: A scope name for the first stage\n        feature extractor.\n      second_stage_feature_extractor_scope: A scope name for the second stage\n        feature extractor.\n\n    Returns:\n      A dict mapping variable names (to load from a checkpoint) to variables in\n      the model graph.\n    "
        variables_to_restore = {}
        for variable in variables_helper.get_global_variables_safely():
            if variable.op.name.startswith(first_stage_feature_extractor_scope):
                var_name = variable.op.name.replace(first_stage_feature_extractor_scope + '/', '')
                variables_to_restore[var_name] = variable
            if variable.op.name.startswith(second_stage_feature_extractor_scope):
                var_name = variable.op.name.replace(second_stage_feature_extractor_scope + '/InceptionResnetV2/Repeat', 'InceptionResnetV2/Repeat_2')
                var_name = var_name.replace(second_stage_feature_extractor_scope + '/', '')
                variables_to_restore[var_name] = variable
        return variables_to_restore