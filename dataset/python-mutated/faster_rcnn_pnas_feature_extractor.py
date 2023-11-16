"""PNASNet Faster R-CNN implementation.

Based on PNASNet model: https://arxiv.org/abs/1712.00559
"""
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import slim as contrib_slim
from object_detection.meta_architectures import faster_rcnn_meta_arch
from object_detection.utils import variables_helper
from nets.nasnet import nasnet_utils
from nets.nasnet import pnasnet
arg_scope = contrib_framework.arg_scope
slim = contrib_slim

def pnasnet_large_arg_scope_for_detection(is_batch_norm_training=False):
    if False:
        while True:
            i = 10
    'Defines the default arg scope for the PNASNet Large for object detection.\n\n  This provides a small edit to switch batch norm training on and off.\n\n  Args:\n    is_batch_norm_training: Boolean indicating whether to train with batch norm.\n\n  Returns:\n    An `arg_scope` to use for the PNASNet Large Model.\n  '
    imagenet_scope = pnasnet.pnasnet_large_arg_scope()
    with arg_scope(imagenet_scope):
        with arg_scope([slim.batch_norm], is_training=is_batch_norm_training) as sc:
            return sc

def _filter_scaling(reduction_indices, start_cell_num):
    if False:
        print('Hello World!')
    'Compute the expected filter scaling at given PNASNet cell start_cell_num.\n\n  In the pnasnet.py code, filter_scaling starts at 1.0. We instead\n  adapt filter scaling to depend on the starting cell.\n  At first cells, before any reduction, filter_scalling is 1.0. With passing\n  any reduction cell, the filter_scaling is multiplied by 2.\n\n  Args:\n    reduction_indices: list of int indices.\n    start_cell_num: int.\n  Returns:\n    filter_scaling: float.\n  '
    filter_scaling = 1.0
    for ind in reduction_indices:
        if ind < start_cell_num:
            filter_scaling *= 2.0
    return filter_scaling

def _build_pnasnet_base(hidden_previous, hidden, normal_cell, hparams, true_cell_num, start_cell_num):
    if False:
        while True:
            i = 10
    'Constructs a PNASNet image model for proposal classifier features.'
    reduction_indices = nasnet_utils.calc_reduction_layers(hparams.num_cells, hparams.num_reduction_layers)
    filter_scaling = _filter_scaling(reduction_indices, start_cell_num)
    cell_outputs = [None, hidden_previous, hidden]
    net = hidden
    for cell_num in range(start_cell_num, hparams.num_cells):
        is_reduction = cell_num in reduction_indices
        stride = 2 if is_reduction else 1
        if is_reduction:
            filter_scaling *= hparams.filter_scaling_rate
        prev_layer = cell_outputs[-2]
        net = normal_cell(net, scope='cell_{}'.format(cell_num), filter_scaling=filter_scaling, stride=stride, prev_layer=prev_layer, cell_num=true_cell_num)
        true_cell_num += 1
        cell_outputs.append(net)
    with tf.variable_scope('final_layer'):
        net = tf.nn.relu(net)
    return net

class FasterRCNNPNASFeatureExtractor(faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
    """Faster R-CNN with PNASNet feature extractor implementation."""

    def __init__(self, is_training, first_stage_features_stride, batch_norm_trainable=False, reuse_weights=None, weight_decay=0.0):
        if False:
            while True:
                i = 10
        'Constructor.\n\n    Args:\n      is_training: See base class.\n      first_stage_features_stride: See base class.\n      batch_norm_trainable: See base class.\n      reuse_weights: See base class.\n      weight_decay: See base class.\n\n    Raises:\n      ValueError: If `first_stage_features_stride` is not 16.\n    '
        if first_stage_features_stride != 16:
            raise ValueError('`first_stage_features_stride` must be 16.')
        super(FasterRCNNPNASFeatureExtractor, self).__init__(is_training, first_stage_features_stride, batch_norm_trainable, reuse_weights, weight_decay)

    def preprocess(self, resized_inputs):
        if False:
            while True:
                i = 10
        'Faster R-CNN with PNAS preprocessing.\n\n    Maps pixel values to the range [-1, 1].\n\n    Args:\n      resized_inputs: A [batch, height_in, width_in, channels] float32 tensor\n        representing a batch of images with values between 0 and 255.0.\n\n    Returns:\n      preprocessed_inputs: A [batch, height_out, width_out, channels] float32\n        tensor representing a batch of images.\n\n    '
        return 2.0 / 255.0 * resized_inputs - 1.0

    def _extract_proposal_features(self, preprocessed_inputs, scope):
        if False:
            print('Hello World!')
        'Extracts first stage RPN features.\n\n    Extracts features using the first half of the PNASNet network.\n    We construct the network in `align_feature_maps=True` mode, which means\n    that all VALID paddings in the network are changed to SAME padding so that\n    the feature maps are aligned.\n\n    Args:\n      preprocessed_inputs: A [batch, height, width, channels] float32 tensor\n        representing a batch of images.\n      scope: A scope name.\n\n    Returns:\n      rpn_feature_map: A tensor with shape [batch, height, width, depth]\n      end_points: A dictionary mapping feature extractor tensor names to tensors\n\n    Raises:\n      ValueError: If the created network is missing the required activation.\n    '
        del scope
        if len(preprocessed_inputs.get_shape().as_list()) != 4:
            raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a tensor of shape %s' % preprocessed_inputs.get_shape())
        with slim.arg_scope(pnasnet_large_arg_scope_for_detection(is_batch_norm_training=self._train_batch_norm)):
            with arg_scope([slim.conv2d, slim.batch_norm, slim.separable_conv2d], reuse=self._reuse_weights):
                (_, end_points) = pnasnet.build_pnasnet_large(preprocessed_inputs, num_classes=None, is_training=self._is_training, final_endpoint='Cell_7')
        rpn_feature_map = tf.concat([end_points['Cell_6'], end_points['Cell_7']], 3)
        batch = preprocessed_inputs.get_shape().as_list()[0]
        shape_without_batch = rpn_feature_map.get_shape().as_list()[1:]
        rpn_feature_map_shape = [batch] + shape_without_batch
        rpn_feature_map.set_shape(rpn_feature_map_shape)
        return (rpn_feature_map, end_points)

    def _extract_box_classifier_features(self, proposal_feature_maps, scope):
        if False:
            return 10
        'Extracts second stage box classifier features.\n\n    This function reconstructs the "second half" of the PNASNet\n    network after the part defined in `_extract_proposal_features`.\n\n    Args:\n      proposal_feature_maps: A 4-D float tensor with shape\n        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]\n        representing the feature map cropped to each proposal.\n      scope: A scope name.\n\n    Returns:\n      proposal_classifier_features: A 4-D float tensor with shape\n        [batch_size * self.max_num_proposals, height, width, depth]\n        representing box classifier features for each proposal.\n    '
        del scope
        num_stem_cells = 2
        (hidden_previous, hidden) = tf.split(proposal_feature_maps, 2, axis=3)
        hparams = pnasnet.large_imagenet_config()
        if not self._is_training:
            hparams.set_hparam('drop_path_keep_prob', 1.0)
        total_num_cells = hparams.num_cells + num_stem_cells
        normal_cell = pnasnet.PNasNetNormalCell(hparams.num_conv_filters, hparams.drop_path_keep_prob, total_num_cells, hparams.total_training_steps)
        with arg_scope([slim.dropout, nasnet_utils.drop_path], is_training=self._is_training):
            with arg_scope([slim.batch_norm], is_training=self._train_batch_norm):
                with arg_scope([slim.avg_pool2d, slim.max_pool2d, slim.conv2d, slim.batch_norm, slim.separable_conv2d, nasnet_utils.factorized_reduction, nasnet_utils.global_avg_pool, nasnet_utils.get_channel_index, nasnet_utils.get_channel_dim], data_format=hparams.data_format):
                    start_cell_num = 8
                    true_cell_num = start_cell_num + num_stem_cells
                    with slim.arg_scope(pnasnet.pnasnet_large_arg_scope()):
                        net = _build_pnasnet_base(hidden_previous, hidden, normal_cell=normal_cell, hparams=hparams, true_cell_num=true_cell_num, start_cell_num=start_cell_num)
        proposal_classifier_features = net
        return proposal_classifier_features

    def restore_from_classification_checkpoint_fn(self, first_stage_feature_extractor_scope, second_stage_feature_extractor_scope):
        if False:
            for i in range(10):
                print('nop')
        'Returns a map of variables to load from a foreign checkpoint.\n\n    Note that this overrides the default implementation in\n    faster_rcnn_meta_arch.FasterRCNNFeatureExtractor which does not work for\n    PNASNet checkpoints.\n\n    Args:\n      first_stage_feature_extractor_scope: A scope name for the first stage\n        feature extractor.\n      second_stage_feature_extractor_scope: A scope name for the second stage\n        feature extractor.\n\n    Returns:\n      A dict mapping variable names (to load from a checkpoint) to variables in\n      the model graph.\n    '
        variables_to_restore = {}
        for variable in variables_helper.get_global_variables_safely():
            if variable.op.name.startswith(first_stage_feature_extractor_scope):
                var_name = variable.op.name.replace(first_stage_feature_extractor_scope + '/', '')
                var_name += '/ExponentialMovingAverage'
                variables_to_restore[var_name] = variable
            if variable.op.name.startswith(second_stage_feature_extractor_scope):
                var_name = variable.op.name.replace(second_stage_feature_extractor_scope + '/', '')
                var_name += '/ExponentialMovingAverage'
                variables_to_restore[var_name] = variable
        return variables_to_restore