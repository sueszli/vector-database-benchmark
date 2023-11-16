"""Contains the definition for the NASNet classification networks.

Paper: https://arxiv.org/abs/1707.07012
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import slim as contrib_slim
from tensorflow.contrib import training as contrib_training
from nets.nasnet import nasnet_utils
arg_scope = contrib_framework.arg_scope
slim = contrib_slim

def cifar_config():
    if False:
        return 10
    return contrib_training.HParams(stem_multiplier=3.0, drop_path_keep_prob=0.6, num_cells=18, use_aux_head=1, num_conv_filters=32, dense_dropout_keep_prob=1.0, filter_scaling_rate=2.0, num_reduction_layers=2, data_format='NHWC', skip_reduction_layer_input=0, total_training_steps=937500, use_bounded_activation=False)

def large_imagenet_config():
    if False:
        while True:
            i = 10
    return contrib_training.HParams(stem_multiplier=3.0, dense_dropout_keep_prob=0.5, num_cells=18, filter_scaling_rate=2.0, num_conv_filters=168, drop_path_keep_prob=0.7, use_aux_head=1, num_reduction_layers=2, data_format='NHWC', skip_reduction_layer_input=1, total_training_steps=250000, use_bounded_activation=False)

def mobile_imagenet_config():
    if False:
        return 10
    return contrib_training.HParams(stem_multiplier=1.0, dense_dropout_keep_prob=0.5, num_cells=12, filter_scaling_rate=2.0, drop_path_keep_prob=1.0, num_conv_filters=44, use_aux_head=1, num_reduction_layers=2, data_format='NHWC', skip_reduction_layer_input=0, total_training_steps=250000, use_bounded_activation=False)

def _update_hparams(hparams, is_training):
    if False:
        while True:
            i = 10
    'Update hparams for given is_training option.'
    if not is_training:
        hparams.set_hparam('drop_path_keep_prob', 1.0)

def nasnet_cifar_arg_scope(weight_decay=0.0005, batch_norm_decay=0.9, batch_norm_epsilon=1e-05):
    if False:
        print('Hello World!')
    'Defines the default arg scope for the NASNet-A Cifar model.\n\n  Args:\n    weight_decay: The weight decay to use for regularizing the model.\n    batch_norm_decay: Decay for batch norm moving average.\n    batch_norm_epsilon: Small float added to variance to avoid dividing by zero\n      in batch norm.\n\n  Returns:\n    An `arg_scope` to use for the NASNet Cifar Model.\n  '
    batch_norm_params = {'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon, 'scale': True, 'fused': True}
    weights_regularizer = contrib_layers.l2_regularizer(weight_decay)
    weights_initializer = contrib_layers.variance_scaling_initializer(mode='FAN_OUT')
    with arg_scope([slim.fully_connected, slim.conv2d, slim.separable_conv2d], weights_regularizer=weights_regularizer, weights_initializer=weights_initializer):
        with arg_scope([slim.fully_connected], activation_fn=None, scope='FC'):
            with arg_scope([slim.conv2d, slim.separable_conv2d], activation_fn=None, biases_initializer=None):
                with arg_scope([slim.batch_norm], **batch_norm_params) as sc:
                    return sc

def nasnet_mobile_arg_scope(weight_decay=4e-05, batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
    if False:
        return 10
    'Defines the default arg scope for the NASNet-A Mobile ImageNet model.\n\n  Args:\n    weight_decay: The weight decay to use for regularizing the model.\n    batch_norm_decay: Decay for batch norm moving average.\n    batch_norm_epsilon: Small float added to variance to avoid dividing by zero\n      in batch norm.\n\n  Returns:\n    An `arg_scope` to use for the NASNet Mobile Model.\n  '
    batch_norm_params = {'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon, 'scale': True, 'fused': True}
    weights_regularizer = contrib_layers.l2_regularizer(weight_decay)
    weights_initializer = contrib_layers.variance_scaling_initializer(mode='FAN_OUT')
    with arg_scope([slim.fully_connected, slim.conv2d, slim.separable_conv2d], weights_regularizer=weights_regularizer, weights_initializer=weights_initializer):
        with arg_scope([slim.fully_connected], activation_fn=None, scope='FC'):
            with arg_scope([slim.conv2d, slim.separable_conv2d], activation_fn=None, biases_initializer=None):
                with arg_scope([slim.batch_norm], **batch_norm_params) as sc:
                    return sc

def nasnet_large_arg_scope(weight_decay=5e-05, batch_norm_decay=0.9997, batch_norm_epsilon=0.001):
    if False:
        for i in range(10):
            print('nop')
    'Defines the default arg scope for the NASNet-A Large ImageNet model.\n\n  Args:\n    weight_decay: The weight decay to use for regularizing the model.\n    batch_norm_decay: Decay for batch norm moving average.\n    batch_norm_epsilon: Small float added to variance to avoid dividing by zero\n      in batch norm.\n\n  Returns:\n    An `arg_scope` to use for the NASNet Large Model.\n  '
    batch_norm_params = {'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon, 'scale': True, 'fused': True}
    weights_regularizer = contrib_layers.l2_regularizer(weight_decay)
    weights_initializer = contrib_layers.variance_scaling_initializer(mode='FAN_OUT')
    with arg_scope([slim.fully_connected, slim.conv2d, slim.separable_conv2d], weights_regularizer=weights_regularizer, weights_initializer=weights_initializer):
        with arg_scope([slim.fully_connected], activation_fn=None, scope='FC'):
            with arg_scope([slim.conv2d, slim.separable_conv2d], activation_fn=None, biases_initializer=None):
                with arg_scope([slim.batch_norm], **batch_norm_params) as sc:
                    return sc

def _build_aux_head(net, end_points, num_classes, hparams, scope):
    if False:
        print('Hello World!')
    'Auxiliary head used for all models across all datasets.'
    activation_fn = tf.nn.relu6 if hparams.use_bounded_activation else tf.nn.relu
    with tf.variable_scope(scope):
        aux_logits = tf.identity(net)
        with tf.variable_scope('aux_logits'):
            aux_logits = slim.avg_pool2d(aux_logits, [5, 5], stride=3, padding='VALID')
            aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='proj')
            aux_logits = slim.batch_norm(aux_logits, scope='aux_bn0')
            aux_logits = activation_fn(aux_logits)
            shape = aux_logits.shape
            if hparams.data_format == 'NHWC':
                shape = shape[1:3]
            else:
                shape = shape[2:4]
            aux_logits = slim.conv2d(aux_logits, 768, shape, padding='VALID')
            aux_logits = slim.batch_norm(aux_logits, scope='aux_bn1')
            aux_logits = activation_fn(aux_logits)
            aux_logits = contrib_layers.flatten(aux_logits)
            aux_logits = slim.fully_connected(aux_logits, num_classes)
            end_points['AuxLogits'] = aux_logits

def _imagenet_stem(inputs, hparams, stem_cell, current_step=None):
    if False:
        print('Hello World!')
    'Stem used for models trained on ImageNet.'
    num_stem_cells = 2
    num_stem_filters = int(32 * hparams.stem_multiplier)
    net = slim.conv2d(inputs, num_stem_filters, [3, 3], stride=2, scope='conv0', padding='VALID')
    net = slim.batch_norm(net, scope='conv0_bn')
    cell_outputs = [None, net]
    filter_scaling = 1.0 / hparams.filter_scaling_rate ** num_stem_cells
    for cell_num in range(num_stem_cells):
        net = stem_cell(net, scope='cell_stem_{}'.format(cell_num), filter_scaling=filter_scaling, stride=2, prev_layer=cell_outputs[-2], cell_num=cell_num, current_step=current_step)
        cell_outputs.append(net)
        filter_scaling *= hparams.filter_scaling_rate
    return (net, cell_outputs)

def _cifar_stem(inputs, hparams):
    if False:
        for i in range(10):
            print('nop')
    'Stem used for models trained on Cifar.'
    num_stem_filters = int(hparams.num_conv_filters * hparams.stem_multiplier)
    net = slim.conv2d(inputs, num_stem_filters, 3, scope='l1_stem_3x3')
    net = slim.batch_norm(net, scope='l1_stem_bn')
    return (net, [None, net])

def build_nasnet_cifar(images, num_classes, is_training=True, config=None, current_step=None):
    if False:
        return 10
    'Build NASNet model for the Cifar Dataset.'
    hparams = cifar_config() if config is None else copy.deepcopy(config)
    _update_hparams(hparams, is_training)
    if tf.test.is_gpu_available() and hparams.data_format == 'NHWC':
        tf.logging.info('A GPU is available on the machine, consider using NCHW data format for increased speed on GPU.')
    if hparams.data_format == 'NCHW':
        images = tf.transpose(images, [0, 3, 1, 2])
    total_num_cells = hparams.num_cells + 2
    normal_cell = nasnet_utils.NasNetANormalCell(hparams.num_conv_filters, hparams.drop_path_keep_prob, total_num_cells, hparams.total_training_steps, hparams.use_bounded_activation)
    reduction_cell = nasnet_utils.NasNetAReductionCell(hparams.num_conv_filters, hparams.drop_path_keep_prob, total_num_cells, hparams.total_training_steps, hparams.use_bounded_activation)
    with arg_scope([slim.dropout, nasnet_utils.drop_path, slim.batch_norm], is_training=is_training):
        with arg_scope([slim.avg_pool2d, slim.max_pool2d, slim.conv2d, slim.batch_norm, slim.separable_conv2d, nasnet_utils.factorized_reduction, nasnet_utils.global_avg_pool, nasnet_utils.get_channel_index, nasnet_utils.get_channel_dim], data_format=hparams.data_format):
            return _build_nasnet_base(images, normal_cell=normal_cell, reduction_cell=reduction_cell, num_classes=num_classes, hparams=hparams, is_training=is_training, stem_type='cifar', current_step=current_step)
build_nasnet_cifar.default_image_size = 32

def build_nasnet_mobile(images, num_classes, is_training=True, final_endpoint=None, config=None, current_step=None):
    if False:
        return 10
    'Build NASNet Mobile model for the ImageNet Dataset.'
    hparams = mobile_imagenet_config() if config is None else copy.deepcopy(config)
    _update_hparams(hparams, is_training)
    if tf.test.is_gpu_available() and hparams.data_format == 'NHWC':
        tf.logging.info('A GPU is available on the machine, consider using NCHW data format for increased speed on GPU.')
    if hparams.data_format == 'NCHW':
        images = tf.transpose(images, [0, 3, 1, 2])
    total_num_cells = hparams.num_cells + 2
    total_num_cells += 2
    normal_cell = nasnet_utils.NasNetANormalCell(hparams.num_conv_filters, hparams.drop_path_keep_prob, total_num_cells, hparams.total_training_steps, hparams.use_bounded_activation)
    reduction_cell = nasnet_utils.NasNetAReductionCell(hparams.num_conv_filters, hparams.drop_path_keep_prob, total_num_cells, hparams.total_training_steps, hparams.use_bounded_activation)
    with arg_scope([slim.dropout, nasnet_utils.drop_path, slim.batch_norm], is_training=is_training):
        with arg_scope([slim.avg_pool2d, slim.max_pool2d, slim.conv2d, slim.batch_norm, slim.separable_conv2d, nasnet_utils.factorized_reduction, nasnet_utils.global_avg_pool, nasnet_utils.get_channel_index, nasnet_utils.get_channel_dim], data_format=hparams.data_format):
            return _build_nasnet_base(images, normal_cell=normal_cell, reduction_cell=reduction_cell, num_classes=num_classes, hparams=hparams, is_training=is_training, stem_type='imagenet', final_endpoint=final_endpoint, current_step=current_step)
build_nasnet_mobile.default_image_size = 224

def build_nasnet_large(images, num_classes, is_training=True, final_endpoint=None, config=None, current_step=None):
    if False:
        print('Hello World!')
    'Build NASNet Large model for the ImageNet Dataset.'
    hparams = large_imagenet_config() if config is None else copy.deepcopy(config)
    _update_hparams(hparams, is_training)
    if tf.test.is_gpu_available() and hparams.data_format == 'NHWC':
        tf.logging.info('A GPU is available on the machine, consider using NCHW data format for increased speed on GPU.')
    if hparams.data_format == 'NCHW':
        images = tf.transpose(images, [0, 3, 1, 2])
    total_num_cells = hparams.num_cells + 2
    total_num_cells += 2
    normal_cell = nasnet_utils.NasNetANormalCell(hparams.num_conv_filters, hparams.drop_path_keep_prob, total_num_cells, hparams.total_training_steps, hparams.use_bounded_activation)
    reduction_cell = nasnet_utils.NasNetAReductionCell(hparams.num_conv_filters, hparams.drop_path_keep_prob, total_num_cells, hparams.total_training_steps, hparams.use_bounded_activation)
    with arg_scope([slim.dropout, nasnet_utils.drop_path, slim.batch_norm], is_training=is_training):
        with arg_scope([slim.avg_pool2d, slim.max_pool2d, slim.conv2d, slim.batch_norm, slim.separable_conv2d, nasnet_utils.factorized_reduction, nasnet_utils.global_avg_pool, nasnet_utils.get_channel_index, nasnet_utils.get_channel_dim], data_format=hparams.data_format):
            return _build_nasnet_base(images, normal_cell=normal_cell, reduction_cell=reduction_cell, num_classes=num_classes, hparams=hparams, is_training=is_training, stem_type='imagenet', final_endpoint=final_endpoint, current_step=current_step)
build_nasnet_large.default_image_size = 331

def _build_nasnet_base(images, normal_cell, reduction_cell, num_classes, hparams, is_training, stem_type, final_endpoint=None, current_step=None):
    if False:
        while True:
            i = 10
    'Constructs a NASNet image model.'
    end_points = {}

    def add_and_check_endpoint(endpoint_name, net):
        if False:
            i = 10
            return i + 15
        end_points[endpoint_name] = net
        return final_endpoint and endpoint_name == final_endpoint
    reduction_indices = nasnet_utils.calc_reduction_layers(hparams.num_cells, hparams.num_reduction_layers)
    stem_cell = reduction_cell
    if stem_type == 'imagenet':
        stem = lambda : _imagenet_stem(images, hparams, stem_cell)
    elif stem_type == 'cifar':
        stem = lambda : _cifar_stem(images, hparams)
    else:
        raise ValueError('Unknown stem_type: ', stem_type)
    (net, cell_outputs) = stem()
    if add_and_check_endpoint('Stem', net):
        return (net, end_points)
    aux_head_cell_idxes = []
    if len(reduction_indices) >= 2:
        aux_head_cell_idxes.append(reduction_indices[1] - 1)
    filter_scaling = 1.0
    true_cell_num = 2 if stem_type == 'imagenet' else 0
    activation_fn = tf.nn.relu6 if hparams.use_bounded_activation else tf.nn.relu
    for cell_num in range(hparams.num_cells):
        stride = 1
        if hparams.skip_reduction_layer_input:
            prev_layer = cell_outputs[-2]
        if cell_num in reduction_indices:
            filter_scaling *= hparams.filter_scaling_rate
            net = reduction_cell(net, scope='reduction_cell_{}'.format(reduction_indices.index(cell_num)), filter_scaling=filter_scaling, stride=2, prev_layer=cell_outputs[-2], cell_num=true_cell_num, current_step=current_step)
            if add_and_check_endpoint('Reduction_Cell_{}'.format(reduction_indices.index(cell_num)), net):
                return (net, end_points)
            true_cell_num += 1
            cell_outputs.append(net)
        if not hparams.skip_reduction_layer_input:
            prev_layer = cell_outputs[-2]
        net = normal_cell(net, scope='cell_{}'.format(cell_num), filter_scaling=filter_scaling, stride=stride, prev_layer=prev_layer, cell_num=true_cell_num, current_step=current_step)
        if add_and_check_endpoint('Cell_{}'.format(cell_num), net):
            return (net, end_points)
        true_cell_num += 1
        if hparams.use_aux_head and cell_num in aux_head_cell_idxes and num_classes and is_training:
            aux_net = activation_fn(net)
            _build_aux_head(aux_net, end_points, num_classes, hparams, scope='aux_{}'.format(cell_num))
        cell_outputs.append(net)
    with tf.variable_scope('final_layer'):
        net = activation_fn(net)
        net = nasnet_utils.global_avg_pool(net)
        if add_and_check_endpoint('global_pool', net) or not num_classes:
            return (net, end_points)
        net = slim.dropout(net, hparams.dense_dropout_keep_prob, scope='dropout')
        logits = slim.fully_connected(net, num_classes)
        if add_and_check_endpoint('Logits', logits):
            return (net, end_points)
        predictions = tf.nn.softmax(logits, name='predictions')
        if add_and_check_endpoint('Predictions', predictions):
            return (net, end_points)
    return (logits, end_points)