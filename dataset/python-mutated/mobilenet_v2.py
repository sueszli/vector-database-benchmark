"""Implementation of Mobilenet V2.

Architecture: https://arxiv.org/abs/1801.04381

The base model gives 72.2% accuracy on ImageNet, with 300MMadds,
3.4 M parameters.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import functools
import tensorflow as tf
from tensorflow.contrib import layers as contrib_layers
from tensorflow.contrib import slim as contrib_slim
from nets.mobilenet import conv_blocks as ops
from nets.mobilenet import mobilenet as lib
slim = contrib_slim
op = lib.op
expand_input = ops.expand_input_by_factor
V2_DEF = dict(defaults={(slim.batch_norm,): {'center': True, 'scale': True}, (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6}, (ops.expanded_conv,): {'expansion_size': expand_input(6), 'split_expansion': 1, 'normalizer_fn': slim.batch_norm, 'residual': True}, (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}}, spec=[op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]), op(ops.expanded_conv, expansion_size=expand_input(1, divisible_by=1), num_outputs=16), op(ops.expanded_conv, stride=2, num_outputs=24), op(ops.expanded_conv, stride=1, num_outputs=24), op(ops.expanded_conv, stride=2, num_outputs=32), op(ops.expanded_conv, stride=1, num_outputs=32), op(ops.expanded_conv, stride=1, num_outputs=32), op(ops.expanded_conv, stride=2, num_outputs=64), op(ops.expanded_conv, stride=1, num_outputs=64), op(ops.expanded_conv, stride=1, num_outputs=64), op(ops.expanded_conv, stride=1, num_outputs=64), op(ops.expanded_conv, stride=1, num_outputs=96), op(ops.expanded_conv, stride=1, num_outputs=96), op(ops.expanded_conv, stride=1, num_outputs=96), op(ops.expanded_conv, stride=2, num_outputs=160), op(ops.expanded_conv, stride=1, num_outputs=160), op(ops.expanded_conv, stride=1, num_outputs=160), op(ops.expanded_conv, stride=1, num_outputs=320), op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)])
V2_DEF_GROUP_NORM = copy.deepcopy(V2_DEF)
V2_DEF_GROUP_NORM['defaults'] = {(contrib_slim.conv2d, contrib_slim.fully_connected, contrib_slim.separable_conv2d): {'normalizer_fn': contrib_layers.group_norm, 'activation_fn': tf.nn.relu6}, (ops.expanded_conv,): {'expansion_size': ops.expand_input_by_factor(6), 'split_expansion': 1, 'normalizer_fn': contrib_layers.group_norm, 'residual': True}, (contrib_slim.conv2d, contrib_slim.separable_conv2d): {'padding': 'SAME'}}

@slim.add_arg_scope
def mobilenet(input_tensor, num_classes=1001, depth_multiplier=1.0, scope='MobilenetV2', conv_defs=None, finegrain_classification_mode=False, min_depth=None, divisible_by=None, activation_fn=None, **kwargs):
    if False:
        return 10
    'Creates mobilenet V2 network.\n\n  Inference mode is created by default. To create training use training_scope\n  below.\n\n  with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):\n     logits, endpoints = mobilenet_v2.mobilenet(input_tensor)\n\n  Args:\n    input_tensor: The input tensor\n    num_classes: number of classes\n    depth_multiplier: The multiplier applied to scale number of\n    channels in each layer.\n    scope: Scope of the operator\n    conv_defs: Allows to override default conv def.\n    finegrain_classification_mode: When set to True, the model\n    will keep the last layer large even for small multipliers. Following\n    https://arxiv.org/abs/1801.04381\n    suggests that it improves performance for ImageNet-type of problems.\n      *Note* ignored if final_endpoint makes the builder exit earlier.\n    min_depth: If provided, will ensure that all layers will have that\n    many channels after application of depth multiplier.\n    divisible_by: If provided will ensure that all layers # channels\n    will be divisible by this number.\n    activation_fn: Activation function to use, defaults to tf.nn.relu6 if not\n      specified.\n    **kwargs: passed directly to mobilenet.mobilenet:\n      prediction_fn- what prediction function to use.\n      reuse-: whether to reuse variables (if reuse set to true, scope\n      must be given).\n  Returns:\n    logits/endpoints pair\n\n  Raises:\n    ValueError: On invalid arguments\n  '
    if conv_defs is None:
        conv_defs = V2_DEF
    if 'multiplier' in kwargs:
        raise ValueError('mobilenetv2 doesn\'t support generic multiplier parameter use "depth_multiplier" instead.')
    if finegrain_classification_mode:
        conv_defs = copy.deepcopy(conv_defs)
        if depth_multiplier < 1:
            conv_defs['spec'][-1].params['num_outputs'] /= depth_multiplier
    if activation_fn:
        conv_defs = copy.deepcopy(conv_defs)
        defaults = conv_defs['defaults']
        conv_defaults = defaults[slim.conv2d, slim.fully_connected, slim.separable_conv2d]
        conv_defaults['activation_fn'] = activation_fn
    depth_args = {}
    if min_depth is not None:
        depth_args['min_depth'] = min_depth
    if divisible_by is not None:
        depth_args['divisible_by'] = divisible_by
    with slim.arg_scope((lib.depth_multiplier,), **depth_args):
        return lib.mobilenet(input_tensor, num_classes=num_classes, conv_defs=conv_defs, scope=scope, multiplier=depth_multiplier, **kwargs)
mobilenet.default_image_size = 224

def wrapped_partial(func, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    partial_func = functools.partial(func, *args, **kwargs)
    functools.update_wrapper(partial_func, func)
    return partial_func
mobilenet_v2_140 = wrapped_partial(mobilenet, depth_multiplier=1.4)
mobilenet_v2_050 = wrapped_partial(mobilenet, depth_multiplier=0.5, finegrain_classification_mode=True)
mobilenet_v2_035 = wrapped_partial(mobilenet, depth_multiplier=0.35, finegrain_classification_mode=True)

@slim.add_arg_scope
def mobilenet_base(input_tensor, depth_multiplier=1.0, **kwargs):
    if False:
        while True:
            i = 10
    'Creates base of the mobilenet (no pooling and no logits) .'
    return mobilenet(input_tensor, depth_multiplier=depth_multiplier, base_only=True, **kwargs)

@slim.add_arg_scope
def mobilenet_base_group_norm(input_tensor, depth_multiplier=1.0, **kwargs):
    if False:
        while True:
            i = 10
    'Creates base of the mobilenet (no pooling and no logits) .'
    kwargs['conv_defs'] = V2_DEF_GROUP_NORM
    kwargs['conv_defs']['defaults'].update({(contrib_layers.group_norm,): {'groups': kwargs.pop('groups', 8)}})
    return mobilenet(input_tensor, depth_multiplier=depth_multiplier, base_only=True, **kwargs)

def training_scope(**kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Defines MobilenetV2 training scope.\n\n  Usage:\n     with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):\n       logits, endpoints = mobilenet_v2.mobilenet(input_tensor)\n\n  with slim.\n\n  Args:\n    **kwargs: Passed to mobilenet.training_scope. The following parameters\n    are supported:\n      weight_decay- The weight decay to use for regularizing the model.\n      stddev-  Standard deviation for initialization, if negative uses xavier.\n      dropout_keep_prob- dropout keep probability\n      bn_decay- decay for the batch norm moving averages.\n\n  Returns:\n    An `arg_scope` to use for the mobilenet v2 model.\n  '
    return lib.training_scope(**kwargs)
__all__ = ['training_scope', 'mobilenet_base', 'mobilenet', 'V2_DEF']