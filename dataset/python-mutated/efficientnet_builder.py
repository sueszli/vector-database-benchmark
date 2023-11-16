"""Model Builder for EfficientNet.

efficientnet-bx (x=0,1,2,3,4,5,6,7) checkpoints are located in:
  https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/ckptsaug/efficientnet-bx.tar.gz
"""
import functools
import os
import re
from absl import logging
import numpy as np
import tensorflow as tf
import utils
from . import efficientnet_model

def efficientnet_params(model_name):
    if False:
        for i in range(10):
            print('nop')
    'Get efficientnet params based on model name.'
    params_dict = {'efficientnet-b0': (1.0, 1.0, 224, 0.2), 'efficientnet-b1': (1.0, 1.1, 240, 0.2), 'efficientnet-b2': (1.1, 1.2, 260, 0.3), 'efficientnet-b3': (1.2, 1.4, 300, 0.3), 'efficientnet-b4': (1.4, 1.8, 380, 0.4), 'efficientnet-b5': (1.6, 2.2, 456, 0.4), 'efficientnet-b6': (1.8, 2.6, 528, 0.5), 'efficientnet-b7': (2.0, 3.1, 600, 0.5), 'efficientnet-b8': (2.2, 3.6, 672, 0.5), 'efficientnet-l2': (4.3, 5.3, 800, 0.5)}
    return params_dict[model_name]

class BlockDecoder(object):
    """Block Decoder for readability."""

    def _decode_block_string(self, block_string):
        if False:
            for i in range(10):
                print('nop')
        'Gets a block through a string notation of arguments.'
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split('(\\d.*)', op)
            if len(splits) >= 2:
                (key, value) = splits[:2]
                options[key] = value
        if 's' not in options or len(options['s']) != 2:
            raise ValueError('Strides options should be a pair of integers.')
        return efficientnet_model.BlockArgs(kernel_size=int(options['k']), num_repeat=int(options['r']), input_filters=int(options['i']), output_filters=int(options['o']), expand_ratio=int(options['e']), id_skip='noskip' not in block_string, se_ratio=float(options['se']) if 'se' in options else None, strides=[int(options['s'][0]), int(options['s'][1])], conv_type=int(options['c']) if 'c' in options else 0, fused_conv=int(options['f']) if 'f' in options else 0, super_pixel=int(options['p']) if 'p' in options else 0, condconv='cc' in block_string)

    def _encode_block_string(self, block):
        if False:
            while True:
                i = 10
        'Encodes a block to a string.'
        args = ['r%d' % block.num_repeat, 'k%d' % block.kernel_size, 's%d%d' % (block.strides[0], block.strides[1]), 'e%s' % block.expand_ratio, 'i%d' % block.input_filters, 'o%d' % block.output_filters, 'c%d' % block.conv_type, 'f%d' % block.fused_conv, 'p%d' % block.super_pixel]
        if block.se_ratio > 0 and block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        if block.condconv:
            args.append('cc')
        return '_'.join(args)

    def decode(self, string_list):
        if False:
            for i in range(10):
                print('nop')
        'Decodes a list of string notations to specify blocks inside the network.\n\n        Args:\n          string_list: a list of strings, each string is a notation of block.\n\n        Returns:\n          A list of namedtuples to represent blocks arguments.\n        '
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(self._decode_block_string(block_string))
        return blocks_args

    def encode(self, blocks_args):
        if False:
            while True:
                i = 10
        'Encodes a list of Blocks to a list of strings.\n\n        Args:\n          blocks_args: A list of namedtuples to represent blocks arguments.\n        Returns:\n          a list of strings, each string is a notation of block.\n        '
        block_strings = []
        for block in blocks_args:
            block_strings.append(self._encode_block_string(block))
        return block_strings

def swish(features, use_native=True, use_hard=False):
    if False:
        print('Hello World!')
    'Computes the Swish activation function.\n\n    We provide three alternatives:\n      - Native tf.nn.swish, use less memory during training than composable swish.\n      - Quantization friendly hard swish.\n      - A composable swish, equivalent to tf.nn.swish, but more general for\n        finetuning and TF-Hub.\n\n    Args:\n      features: A `Tensor` representing preactivation values.\n      use_native: Whether to use the native swish from tf.nn that uses a custom\n        gradient to reduce memory usage, or to use customized swish that uses\n        default TensorFlow gradient computation.\n      use_hard: Whether to use quantization-friendly hard swish.\n\n    Returns:\n      The activation value.\n    '
    if use_native and use_hard:
        raise ValueError('Cannot specify both use_native and use_hard.')
    if use_native:
        return tf.nn.swish(features)
    if use_hard:
        return features * tf.nn.relu6(features + np.float32(3)) * (1.0 / 6.0)
    features = tf.convert_to_tensor(features, name='features')
    return features * tf.nn.sigmoid(features)
_DEFAULT_BLOCKS_ARGS = ['r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25', 'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25', 'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25', 'r1_k3_s11_e6_i192_o320_se0.25']

def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2, survival_prob=0.8):
    if False:
        print('Hello World!')
    'Creates a efficientnet model.'
    global_params = efficientnet_model.GlobalParams(blocks_args=_DEFAULT_BLOCKS_ARGS, batch_norm_momentum=0.99, batch_norm_epsilon=0.001, dropout_rate=dropout_rate, survival_prob=survival_prob, data_format='channels_last', num_classes=1000, width_coefficient=width_coefficient, depth_coefficient=depth_coefficient, depth_divisor=8, min_depth=None, relu_fn=tf.nn.swish, batch_norm=utils.BatchNormalization, use_se=True, clip_projection_output=False)
    return global_params

def get_model_params(model_name, override_params):
    if False:
        return 10
    'Get the block args and global params for a given model.'
    if model_name.startswith('efficientnet'):
        (width_coefficient, depth_coefficient, _, dropout_rate) = efficientnet_params(model_name)
        global_params = efficientnet(width_coefficient, depth_coefficient, dropout_rate)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        global_params = global_params._replace(**override_params)
    decoder = BlockDecoder()
    blocks_args = decoder.decode(global_params.blocks_args)
    logging.info('global_params= %s', global_params)
    return (blocks_args, global_params)

def get_model(model_name, override_params={}):
    if False:
        while True:
            i = 10
    'A helper function to create and return model.\n\n    Args:\n      model_name: string, the predefined model name.\n      override_params: A dictionary of params for overriding. Fields must exist in\n        efficientnet_model.GlobalParams.\n\n    Returns:\n      created model\n\n    Raises:\n      When model_name specified an undefined model, raises NotImplementedError.\n      When override_params has invalid fields, raises ValueError.\n    '
    if model_name.startswith('efficientnet-'):
        (blocks_args, global_params) = get_model_params(model_name, override_params)
        return efficientnet_model.Model(blocks_args, global_params, model_name)
    else:
        raise ValueError('Unknown model name {}'.format(model_name))