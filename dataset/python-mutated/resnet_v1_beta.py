"""Resnet v1 model variants.

Code branched out from slim/nets/resnet_v1.py, and please refer to it for
more details.

The original version ResNets-v1 were proposed by:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
from six.moves import range
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from deeplab.core import conv2d_ws
from deeplab.core import utils
from tensorflow.contrib.slim.nets import resnet_utils
slim = contrib_slim
_DEFAULT_MULTI_GRID = [1, 1, 1]
_DEFAULT_MULTI_GRID_RESNET_18 = [1, 1]

@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, unit_rate=1, rate=1, outputs_collections=None, scope=None):
    if False:
        i = 10
        return i + 15
    "Bottleneck residual unit variant with BN after convolutions.\n\n  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for\n  its definition. Note that we use here the bottleneck variant which has an\n  extra bottleneck layer.\n\n  When putting together two consecutive ResNet blocks that use this unit, one\n  should use stride = 2 in the last unit of the first block.\n\n  Args:\n    inputs: A tensor of size [batch, height, width, channels].\n    depth: The depth of the ResNet unit output.\n    depth_bottleneck: The depth of the bottleneck layers.\n    stride: The ResNet unit's stride. Determines the amount of downsampling of\n      the units output compared to its input.\n    unit_rate: An integer, unit rate for atrous convolution.\n    rate: An integer, rate for atrous convolution.\n    outputs_collections: Collection to add the ResNet unit output.\n    scope: Optional variable_scope.\n\n  Returns:\n    The ResNet unit's output.\n  "
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = conv2d_ws.conv2d(inputs, depth, [1, 1], stride=stride, activation_fn=None, scope='shortcut')
        residual = conv2d_ws.conv2d(inputs, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = conv2d_ws.conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate * unit_rate, scope='conv2')
        residual = conv2d_ws.conv2d(residual, depth, [1, 1], stride=1, activation_fn=None, scope='conv3')
        output = tf.nn.relu(shortcut + residual)
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)

@slim.add_arg_scope
def lite_bottleneck(inputs, depth, stride, unit_rate=1, rate=1, outputs_collections=None, scope=None):
    if False:
        while True:
            i = 10
    "Bottleneck residual unit variant with BN after convolutions.\n\n  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for\n  its definition. Note that we use here the bottleneck variant which has an\n  extra bottleneck layer.\n\n  When putting together two consecutive ResNet blocks that use this unit, one\n  should use stride = 2 in the last unit of the first block.\n\n  Args:\n    inputs: A tensor of size [batch, height, width, channels].\n    depth: The depth of the ResNet unit output.\n    stride: The ResNet unit's stride. Determines the amount of downsampling of\n      the units output compared to its input.\n    unit_rate: An integer, unit rate for atrous convolution.\n    rate: An integer, rate for atrous convolution.\n    outputs_collections: Collection to add the ResNet unit output.\n    scope: Optional variable_scope.\n\n  Returns:\n    The ResNet unit's output.\n  "
    with tf.variable_scope(scope, 'lite_bottleneck_v1', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = conv2d_ws.conv2d(inputs, depth, [1, 1], stride=stride, activation_fn=None, scope='shortcut')
        residual = conv2d_ws.conv2d_same(inputs, depth, 3, 1, rate=rate * unit_rate, scope='conv1')
        with slim.arg_scope([conv2d_ws.conv2d], activation_fn=None):
            residual = conv2d_ws.conv2d_same(residual, depth, 3, stride, rate=rate * unit_rate, scope='conv2')
        output = tf.nn.relu(shortcut + residual)
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)

def root_block_fn_for_beta_variant(net, depth_multiplier=1.0):
    if False:
        while True:
            i = 10
    'Gets root_block_fn for beta variant.\n\n  ResNet-v1 beta variant modifies the first original 7x7 convolution to three\n  3x3 convolutions.\n\n  Args:\n    net: A tensor of size [batch, height, width, channels], input to the model.\n    depth_multiplier: Controls the number of convolution output channels for\n      each input channel. The total number of depthwise convolution output\n      channels will be equal to `num_filters_out * depth_multiplier`.\n\n  Returns:\n    A tensor after three 3x3 convolutions.\n  '
    net = conv2d_ws.conv2d_same(net, int(64 * depth_multiplier), 3, stride=2, scope='conv1_1')
    net = conv2d_ws.conv2d_same(net, int(64 * depth_multiplier), 3, stride=1, scope='conv1_2')
    net = conv2d_ws.conv2d_same(net, int(128 * depth_multiplier), 3, stride=1, scope='conv1_3')
    return net

def resnet_v1_beta(inputs, blocks, num_classes=None, is_training=None, global_pool=True, output_stride=None, root_block_fn=None, reuse=None, scope=None, sync_batch_norm_method='None'):
    if False:
        return 10
    "Generator for v1 ResNet models (beta variant).\n\n  This function generates a family of modified ResNet v1 models. In particular,\n  the first original 7x7 convolution is replaced with three 3x3 convolutions.\n  See the resnet_v1_*() methods for specific model instantiations, obtained by\n  selecting different block instantiations that produce ResNets of various\n  depths.\n\n  The code is modified from slim/nets/resnet_v1.py, and please refer to it for\n  more details.\n\n  Args:\n    inputs: A tensor of size [batch, height_in, width_in, channels].\n    blocks: A list of length equal to the number of ResNet blocks. Each element\n      is a resnet_utils.Block object describing the units in the block.\n    num_classes: Number of predicted classes for classification tasks. If None\n      we return the features before the logit layer.\n    is_training: Enable/disable is_training for batch normalization.\n    global_pool: If True, we perform global average pooling before computing the\n      logits. Set to True for image classification, False for dense prediction.\n    output_stride: If None, then the output will be computed at the nominal\n      network stride. If output_stride is not None, it specifies the requested\n      ratio of input to output spatial resolution.\n    root_block_fn: The function consisting of convolution operations applied to\n      the root input. If root_block_fn is None, use the original setting of\n      RseNet-v1, which is simply one convolution with 7x7 kernel and stride=2.\n    reuse: whether or not the network and its variables should be reused. To be\n      able to reuse 'scope' must be given.\n    scope: Optional variable_scope.\n    sync_batch_norm_method: String, sync batchnorm method.\n\n  Returns:\n    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].\n      If global_pool is False, then height_out and width_out are reduced by a\n      factor of output_stride compared to the respective height_in and width_in,\n      else both height_out and width_out equal one. If num_classes is None, then\n      net is the output of the last ResNet block, potentially after global\n      average pooling. If num_classes is not None, net contains the pre-softmax\n      activations.\n    end_points: A dictionary from components of the network to the corresponding\n      activation.\n\n  Raises:\n    ValueError: If the target output_stride is not valid.\n  "
    if root_block_fn is None:
        root_block_fn = functools.partial(conv2d_ws.conv2d_same, num_outputs=64, kernel_size=7, stride=2, scope='conv1')
    batch_norm = utils.get_batch_norm_fn(sync_batch_norm_method)
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([conv2d_ws.conv2d, bottleneck, lite_bottleneck, resnet_utils.stack_blocks_dense], outputs_collections=end_points_collection):
            if is_training is not None:
                arg_scope = slim.arg_scope([batch_norm], is_training=is_training)
            else:
                arg_scope = slim.arg_scope([])
            with arg_scope:
                net = inputs
                if output_stride is not None:
                    if output_stride % 4 != 0:
                        raise ValueError('The output_stride needs to be a multiple of 4.')
                    output_stride //= 4
                net = root_block_fn(net)
                net = slim.max_pool2d(net, 3, stride=2, padding='SAME', scope='pool1')
                net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keepdims=True)
                if num_classes is not None:
                    net = conv2d_ws.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits', use_weight_standardization=False)
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if num_classes is not None:
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return (net, end_points)

def resnet_v1_beta_block(scope, base_depth, num_units, stride):
    if False:
        while True:
            i = 10
    'Helper function for creating a resnet_v1 beta variant bottleneck block.\n\n  Args:\n    scope: The scope of the block.\n    base_depth: The depth of the bottleneck layer for each unit.\n    num_units: The number of units in the block.\n    stride: The stride of the block, implemented as a stride in the last unit.\n      All other units have stride=1.\n\n  Returns:\n    A resnet_v1 bottleneck block.\n  '
    return resnet_utils.Block(scope, bottleneck, [{'depth': base_depth * 4, 'depth_bottleneck': base_depth, 'stride': 1, 'unit_rate': 1}] * (num_units - 1) + [{'depth': base_depth * 4, 'depth_bottleneck': base_depth, 'stride': stride, 'unit_rate': 1}])

def resnet_v1_small_beta_block(scope, base_depth, num_units, stride):
    if False:
        i = 10
        return i + 15
    'Helper function for creating a resnet_18 beta variant bottleneck block.\n\n  Args:\n    scope: The scope of the block.\n    base_depth: The depth of the bottleneck layer for each unit.\n    num_units: The number of units in the block.\n    stride: The stride of the block, implemented as a stride in the last unit.\n      All other units have stride=1.\n\n  Returns:\n    A resnet_18 bottleneck block.\n  '
    block_args = []
    for _ in range(num_units - 1):
        block_args.append({'depth': base_depth, 'stride': 1, 'unit_rate': 1})
    block_args.append({'depth': base_depth, 'stride': stride, 'unit_rate': 1})
    return resnet_utils.Block(scope, lite_bottleneck, block_args)

def resnet_v1_18(inputs, num_classes=None, is_training=None, global_pool=False, output_stride=None, multi_grid=None, reuse=None, scope='resnet_v1_18', sync_batch_norm_method='None'):
    if False:
        for i in range(10):
            print('nop')
    "Resnet v1 18.\n\n  Args:\n    inputs: A tensor of size [batch, height_in, width_in, channels].\n    num_classes: Number of predicted classes for classification tasks. If None\n      we return the features before the logit layer.\n    is_training: Enable/disable is_training for batch normalization.\n    global_pool: If True, we perform global average pooling before computing the\n      logits. Set to True for image classification, False for dense prediction.\n    output_stride: If None, then the output will be computed at the nominal\n      network stride. If output_stride is not None, it specifies the requested\n      ratio of input to output spatial resolution.\n    multi_grid: Employ a hierarchy of different atrous rates within network.\n    reuse: whether or not the network and its variables should be reused. To be\n      able to reuse 'scope' must be given.\n    scope: Optional variable_scope.\n    sync_batch_norm_method: String, sync batchnorm method.\n\n  Returns:\n    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].\n      If global_pool is False, then height_out and width_out are reduced by a\n      factor of output_stride compared to the respective height_in and width_in,\n      else both height_out and width_out equal one. If num_classes is None, then\n      net is the output of the last ResNet block, potentially after global\n      average pooling. If num_classes is not None, net contains the pre-softmax\n      activations.\n    end_points: A dictionary from components of the network to the corresponding\n      activation.\n\n  Raises:\n    ValueError: if multi_grid is not None and does not have length = 3.\n  "
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID_RESNET_18
    elif len(multi_grid) != 2:
        raise ValueError('Expect multi_grid to have length 2.')
    block4_args = []
    for rate in multi_grid:
        block4_args.append({'depth': 512, 'stride': 1, 'unit_rate': rate})
    blocks = [resnet_v1_small_beta_block('block1', base_depth=64, num_units=2, stride=2), resnet_v1_small_beta_block('block2', base_depth=128, num_units=2, stride=2), resnet_v1_small_beta_block('block3', base_depth=256, num_units=2, stride=2), resnet_utils.Block('block4', lite_bottleneck, block4_args)]
    return resnet_v1_beta(inputs, blocks=blocks, num_classes=num_classes, is_training=is_training, global_pool=global_pool, output_stride=output_stride, reuse=reuse, scope=scope, sync_batch_norm_method=sync_batch_norm_method)

def resnet_v1_18_beta(inputs, num_classes=None, is_training=None, global_pool=False, output_stride=None, multi_grid=None, root_depth_multiplier=0.25, reuse=None, scope='resnet_v1_18', sync_batch_norm_method='None'):
    if False:
        while True:
            i = 10
    "Resnet v1 18 beta variant.\n\n  This variant modifies the first convolution layer of ResNet-v1-18. In\n  particular, it changes the original one 7x7 convolution to three 3x3\n  convolutions.\n\n  Args:\n    inputs: A tensor of size [batch, height_in, width_in, channels].\n    num_classes: Number of predicted classes for classification tasks. If None\n      we return the features before the logit layer.\n    is_training: Enable/disable is_training for batch normalization.\n    global_pool: If True, we perform global average pooling before computing the\n      logits. Set to True for image classification, False for dense prediction.\n    output_stride: If None, then the output will be computed at the nominal\n      network stride. If output_stride is not None, it specifies the requested\n      ratio of input to output spatial resolution.\n    multi_grid: Employ a hierarchy of different atrous rates within network.\n    root_depth_multiplier: Float, depth multiplier used for the first three\n      convolution layers that replace the 7x7 convolution.\n    reuse: whether or not the network and its variables should be reused. To be\n      able to reuse 'scope' must be given.\n    scope: Optional variable_scope.\n    sync_batch_norm_method: String, sync batchnorm method.\n\n  Returns:\n    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].\n      If global_pool is False, then height_out and width_out are reduced by a\n      factor of output_stride compared to the respective height_in and width_in,\n      else both height_out and width_out equal one. If num_classes is None, then\n      net is the output of the last ResNet block, potentially after global\n      average pooling. If num_classes is not None, net contains the pre-softmax\n      activations.\n    end_points: A dictionary from components of the network to the corresponding\n      activation.\n\n  Raises:\n    ValueError: if multi_grid is not None and does not have length = 3.\n  "
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID_RESNET_18
    elif len(multi_grid) != 2:
        raise ValueError('Expect multi_grid to have length 2.')
    block4_args = []
    for rate in multi_grid:
        block4_args.append({'depth': 512, 'stride': 1, 'unit_rate': rate})
    blocks = [resnet_v1_small_beta_block('block1', base_depth=64, num_units=2, stride=2), resnet_v1_small_beta_block('block2', base_depth=128, num_units=2, stride=2), resnet_v1_small_beta_block('block3', base_depth=256, num_units=2, stride=2), resnet_utils.Block('block4', lite_bottleneck, block4_args)]
    return resnet_v1_beta(inputs, blocks=blocks, num_classes=num_classes, is_training=is_training, global_pool=global_pool, output_stride=output_stride, root_block_fn=functools.partial(root_block_fn_for_beta_variant, depth_multiplier=root_depth_multiplier), reuse=reuse, scope=scope, sync_batch_norm_method=sync_batch_norm_method)

def resnet_v1_50(inputs, num_classes=None, is_training=None, global_pool=False, output_stride=None, multi_grid=None, reuse=None, scope='resnet_v1_50', sync_batch_norm_method='None'):
    if False:
        for i in range(10):
            print('nop')
    "Resnet v1 50.\n\n  Args:\n    inputs: A tensor of size [batch, height_in, width_in, channels].\n    num_classes: Number of predicted classes for classification tasks. If None\n      we return the features before the logit layer.\n    is_training: Enable/disable is_training for batch normalization.\n    global_pool: If True, we perform global average pooling before computing the\n      logits. Set to True for image classification, False for dense prediction.\n    output_stride: If None, then the output will be computed at the nominal\n      network stride. If output_stride is not None, it specifies the requested\n      ratio of input to output spatial resolution.\n    multi_grid: Employ a hierarchy of different atrous rates within network.\n    reuse: whether or not the network and its variables should be reused. To be\n      able to reuse 'scope' must be given.\n    scope: Optional variable_scope.\n    sync_batch_norm_method: String, sync batchnorm method.\n\n  Returns:\n    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].\n      If global_pool is False, then height_out and width_out are reduced by a\n      factor of output_stride compared to the respective height_in and width_in,\n      else both height_out and width_out equal one. If num_classes is None, then\n      net is the output of the last ResNet block, potentially after global\n      average pooling. If num_classes is not None, net contains the pre-softmax\n      activations.\n    end_points: A dictionary from components of the network to the corresponding\n      activation.\n\n  Raises:\n    ValueError: if multi_grid is not None and does not have length = 3.\n  "
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID
    elif len(multi_grid) != 3:
        raise ValueError('Expect multi_grid to have length 3.')
    blocks = [resnet_v1_beta_block('block1', base_depth=64, num_units=3, stride=2), resnet_v1_beta_block('block2', base_depth=128, num_units=4, stride=2), resnet_v1_beta_block('block3', base_depth=256, num_units=6, stride=2), resnet_utils.Block('block4', bottleneck, [{'depth': 2048, 'depth_bottleneck': 512, 'stride': 1, 'unit_rate': rate} for rate in multi_grid])]
    return resnet_v1_beta(inputs, blocks=blocks, num_classes=num_classes, is_training=is_training, global_pool=global_pool, output_stride=output_stride, reuse=reuse, scope=scope, sync_batch_norm_method=sync_batch_norm_method)

def resnet_v1_50_beta(inputs, num_classes=None, is_training=None, global_pool=False, output_stride=None, multi_grid=None, reuse=None, scope='resnet_v1_50', sync_batch_norm_method='None'):
    if False:
        print('Hello World!')
    "Resnet v1 50 beta variant.\n\n  This variant modifies the first convolution layer of ResNet-v1-50. In\n  particular, it changes the original one 7x7 convolution to three 3x3\n  convolutions.\n\n  Args:\n    inputs: A tensor of size [batch, height_in, width_in, channels].\n    num_classes: Number of predicted classes for classification tasks. If None\n      we return the features before the logit layer.\n    is_training: Enable/disable is_training for batch normalization.\n    global_pool: If True, we perform global average pooling before computing the\n      logits. Set to True for image classification, False for dense prediction.\n    output_stride: If None, then the output will be computed at the nominal\n      network stride. If output_stride is not None, it specifies the requested\n      ratio of input to output spatial resolution.\n    multi_grid: Employ a hierarchy of different atrous rates within network.\n    reuse: whether or not the network and its variables should be reused. To be\n      able to reuse 'scope' must be given.\n    scope: Optional variable_scope.\n    sync_batch_norm_method: String, sync batchnorm method.\n\n  Returns:\n    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].\n      If global_pool is False, then height_out and width_out are reduced by a\n      factor of output_stride compared to the respective height_in and width_in,\n      else both height_out and width_out equal one. If num_classes is None, then\n      net is the output of the last ResNet block, potentially after global\n      average pooling. If num_classes is not None, net contains the pre-softmax\n      activations.\n    end_points: A dictionary from components of the network to the corresponding\n      activation.\n\n  Raises:\n    ValueError: if multi_grid is not None and does not have length = 3.\n  "
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID
    elif len(multi_grid) != 3:
        raise ValueError('Expect multi_grid to have length 3.')
    blocks = [resnet_v1_beta_block('block1', base_depth=64, num_units=3, stride=2), resnet_v1_beta_block('block2', base_depth=128, num_units=4, stride=2), resnet_v1_beta_block('block3', base_depth=256, num_units=6, stride=2), resnet_utils.Block('block4', bottleneck, [{'depth': 2048, 'depth_bottleneck': 512, 'stride': 1, 'unit_rate': rate} for rate in multi_grid])]
    return resnet_v1_beta(inputs, blocks=blocks, num_classes=num_classes, is_training=is_training, global_pool=global_pool, output_stride=output_stride, root_block_fn=functools.partial(root_block_fn_for_beta_variant), reuse=reuse, scope=scope, sync_batch_norm_method=sync_batch_norm_method)

def resnet_v1_101(inputs, num_classes=None, is_training=None, global_pool=False, output_stride=None, multi_grid=None, reuse=None, scope='resnet_v1_101', sync_batch_norm_method='None'):
    if False:
        while True:
            i = 10
    "Resnet v1 101.\n\n  Args:\n    inputs: A tensor of size [batch, height_in, width_in, channels].\n    num_classes: Number of predicted classes for classification tasks. If None\n      we return the features before the logit layer.\n    is_training: Enable/disable is_training for batch normalization.\n    global_pool: If True, we perform global average pooling before computing the\n      logits. Set to True for image classification, False for dense prediction.\n    output_stride: If None, then the output will be computed at the nominal\n      network stride. If output_stride is not None, it specifies the requested\n      ratio of input to output spatial resolution.\n    multi_grid: Employ a hierarchy of different atrous rates within network.\n    reuse: whether or not the network and its variables should be reused. To be\n      able to reuse 'scope' must be given.\n    scope: Optional variable_scope.\n    sync_batch_norm_method: String, sync batchnorm method.\n\n  Returns:\n    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].\n      If global_pool is False, then height_out and width_out are reduced by a\n      factor of output_stride compared to the respective height_in and width_in,\n      else both height_out and width_out equal one. If num_classes is None, then\n      net is the output of the last ResNet block, potentially after global\n      average pooling. If num_classes is not None, net contains the pre-softmax\n      activations.\n    end_points: A dictionary from components of the network to the corresponding\n      activation.\n\n  Raises:\n    ValueError: if multi_grid is not None and does not have length = 3.\n  "
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID
    elif len(multi_grid) != 3:
        raise ValueError('Expect multi_grid to have length 3.')
    blocks = [resnet_v1_beta_block('block1', base_depth=64, num_units=3, stride=2), resnet_v1_beta_block('block2', base_depth=128, num_units=4, stride=2), resnet_v1_beta_block('block3', base_depth=256, num_units=23, stride=2), resnet_utils.Block('block4', bottleneck, [{'depth': 2048, 'depth_bottleneck': 512, 'stride': 1, 'unit_rate': rate} for rate in multi_grid])]
    return resnet_v1_beta(inputs, blocks=blocks, num_classes=num_classes, is_training=is_training, global_pool=global_pool, output_stride=output_stride, reuse=reuse, scope=scope, sync_batch_norm_method=sync_batch_norm_method)

def resnet_v1_101_beta(inputs, num_classes=None, is_training=None, global_pool=False, output_stride=None, multi_grid=None, reuse=None, scope='resnet_v1_101', sync_batch_norm_method='None'):
    if False:
        while True:
            i = 10
    "Resnet v1 101 beta variant.\n\n  This variant modifies the first convolution layer of ResNet-v1-101. In\n  particular, it changes the original one 7x7 convolution to three 3x3\n  convolutions.\n\n  Args:\n    inputs: A tensor of size [batch, height_in, width_in, channels].\n    num_classes: Number of predicted classes for classification tasks. If None\n      we return the features before the logit layer.\n    is_training: Enable/disable is_training for batch normalization.\n    global_pool: If True, we perform global average pooling before computing the\n      logits. Set to True for image classification, False for dense prediction.\n    output_stride: If None, then the output will be computed at the nominal\n      network stride. If output_stride is not None, it specifies the requested\n      ratio of input to output spatial resolution.\n    multi_grid: Employ a hierarchy of different atrous rates within network.\n    reuse: whether or not the network and its variables should be reused. To be\n      able to reuse 'scope' must be given.\n    scope: Optional variable_scope.\n    sync_batch_norm_method: String, sync batchnorm method.\n\n  Returns:\n    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].\n      If global_pool is False, then height_out and width_out are reduced by a\n      factor of output_stride compared to the respective height_in and width_in,\n      else both height_out and width_out equal one. If num_classes is None, then\n      net is the output of the last ResNet block, potentially after global\n      average pooling. If num_classes is not None, net contains the pre-softmax\n      activations.\n    end_points: A dictionary from components of the network to the corresponding\n      activation.\n\n  Raises:\n    ValueError: if multi_grid is not None and does not have length = 3.\n  "
    if multi_grid is None:
        multi_grid = _DEFAULT_MULTI_GRID
    elif len(multi_grid) != 3:
        raise ValueError('Expect multi_grid to have length 3.')
    blocks = [resnet_v1_beta_block('block1', base_depth=64, num_units=3, stride=2), resnet_v1_beta_block('block2', base_depth=128, num_units=4, stride=2), resnet_v1_beta_block('block3', base_depth=256, num_units=23, stride=2), resnet_utils.Block('block4', bottleneck, [{'depth': 2048, 'depth_bottleneck': 512, 'stride': 1, 'unit_rate': rate} for rate in multi_grid])]
    return resnet_v1_beta(inputs, blocks=blocks, num_classes=num_classes, is_training=is_training, global_pool=global_pool, output_stride=output_stride, root_block_fn=functools.partial(root_block_fn_for_beta_variant), reuse=reuse, scope=scope, sync_batch_norm_method=sync_batch_norm_method)

def resnet_arg_scope(weight_decay=0.0001, batch_norm_decay=0.997, batch_norm_epsilon=1e-05, batch_norm_scale=True, activation_fn=tf.nn.relu, use_batch_norm=True, sync_batch_norm_method='None', normalization_method='unspecified', use_weight_standardization=False):
    if False:
        while True:
            i = 10
    'Defines the default ResNet arg scope.\n\n  Args:\n    weight_decay: The weight decay to use for regularizing the model.\n    batch_norm_decay: The moving average decay when estimating layer activation\n      statistics in batch normalization.\n    batch_norm_epsilon: Small constant to prevent division by zero when\n      normalizing activations by their variance in batch normalization.\n    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the\n      activations in the batch normalization layer.\n    activation_fn: The activation function which is used in ResNet.\n    use_batch_norm: Deprecated in favor of normalization_method.\n    sync_batch_norm_method: String, sync batchnorm method.\n    normalization_method: String, one of `batch`, `none`, or `group`, to use\n      batch normalization, no normalization, or group normalization.\n    use_weight_standardization: Boolean, whether to use weight standardization.\n\n  Returns:\n    An `arg_scope` to use for the resnet models.\n  '
    batch_norm_params = {'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale}
    batch_norm = utils.get_batch_norm_fn(sync_batch_norm_method)
    if normalization_method == 'batch':
        normalizer_fn = batch_norm
    elif normalization_method == 'none':
        normalizer_fn = None
    elif normalization_method == 'group':
        normalizer_fn = slim.group_norm
    elif normalization_method == 'unspecified':
        normalizer_fn = batch_norm if use_batch_norm else None
    else:
        raise ValueError('Unrecognized normalization_method %s' % normalization_method)
    with slim.arg_scope([conv2d_ws.conv2d], weights_regularizer=slim.l2_regularizer(weight_decay), weights_initializer=slim.variance_scaling_initializer(), activation_fn=activation_fn, normalizer_fn=normalizer_fn, use_weight_standardization=use_weight_standardization):
        with slim.arg_scope([batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc