"""Contains definitions for the original form of Residual Networks.

The 'v1' residual networks (ResNets) implemented in this module were proposed
by:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Other variants were introduced in:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The networks defined in this module utilize the bottleneck building block of
[1] with projection shortcuts only for increasing depths. They employ batch
normalization *after* every weight layer. This is the architecture used by
MSRA in the Imagenet and MSCOCO 2016 competition models ResNet-101 and
ResNet-152. See [2; Fig. 1a] for a comparison between the current 'v1'
architecture and the alternative 'v2' architecture of [2] which uses batch
normalization *before* every weight layer in the so-called full pre-activation
units.

Typical use:

   from tensorflow.contrib.slim.nets import resnet_v1

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs, 1000, is_training=False)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from nets import resnet_utils
resnet_arg_scope = resnet_utils.resnet_arg_scope
slim = contrib_slim

class NoOpScope(object):
    """No-op context manager."""

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            for i in range(10):
                print('nop')
        return False

@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1, outputs_collections=None, scope=None, use_bounded_activations=False):
    if False:
        for i in range(10):
            print('nop')
    "Bottleneck residual unit variant with BN after convolutions.\n\n  This is the original residual unit proposed in [1]. See Fig. 1(a) of [2] for\n  its definition. Note that we use here the bottleneck variant which has an\n  extra bottleneck layer.\n\n  When putting together two consecutive ResNet blocks that use this unit, one\n  should use stride = 2 in the last unit of the first block.\n\n  Args:\n    inputs: A tensor of size [batch, height, width, channels].\n    depth: The depth of the ResNet unit output.\n    depth_bottleneck: The depth of the bottleneck layers.\n    stride: The ResNet unit's stride. Determines the amount of downsampling of\n      the units output compared to its input.\n    rate: An integer, rate for atrous convolution.\n    outputs_collections: Collection to add the ResNet unit output.\n    scope: Optional variable_scope.\n    use_bounded_activations: Whether or not to use bounded activations. Bounded\n      activations better lend themselves to quantized inference.\n\n  Returns:\n    The ResNet unit's output.\n  "
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(inputs, depth, [1, 1], stride=stride, activation_fn=tf.nn.relu6 if use_bounded_activations else None, scope='shortcut')
        residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, activation_fn=None, scope='conv3')
        if use_bounded_activations:
            residual = tf.clip_by_value(residual, -6.0, 6.0)
            output = tf.nn.relu6(shortcut + residual)
        else:
            output = tf.nn.relu(shortcut + residual)
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)

def resnet_v1(inputs, blocks, num_classes=None, is_training=True, global_pool=True, output_stride=None, include_root_block=True, spatial_squeeze=True, store_non_strided_activations=False, reuse=None, scope=None):
    if False:
        print('Hello World!')
    "Generator for v1 ResNet models.\n\n  This function generates a family of ResNet v1 models. See the resnet_v1_*()\n  methods for specific model instantiations, obtained by selecting different\n  block instantiations that produce ResNets of various depths.\n\n  Training for image classification on Imagenet is usually done with [224, 224]\n  inputs, resulting in [7, 7] feature maps at the output of the last ResNet\n  block for the ResNets defined in [1] that have nominal stride equal to 32.\n  However, for dense prediction tasks we advise that one uses inputs with\n  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In\n  this case the feature maps at the ResNet output will have spatial shape\n  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]\n  and corners exactly aligned with the input image corners, which greatly\n  facilitates alignment of the features to the image. Using as input [225, 225]\n  images results in [8, 8] feature maps at the output of the last ResNet block.\n\n  For dense prediction tasks, the ResNet needs to run in fully-convolutional\n  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all\n  have nominal stride equal to 32 and a good choice in FCN mode is to use\n  output_stride=16 in order to increase the density of the computed features at\n  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.\n\n  Args:\n    inputs: A tensor of size [batch, height_in, width_in, channels].\n    blocks: A list of length equal to the number of ResNet blocks. Each element\n      is a resnet_utils.Block object describing the units in the block.\n    num_classes: Number of predicted classes for classification tasks.\n      If 0 or None, we return the features before the logit layer.\n    is_training: whether batch_norm layers are in training mode. If this is set\n      to None, the callers can specify slim.batch_norm's is_training parameter\n      from an outer slim.arg_scope.\n    global_pool: If True, we perform global average pooling before computing the\n      logits. Set to True for image classification, False for dense prediction.\n    output_stride: If None, then the output will be computed at the nominal\n      network stride. If output_stride is not None, it specifies the requested\n      ratio of input to output spatial resolution.\n    include_root_block: If True, include the initial convolution followed by\n      max-pooling, if False excludes it.\n    spatial_squeeze: if True, logits is of shape [B, C], if false logits is\n        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.\n        To use this parameter, the input images must be smaller than 300x300\n        pixels, in which case the output logit layer does not contain spatial\n        information and can be removed.\n    store_non_strided_activations: If True, we compute non-strided (undecimated)\n      activations at the last unit of each block and store them in the\n      `outputs_collections` before subsampling them. This gives us access to\n      higher resolution intermediate activations which are useful in some\n      dense prediction problems but increases 4x the computation and memory cost\n      at the last unit of each block.\n    reuse: whether or not the network and its variables should be reused. To be\n      able to reuse 'scope' must be given.\n    scope: Optional variable_scope.\n\n  Returns:\n    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].\n      If global_pool is False, then height_out and width_out are reduced by a\n      factor of output_stride compared to the respective height_in and width_in,\n      else both height_out and width_out equal one. If num_classes is 0 or None,\n      then net is the output of the last ResNet block, potentially after global\n      average pooling. If num_classes a non-zero integer, net contains the\n      pre-softmax activations.\n    end_points: A dictionary from components of the network to the corresponding\n      activation.\n\n  Raises:\n    ValueError: If the target output_stride is not valid.\n  "
    with tf.variable_scope(scope, 'resnet_v1', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, resnet_utils.stack_blocks_dense], outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=is_training) if is_training is not None else NoOpScope():
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                net = resnet_utils.stack_blocks_dense(net, blocks, output_stride, store_non_strided_activations)
                end_points = slim.utils.convert_collection_to_dict(end_points_collection)
                if global_pool:
                    net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
                    end_points['global_pool'] = net
                if num_classes:
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
                    end_points[sc.name + '/logits'] = net
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                        end_points[sc.name + '/spatial_squeeze'] = net
                    end_points['predictions'] = slim.softmax(net, scope='predictions')
                return (net, end_points)
resnet_v1.default_image_size = 224

def resnet_v1_block(scope, base_depth, num_units, stride):
    if False:
        i = 10
        return i + 15
    'Helper function for creating a resnet_v1 bottleneck block.\n\n  Args:\n    scope: The scope of the block.\n    base_depth: The depth of the bottleneck layer for each unit.\n    num_units: The number of units in the block.\n    stride: The stride of the block, implemented as a stride in the last unit.\n      All other units have stride=1.\n\n  Returns:\n    A resnet_v1 bottleneck block.\n  '
    return resnet_utils.Block(scope, bottleneck, [{'depth': base_depth * 4, 'depth_bottleneck': base_depth, 'stride': 1}] * (num_units - 1) + [{'depth': base_depth * 4, 'depth_bottleneck': base_depth, 'stride': stride}])

def resnet_v1_50(inputs, num_classes=None, is_training=True, global_pool=True, output_stride=None, spatial_squeeze=True, store_non_strided_activations=False, min_base_depth=8, depth_multiplier=1, reuse=None, scope='resnet_v1_50'):
    if False:
        return 10
    'ResNet-50 model of [1]. See resnet_v1() for arg and return description.'
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    blocks = [resnet_v1_block('block1', base_depth=depth_func(64), num_units=3, stride=2), resnet_v1_block('block2', base_depth=depth_func(128), num_units=4, stride=2), resnet_v1_block('block3', base_depth=depth_func(256), num_units=6, stride=2), resnet_v1_block('block4', base_depth=depth_func(512), num_units=3, stride=1)]
    return resnet_v1(inputs, blocks, num_classes, is_training, global_pool=global_pool, output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze, store_non_strided_activations=store_non_strided_activations, reuse=reuse, scope=scope)
resnet_v1_50.default_image_size = resnet_v1.default_image_size

def resnet_v1_101(inputs, num_classes=None, is_training=True, global_pool=True, output_stride=None, spatial_squeeze=True, store_non_strided_activations=False, min_base_depth=8, depth_multiplier=1, reuse=None, scope='resnet_v1_101'):
    if False:
        i = 10
        return i + 15
    'ResNet-101 model of [1]. See resnet_v1() for arg and return description.'
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    blocks = [resnet_v1_block('block1', base_depth=depth_func(64), num_units=3, stride=2), resnet_v1_block('block2', base_depth=depth_func(128), num_units=4, stride=2), resnet_v1_block('block3', base_depth=depth_func(256), num_units=23, stride=2), resnet_v1_block('block4', base_depth=depth_func(512), num_units=3, stride=1)]
    return resnet_v1(inputs, blocks, num_classes, is_training, global_pool=global_pool, output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze, store_non_strided_activations=store_non_strided_activations, reuse=reuse, scope=scope)
resnet_v1_101.default_image_size = resnet_v1.default_image_size

def resnet_v1_152(inputs, num_classes=None, is_training=True, global_pool=True, output_stride=None, store_non_strided_activations=False, spatial_squeeze=True, min_base_depth=8, depth_multiplier=1, reuse=None, scope='resnet_v1_152'):
    if False:
        return 10
    'ResNet-152 model of [1]. See resnet_v1() for arg and return description.'
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    blocks = [resnet_v1_block('block1', base_depth=depth_func(64), num_units=3, stride=2), resnet_v1_block('block2', base_depth=depth_func(128), num_units=8, stride=2), resnet_v1_block('block3', base_depth=depth_func(256), num_units=36, stride=2), resnet_v1_block('block4', base_depth=depth_func(512), num_units=3, stride=1)]
    return resnet_v1(inputs, blocks, num_classes, is_training, global_pool=global_pool, output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze, store_non_strided_activations=store_non_strided_activations, reuse=reuse, scope=scope)
resnet_v1_152.default_image_size = resnet_v1.default_image_size

def resnet_v1_200(inputs, num_classes=None, is_training=True, global_pool=True, output_stride=None, store_non_strided_activations=False, spatial_squeeze=True, min_base_depth=8, depth_multiplier=1, reuse=None, scope='resnet_v1_200'):
    if False:
        while True:
            i = 10
    'ResNet-200 model of [2]. See resnet_v1() for arg and return description.'
    depth_func = lambda d: max(int(d * depth_multiplier), min_base_depth)
    blocks = [resnet_v1_block('block1', base_depth=depth_func(64), num_units=3, stride=2), resnet_v1_block('block2', base_depth=depth_func(128), num_units=24, stride=2), resnet_v1_block('block3', base_depth=depth_func(256), num_units=36, stride=2), resnet_v1_block('block4', base_depth=depth_func(512), num_units=3, stride=1)]
    return resnet_v1(inputs, blocks, num_classes, is_training, global_pool=global_pool, output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze, store_non_strided_activations=store_non_strided_activations, reuse=reuse, scope=scope)
resnet_v1_200.default_image_size = resnet_v1.default_image_size