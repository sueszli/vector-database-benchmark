"""Contains definitions for the preactivation form of Residual Networks.

Residual networks (ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant implemented in this module was
introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer.

Typical use:

   from tensorflow.contrib.slim.nets import resnet_v2

ResNet-101 for image classification into 1000 classes:

   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      net, end_points = resnet_v2.resnet_v2_101(inputs, 1000, is_training=False)

ResNet-101 for semantic segmentation into 21 classes:

   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v2.resnet_arg_scope()):
      net, end_points = resnet_v2.resnet_v2_101(inputs,
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
slim = contrib_slim
resnet_arg_scope = resnet_utils.resnet_arg_scope

@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1, outputs_collections=None, scope=None):
    if False:
        return 10
    "Bottleneck residual unit variant with BN before convolutions.\n\n  This is the full preactivation residual unit variant proposed in [2]. See\n  Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck\n  variant which has an extra bottleneck layer.\n\n  When putting together two consecutive ResNet blocks that use this unit, one\n  should use stride = 2 in the last unit of the first block.\n\n  Args:\n    inputs: A tensor of size [batch, height, width, channels].\n    depth: The depth of the ResNet unit output.\n    depth_bottleneck: The depth of the bottleneck layers.\n    stride: The ResNet unit's stride. Determines the amount of downsampling of\n      the units output compared to its input.\n    rate: An integer, rate for atrous convolution.\n    outputs_collections: Collection to add the ResNet unit output.\n    scope: Optional variable_scope.\n\n  Returns:\n    The ResNet unit's output.\n  "
    with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride, normalizer_fn=None, activation_fn=None, scope='shortcut')
        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, normalizer_fn=None, activation_fn=None, scope='conv3')
        output = shortcut + residual
        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)

def resnet_v2(inputs, blocks, num_classes=None, is_training=True, global_pool=True, output_stride=None, include_root_block=True, spatial_squeeze=True, reuse=None, scope=None):
    if False:
        print('Hello World!')
    "Generator for v2 (preactivation) ResNet models.\n\n  This function generates a family of ResNet v2 models. See the resnet_v2_*()\n  methods for specific model instantiations, obtained by selecting different\n  block instantiations that produce ResNets of various depths.\n\n  Training for image classification on Imagenet is usually done with [224, 224]\n  inputs, resulting in [7, 7] feature maps at the output of the last ResNet\n  block for the ResNets defined in [1] that have nominal stride equal to 32.\n  However, for dense prediction tasks we advise that one uses inputs with\n  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In\n  this case the feature maps at the ResNet output will have spatial shape\n  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]\n  and corners exactly aligned with the input image corners, which greatly\n  facilitates alignment of the features to the image. Using as input [225, 225]\n  images results in [8, 8] feature maps at the output of the last ResNet block.\n\n  For dense prediction tasks, the ResNet needs to run in fully-convolutional\n  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all\n  have nominal stride equal to 32 and a good choice in FCN mode is to use\n  output_stride=16 in order to increase the density of the computed features at\n  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.\n\n  Args:\n    inputs: A tensor of size [batch, height_in, width_in, channels].\n    blocks: A list of length equal to the number of ResNet blocks. Each element\n      is a resnet_utils.Block object describing the units in the block.\n    num_classes: Number of predicted classes for classification tasks.\n      If 0 or None, we return the features before the logit layer.\n    is_training: whether batch_norm layers are in training mode.\n    global_pool: If True, we perform global average pooling before computing the\n      logits. Set to True for image classification, False for dense prediction.\n    output_stride: If None, then the output will be computed at the nominal\n      network stride. If output_stride is not None, it specifies the requested\n      ratio of input to output spatial resolution.\n    include_root_block: If True, include the initial convolution followed by\n      max-pooling, if False excludes it. If excluded, `inputs` should be the\n      results of an activation-less convolution.\n    spatial_squeeze: if True, logits is of shape [B, C], if false logits is\n        of shape [B, 1, 1, C], where B is batch_size and C is number of classes.\n        To use this parameter, the input images must be smaller than 300x300\n        pixels, in which case the output logit layer does not contain spatial\n        information and can be removed.\n    reuse: whether or not the network and its variables should be reused. To be\n      able to reuse 'scope' must be given.\n    scope: Optional variable_scope.\n\n\n  Returns:\n    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].\n      If global_pool is False, then height_out and width_out are reduced by a\n      factor of output_stride compared to the respective height_in and width_in,\n      else both height_out and width_out equal one. If num_classes is 0 or None,\n      then net is the output of the last ResNet block, potentially after global\n      average pooling. If num_classes is a non-zero integer, net contains the\n      pre-softmax activations.\n    end_points: A dictionary from components of the network to the corresponding\n      activation.\n\n  Raises:\n    ValueError: If the target output_stride is not valid.\n  "
    with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck, resnet_utils.stack_blocks_dense], outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                        net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
                net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
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
resnet_v2.default_image_size = 224

def resnet_v2_block(scope, base_depth, num_units, stride):
    if False:
        return 10
    'Helper function for creating a resnet_v2 bottleneck block.\n\n  Args:\n    scope: The scope of the block.\n    base_depth: The depth of the bottleneck layer for each unit.\n    num_units: The number of units in the block.\n    stride: The stride of the block, implemented as a stride in the last unit.\n      All other units have stride=1.\n\n  Returns:\n    A resnet_v2 bottleneck block.\n  '
    return resnet_utils.Block(scope, bottleneck, [{'depth': base_depth * 4, 'depth_bottleneck': base_depth, 'stride': 1}] * (num_units - 1) + [{'depth': base_depth * 4, 'depth_bottleneck': base_depth, 'stride': stride}])
resnet_v2.default_image_size = 224

def resnet_v2_50(inputs, num_classes=None, is_training=True, global_pool=True, output_stride=None, spatial_squeeze=True, reuse=None, scope='resnet_v2_50'):
    if False:
        for i in range(10):
            print('nop')
    'ResNet-50 model of [1]. See resnet_v2() for arg and return description.'
    blocks = [resnet_v2_block('block1', base_depth=64, num_units=3, stride=2), resnet_v2_block('block2', base_depth=128, num_units=4, stride=2), resnet_v2_block('block3', base_depth=256, num_units=6, stride=2), resnet_v2_block('block4', base_depth=512, num_units=3, stride=1)]
    return resnet_v2(inputs, blocks, num_classes, is_training=is_training, global_pool=global_pool, output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze, reuse=reuse, scope=scope)
resnet_v2_50.default_image_size = resnet_v2.default_image_size

def resnet_v2_101(inputs, num_classes=None, is_training=True, global_pool=True, output_stride=None, spatial_squeeze=True, reuse=None, scope='resnet_v2_101'):
    if False:
        for i in range(10):
            print('nop')
    'ResNet-101 model of [1]. See resnet_v2() for arg and return description.'
    blocks = [resnet_v2_block('block1', base_depth=64, num_units=3, stride=2), resnet_v2_block('block2', base_depth=128, num_units=4, stride=2), resnet_v2_block('block3', base_depth=256, num_units=23, stride=2), resnet_v2_block('block4', base_depth=512, num_units=3, stride=1)]
    return resnet_v2(inputs, blocks, num_classes, is_training=is_training, global_pool=global_pool, output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze, reuse=reuse, scope=scope)
resnet_v2_101.default_image_size = resnet_v2.default_image_size

def resnet_v2_152(inputs, num_classes=None, is_training=True, global_pool=True, output_stride=None, spatial_squeeze=True, reuse=None, scope='resnet_v2_152'):
    if False:
        for i in range(10):
            print('nop')
    'ResNet-152 model of [1]. See resnet_v2() for arg and return description.'
    blocks = [resnet_v2_block('block1', base_depth=64, num_units=3, stride=2), resnet_v2_block('block2', base_depth=128, num_units=8, stride=2), resnet_v2_block('block3', base_depth=256, num_units=36, stride=2), resnet_v2_block('block4', base_depth=512, num_units=3, stride=1)]
    return resnet_v2(inputs, blocks, num_classes, is_training=is_training, global_pool=global_pool, output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze, reuse=reuse, scope=scope)
resnet_v2_152.default_image_size = resnet_v2.default_image_size

def resnet_v2_200(inputs, num_classes=None, is_training=True, global_pool=True, output_stride=None, spatial_squeeze=True, reuse=None, scope='resnet_v2_200'):
    if False:
        i = 10
        return i + 15
    'ResNet-200 model of [2]. See resnet_v2() for arg and return description.'
    blocks = [resnet_v2_block('block1', base_depth=64, num_units=3, stride=2), resnet_v2_block('block2', base_depth=128, num_units=24, stride=2), resnet_v2_block('block3', base_depth=256, num_units=36, stride=2), resnet_v2_block('block4', base_depth=512, num_units=3, stride=1)]
    return resnet_v2(inputs, blocks, num_classes, is_training=is_training, global_pool=global_pool, output_stride=output_stride, include_root_block=True, spatial_squeeze=spatial_squeeze, reuse=reuse, scope=scope)
resnet_v2_200.default_image_size = resnet_v2.default_image_size