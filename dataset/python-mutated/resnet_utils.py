"""Contains building blocks for various versions of Residual Networks.

Residual networks (ResNets) were proposed in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Deep Residual Learning for Image Recognition. arXiv:1512.03385, 2015

More variants were introduced in:
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
  Identity Mappings in Deep Residual Networks. arXiv: 1603.05027, 2016

We can obtain different ResNet variants by changing the network depth, width,
and form of residual unit. This module implements the infrastructure for
building them. Concrete ResNet units and full ResNet networks are implemented in
the accompanying resnet_v1.py and resnet_v2.py modules.

Compared to https://github.com/KaimingHe/deep-residual-networks, in the current
implementation we subsample the output activations in the last residual unit of
each block, instead of subsampling the input activations in the first residual
unit of each block. The two implementations give identical results but our
implementation is more memory efficient.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
slim = contrib_slim

class Block(collections.namedtuple('Block', ['scope', 'unit_fn', 'args'])):
    """A named tuple describing a ResNet block.

  Its parts are:
    scope: The scope of the `Block`.
    unit_fn: The ResNet unit function which takes as input a `Tensor` and
      returns another `Tensor` with the output of the ResNet unit.
    args: A list of length equal to the number of units in the `Block`. The list
      contains one (depth, depth_bottleneck, stride) tuple for each unit in the
      block to serve as argument to unit_fn.
  """

def subsample(inputs, factor, scope=None):
    if False:
        print('Hello World!')
    'Subsamples the input along the spatial dimensions.\n\n  Args:\n    inputs: A `Tensor` of size [batch, height_in, width_in, channels].\n    factor: The subsampling factor.\n    scope: Optional variable_scope.\n\n  Returns:\n    output: A `Tensor` of size [batch, height_out, width_out, channels] with the\n      input, either intact (if factor == 1) or subsampled (if factor > 1).\n  '
    if factor == 1:
        return inputs
    else:
        return slim.max_pool2d(inputs, [1, 1], stride=factor, scope=scope)

def conv2d_same(inputs, num_outputs, kernel_size, stride, rate=1, scope=None):
    if False:
        return 10
    "Strided 2-D convolution with 'SAME' padding.\n\n  When stride > 1, then we do explicit zero-padding, followed by conv2d with\n  'VALID' padding.\n\n  Note that\n\n     net = conv2d_same(inputs, num_outputs, 3, stride=stride)\n\n  is equivalent to\n\n     net = slim.conv2d(inputs, num_outputs, 3, stride=1, padding='SAME')\n     net = subsample(net, factor=stride)\n\n  whereas\n\n     net = slim.conv2d(inputs, num_outputs, 3, stride=stride, padding='SAME')\n\n  is different when the input's height or width is even, which is why we add the\n  current function. For more details, see ResnetUtilsTest.testConv2DSameEven().\n\n  Args:\n    inputs: A 4-D tensor of size [batch, height_in, width_in, channels].\n    num_outputs: An integer, the number of output filters.\n    kernel_size: An int with the kernel_size of the filters.\n    stride: An integer, the output stride.\n    rate: An integer, rate for atrous convolution.\n    scope: Scope.\n\n  Returns:\n    output: A 4-D tensor of size [batch, height_out, width_out, channels] with\n      the convolution output.\n  "
    if stride == 1:
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=1, rate=rate, padding='SAME', scope=scope)
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
        return slim.conv2d(inputs, num_outputs, kernel_size, stride=stride, rate=rate, padding='VALID', scope=scope)

@slim.add_arg_scope
def stack_blocks_dense(net, blocks, output_stride=None, store_non_strided_activations=False, outputs_collections=None):
    if False:
        print('Hello World!')
    "Stacks ResNet `Blocks` and controls output feature density.\n\n  First, this function creates scopes for the ResNet in the form of\n  'block_name/unit_1', 'block_name/unit_2', etc.\n\n  Second, this function allows the user to explicitly control the ResNet\n  output_stride, which is the ratio of the input to output spatial resolution.\n  This is useful for dense prediction tasks such as semantic segmentation or\n  object detection.\n\n  Most ResNets consist of 4 ResNet blocks and subsample the activations by a\n  factor of 2 when transitioning between consecutive ResNet blocks. This results\n  to a nominal ResNet output_stride equal to 8. If we set the output_stride to\n  half the nominal network stride (e.g., output_stride=4), then we compute\n  responses twice.\n\n  Control of the output feature density is implemented by atrous convolution.\n\n  Args:\n    net: A `Tensor` of size [batch, height, width, channels].\n    blocks: A list of length equal to the number of ResNet `Blocks`. Each\n      element is a ResNet `Block` object describing the units in the `Block`.\n    output_stride: If `None`, then the output will be computed at the nominal\n      network stride. If output_stride is not `None`, it specifies the requested\n      ratio of input to output spatial resolution, which needs to be equal to\n      the product of unit strides from the start up to some level of the ResNet.\n      For example, if the ResNet employs units with strides 1, 2, 1, 3, 4, 1,\n      then valid values for the output_stride are 1, 2, 6, 24 or None (which\n      is equivalent to output_stride=24).\n    store_non_strided_activations: If True, we compute non-strided (undecimated)\n      activations at the last unit of each block and store them in the\n      `outputs_collections` before subsampling them. This gives us access to\n      higher resolution intermediate activations which are useful in some\n      dense prediction problems but increases 4x the computation and memory cost\n      at the last unit of each block.\n    outputs_collections: Collection to add the ResNet block outputs.\n\n  Returns:\n    net: Output tensor with stride equal to the specified output_stride.\n\n  Raises:\n    ValueError: If the target output_stride is not valid.\n  "
    current_stride = 1
    rate = 1
    for block in blocks:
        with tf.variable_scope(block.scope, 'block', [net]) as sc:
            block_stride = 1
            for (i, unit) in enumerate(block.args):
                if store_non_strided_activations and i == len(block.args) - 1:
                    block_stride = unit.get('stride', 1)
                    unit = dict(unit, stride=1)
                with tf.variable_scope('unit_%d' % (i + 1), values=[net]):
                    if output_stride is not None and current_stride == output_stride:
                        net = block.unit_fn(net, rate=rate, **dict(unit, stride=1))
                        rate *= unit.get('stride', 1)
                    else:
                        net = block.unit_fn(net, rate=1, **unit)
                        current_stride *= unit.get('stride', 1)
                        if output_stride is not None and current_stride > output_stride:
                            raise ValueError('The target output_stride cannot be reached.')
            net = slim.utils.collect_named_outputs(outputs_collections, sc.name, net)
            if output_stride is not None and current_stride == output_stride:
                rate *= block_stride
            else:
                net = subsample(net, block_stride)
                current_stride *= block_stride
                if output_stride is not None and current_stride > output_stride:
                    raise ValueError('The target output_stride cannot be reached.')
    if output_stride is not None and current_stride != output_stride:
        raise ValueError('The target output_stride cannot be reached.')
    return net

def resnet_arg_scope(weight_decay=0.0001, batch_norm_decay=0.997, batch_norm_epsilon=1e-05, batch_norm_scale=True, activation_fn=tf.nn.relu, use_batch_norm=True, batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):
    if False:
        for i in range(10):
            print('nop')
    'Defines the default ResNet arg scope.\n\n  TODO(gpapan): The batch-normalization related default values above are\n    appropriate for use in conjunction with the reference ResNet models\n    released at https://github.com/KaimingHe/deep-residual-networks. When\n    training ResNets from scratch, they might need to be tuned.\n\n  Args:\n    weight_decay: The weight decay to use for regularizing the model.\n    batch_norm_decay: The moving average decay when estimating layer activation\n      statistics in batch normalization.\n    batch_norm_epsilon: Small constant to prevent division by zero when\n      normalizing activations by their variance in batch normalization.\n    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the\n      activations in the batch normalization layer.\n    activation_fn: The activation function which is used in ResNet.\n    use_batch_norm: Whether or not to use batch normalization.\n    batch_norm_updates_collections: Collection for the update ops for\n      batch norm.\n\n  Returns:\n    An `arg_scope` to use for the resnet models.\n  '
    batch_norm_params = {'decay': batch_norm_decay, 'epsilon': batch_norm_epsilon, 'scale': batch_norm_scale, 'updates_collections': batch_norm_updates_collections, 'fused': None}
    with slim.arg_scope([slim.conv2d], weights_regularizer=slim.l2_regularizer(weight_decay), weights_initializer=slim.variance_scaling_initializer(), activation_fn=activation_fn, normalizer_fn=slim.batch_norm if use_batch_norm else None, normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc