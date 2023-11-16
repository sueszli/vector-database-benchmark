"""Dense Prediction Cell class that can be evolved in semantic segmentation.

DensePredictionCell is used as a `layer` in semantic segmentation whose
architecture is determined by the `config`, a dictionary specifying
the architecture.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from deeplab.core import utils
slim = contrib_slim
_META_ARCHITECTURE_SCOPE = 'meta_architecture'
_CONCAT_PROJECTION_SCOPE = 'concat_projection'
_OP = 'op'
_CONV = 'conv'
_PYRAMID_POOLING = 'pyramid_pooling'
_KERNEL = 'kernel'
_RATE = 'rate'
_GRID_SIZE = 'grid_size'
_TARGET_SIZE = 'target_size'
_INPUT = 'input'

def dense_prediction_cell_hparams():
    if False:
        i = 10
        return i + 15
    "DensePredictionCell HParams.\n\n  Returns:\n    A dictionary of hyper-parameters used for dense prediction cell with keys:\n      - reduction_size: Integer, the number of output filters for each operation\n          inside the cell.\n      - dropout_on_concat_features: Boolean, apply dropout on the concatenated\n          features or not.\n      - dropout_on_projection_features: Boolean, apply dropout on the projection\n          features or not.\n      - dropout_keep_prob: Float, when `dropout_on_concat_features' or\n          `dropout_on_projection_features' is True, the `keep_prob` value used\n          in the dropout operation.\n      - concat_channels: Integer, the concatenated features will be\n          channel-reduced to `concat_channels` channels.\n      - conv_rate_multiplier: Integer, used to multiply the convolution rates.\n          This is useful in the case when the output_stride is changed from 16\n          to 8, we need to double the convolution rates correspondingly.\n  "
    return {'reduction_size': 256, 'dropout_on_concat_features': True, 'dropout_on_projection_features': False, 'dropout_keep_prob': 0.9, 'concat_channels': 256, 'conv_rate_multiplier': 1}

class DensePredictionCell(object):
    """DensePredictionCell class used as a 'layer' in semantic segmentation."""

    def __init__(self, config, hparams=None):
        if False:
            i = 10
            return i + 15
        'Initializes the dense prediction cell.\n\n    Args:\n      config: A dictionary storing the architecture of a dense prediction cell.\n      hparams: A dictionary of hyper-parameters, provided by users. This\n        dictionary will be used to update the default dictionary returned by\n        dense_prediction_cell_hparams().\n\n    Raises:\n       ValueError: If `conv_rate_multiplier` has value < 1.\n    '
        self.hparams = dense_prediction_cell_hparams()
        if hparams is not None:
            self.hparams.update(hparams)
        self.config = config
        if self.hparams['conv_rate_multiplier'] < 1:
            raise ValueError('conv_rate_multiplier cannot have value < 1.')

    def _get_pyramid_pooling_arguments(self, crop_size, output_stride, image_grid, image_pooling_crop_size=None):
        if False:
            i = 10
            return i + 15
        'Gets arguments for pyramid pooling.\n\n    Args:\n      crop_size: A list of two integers, [crop_height, crop_width] specifying\n        whole patch crop size.\n      output_stride: Integer, output stride value for extracted features.\n      image_grid: A list of two integers, [image_grid_height, image_grid_width],\n        specifying the grid size of how the pyramid pooling will be performed.\n      image_pooling_crop_size: A list of two integers, [crop_height, crop_width]\n        specifying the crop size for image pooling operations. Note that we\n        decouple whole patch crop_size and image_pooling_crop_size as one could\n        perform the image_pooling with different crop sizes.\n\n    Returns:\n      A list of (resize_value, pooled_kernel)\n    '
        resize_height = utils.scale_dimension(crop_size[0], 1.0 / output_stride)
        resize_width = utils.scale_dimension(crop_size[1], 1.0 / output_stride)
        if image_pooling_crop_size is None:
            image_pooling_crop_size = crop_size
        pooled_height = utils.scale_dimension(image_pooling_crop_size[0], 1.0 / (output_stride * image_grid[0]))
        pooled_width = utils.scale_dimension(image_pooling_crop_size[1], 1.0 / (output_stride * image_grid[1]))
        return ([resize_height, resize_width], [pooled_height, pooled_width])

    def _parse_operation(self, config, crop_size, output_stride, image_pooling_crop_size=None):
        if False:
            while True:
                i = 10
        "Parses one operation.\n\n    When 'operation' is 'pyramid_pooling', we compute the required\n    hyper-parameters and save in config.\n\n    Args:\n      config: A dictionary storing required hyper-parameters for one\n        operation.\n      crop_size: A list of two integers, [crop_height, crop_width] specifying\n        whole patch crop size.\n      output_stride: Integer, output stride value for extracted features.\n      image_pooling_crop_size: A list of two integers, [crop_height, crop_width]\n        specifying the crop size for image pooling operations. Note that we\n        decouple whole patch crop_size and image_pooling_crop_size as one could\n        perform the image_pooling with different crop sizes.\n\n    Returns:\n      A dictionary stores the related information for the operation.\n    "
        if config[_OP] == _PYRAMID_POOLING:
            (config[_TARGET_SIZE], config[_KERNEL]) = self._get_pyramid_pooling_arguments(crop_size=crop_size, output_stride=output_stride, image_grid=config[_GRID_SIZE], image_pooling_crop_size=image_pooling_crop_size)
        return config

    def build_cell(self, features, output_stride=16, crop_size=None, image_pooling_crop_size=None, weight_decay=4e-05, reuse=None, is_training=False, fine_tune_batch_norm=False, scope=None):
        if False:
            while True:
                i = 10
        'Builds the dense prediction cell based on the config.\n\n    Args:\n      features: Input feature map of size [batch, height, width, channels].\n      output_stride: Int, output stride at which the features were extracted.\n      crop_size: A list [crop_height, crop_width], determining the input\n        features resolution.\n      image_pooling_crop_size: A list of two integers, [crop_height, crop_width]\n        specifying the crop size for image pooling operations. Note that we\n        decouple whole patch crop_size and image_pooling_crop_size as one could\n        perform the image_pooling with different crop sizes.\n      weight_decay: Float, the weight decay for model variables.\n      reuse: Reuse the model variables or not.\n      is_training: Boolean, is training or not.\n      fine_tune_batch_norm: Boolean, fine-tuning batch norm parameters or not.\n      scope: Optional string, specifying the variable scope.\n\n    Returns:\n      Features after passing through the constructed dense prediction cell with\n        shape = [batch, height, width, channels] where channels are determined\n        by `reduction_size` returned by dense_prediction_cell_hparams().\n\n    Raises:\n      ValueError: Use Convolution with kernel size not equal to 1x1 or 3x3 or\n        the operation is not recognized.\n    '
        batch_norm_params = {'is_training': is_training and fine_tune_batch_norm, 'decay': 0.9997, 'epsilon': 1e-05, 'scale': True}
        hparams = self.hparams
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], weights_regularizer=slim.l2_regularizer(weight_decay), activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm, padding='SAME', stride=1, reuse=reuse):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with tf.variable_scope(scope, _META_ARCHITECTURE_SCOPE, [features]):
                    depth = hparams['reduction_size']
                    branch_logits = []
                    for (i, current_config) in enumerate(self.config):
                        scope = 'branch%d' % i
                        current_config = self._parse_operation(config=current_config, crop_size=crop_size, output_stride=output_stride, image_pooling_crop_size=image_pooling_crop_size)
                        tf.logging.info(current_config)
                        if current_config[_INPUT] < 0:
                            operation_input = features
                        else:
                            operation_input = branch_logits[current_config[_INPUT]]
                        if current_config[_OP] == _CONV:
                            if current_config[_KERNEL] == [1, 1] or current_config[_KERNEL] == 1:
                                branch_logits.append(slim.conv2d(operation_input, depth, 1, scope=scope))
                            else:
                                conv_rate = [r * hparams['conv_rate_multiplier'] for r in current_config[_RATE]]
                                branch_logits.append(utils.split_separable_conv2d(operation_input, filters=depth, kernel_size=current_config[_KERNEL], rate=conv_rate, weight_decay=weight_decay, scope=scope))
                        elif current_config[_OP] == _PYRAMID_POOLING:
                            pooled_features = slim.avg_pool2d(operation_input, kernel_size=current_config[_KERNEL], stride=[1, 1], padding='VALID')
                            pooled_features = slim.conv2d(pooled_features, depth, 1, scope=scope)
                            pooled_features = tf.image.resize_bilinear(pooled_features, current_config[_TARGET_SIZE], align_corners=True)
                            resize_height = current_config[_TARGET_SIZE][0]
                            resize_width = current_config[_TARGET_SIZE][1]
                            if isinstance(resize_height, tf.Tensor):
                                resize_height = None
                            if isinstance(resize_width, tf.Tensor):
                                resize_width = None
                            pooled_features.set_shape([None, resize_height, resize_width, depth])
                            branch_logits.append(pooled_features)
                        else:
                            raise ValueError('Unrecognized operation.')
                    concat_logits = tf.concat(branch_logits, 3)
                    if self.hparams['dropout_on_concat_features']:
                        concat_logits = slim.dropout(concat_logits, keep_prob=self.hparams['dropout_keep_prob'], is_training=is_training, scope=_CONCAT_PROJECTION_SCOPE + '_dropout')
                    concat_logits = slim.conv2d(concat_logits, self.hparams['concat_channels'], 1, scope=_CONCAT_PROJECTION_SCOPE)
                    if self.hparams['dropout_on_projection_features']:
                        concat_logits = slim.dropout(concat_logits, keep_prob=self.hparams['dropout_keep_prob'], is_training=is_training, scope=_CONCAT_PROJECTION_SCOPE + '_dropout')
                    return concat_logits