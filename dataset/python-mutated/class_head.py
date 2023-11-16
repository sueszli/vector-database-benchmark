"""Class Head.

Contains Class prediction head classes for different meta architectures.
All the class prediction heads have a predict function that receives the
`features` as the first argument and returns class predictions with background.
"""
import functools
import tensorflow as tf
from tensorflow.contrib import slim as contrib_slim
from object_detection.predictors.heads import head
slim = contrib_slim

class MaskRCNNClassHead(head.Head):
    """Mask RCNN class prediction head.

  Please refer to Mask RCNN paper:
  https://arxiv.org/abs/1703.06870
  """

    def __init__(self, is_training, num_class_slots, fc_hyperparams_fn, use_dropout, dropout_keep_prob, scope='ClassPredictor'):
        if False:
            while True:
                i = 10
        'Constructor.\n\n    Args:\n      is_training: Indicates whether the BoxPredictor is in training mode.\n      num_class_slots: number of class slots. Note that num_class_slots may or\n        may not include an implicit background category.\n      fc_hyperparams_fn: A function to generate tf-slim arg_scope with\n        hyperparameters for fully connected ops.\n      use_dropout: Option to use dropout or not.  Note that a single dropout\n        op is applied here prior to both box and class predictions, which stands\n        in contrast to the ConvolutionalBoxPredictor below.\n      dropout_keep_prob: Keep probability for dropout.\n        This is only used if use_dropout is True.\n      scope: Scope name for the convolution operation.\n    '
        super(MaskRCNNClassHead, self).__init__()
        self._is_training = is_training
        self._num_class_slots = num_class_slots
        self._fc_hyperparams_fn = fc_hyperparams_fn
        self._use_dropout = use_dropout
        self._dropout_keep_prob = dropout_keep_prob
        self._scope = scope

    def predict(self, features, num_predictions_per_location=1):
        if False:
            return 10
        'Predicts boxes and class scores.\n\n    Args:\n      features: A float tensor of shape [batch_size, height, width, channels]\n        containing features for a batch of images.\n      num_predictions_per_location: Int containing number of predictions per\n        location.\n\n    Returns:\n      class_predictions_with_background: A float tensor of shape\n        [batch_size, 1, num_class_slots] representing the class predictions for\n        the proposals.\n\n    Raises:\n      ValueError: If num_predictions_per_location is not 1.\n    '
        if num_predictions_per_location != 1:
            raise ValueError('Only num_predictions_per_location=1 is supported')
        spatial_averaged_roi_pooled_features = tf.reduce_mean(features, [1, 2], keep_dims=True, name='AvgPool')
        flattened_roi_pooled_features = slim.flatten(spatial_averaged_roi_pooled_features)
        if self._use_dropout:
            flattened_roi_pooled_features = slim.dropout(flattened_roi_pooled_features, keep_prob=self._dropout_keep_prob, is_training=self._is_training)
        with slim.arg_scope(self._fc_hyperparams_fn()):
            class_predictions_with_background = slim.fully_connected(flattened_roi_pooled_features, self._num_class_slots, activation_fn=None, scope=self._scope)
        class_predictions_with_background = tf.reshape(class_predictions_with_background, [-1, 1, self._num_class_slots])
        return class_predictions_with_background

class ConvolutionalClassHead(head.Head):
    """Convolutional class prediction head."""

    def __init__(self, is_training, num_class_slots, use_dropout, dropout_keep_prob, kernel_size, apply_sigmoid_to_scores=False, class_prediction_bias_init=0.0, use_depthwise=False, scope='ClassPredictor'):
        if False:
            print('Hello World!')
        'Constructor.\n\n    Args:\n      is_training: Indicates whether the BoxPredictor is in training mode.\n      num_class_slots: number of class slots. Note that num_class_slots may or\n        may not include an implicit background category.\n      use_dropout: Option to use dropout or not.  Note that a single dropout\n        op is applied here prior to both box and class predictions, which stands\n        in contrast to the ConvolutionalBoxPredictor below.\n      dropout_keep_prob: Keep probability for dropout.\n        This is only used if use_dropout is True.\n      kernel_size: Size of final convolution kernel.  If the\n        spatial resolution of the feature map is smaller than the kernel size,\n        then the kernel size is automatically set to be\n        min(feature_width, feature_height).\n      apply_sigmoid_to_scores: if True, apply the sigmoid on the output\n        class_predictions.\n      class_prediction_bias_init: constant value to initialize bias of the last\n        conv2d layer before class prediction.\n      use_depthwise: Whether to use depthwise convolutions for prediction\n        steps. Default is False.\n      scope: Scope name for the convolution operation.\n\n    Raises:\n      ValueError: if min_depth > max_depth.\n      ValueError: if use_depthwise is True and kernel_size is 1.\n    '
        if use_depthwise and kernel_size == 1:
            raise ValueError('Should not use 1x1 kernel when using depthwise conv')
        super(ConvolutionalClassHead, self).__init__()
        self._is_training = is_training
        self._num_class_slots = num_class_slots
        self._use_dropout = use_dropout
        self._dropout_keep_prob = dropout_keep_prob
        self._kernel_size = kernel_size
        self._apply_sigmoid_to_scores = apply_sigmoid_to_scores
        self._class_prediction_bias_init = class_prediction_bias_init
        self._use_depthwise = use_depthwise
        self._scope = scope

    def predict(self, features, num_predictions_per_location):
        if False:
            while True:
                i = 10
        'Predicts boxes.\n\n    Args:\n      features: A float tensor of shape [batch_size, height, width, channels]\n        containing image features.\n      num_predictions_per_location: Number of box predictions to be made per\n        spatial location.\n\n    Returns:\n      class_predictions_with_background: A float tensors of shape\n        [batch_size, num_anchors, num_class_slots] representing the class\n        predictions for the proposals.\n    '
        net = features
        if self._use_dropout:
            net = slim.dropout(net, keep_prob=self._dropout_keep_prob)
        if self._use_depthwise:
            depthwise_scope = self._scope + '_depthwise'
            class_predictions_with_background = slim.separable_conv2d(net, None, [self._kernel_size, self._kernel_size], padding='SAME', depth_multiplier=1, stride=1, rate=1, scope=depthwise_scope)
            class_predictions_with_background = slim.conv2d(class_predictions_with_background, num_predictions_per_location * self._num_class_slots, [1, 1], activation_fn=None, normalizer_fn=None, normalizer_params=None, scope=self._scope)
        else:
            class_predictions_with_background = slim.conv2d(net, num_predictions_per_location * self._num_class_slots, [self._kernel_size, self._kernel_size], activation_fn=None, normalizer_fn=None, normalizer_params=None, scope=self._scope, biases_initializer=tf.constant_initializer(self._class_prediction_bias_init))
        if self._apply_sigmoid_to_scores:
            class_predictions_with_background = tf.sigmoid(class_predictions_with_background)
        batch_size = features.get_shape().as_list()[0]
        if batch_size is None:
            batch_size = tf.shape(features)[0]
        class_predictions_with_background = tf.reshape(class_predictions_with_background, [batch_size, -1, self._num_class_slots])
        return class_predictions_with_background

class WeightSharedConvolutionalClassHead(head.Head):
    """Weight shared convolutional class prediction head.

  This head allows sharing the same set of parameters (weights) when called more
  then once on different feature maps.
  """

    def __init__(self, num_class_slots, kernel_size=3, class_prediction_bias_init=0.0, use_dropout=False, dropout_keep_prob=0.8, use_depthwise=False, score_converter_fn=tf.identity, return_flat_predictions=True, scope='ClassPredictor'):
        if False:
            return 10
        'Constructor.\n\n    Args:\n      num_class_slots: number of class slots. Note that num_class_slots may or\n        may not include an implicit background category.\n      kernel_size: Size of final convolution kernel.\n      class_prediction_bias_init: constant value to initialize bias of the last\n        conv2d layer before class prediction.\n      use_dropout: Whether to apply dropout to class prediction head.\n      dropout_keep_prob: Probability of keeping activiations.\n      use_depthwise: Whether to use depthwise convolutions for prediction\n        steps. Default is False.\n      score_converter_fn: Callable elementwise nonlinearity (that takes tensors\n        as inputs and returns tensors).\n      return_flat_predictions: If true, returns flattened prediction tensor\n        of shape [batch, height * width * num_predictions_per_location,\n        box_coder]. Otherwise returns the prediction tensor before reshaping,\n        whose shape is [batch, height, width, num_predictions_per_location *\n        num_class_slots].\n      scope: Scope name for the convolution operation.\n\n    Raises:\n      ValueError: if use_depthwise is True and kernel_size is 1.\n    '
        if use_depthwise and kernel_size == 1:
            raise ValueError('Should not use 1x1 kernel when using depthwise conv')
        super(WeightSharedConvolutionalClassHead, self).__init__()
        self._num_class_slots = num_class_slots
        self._kernel_size = kernel_size
        self._class_prediction_bias_init = class_prediction_bias_init
        self._use_dropout = use_dropout
        self._dropout_keep_prob = dropout_keep_prob
        self._use_depthwise = use_depthwise
        self._score_converter_fn = score_converter_fn
        self._return_flat_predictions = return_flat_predictions
        self._scope = scope

    def predict(self, features, num_predictions_per_location):
        if False:
            print('Hello World!')
        'Predicts boxes.\n\n    Args:\n      features: A float tensor of shape [batch_size, height, width, channels]\n        containing image features.\n      num_predictions_per_location: Number of box predictions to be made per\n        spatial location.\n\n    Returns:\n      class_predictions_with_background: A tensor of shape\n        [batch_size, num_anchors, num_class_slots] representing the class\n        predictions for the proposals, or a tensor of shape [batch, height,\n        width, num_predictions_per_location * num_class_slots] representing\n        class predictions before reshaping if self._return_flat_predictions is\n        False.\n    '
        class_predictions_net = features
        if self._use_dropout:
            class_predictions_net = slim.dropout(class_predictions_net, keep_prob=self._dropout_keep_prob)
        if self._use_depthwise:
            conv_op = functools.partial(slim.separable_conv2d, depth_multiplier=1)
        else:
            conv_op = slim.conv2d
        class_predictions_with_background = conv_op(class_predictions_net, num_predictions_per_location * self._num_class_slots, [self._kernel_size, self._kernel_size], activation_fn=None, stride=1, padding='SAME', normalizer_fn=None, biases_initializer=tf.constant_initializer(self._class_prediction_bias_init), scope=self._scope)
        batch_size = features.get_shape().as_list()[0]
        if batch_size is None:
            batch_size = tf.shape(features)[0]
        class_predictions_with_background = self._score_converter_fn(class_predictions_with_background)
        if self._return_flat_predictions:
            class_predictions_with_background = tf.reshape(class_predictions_with_background, [batch_size, -1, self._num_class_slots])
        return class_predictions_with_background