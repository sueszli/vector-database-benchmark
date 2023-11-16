"""RFCN Box Predictor."""
import tensorflow as tf
from object_detection.core import box_predictor
from object_detection.utils import ops
BOX_ENCODINGS = box_predictor.BOX_ENCODINGS
CLASS_PREDICTIONS_WITH_BACKGROUND = box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND
MASK_PREDICTIONS = box_predictor.MASK_PREDICTIONS

class RfcnKerasBoxPredictor(box_predictor.KerasBoxPredictor):
    """RFCN Box Predictor.

  Applies a position sensitive ROI pooling on position sensitive feature maps to
  predict classes and refined locations. See https://arxiv.org/abs/1605.06409
  for details.

  This is used for the second stage of the RFCN meta architecture. Notice that
  locations are *not* shared across classes, thus for each anchor, a separate
  prediction is made for each class.
  """

    def __init__(self, is_training, num_classes, conv_hyperparams, freeze_batchnorm, num_spatial_bins, depth, crop_size, box_code_size, name=None):
        if False:
            for i in range(10):
                print('nop')
        'Constructor.\n\n    Args:\n      is_training: Indicates whether the BoxPredictor is in training mode.\n      num_classes: number of classes.  Note that num_classes *does not*\n        include the background category, so if groundtruth labels take values\n        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the\n        assigned classification targets can range from {0,... K}).\n      conv_hyperparams: A `hyperparams_builder.KerasLayerHyperparams` object\n        containing hyperparameters for convolution ops.\n      freeze_batchnorm: Whether to freeze batch norm parameters during\n        training or not. When training with a small batch size (e.g. 1), it is\n        desirable to freeze batch norm update and use pretrained batch norm\n        params.\n      num_spatial_bins: A list of two integers `[spatial_bins_y,\n        spatial_bins_x]`.\n      depth: Target depth to reduce the input feature maps to.\n      crop_size: A list of two integers `[crop_height, crop_width]`.\n      box_code_size: Size of encoding for each box.\n      name: A string name scope to assign to the box predictor. If `None`, Keras\n        will auto-generate one from the class name.\n    '
        super(RfcnKerasBoxPredictor, self).__init__(is_training, num_classes, freeze_batchnorm=freeze_batchnorm, inplace_batchnorm_update=False, name=name)
        self._freeze_batchnorm = freeze_batchnorm
        self._conv_hyperparams = conv_hyperparams
        self._num_spatial_bins = num_spatial_bins
        self._depth = depth
        self._crop_size = crop_size
        self._box_code_size = box_code_size
        self._shared_conv_layers = []
        self._shared_conv_layers.append(tf.keras.layers.Conv2D(self._depth, [1, 1], padding='SAME', name='reduce_depth_conv', **self._conv_hyperparams.params()))
        self._shared_conv_layers.append(self._conv_hyperparams.build_batch_norm(training=self._is_training and (not self._freeze_batchnorm), name='reduce_depth_batchnorm'))
        self._shared_conv_layers.append(self._conv_hyperparams.build_activation_layer(name='reduce_depth_activation'))
        self._box_encoder_layers = []
        location_feature_map_depth = self._num_spatial_bins[0] * self._num_spatial_bins[1] * self.num_classes * self._box_code_size
        self._box_encoder_layers.append(tf.keras.layers.Conv2D(location_feature_map_depth, [1, 1], padding='SAME', name='refined_locations_conv', **self._conv_hyperparams.params()))
        self._box_encoder_layers.append(self._conv_hyperparams.build_batch_norm(training=self._is_training and (not self._freeze_batchnorm), name='refined_locations_batchnorm'))
        self._class_predictor_layers = []
        self._total_classes = self.num_classes + 1
        class_feature_map_depth = self._num_spatial_bins[0] * self._num_spatial_bins[1] * self._total_classes
        self._class_predictor_layers.append(tf.keras.layers.Conv2D(class_feature_map_depth, [1, 1], padding='SAME', name='class_predictions_conv', **self._conv_hyperparams.params()))
        self._class_predictor_layers.append(self._conv_hyperparams.build_batch_norm(training=self._is_training and (not self._freeze_batchnorm), name='class_predictions_batchnorm'))

    @property
    def num_classes(self):
        if False:
            return 10
        return self._num_classes

    def _predict(self, image_features, proposal_boxes, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Computes encoded object locations and corresponding confidences.\n\n    Args:\n      image_features: A list of float tensors of shape [batch_size, height_i,\n      width_i, channels_i] containing features for a batch of images.\n      proposal_boxes: A float tensor of shape [batch_size, num_proposals,\n        box_code_size].\n      **kwargs: Unused Keyword args\n\n    Returns:\n      box_encodings: A list of float tensors of shape\n        [batch_size, num_anchors_i, q, code_size] representing the location of\n        the objects, where q is 1 or the number of classes. Each entry in the\n        list corresponds to a feature map in the input `image_features` list.\n      class_predictions_with_background: A list of float tensors of shape\n        [batch_size, num_anchors_i, num_classes + 1] representing the class\n        predictions for the proposals. Each entry in the list corresponds to a\n        feature map in the input `image_features` list.\n\n    Raises:\n      ValueError: if num_predictions_per_location is not 1 or if\n        len(image_features) is not 1.\n    '
        if len(image_features) != 1:
            raise ValueError('length of `image_features` must be 1. Found {}'.format(len(image_features)))
        image_feature = image_features[0]
        batch_size = tf.shape(proposal_boxes)[0]
        num_boxes = tf.shape(proposal_boxes)[1]
        net = image_feature
        for layer in self._shared_conv_layers:
            net = layer(net)
        box_net = net
        for layer in self._box_encoder_layers:
            box_net = layer(box_net)
        box_encodings = ops.batch_position_sensitive_crop_regions(box_net, boxes=proposal_boxes, crop_size=self._crop_size, num_spatial_bins=self._num_spatial_bins, global_pool=True)
        box_encodings = tf.squeeze(box_encodings, axis=[2, 3])
        box_encodings = tf.reshape(box_encodings, [batch_size * num_boxes, 1, self.num_classes, self._box_code_size])
        class_net = net
        for layer in self._class_predictor_layers:
            class_net = layer(class_net)
        class_predictions_with_background = ops.batch_position_sensitive_crop_regions(class_net, boxes=proposal_boxes, crop_size=self._crop_size, num_spatial_bins=self._num_spatial_bins, global_pool=True)
        class_predictions_with_background = tf.squeeze(class_predictions_with_background, axis=[2, 3])
        class_predictions_with_background = tf.reshape(class_predictions_with_background, [batch_size * num_boxes, 1, self._total_classes])
        return {BOX_ENCODINGS: [box_encodings], CLASS_PREDICTIONS_WITH_BACKGROUND: [class_predictions_with_background]}