"""RFCN Box Predictor."""
import tensorflow as tf
from object_detection.core import box_predictor
from object_detection.utils import ops
slim = tf.contrib.slim
BOX_ENCODINGS = box_predictor.BOX_ENCODINGS
CLASS_PREDICTIONS_WITH_BACKGROUND = box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND
MASK_PREDICTIONS = box_predictor.MASK_PREDICTIONS

class RfcnBoxPredictor(box_predictor.BoxPredictor):
    """RFCN Box Predictor.

  Applies a position sensitive ROI pooling on position sensitive feature maps to
  predict classes and refined locations. See https://arxiv.org/abs/1605.06409
  for details.

  This is used for the second stage of the RFCN meta architecture. Notice that
  locations are *not* shared across classes, thus for each anchor, a separate
  prediction is made for each class.
  """

    def __init__(self, is_training, num_classes, conv_hyperparams_fn, num_spatial_bins, depth, crop_size, box_code_size):
        if False:
            print('Hello World!')
        'Constructor.\n\n    Args:\n      is_training: Indicates whether the BoxPredictor is in training mode.\n      num_classes: number of classes.  Note that num_classes *does not*\n        include the background category, so if groundtruth labels take values\n        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the\n        assigned classification targets can range from {0,... K}).\n      conv_hyperparams_fn: A function to construct tf-slim arg_scope with\n        hyperparameters for convolutional layers.\n      num_spatial_bins: A list of two integers `[spatial_bins_y,\n        spatial_bins_x]`.\n      depth: Target depth to reduce the input feature maps to.\n      crop_size: A list of two integers `[crop_height, crop_width]`.\n      box_code_size: Size of encoding for each box.\n    '
        super(RfcnBoxPredictor, self).__init__(is_training, num_classes)
        self._conv_hyperparams_fn = conv_hyperparams_fn
        self._num_spatial_bins = num_spatial_bins
        self._depth = depth
        self._crop_size = crop_size
        self._box_code_size = box_code_size

    @property
    def num_classes(self):
        if False:
            return 10
        return self._num_classes

    def _predict(self, image_features, num_predictions_per_location, proposal_boxes):
        if False:
            print('Hello World!')
        'Computes encoded object locations and corresponding confidences.\n\n    Args:\n      image_features: A list of float tensors of shape [batch_size, height_i,\n      width_i, channels_i] containing features for a batch of images.\n      num_predictions_per_location: A list of integers representing the number\n        of box predictions to be made per spatial location for each feature map.\n        Currently, this must be set to [1], or an error will be raised.\n      proposal_boxes: A float tensor of shape [batch_size, num_proposals,\n        box_code_size].\n\n    Returns:\n      box_encodings: A list of float tensors of shape\n        [batch_size, num_anchors_i, q, code_size] representing the location of\n        the objects, where q is 1 or the number of classes. Each entry in the\n        list corresponds to a feature map in the input `image_features` list.\n      class_predictions_with_background: A list of float tensors of shape\n        [batch_size, num_anchors_i, num_classes + 1] representing the class\n        predictions for the proposals. Each entry in the list corresponds to a\n        feature map in the input `image_features` list.\n\n    Raises:\n      ValueError: if num_predictions_per_location is not 1 or if\n        len(image_features) is not 1.\n    '
        if len(num_predictions_per_location) != 1 or num_predictions_per_location[0] != 1:
            raise ValueError('Currently RfcnBoxPredictor only supports predicting a single box per class per location.')
        if len(image_features) != 1:
            raise ValueError('length of `image_features` must be 1. Found {}'.format(len(image_features)))
        image_feature = image_features[0]
        num_predictions_per_location = num_predictions_per_location[0]
        batch_size = tf.shape(proposal_boxes)[0]
        num_boxes = tf.shape(proposal_boxes)[1]
        net = image_feature
        with slim.arg_scope(self._conv_hyperparams_fn()):
            net = slim.conv2d(net, self._depth, [1, 1], scope='reduce_depth')
            location_feature_map_depth = self._num_spatial_bins[0] * self._num_spatial_bins[1] * self.num_classes * self._box_code_size
            location_feature_map = slim.conv2d(net, location_feature_map_depth, [1, 1], activation_fn=None, scope='refined_locations')
            box_encodings = ops.batch_position_sensitive_crop_regions(location_feature_map, boxes=proposal_boxes, crop_size=self._crop_size, num_spatial_bins=self._num_spatial_bins, global_pool=True)
            box_encodings = tf.squeeze(box_encodings, axis=[2, 3])
            box_encodings = tf.reshape(box_encodings, [batch_size * num_boxes, 1, self.num_classes, self._box_code_size])
            total_classes = self.num_classes + 1
            class_feature_map_depth = self._num_spatial_bins[0] * self._num_spatial_bins[1] * total_classes
            class_feature_map = slim.conv2d(net, class_feature_map_depth, [1, 1], activation_fn=None, scope='class_predictions')
            class_predictions_with_background = ops.batch_position_sensitive_crop_regions(class_feature_map, boxes=proposal_boxes, crop_size=self._crop_size, num_spatial_bins=self._num_spatial_bins, global_pool=True)
            class_predictions_with_background = tf.squeeze(class_predictions_with_background, axis=[2, 3])
            class_predictions_with_background = tf.reshape(class_predictions_with_background, [batch_size * num_boxes, 1, total_classes])
        return {BOX_ENCODINGS: [box_encodings], CLASS_PREDICTIONS_WITH_BACKGROUND: [class_predictions_with_background]}