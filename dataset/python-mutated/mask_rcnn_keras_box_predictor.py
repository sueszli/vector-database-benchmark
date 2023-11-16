"""Mask R-CNN Box Predictor."""
from object_detection.core import box_predictor
BOX_ENCODINGS = box_predictor.BOX_ENCODINGS
CLASS_PREDICTIONS_WITH_BACKGROUND = box_predictor.CLASS_PREDICTIONS_WITH_BACKGROUND
MASK_PREDICTIONS = box_predictor.MASK_PREDICTIONS

class MaskRCNNKerasBoxPredictor(box_predictor.KerasBoxPredictor):
    """Mask R-CNN Box Predictor.

  See Mask R-CNN: He, K., Gkioxari, G., Dollar, P., & Girshick, R. (2017).
  Mask R-CNN. arXiv preprint arXiv:1703.06870.

  This is used for the second stage of the Mask R-CNN detector where proposals
  cropped from an image are arranged along the batch dimension of the input
  image_features tensor. Notice that locations are *not* shared across classes,
  thus for each anchor, a separate prediction is made for each class.

  In addition to predicting boxes and classes, optionally this class allows
  predicting masks and/or keypoints inside detection boxes.

  Currently this box predictor makes per-class predictions; that is, each
  anchor makes a separate box prediction for each class.
  """

    def __init__(self, is_training, num_classes, freeze_batchnorm, box_prediction_head, class_prediction_head, third_stage_heads, name=None):
        if False:
            i = 10
            return i + 15
        'Constructor.\n\n    Args:\n      is_training: Indicates whether the BoxPredictor is in training mode.\n      num_classes: number of classes.  Note that num_classes *does not*\n        include the background category, so if groundtruth labels take values\n        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the\n        assigned classification targets can range from {0,... K}).\n      freeze_batchnorm: Whether to freeze batch norm parameters during\n        training or not. When training with a small batch size (e.g. 1), it is\n        desirable to freeze batch norm update and use pretrained batch norm\n        params.\n      box_prediction_head: The head that predicts the boxes in second stage.\n      class_prediction_head: The head that predicts the classes in second stage.\n      third_stage_heads: A dictionary mapping head names to mask rcnn head\n        classes.\n      name: A string name scope to assign to the model. If `None`, Keras\n        will auto-generate one from the class name.\n    '
        super(MaskRCNNKerasBoxPredictor, self).__init__(is_training, num_classes, freeze_batchnorm=freeze_batchnorm, inplace_batchnorm_update=False, name=name)
        self._box_prediction_head = box_prediction_head
        self._class_prediction_head = class_prediction_head
        self._third_stage_heads = third_stage_heads

    @property
    def num_classes(self):
        if False:
            return 10
        return self._num_classes

    def get_second_stage_prediction_heads(self):
        if False:
            for i in range(10):
                print('nop')
        return (BOX_ENCODINGS, CLASS_PREDICTIONS_WITH_BACKGROUND)

    def get_third_stage_prediction_heads(self):
        if False:
            i = 10
            return i + 15
        return sorted(self._third_stage_heads.keys())

    def _predict(self, image_features, prediction_stage=2, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Optionally computes encoded object locations, confidences, and masks.\n\n    Predicts the heads belonging to the given prediction stage.\n\n    Args:\n      image_features: A list of float tensors of shape\n        [batch_size, height_i, width_i, channels_i] containing roi pooled\n        features for each image. The length of the list should be 1 otherwise\n        a ValueError will be raised.\n      prediction_stage: Prediction stage. Acceptable values are 2 and 3.\n      **kwargs: Unused Keyword args\n\n    Returns:\n      A dictionary containing the predicted tensors that are listed in\n      self._prediction_heads. A subset of the following keys will exist in the\n      dictionary:\n        BOX_ENCODINGS: A float tensor of shape\n          [batch_size, 1, num_classes, code_size] representing the\n          location of the objects.\n        CLASS_PREDICTIONS_WITH_BACKGROUND: A float tensor of shape\n          [batch_size, 1, num_classes + 1] representing the class\n          predictions for the proposals.\n        MASK_PREDICTIONS: A float tensor of shape\n          [batch_size, 1, num_classes, image_height, image_width]\n\n    Raises:\n      ValueError: If num_predictions_per_location is not 1 or if\n        len(image_features) is not 1.\n      ValueError: if prediction_stage is not 2 or 3.\n    '
        if len(image_features) != 1:
            raise ValueError('length of `image_features` must be 1. Found {}'.format(len(image_features)))
        image_feature = image_features[0]
        predictions_dict = {}
        if prediction_stage == 2:
            predictions_dict[BOX_ENCODINGS] = self._box_prediction_head(image_feature)
            predictions_dict[CLASS_PREDICTIONS_WITH_BACKGROUND] = self._class_prediction_head(image_feature)
        elif prediction_stage == 3:
            for prediction_head in self.get_third_stage_prediction_heads():
                head_object = self._third_stage_heads[prediction_head]
                predictions_dict[prediction_head] = head_object(image_feature)
        else:
            raise ValueError('prediction_stage should be either 2 or 3.')
        return predictions_dict