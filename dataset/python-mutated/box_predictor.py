"""Box predictor for object detectors.

Box predictors are classes that take a high level
image feature map as input and produce two predictions,
(1) a tensor encoding box locations, and
(2) a tensor encoding classes for each box.

These components are passed directly to loss functions
in our detection models.

These modules are separated from the main model since the same
few box predictor architectures are shared across many models.
"""
from abc import abstractmethod
import tensorflow as tf
BOX_ENCODINGS = 'box_encodings'
CLASS_PREDICTIONS_WITH_BACKGROUND = 'class_predictions_with_background'
MASK_PREDICTIONS = 'mask_predictions'

class BoxPredictor(object):
    """BoxPredictor."""

    def __init__(self, is_training, num_classes):
        if False:
            i = 10
            return i + 15
        'Constructor.\n\n    Args:\n      is_training: Indicates whether the BoxPredictor is in training mode.\n      num_classes: number of classes.  Note that num_classes *does not*\n        include the background category, so if groundtruth labels take values\n        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the\n        assigned classification targets can range from {0,... K}).\n    '
        self._is_training = is_training
        self._num_classes = num_classes

    @property
    def is_keras_model(self):
        if False:
            while True:
                i = 10
        return False

    @property
    def num_classes(self):
        if False:
            return 10
        return self._num_classes

    def predict(self, image_features, num_predictions_per_location, scope=None, **params):
        if False:
            i = 10
            return i + 15
        'Computes encoded object locations and corresponding confidences.\n\n    Takes a list of high level image feature maps as input and produces a list\n    of box encodings and a list of class scores where each element in the output\n    lists correspond to the feature maps in the input list.\n\n    Args:\n      image_features: A list of float tensors of shape [batch_size, height_i,\n      width_i, channels_i] containing features for a batch of images.\n      num_predictions_per_location: A list of integers representing the number\n        of box predictions to be made per spatial location for each feature map.\n      scope: Variable and Op scope name.\n      **params: Additional keyword arguments for specific implementations of\n              BoxPredictor.\n\n    Returns:\n      A dictionary containing at least the following tensors.\n        box_encodings: A list of float tensors. Each entry in the list\n          corresponds to a feature map in the input `image_features` list. All\n          tensors in the list have one of the two following shapes:\n          a. [batch_size, num_anchors_i, q, code_size] representing the location\n            of the objects, where q is 1 or the number of classes.\n          b. [batch_size, num_anchors_i, code_size].\n        class_predictions_with_background: A list of float tensors of shape\n          [batch_size, num_anchors_i, num_classes + 1] representing the class\n          predictions for the proposals. Each entry in the list corresponds to a\n          feature map in the input `image_features` list.\n\n    Raises:\n      ValueError: If length of `image_features` is not equal to length of\n        `num_predictions_per_location`.\n    '
        if len(image_features) != len(num_predictions_per_location):
            raise ValueError('image_feature and num_predictions_per_location must be of same length, found: {} vs {}'.format(len(image_features), len(num_predictions_per_location)))
        if scope is not None:
            with tf.variable_scope(scope):
                return self._predict(image_features, num_predictions_per_location, **params)
        return self._predict(image_features, num_predictions_per_location, **params)

    @abstractmethod
    def _predict(self, image_features, num_predictions_per_location, **params):
        if False:
            return 10
        'Implementations must override this method.\n\n    Args:\n      image_features: A list of float tensors of shape [batch_size, height_i,\n        width_i, channels_i] containing features for a batch of images.\n      num_predictions_per_location: A list of integers representing the number\n        of box predictions to be made per spatial location for each feature map.\n      **params: Additional keyword arguments for specific implementations of\n              BoxPredictor.\n\n    Returns:\n      A dictionary containing at least the following tensors.\n        box_encodings: A list of float tensors. Each entry in the list\n          corresponds to a feature map in the input `image_features` list. All\n          tensors in the list have one of the two following shapes:\n          a. [batch_size, num_anchors_i, q, code_size] representing the location\n            of the objects, where q is 1 or the number of classes.\n          b. [batch_size, num_anchors_i, code_size].\n        class_predictions_with_background: A list of float tensors of shape\n          [batch_size, num_anchors_i, num_classes + 1] representing the class\n          predictions for the proposals. Each entry in the list corresponds to a\n          feature map in the input `image_features` list.\n    '
        pass

class KerasBoxPredictor(tf.keras.Model):
    """Keras-based BoxPredictor."""

    def __init__(self, is_training, num_classes, freeze_batchnorm, inplace_batchnorm_update, name=None):
        if False:
            i = 10
            return i + 15
        'Constructor.\n\n    Args:\n      is_training: Indicates whether the BoxPredictor is in training mode.\n      num_classes: number of classes.  Note that num_classes *does not*\n        include the background category, so if groundtruth labels take values\n        in {0, 1, .., K-1}, num_classes=K (and not K+1, even though the\n        assigned classification targets can range from {0,... K}).\n      freeze_batchnorm: Whether to freeze batch norm parameters during\n        training or not. When training with a small batch size (e.g. 1), it is\n        desirable to freeze batch norm update and use pretrained batch norm\n        params.\n      inplace_batchnorm_update: Whether to update batch norm moving average\n        values inplace. When this is false train op must add a control\n        dependency on tf.graphkeys.UPDATE_OPS collection in order to update\n        batch norm statistics.\n      name: A string name scope to assign to the model. If `None`, Keras\n        will auto-generate one from the class name.\n    '
        super(KerasBoxPredictor, self).__init__(name=name)
        self._is_training = is_training
        self._num_classes = num_classes
        self._freeze_batchnorm = freeze_batchnorm
        self._inplace_batchnorm_update = inplace_batchnorm_update

    @property
    def is_keras_model(self):
        if False:
            return 10
        return True

    @property
    def num_classes(self):
        if False:
            print('Hello World!')
        return self._num_classes

    def call(self, image_features, **kwargs):
        if False:
            return 10
        'Computes encoded object locations and corresponding confidences.\n\n    Takes a list of high level image feature maps as input and produces a list\n    of box encodings and a list of class scores where each element in the output\n    lists correspond to the feature maps in the input list.\n\n    Args:\n      image_features: A list of float tensors of shape [batch_size, height_i,\n      width_i, channels_i] containing features for a batch of images.\n      **kwargs: Additional keyword arguments for specific implementations of\n            BoxPredictor.\n\n    Returns:\n      A dictionary containing at least the following tensors.\n        box_encodings: A list of float tensors. Each entry in the list\n          corresponds to a feature map in the input `image_features` list. All\n          tensors in the list have one of the two following shapes:\n          a. [batch_size, num_anchors_i, q, code_size] representing the location\n            of the objects, where q is 1 or the number of classes.\n          b. [batch_size, num_anchors_i, code_size].\n        class_predictions_with_background: A list of float tensors of shape\n          [batch_size, num_anchors_i, num_classes + 1] representing the class\n          predictions for the proposals. Each entry in the list corresponds to a\n          feature map in the input `image_features` list.\n    '
        return self._predict(image_features, **kwargs)

    @abstractmethod
    def _predict(self, image_features, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Implementations must override this method.\n\n    Args:\n      image_features: A list of float tensors of shape [batch_size, height_i,\n        width_i, channels_i] containing features for a batch of images.\n      **kwargs: Additional keyword arguments for specific implementations of\n              BoxPredictor.\n\n    Returns:\n      A dictionary containing at least the following tensors.\n        box_encodings: A list of float tensors. Each entry in the list\n          corresponds to a feature map in the input `image_features` list. All\n          tensors in the list have one of the two following shapes:\n          a. [batch_size, num_anchors_i, q, code_size] representing the location\n            of the objects, where q is 1 or the number of classes.\n          b. [batch_size, num_anchors_i, code_size].\n        class_predictions_with_background: A list of float tensors of shape\n          [batch_size, num_anchors_i, num_classes + 1] representing the class\n          predictions for the proposals. Each entry in the list corresponds to a\n          feature map in the input `image_features` list.\n    '
        raise NotImplementedError