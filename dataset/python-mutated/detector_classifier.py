"""
This module implements the base class `DetectorClassifier` for classifier and detector combinations.

Paper link:
    https://arxiv.org/abs/1705.07263
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import List, Optional, Union, Tuple, TYPE_CHECKING
import numpy as np
from art.estimators.classification.classifier import ClassifierNeuralNetwork
if TYPE_CHECKING:
    from art.utils import PREPROCESSING_TYPE
    from art.data_generators import DataGenerator
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor
logger = logging.getLogger(__name__)

class DetectorClassifier(ClassifierNeuralNetwork):
    """
    This class implements a Classifier extension that wraps a classifier and a detector.
    More details in https://arxiv.org/abs/1705.07263
    """
    estimator_params = ClassifierNeuralNetwork.estimator_params + ['classifier', 'detector']

    def __init__(self, classifier: ClassifierNeuralNetwork, detector: ClassifierNeuralNetwork, preprocessing_defences: Union['Preprocessor', List['Preprocessor'], None]=None, postprocessing_defences: Union['Postprocessor', List['Postprocessor'], None]=None, preprocessing: 'PREPROCESSING_TYPE'=(0.0, 1.0)) -> None:
        if False:
            return 10
        '\n        Initialization for the DetectorClassifier.\n\n        :param classifier: A trained classifier.\n        :param detector: A trained detector applied for the binary classification.\n        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier. Not applicable\n               in this classifier.\n        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.\n        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be\n               used for data preprocessing. The first value will be subtracted from the input. The input will then\n               be divided by the second one. Not applicable in this classifier.\n        '
        if preprocessing_defences is not None:
            raise NotImplementedError('Preprocessing is not applicable in this classifier.')
        super().__init__(model=None, clip_values=classifier.clip_values, preprocessing=preprocessing, channels_first=classifier.channels_first, preprocessing_defences=preprocessing_defences, postprocessing_defences=postprocessing_defences)
        self.classifier = classifier
        self.detector = detector
        self.nb_classes = classifier.nb_classes + 1
        self._input_shape = classifier.input_shape

    @property
    def input_shape(self) -> Tuple[int, ...]:
        if False:
            return 10
        '\n        Return the shape of one input sample.\n\n        :return: Shape of one input sample.\n        '
        return self._input_shape

    def predict(self, x: np.ndarray, batch_size: int=128, **kwargs) -> np.ndarray:
        if False:
            return 10
        '\n        Perform prediction for a batch of inputs.\n\n        :param x: Input samples.\n        :param batch_size: Size of batches.\n        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.\n        '
        classifier_outputs = self.classifier.predict(x=x, batch_size=batch_size)
        detector_outputs = self.detector.predict(x=x, batch_size=batch_size)
        detector_outputs = (np.reshape(detector_outputs, [-1]) + 1) * np.max(classifier_outputs, axis=1)
        detector_outputs = np.reshape(detector_outputs, [-1, 1])
        combined_outputs = np.concatenate([classifier_outputs, detector_outputs], axis=1)
        predictions = self._apply_postprocessing(preds=combined_outputs, fit=False)
        return predictions

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int=128, nb_epochs: int=10, **kwargs) -> None:
        if False:
            while True:
                i = 10
        '\n        Fit the classifier on the training set `(x, y)`.\n\n        :param x: Training data.\n        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes).\n        :param batch_size: Size of batches.\n        :param nb_epochs: Number of epochs to use for training.\n        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch\n               and providing it takes no effect.\n        :raises `NotImplementedException`: This method is not supported for detector-classifiers.\n        '
        raise NotImplementedError

    def fit_generator(self, generator: 'DataGenerator', nb_epochs: int=20, **kwargs) -> None:
        if False:
            print('Hello World!')
        '\n        Fit the classifier using the generator that yields batches as specified.\n\n        :param generator: Batch generator providing `(x, y)` for each epoch.\n        :param nb_epochs: Number of epochs to use for training.\n        :param kwargs: Dictionary of framework-specific arguments. This parameter is not currently supported for PyTorch\n               and providing it takes no effect.\n        :raises `NotImplementedException`: This method is not supported for detector-classifiers.\n        '
        raise NotImplementedError

    def class_gradient(self, x: np.ndarray, label: Union[int, List[int], np.ndarray, None]=None, training_mode: bool=False, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        "\n        Compute per-class derivatives w.r.t. `x`.\n\n        :param x: Sample input with shape as expected by the model.\n        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class\n                      output is computed for all samples. If multiple values as provided, the first dimension should\n                      match the batch size of `x`, and each value will be used as target for its corresponding sample in\n                      `x`. If `None`, then gradients for all classes will be computed for each sample.\n        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.\n        :return: Array of gradients of input features w.r.t. each class in the form\n                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes\n                 `(batch_size, 1, input_shape)` when `label` parameter is specified.\n        "
        if not (label is None or (isinstance(label, int) and label in range(self.nb_classes)) or (isinstance(label, np.ndarray) and len(label.shape) == 1 and (label < self.nb_classes).all() and (label.shape[0] == x.shape[0]))):
            raise ValueError(f'Label {label} is out of range.')
        if label is None:
            combined_grads = self._compute_combined_grads(x, label=None)
        elif isinstance(label, int):
            if label < self.nb_classes - 1:
                combined_grads = self.classifier.class_gradient(x=x, label=label, training_mode=training_mode, **kwargs)
            else:
                classifier_grads = self.classifier.class_gradient(x=x, label=None, training_mode=training_mode, **kwargs)
                detector_grads = self.detector.class_gradient(x=x, label=0, training_mode=training_mode, **kwargs)
                classifier_preds = self.classifier.predict(x=x)
                maxind_classifier_preds = np.argmax(classifier_preds, axis=1)
                max_classifier_preds = classifier_preds[np.arange(x.shape[0]), maxind_classifier_preds]
                first_detector_grads = max_classifier_preds[:, None, None, None, None] * detector_grads
                max_classifier_grads = classifier_grads[np.arange(len(classifier_grads)), maxind_classifier_preds]
                detector_preds = self.detector.predict(x=x)
                second_detector_grads = max_classifier_grads * (detector_preds + 1)[:, None, None]
                second_detector_grads = second_detector_grads[None, ...]
                second_detector_grads = np.swapaxes(second_detector_grads, 0, 1)
                combined_grads = first_detector_grads + second_detector_grads
        else:
            classifier_idx = np.where(label < self.nb_classes - 1)
            detector_idx = np.where(label == self.nb_classes - 1)
            combined_grads = np.zeros(shape=(x.shape[0], 1, x.shape[1], x.shape[2], x.shape[3]))
            if classifier_idx:
                combined_grads[classifier_idx] = self.classifier.class_gradient(x=x[classifier_idx], label=label[classifier_idx], training_mode=training_mode, **kwargs)
            if detector_idx:
                classifier_grads = self.classifier.class_gradient(x=x[detector_idx], label=None, training_mode=training_mode, **kwargs)
                detector_grads = self.detector.class_gradient(x=x[detector_idx], label=0, training_mode=training_mode, **kwargs)
                classifier_preds = self.classifier.predict(x=x[detector_idx])
                maxind_classifier_preds = np.argmax(classifier_preds, axis=1)
                max_classifier_preds = classifier_preds[np.arange(len(detector_idx)), maxind_classifier_preds]
                first_detector_grads = max_classifier_preds[:, None, None, None, None] * detector_grads
                max_classifier_grads = classifier_grads[np.arange(len(classifier_grads)), maxind_classifier_preds]
                detector_preds = self.detector.predict(x=x[detector_idx])
                second_detector_grads = max_classifier_grads * (detector_preds + 1)[:, None, None]
                second_detector_grads = second_detector_grads[None, ...]
                second_detector_grads = np.swapaxes(second_detector_grads, 0, 1)
                detector_grads = first_detector_grads + second_detector_grads
                combined_grads[detector_idx] = detector_grads
        return combined_grads

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute the loss of the neural network for samples `x`.\n\n        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,\n                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).\n        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices\n                  of shape `(nb_samples,)`.\n        :return: Loss values.\n        :rtype: Format as expected by the `model`\n        '
        raise NotImplementedError

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, training_mode: bool=False, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        "\n        Compute the gradient of the loss function w.r.t. `x`.\n\n        :param x: Sample input with shape as expected by the model.\n        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape\n                  (nb_samples,).\n        :param training_mode: `True` for model set to training mode and `'False` for model set to evaluation mode.\n        :return: Array of gradients of the same shape as `x`.\n        :raises `NotImplementedException`: This method is not supported for detector-classifiers.\n        "
        raise NotImplementedError

    @property
    def layer_names(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the hidden layers in the model, if applicable. This function is not supported for the\n        Classifier and Detector classes.\n\n        :return: The hidden layers in the model, input and output layers excluded.\n        :raises `NotImplementedException`: This method is not supported for detector-classifiers.\n        '
        raise NotImplementedError

    def get_activations(self, x: np.ndarray, layer: Union[int, str], batch_size: int=128, framework: bool=False) -> np.ndarray:
        if False:
            return 10
        '\n        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and\n        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by\n        calling `layer_names`.\n\n        :param x: Input for computing the activations.\n        :param layer: Layer for computing the activations.\n        :param batch_size: Size of batches.\n        :param framework: If true, return the intermediate tensor representation of the activation.\n        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.\n        :raises `NotImplementedException`: This method is not supported for detector-classifiers.\n        '
        raise NotImplementedError

    def save(self, filename: str, path: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Save a model to file in the format specific to the backend framework.\n\n        :param filename: Name of the file where to store the model.\n        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in\n                     the default data location of the library `ART_DATA_PATH`.\n        '
        self.classifier.save(filename=filename + '_classifier', path=path)
        self.detector.save(filename=filename + '_detector', path=path)

    def clone_for_refitting(self) -> 'DetectorClassifier':
        if False:
            return 10
        '\n        Clone classifier for refitting.\n        '
        raise NotImplementedError

    def __repr__(self):
        if False:
            return 10
        repr_ = f"{self.__module__ + '.' + self.__class__.__name__}(classifier={self.classifier}, detector={self.detector}, postprocessing_defences={self.postprocessing_defences}, preprocessing={self.preprocessing}"
        return repr_

    def _compute_combined_grads(self, x: np.ndarray, label: Optional[Union[int, List[int]]]=None) -> np.ndarray:
        if False:
            print('Hello World!')
        classifier_grads = self.classifier.class_gradient(x=x, label=label)
        detector_grads = self.detector.class_gradient(x=x, label=label)
        classifier_preds = self.classifier.predict(x=x)
        maxind_classifier_preds = np.argmax(classifier_preds, axis=1)
        max_classifier_preds = classifier_preds[np.arange(classifier_preds.shape[0]), maxind_classifier_preds]
        first_detector_grads = max_classifier_preds[:, None, None, None, None] * detector_grads
        max_classifier_grads = classifier_grads[np.arange(len(classifier_grads)), maxind_classifier_preds]
        detector_preds = self.detector.predict(x=x)
        second_detector_grads = max_classifier_grads * (detector_preds + 1)[:, None, None]
        second_detector_grads = second_detector_grads[None, ...]
        second_detector_grads = np.swapaxes(second_detector_grads, 0, 1)
        detector_grads = first_detector_grads + second_detector_grads
        combined_logits_grads = np.concatenate([classifier_grads, detector_grads], axis=1)
        return combined_logits_grads