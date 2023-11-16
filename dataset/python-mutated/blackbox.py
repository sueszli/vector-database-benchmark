"""
This module implements the classifier `BlackBoxClassifier` for black-box classifiers.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from functools import total_ordering
import logging
from typing import Callable, List, Optional, Union, Tuple, TYPE_CHECKING
import numpy as np
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
from art.estimators.classification.classifier import ClassifierMixin, Classifier
if TYPE_CHECKING:
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor
logger = logging.getLogger(__name__)

class BlackBoxClassifier(ClassifierMixin, BaseEstimator):
    """
    Class for black-box classifiers.
    """
    estimator_params = Classifier.estimator_params + ['nb_classes', 'input_shape', 'predict_fn']

    def __init__(self, predict_fn: Union[Callable, Tuple[np.ndarray, np.ndarray]], input_shape: Tuple[int, ...], nb_classes: int, clip_values: Optional['CLIP_VALUES_TYPE']=None, preprocessing_defences: Union['Preprocessor', List['Preprocessor'], None]=None, postprocessing_defences: Union['Postprocessor', List['Postprocessor'], None]=None, preprocessing: 'PREPROCESSING_TYPE'=(0.0, 1.0), fuzzy_float_compare: bool=False):
        if False:
            i = 10
            return i + 15
        '\n        Create a `Classifier` instance for a black-box model.\n\n        :param predict_fn: Function that takes in an `np.ndarray` of input data and returns the one-hot encoded matrix\n               of predicted classes or tuple of the form `(inputs, labels)` containing the predicted labels for each\n               input.\n        :param input_shape: Size of input.\n        :param nb_classes: Number of prediction classes.\n        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and\n               maximum values allowed for features. If floats are provided, these will be used as the range of all\n               features. If arrays are provided, each value will be considered the bound for a feature, thus\n               the shape of clip values needs to match the total number of features.\n        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.\n        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.\n        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be\n               used for data preprocessing. The first value will be subtracted from the input. The input will then\n               be divided by the second one.\n        :param fuzzy_float_compare: If `predict_fn` is a tuple mapping inputs to labels, and this is True, looking up\n               inputs in the table will be done using `numpy.isclose`. Only set to True if really needed, since this\n               severely affects performance.\n        '
        super().__init__(model=None, clip_values=clip_values, preprocessing_defences=preprocessing_defences, postprocessing_defences=postprocessing_defences, preprocessing=preprocessing)
        if callable(predict_fn):
            self._predict_fn = predict_fn
        else:
            self._predict_fn = _make_lookup_predict_fn(predict_fn, fuzzy_float_compare)
        self._input_shape = input_shape
        self.nb_classes = nb_classes

    @property
    def input_shape(self) -> Tuple[int, ...]:
        if False:
            print('Hello World!')
        '\n        Return the shape of one input sample.\n\n        :return: Shape of one input sample.\n        '
        return self._input_shape

    @property
    def predict_fn(self) -> Callable:
        if False:
            print('Hello World!')
        '\n        Return the prediction function.\n\n        :return: The prediction function.\n        '
        return self._predict_fn

    def predict(self, x: np.ndarray, batch_size: int=128, **kwargs) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Perform prediction for a batch of inputs.\n\n        :param x: Input samples.\n        :param batch_size: Size of batches.\n        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.\n        '
        from art.config import ART_NUMPY_DTYPE
        (x_preprocessed, _) = self._apply_preprocessing(x, y=None, fit=False)
        predictions = np.zeros((x_preprocessed.shape[0], self.nb_classes), dtype=ART_NUMPY_DTYPE)
        for batch_index in range(int(np.ceil(x_preprocessed.shape[0] / float(batch_size)))):
            (begin, end) = (batch_index * batch_size, min((batch_index + 1) * batch_size, x_preprocessed.shape[0]))
            predictions[begin:end] = self.predict_fn(x_preprocessed[begin:end])
        predictions = self._apply_postprocessing(preds=predictions, fit=False)
        return predictions

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Fit the classifier on the training set `(x, y)`.\n\n        :param x: Training data.\n        :param y: Labels, one-vs-rest encoding.\n        :param kwargs: Dictionary of framework-specific arguments. These should be parameters supported by the\n               `fit_generator` function in Keras and will be passed to this function as such. Including the number of\n               epochs or the number of steps per epoch as part of this argument will result in as error.\n        :raises `NotImplementedException`: This method is not supported for black-box classifiers.\n        '
        raise NotImplementedError

    def save(self, filename: str, path: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Save a model to file in the format specific to the backend framework. For Keras, .h5 format is used.\n\n        :param filename: Name of the file where to store the model.\n        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in\n                     the default data location of the library `ART_DATA_PATH`.\n        :raises `NotImplementedException`: This method is not supported for black-box classifiers.\n        '
        raise NotImplementedError

class BlackBoxClassifierNeuralNetwork(NeuralNetworkMixin, ClassifierMixin, BaseEstimator):
    """
    Class for black-box neural network classifiers.
    """
    estimator_params = NeuralNetworkMixin.estimator_params + ClassifierMixin.estimator_params + BaseEstimator.estimator_params + ['nb_classes', 'input_shape', 'predict_fn']

    def __init__(self, predict_fn: Union[Callable, Tuple[np.ndarray, np.ndarray]], input_shape: Tuple[int, ...], nb_classes: int, channels_first: bool=True, clip_values: Optional['CLIP_VALUES_TYPE']=None, preprocessing_defences: Union['Preprocessor', List['Preprocessor'], None]=None, postprocessing_defences: Union['Postprocessor', List['Postprocessor'], None]=None, preprocessing: 'PREPROCESSING_TYPE'=(0, 1), fuzzy_float_compare: bool=False):
        if False:
            print('Hello World!')
        '\n        Create a `Classifier` instance for a black-box model.\n\n        :param predict_fn: Function that takes in an `np.ndarray` of input data and returns the one-hot encoded matrix\n               of predicted classes or tuple of the form `(inputs, labels)` containing the predicted labels for each\n               input.\n        :param input_shape: Size of input.\n        :param nb_classes: Number of prediction classes.\n        :param channels_first: Set channels first or last.\n        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and\n               maximum values allowed for features. If floats are provided, these will be used as the range of all\n               features. If arrays are provided, each value will be considered the bound for a feature, thus\n               the shape of clip values needs to match the total number of features.\n        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.\n        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.\n        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be\n               used for data preprocessing. The first value will be subtracted from the input. The input will then\n               be divided by the second one.\n        :param fuzzy_float_compare: If `predict_fn` is a tuple mapping inputs to labels, and this is True, looking up\n               inputs in the table will be done using `numpy.isclose`. Only set to True if really needed, since this\n               severely affects performance.\n        '
        super().__init__(model=None, channels_first=channels_first, clip_values=clip_values, preprocessing_defences=preprocessing_defences, postprocessing_defences=postprocessing_defences, preprocessing=preprocessing)
        if callable(predict_fn):
            self._predict_fn = predict_fn
        else:
            self._predict_fn = _make_lookup_predict_fn(predict_fn, fuzzy_float_compare)
        self._input_shape = input_shape
        self.nb_classes = nb_classes
        self._learning_phase = None
        self._layer_names = None

    @property
    def input_shape(self) -> Tuple[int, ...]:
        if False:
            return 10
        '\n        Return the shape of one input sample.\n\n        :return: Shape of one input sample.\n        '
        return self._input_shape

    def predict(self, x: np.ndarray, batch_size: int=128, **kwargs):
        if False:
            print('Hello World!')
        '\n        Perform prediction for a batch of inputs.\n\n        :param x: Test set.\n        :param batch_size: Size of batches.\n        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.\n        '
        from art.config import ART_NUMPY_DTYPE
        (x_preprocessed, _) = self._apply_preprocessing(x, y=None, fit=False)
        predictions = np.zeros((x_preprocessed.shape[0], self.nb_classes), dtype=ART_NUMPY_DTYPE)
        for batch_index in range(int(np.ceil(x_preprocessed.shape[0] / float(batch_size)))):
            (begin, end) = (batch_index * batch_size, min((batch_index + 1) * batch_size, x_preprocessed.shape[0]))
            predictions[begin:end] = self._predict_fn(x_preprocessed[begin:end])
        predictions = self._apply_postprocessing(preds=predictions, fit=False)
        return predictions

    def fit(self, x: np.ndarray, y, batch_size: int=128, nb_epochs: int=20, **kwargs) -> None:
        if False:
            return 10
        '\n        Fit the model of the estimator on the training data `x` and `y`.\n\n        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,\n                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).\n        :param y: Target values.\n        :type y: Format as expected by the `model`\n        :param batch_size: Batch size.\n        :param nb_epochs: Number of training epochs.\n        '
        raise NotImplementedError

    def get_activations(self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool=False) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Return the output of a specific layer for samples `x` where `layer` is the index of the layer between 0 and\n        `nb_layers - 1 or the name of the layer. The number of layers can be determined by counting the results\n        returned by calling `layer_names`.\n\n        :param x: Samples\n        :param layer: Index or name of the layer.\n        :param batch_size: Batch size.\n        :param framework: If true, return the intermediate tensor representation of the activation.\n        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.\n        '
        raise NotImplementedError

    def loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Compute the loss of the neural network for samples `x`.\n\n        :param x: Samples of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,\n                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).\n        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices\n                  of shape `(nb_samples,)`.\n        :return: Loss values.\n        :rtype: Format as expected by the `model`\n        '
        raise NotImplementedError

    def compute_loss(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

@total_ordering
class FuzzyMapping:
    """
    Class for a sample/label pair to be used in a `SortedList`.
    """

    def __init__(self, key: np.ndarray, value=None):
        if False:
            while True:
                i = 10
        '\n        Create an instance of a key/value to pair to be used in a `SortedList`.\n\n        :param key: The sample to be matched against.\n        :param value: The mapped value.\n        '
        self.key = key
        self.value = value

    def __eq__(self, other):
        if False:
            return 10
        return np.all(np.isclose(self.key, other.key))

    def __ge__(self, other):
        if False:
            while True:
                i = 10
        close_cells = np.isclose(self.key, other.key)
        if np.all(close_cells):
            return True
        compare_idx = np.unravel_index(np.argmin(close_cells), shape=self.key.shape)
        return self.key[compare_idx] >= other.key[compare_idx]

def _make_lookup_predict_fn(existing_predictions: Tuple[np.ndarray, np.ndarray], fuzzy_float_compare: bool) -> Callable:
    if False:
        print('Hello World!')
    '\n    Makes a predict_fn callback based on a table of existing predictions.\n\n    :param existing_predictions: Tuple of (samples, labels).\n    :param fuzzy_float_compare: Look up predictions using `np.isclose`, only set to True if really needed, since this\n                                severely affects performance.\n    :return: Prediction function.\n    '
    (samples, labels) = existing_predictions
    if fuzzy_float_compare:
        from sortedcontainers import SortedList
        sorted_predictions = SortedList([FuzzyMapping(key, value) for (key, value) in zip(samples, labels)])

        def fuzzy_predict_fn(batch):
            if False:
                for i in range(10):
                    print('nop')
            predictions = []
            for row in batch:
                try:
                    match_idx = sorted_predictions.index(FuzzyMapping(row))
                except ValueError as err:
                    raise ValueError('No existing prediction for queried input') from err
                predictions.append(sorted_predictions[match_idx].value)
            return np.array(predictions)
        return fuzzy_predict_fn
    mapping = {}
    for (x, y) in zip(samples, labels):
        mapping[x.tobytes()] = y

    def predict_fn(batch):
        if False:
            i = 10
            return i + 15
        predictions = []
        for row in batch:
            row_bytes = row.tobytes()
            if row.tobytes() not in mapping:
                raise ValueError('No existing prediction for queried input')
            predictions.append(mapping[row_bytes])
        return np.array(predictions)
    return predict_fn