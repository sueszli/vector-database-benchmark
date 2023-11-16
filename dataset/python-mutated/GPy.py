"""
This module implements a wrapper class for GPy Gaussian Process classification models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
import os
from typing import List, Optional, Union, Tuple, TYPE_CHECKING
import numpy as np
from art.estimators.classification.classifier import ClassifierClassLossGradients
from art import config
if TYPE_CHECKING:
    from GPy.models import GPClassification
    from art.utils import CLIP_VALUES_TYPE, PREPROCESSING_TYPE
    from art.defences.preprocessor import Preprocessor
    from art.defences.postprocessor import Postprocessor
logger = logging.getLogger(__name__)

class GPyGaussianProcessClassifier(ClassifierClassLossGradients):
    """
    Wrapper class for GPy Gaussian Process classification models.
    """

    def __init__(self, model: Optional['GPClassification']=None, clip_values: Optional['CLIP_VALUES_TYPE']=None, preprocessing_defences: Union['Preprocessor', List['Preprocessor'], None]=None, postprocessing_defences: Union['Postprocessor', List['Postprocessor'], None]=None, preprocessing: 'PREPROCESSING_TYPE'=(0.0, 1.0)) -> None:
        if False:
            print('Hello World!')
        '\n        Create a `Classifier` instance GPY Gaussian Process classification models.\n\n        :param model: GPY Gaussian Process Classification model.\n        :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed\n               for features.\n        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.\n        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.\n        :param preprocessing: Tuple of the form `(subtrahend, divisor)` of floats or `np.ndarray` of values to be\n               used for data preprocessing. The first value will be subtracted from the input. The input will then\n               be divided by the second one.\n        '
        from GPy.models import GPClassification
        if not isinstance(model, GPClassification):
            raise TypeError('Model must be of type GPy.models.GPClassification')
        super().__init__(model=model, clip_values=clip_values, preprocessing_defences=preprocessing_defences, postprocessing_defences=postprocessing_defences, preprocessing=preprocessing)
        self.nb_classes = 2

    @property
    def input_shape(self) -> Tuple[int, ...]:
        if False:
            return 10
        '\n        Return the shape of one input sample.\n\n        :return: Shape of one input sample.\n        '
        return self._input_shape

    def class_gradient(self, x: np.ndarray, label: Union[int, List[int], None]=None, eps: float=0.0001, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Compute per-class derivatives w.r.t. `x`.\n\n        :param x: Sample input with shape as expected by the model.\n        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class\n                      output is computed for all samples. If multiple values as provided, the first dimension should\n                      match the batch size of `x`, and each value will be used as target for its corresponding sample in\n                      `x`. If `None`, then gradients for all classes will be computed for each sample.\n        :param eps: Fraction added to the diagonal elements of the input `x`.\n        :return: Array of gradients of input features w.r.t. each class in the form\n                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes\n                 `(batch_size, 1, input_shape)` when `label` parameter is specified.\n        '
        (x_preprocessed, _) = self._apply_preprocessing(x, y=None, fit=False)
        grads = np.zeros((np.shape(x_preprocessed)[0], 2, np.shape(x)[1]))
        for i in range(np.shape(x_preprocessed)[0]):
            for i_c in range(2):
                ind = self.predict(x[i].reshape(1, -1))[0, i_c]
                sur = self.predict(np.repeat(x_preprocessed[i].reshape(1, -1), np.shape(x_preprocessed)[1], 0) + eps * np.eye(np.shape(x_preprocessed)[1]))[:, i_c]
                grads[i, i_c] = ((sur - ind) * eps).reshape(1, -1)
        grads = self._apply_preprocessing_gradient(x, grads)
        if label is not None:
            return grads[:, label, :].reshape(np.shape(x_preprocessed)[0], 1, np.shape(x_preprocessed)[1])
        return grads

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Compute the gradient of the loss function w.r.t. `x`.\n\n        :param x: Sample input with shape as expected by the model.\n        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape\n                  `(nb_samples,)`.\n        :return: Array of gradients of the same shape as `x`.\n        '
        (x_preprocessed, _) = self._apply_preprocessing(x, y, fit=False)
        eps = 1e-05
        grads = np.zeros(np.shape(x))
        for i in range(np.shape(x)[0]):
            ind = 1.0 - self.predict(x_preprocessed[i].reshape(1, -1))[0, np.argmax(y[i])]
            sur = 1.0 - self.predict(np.repeat(x_preprocessed[i].reshape(1, -1), np.shape(x_preprocessed)[1], 0) + eps * np.eye(np.shape(x_preprocessed)[1]))[:, np.argmax(y[i])]
            grads[i] = ((sur - ind) * eps).reshape(1, -1)
        grads = self._apply_preprocessing_gradient(x, grads)
        return grads

    def predict(self, x: np.ndarray, logits: bool=False, **kwargs) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Perform prediction for a batch of inputs.\n\n        :param x: Input samples.\n        :param logits: `True` if the prediction should be done without squashing function.\n        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.\n        '
        (x_preprocessed, _) = self._apply_preprocessing(x, y=None, fit=False)
        out = np.zeros((np.shape(x_preprocessed)[0], 2))
        if logits:
            out[:, 0] = self.model.predict_noiseless(x_preprocessed)[0].reshape(-1)
            out[:, 1] = -1.0 * out[:, 0]
        else:
            out[:, 0] = self.model.predict(x_preprocessed)[0].reshape(-1)
            out[:, 1] = 1.0 - out[:, 0]
        predictions = self._apply_postprocessing(preds=out, fit=False)
        return predictions

    def predict_uncertainty(self, x: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Perform uncertainty prediction for a batch of inputs.\n\n        :param x: Input samples.\n        :return: Array of uncertainty predictions of shape `(nb_inputs)`.\n        '
        (x_preprocessed, _) = self._apply_preprocessing(x, y=None, fit=False)
        out = self.model.predict_noiseless(x_preprocessed)[1]
        predictions = self._apply_postprocessing(preds=out, fit=False)
        return predictions

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Fit the classifier on the training set `(x, y)`.\n\n        :param x: Training data. Not used, as given to model in initialized earlier.\n        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes).\n        '
        raise NotImplementedError

    def save(self, filename: str, path: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        '\n        Save a model to file in the format specific to the backend framework.\n\n        :param filename: Name of the file where to store the model.\n        :param path: Path of the folder where to store the model. If no path is specified, the model will be stored in\n                     the default data location of the library `ART_DATA_PATH`.\n        '
        if path is None:
            full_path = os.path.join(config.ART_DATA_PATH, filename)
        else:
            full_path = os.path.join(path, filename)
        folder = os.path.split(full_path)[0]
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.model.save_model(full_path, save_data=False)