"""
Provides black-box gradient estimation using NES.
"""
import logging
from typing import List, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
from scipy.stats import entropy
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin, ClassifierLossGradients
from art.utils import clip_and_round
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE
logger = logging.getLogger(__name__)

class QueryEfficientGradientEstimationClassifier(ClassifierLossGradients, ClassifierMixin, BaseEstimator):
    """
    Implementation of Query-Efficient Black-box Adversarial Examples. The attack approximates the gradient by
    maximizing the loss function over samples drawn from random Gaussian noise around the input.

    | Paper link: https://arxiv.org/abs/1712.07113
    """
    estimator_params = ['num_basis', 'sigma', 'round_samples']

    def __init__(self, classifier: 'CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE', num_basis: int, sigma: float, round_samples: float=0.0) -> None:
        if False:
            return 10
        '\n        :param classifier: An instance of a classification estimator whose loss_gradient is being approximated.\n        :param num_basis:  The number of samples to draw to approximate the gradient.\n        :param sigma: Scaling on the Gaussian noise N(0,1).\n        :param round_samples: The resolution of the input domain to round the data to, e.g., 1.0, or 1/255. Set to 0 to\n                              disable.\n        '
        super().__init__(model=classifier.model, clip_values=classifier.clip_values)
        self._classifier = classifier
        self.num_basis = num_basis
        self.sigma = sigma
        self.round_samples = round_samples
        self._nb_classes = self._classifier.nb_classes

    @property
    def input_shape(self) -> Tuple[int, ...]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the shape of one input sample.\n\n        :return: Shape of one input sample.\n        '
        return self._classifier.input_shape

    def predict(self, x: np.ndarray, batch_size: int=128, **kwargs) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Perform prediction of the classifier for input `x`. Rounds results first.\n\n        :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,\n                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).\n        :param batch_size: Size of batches.\n        :return: Array of predictions of shape `(nb_inputs, nb_classes)`.\n        '
        return self._classifier.predict(clip_and_round(x, self.clip_values, self.round_samples), batch_size=batch_size)

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Fit the classifier using the training data `(x, y)`.\n\n        :param x: Features in array of shape (nb_samples, nb_features) or (nb_samples, nb_pixels_1, nb_pixels_2,\n                  nb_channels) or (nb_samples, nb_channels, nb_pixels_1, nb_pixels_2).\n        :param y: Target values (class labels in classification) in array of shape (nb_samples, nb_classes) in\n                  one-hot encoding format.\n        :param kwargs: Dictionary of framework-specific arguments.\n        '
        raise NotImplementedError

    def _generate_samples(self, x: np.ndarray, epsilon_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            i = 10
            return i + 15
        '\n        Generate samples around the current image.\n\n        :param x: Sample input with shape as expected by the model.\n        :param epsilon_map: Samples drawn from search space.\n        :return: Two arrays of new input samples to approximate gradient.\n        '
        minus = clip_and_round(np.repeat(x, self.num_basis, axis=0) - epsilon_map, self.clip_values, self.round_samples)
        plus = clip_and_round(np.repeat(x, self.num_basis, axis=0) + epsilon_map, self.clip_values, self.round_samples)
        return (minus, plus)

    def class_gradient(self, x: np.ndarray, label: Union[int, List[int], None]=None, **kwargs) -> np.ndarray:
        if False:
            return 10
        "\n        Compute per-class derivatives w.r.t. `x`.\n\n        :param x: Input with shape as expected by the classifier's model.\n        :param label: Index of a specific per-class derivative. If an integer is provided, the gradient of that class\n                      output is computed for all samples. If multiple values as provided, the first dimension should\n                      match the batch size of `x`, and each value will be used as target for its corresponding sample in\n                      `x`. If `None`, then gradients for all classes will be computed for each sample.\n        :return: Array of gradients of input features w.r.t. each class in the form\n                 `(batch_size, nb_classes, input_shape)` when computing for all classes, otherwise shape becomes\n                 `(batch_size, 1, input_shape)` when `label` parameter is specified.\n        "
        raise NotImplementedError

    def loss_gradient(self, x: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Compute the gradient of the loss function w.r.t. `x`.\n\n        :param x: Sample input with shape as expected by the model.\n        :param y: Correct labels, one-vs-rest encoding.\n        :return: Array of gradients of the same shape as `x`.\n        '
        epsilon_map = self.sigma * np.random.normal(size=[self.num_basis] + list(self.input_shape))
        grads = []
        for i in range(len(x)):
            (minus, plus) = self._generate_samples(x[i:i + 1], epsilon_map)
            new_y_minus = np.array([entropy(y[i], p) for p in self.predict(minus)])
            new_y_plus = np.array([entropy(y[i], p) for p in self.predict(plus)])
            query_efficient_grad = 2 * np.mean(np.multiply(epsilon_map.reshape(self.num_basis, -1), (new_y_plus - new_y_minus).reshape(self.num_basis, -1) / (2 * self.sigma)).reshape([-1] + list(self.input_shape)), axis=0)
            grads.append(query_efficient_grad)
        grads_array = self._apply_preprocessing_gradient(x, np.array(grads))
        return grads_array

    def get_activations(self, x: np.ndarray, layer: Union[int, str], batch_size: int) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Return the output of the specified layer for input `x`. `layer` is specified by layer index (between 0 and\n        `nb_layers - 1`) or by name. The number of layers can be determined by counting the results returned by\n        calling `layer_names`.\n\n        :param x: Input for computing the activations.\n        :param layer: Layer for computing the activations.\n        :param batch_size: Size of batches.\n        :return: The output of `layer`, where the first dimension is the batch size corresponding to `x`.\n        '
        raise NotImplementedError

    def save(self, filename: str, path: Optional[str]=None) -> None:
        if False:
            return 10
        '\n        Save a model to file specific to the backend framework.\n\n        :param filename: Name of the file where to save the model.\n        :param path: Path of the directory where to save the model. If no path is specified, the model will be stored in\n                     the default data location of ART at `ART_DATA_PATH`.\n        '
        raise NotImplementedError