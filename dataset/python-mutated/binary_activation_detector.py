"""
Module containing different methods for the detection of adversarial examples. All models are considered to be binary
detectors.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Tuple, Union, TYPE_CHECKING
import numpy as np
from art.defences.detector.evasion.evasion_detector import EvasionDetector
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE
logger = logging.getLogger(__name__)

class BinaryActivationDetector(EvasionDetector):
    """
    Binary detector of adversarial samples coming from evasion attacks. The detector uses an architecture provided by
    the user and is trained on the values of the activations of a classifier at a given layer.
    """
    defence_params = ['classifier', 'detector', 'layer']

    def __init__(self, classifier: 'CLASSIFIER_NEURALNETWORK_TYPE', detector: 'CLASSIFIER_NEURALNETWORK_TYPE', layer: Union[int, str]) -> None:
        if False:
            return 10
        '\n        Create a `BinaryActivationDetector` instance which performs binary classification on activation information.\n        The shape of the input of the detector has to match that of the output of the chosen layer.\n\n        :param classifier: The classifier of which the activation information is to be used for detection.\n        :param detector: The detector architecture to be trained and applied for the binary classification.\n        :param layer: Layer for computing the activations to use for training the detector.\n        '
        super().__init__()
        self.classifier = classifier
        self.detector = detector
        if classifier.layer_names is None:
            raise ValueError('No layer names identified.')
        if isinstance(layer, int):
            if layer < 0 or layer >= len(classifier.layer_names):
                raise ValueError(f'Layer index {layer} is outside of range (0 to {len(classifier.layer_names) - 1} included).')
            self._layer_name = classifier.layer_names[layer]
        else:
            if layer not in classifier.layer_names:
                raise ValueError(f'Layer name {layer} is not part of the graph.')
            self._layer_name = layer

    def _get_activations(self, x: np.ndarray, layer: Union[int, str], batch_size: int, framework: bool=False) -> np.ndarray:
        if False:
            return 10
        x_activations = self.classifier.get_activations(x, layer, batch_size, framework)
        if x_activations is None:
            raise ValueError('Classifier activations are null.')
        if isinstance(x_activations, np.ndarray):
            return x_activations
        return x_activations.numpy()

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int=128, nb_epochs: int=20, **kwargs) -> None:
        if False:
            while True:
                i = 10
        '\n        Fit the detector using training data.\n\n        :param x: Training set to fit the detector.\n        :param y: Labels for the training set.\n        :param batch_size: Size of batches.\n        :param nb_epochs: Number of epochs to use for training.\n        :param kwargs: Other parameters.\n        '
        x_activations: np.ndarray = self._get_activations(x, self._layer_name, batch_size)
        self.detector.fit(x_activations, y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)

    def detect(self, x: np.ndarray, batch_size: int=128, **kwargs) -> Tuple[dict, np.ndarray]:
        if False:
            i = 10
            return i + 15
        '\n        Perform detection of adversarial data and return prediction as tuple.\n\n        :param x: Data sample on which to perform detection.\n        :param batch_size: Size of batches.\n        :return: (report, is_adversarial):\n                where report is a dictionary containing the detector model output predictions;\n                where is_adversarial is a boolean list of per-sample prediction whether the sample is adversarial\n                or not and has the same `batch_size` (first dimension) as `x`.\n        '
        x_activations: np.ndarray = self._get_activations(x, self._layer_name, batch_size)
        predictions = self.detector.predict(x_activations, batch_size=batch_size)
        is_adversarial = np.argmax(predictions, axis=1).astype(bool)
        report = {'predictions': predictions}
        return (report, is_adversarial)