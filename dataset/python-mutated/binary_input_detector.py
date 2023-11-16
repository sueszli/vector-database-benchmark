"""
Module containing different methods for the detection of adversarial examples. All models are considered to be binary
detectors.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Tuple, TYPE_CHECKING
import numpy as np
from art.defences.detector.evasion.evasion_detector import EvasionDetector
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_NEURALNETWORK_TYPE
logger = logging.getLogger(__name__)

class BinaryInputDetector(EvasionDetector):
    """
    Binary detector of adversarial samples coming from evasion attacks. The detector uses an architecture provided by
    the user and trains it on data labeled as clean (label 0) or adversarial (label 1).
    """
    defence_params = ['detector']

    def __init__(self, detector: 'CLASSIFIER_NEURALNETWORK_TYPE') -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a `BinaryInputDetector` instance which performs binary classification on input data.\n\n        :param detector: The detector architecture to be trained and applied for the binary classification.\n        '
        super().__init__()
        self.detector = detector

    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int=128, nb_epochs: int=20, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Fit the detector using clean and adversarial samples.\n\n        :param x: Training set to fit the detector.\n        :param y: Labels for the training set.\n        :param batch_size: Size of batches.\n        :param nb_epochs: Number of epochs to use for training.\n        :param kwargs: Other parameters.\n        '
        self.detector.fit(x, y, batch_size=batch_size, nb_epochs=nb_epochs, **kwargs)

    def detect(self, x: np.ndarray, batch_size: int=128, **kwargs) -> Tuple[dict, np.ndarray]:
        if False:
            i = 10
            return i + 15
        '\n        Perform detection of adversarial data and return prediction as tuple.\n\n        :param x: Data sample on which to perform detection.\n        :param batch_size: Size of batches.\n        :return: (report, is_adversarial):\n                where report is a dictionary containing the detector model output predictions;\n                where is_adversarial is a boolean list of per-sample prediction whether the sample is adversarial\n                or not and has the same `batch_size` (first dimension) as `x`.\n        '
        predictions = self.detector.predict(x, batch_size=batch_size)
        is_adversarial = np.argmax(predictions, axis=1).astype(bool)
        report = {'predictions': predictions}
        return (report, is_adversarial)