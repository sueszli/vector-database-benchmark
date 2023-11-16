"""
This module implements the abstract base class for all evasion detectors.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
from typing import Any, Dict, List, Tuple
import numpy as np

class EvasionDetector(abc.ABC):
    """
    Abstract base class for all evasion detectors.
    """
    defence_params: List[str] = []

    def __init__(self) -> None:
        if False:
            return 10
        '\n        Create an evasion detector object.\n        '
        pass

    @abc.abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, batch_size: int=128, nb_epochs: int=20, **kwargs) -> None:
        if False:
            return 10
        '\n        Fit the detection classifier if necessary.\n\n        :param x: Training set to fit the detector.\n        :param y: Labels for the training set.\n        :param batch_size: Size of batches.\n        :param nb_epochs: Number of epochs to use for training.\n        :param kwargs: Other parameters.\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def detect(self, x: np.ndarray, batch_size: int=128, **kwargs) -> Tuple[dict, np.ndarray]:
        if False:
            i = 10
            return i + 15
        '\n        Perform detection of adversarial data and return prediction as tuple.\n\n        :param x: Data sample on which to perform detection.\n        :param batch_size: Size of batches.\n        :param kwargs: Defence-specific parameters used by child classes.\n        :return: (report, is_adversarial):\n                where report is a dictionary containing information specific to the detection defence;\n                where is_adversarial is a boolean list of per-sample prediction whether the sample is adversarial\n        '
        raise NotImplementedError

    def set_params(self, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Take in a dictionary of parameters and apply defence-specific checks before saving them as attributes.\n\n        :param kwargs: A dictionary of defence-specific parameters.\n        '
        for (key, value) in kwargs.items():
            if key in self.defence_params:
                setattr(self, key, value)
        self._check_params()

    def get_params(self) -> Dict[str, Any]:
        if False:
            return 10
        '\n        Returns dictionary of parameters used to run defence.\n\n        :return: Dictionary of parameters of the method.\n        '
        dictionary = {param: getattr(self, param) for param in self.defence_params}
        return dictionary

    def _check_params(self) -> None:
        if False:
            return 10
        pass