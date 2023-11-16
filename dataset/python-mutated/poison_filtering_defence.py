"""
This module implements the abstract base class for all poison filtering defences.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import abc
import sys
from typing import Any, Dict, List, Tuple, TYPE_CHECKING
import numpy as np
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

class PoisonFilteringDefence(ABC):
    """
    Base class for all poison filtering defences.
    """
    defence_params = ['classifier']

    def __init__(self, classifier: 'CLASSIFIER_TYPE', x_train: np.ndarray, y_train: np.ndarray) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create an :class:`.ActivationDefence` object with the provided classifier.\n\n        :param classifier: Model evaluated for poison.\n        :param x_train: dataset used to train the classifier.\n        :param y_train: labels used to train the classifier.\n        '
        self.classifier = classifier
        self.x_train = x_train
        self.y_train = y_train

    @abc.abstractmethod
    def detect_poison(self, **kwargs) -> Tuple[dict, List[int]]:
        if False:
            while True:
                i = 10
        '\n        Detect poison.\n\n        :param kwargs: Defence-specific parameters used by child classes.\n        :return: Dictionary with report and list with items identified as poison.\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_defence(self, is_clean: np.ndarray, **kwargs) -> str:
        if False:
            while True:
                i = 10
        "\n        Evaluate the defence given the labels specifying if the data is poisoned or not.\n\n        :param is_clean: 1-D array where is_clean[i]=1 means x_train[i] is clean and is_clean[i]=0 that it's poison.\n        :param kwargs: Defence-specific parameters used by child classes.\n        :return: JSON object with confusion matrix.\n        "
        raise NotImplementedError

    def set_params(self, **kwargs) -> None:
        if False:
            print('Hello World!')
        '\n        Take in a dictionary of parameters and apply attack-specific checks before saving them as attributes.\n\n        :param kwargs: A dictionary of defence-specific parameters.\n        '
        for (key, value) in kwargs.items():
            if key in self.defence_params:
                setattr(self, key, value)
        self._check_params()

    def get_params(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Returns dictionary of parameters used to run defence.\n\n        :return: Dictionary of parameters of the method.\n        '
        dictionary = {param: getattr(self, param) for param in self.defence_params}
        return dictionary

    def _check_params(self) -> None:
        if False:
            return 10
        pass