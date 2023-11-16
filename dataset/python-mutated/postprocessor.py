"""
This module implements the abstract base class for defences that post-process classifier output.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from typing import List
import abc
import numpy as np

class Postprocessor(abc.ABC):
    """
    Abstract base class for postprocessing defences. Postprocessing defences are not included in the loss function
    evaluation for loss gradients or the calculation of class gradients.
    """
    params: List[str] = []

    def __init__(self, is_fitted: bool=False, apply_fit: bool=True, apply_predict: bool=True) -> None:
        if False:
            while True:
                i = 10
        '\n        Create a postprocessing object.\n\n        Optionally, set attributes.\n        '
        self._is_fitted = bool(is_fitted)
        self._apply_fit = bool(apply_fit)
        self._apply_predict = bool(apply_predict)
        Postprocessor._check_params(self)

    @property
    def is_fitted(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the state of the postprocessing object.\n\n        :return: `True` if the postprocessing model has been fitted (if this applies).\n        '
        return self._is_fitted

    @property
    def apply_fit(self) -> bool:
        if False:
            return 10
        '\n        Property of the defence indicating if it should be applied at training time.\n\n        :return: `True` if the defence should be applied when fitting a model, `False` otherwise.\n        '
        return self._apply_fit

    @property
    def apply_predict(self) -> bool:
        if False:
            print('Hello World!')
        '\n        Property of the defence indicating if it should be applied at test time.\n\n        :return: `True` if the defence should be applied at prediction time, `False` otherwise.\n        '
        return self._apply_predict

    @abc.abstractmethod
    def __call__(self, preds: np.ndarray) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform model postprocessing and return postprocessed output.\n\n        :param preds: model output to be postprocessed.\n        :return: Postprocessed model output.\n        '
        raise NotImplementedError

    def fit(self, preds: np.ndarray, **kwargs) -> None:
        if False:
            print('Hello World!')
        '\n        Fit the parameters of the postprocessor if it has any.\n\n        :param preds: Training set to fit the postprocessor.\n        :param kwargs: Other parameters.\n        '
        pass

    def set_params(self, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Take in a dictionary of parameters and apply checks before saving them as attributes.\n        '
        for (key, value) in kwargs.items():
            if key in self.params:
                setattr(self, key, value)
        self._check_params()

    def _check_params(self) -> None:
        if False:
            while True:
                i = 10
        pass