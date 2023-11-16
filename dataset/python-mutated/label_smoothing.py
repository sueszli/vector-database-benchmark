"""
This module implements the label smoothing defence in `LabelSmoothing`. It computes a vector of smooth labels from a
vector of hard labels.

| Paper link: https://pdfs.semanticscholar.org/b5ec/486044c6218dd41b17d8bba502b32a12b91a.pdf

| Please keep in mind the limitations of defences. For details on how to evaluate classifier security in general,
    see https://arxiv.org/abs/1902.06705  .
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Tuple
import numpy as np
from art.defences.preprocessor.preprocessor import Preprocessor
logger = logging.getLogger(__name__)

class LabelSmoothing(Preprocessor):
    """
    Computes a vector of smooth labels from a vector of hard ones. The hard labels have to contain ones for the
    correct classes and zeros for all the others. The remaining probability mass between `max_value` and 1 is
    distributed uniformly between the incorrect classes for each instance.


    | Paper link: https://pdfs.semanticscholar.org/b5ec/486044c6218dd41b17d8bba502b32a12b91a.pdf

    | Please keep in mind the limitations of defences. For details on how to evaluate classifier security in general,
        see https://arxiv.org/abs/1902.06705  .
    """
    params = ['max_value']

    def __init__(self, max_value: float=0.9, apply_fit: bool=True, apply_predict: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Create an instance of label smoothing.\n\n        :param max_value: Value to affect to correct label\n        :param apply_fit: True if applied during fitting/training.\n        :param apply_predict: True if applied during predicting.\n        '
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.max_value = max_value
        self._check_params()

    def __call__(self, x: np.ndarray, y: Optional[np.ndarray]=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if False:
            return 10
        '\n        Apply label smoothing.\n\n        :param x: Input data, will not be modified by this method.\n        :param y: Original vector of label probabilities (one-vs-rest).\n        :return: Unmodified input data and the vector of smooth probabilities as correct labels.\n        :raises `ValueError`: If no labels are provided.\n        '
        if y is None:
            raise ValueError('Labels `y` cannot be None.')
        min_value = (1 - self.max_value) / (y.shape[1] - 1)
        assert self.max_value >= min_value
        smooth_y = y.copy()
        smooth_y[smooth_y == 1.0] = self.max_value
        smooth_y[smooth_y == 0.0] = min_value
        return (x, smooth_y)

    def _check_params(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.max_value <= 0 or self.max_value > 1:
            raise ValueError('The maximum value for correct labels must be between 0 and 1.')