"""
This module implements a rounding to the classifier output.
"""
import logging
import numpy as np
from art.defences.postprocessor.postprocessor import Postprocessor
logger = logging.getLogger(__name__)

class Rounded(Postprocessor):
    """
    Implementation of a postprocessor based on rounding classifier output.
    """
    params = ['decimals']

    def __init__(self, decimals: int=3, apply_fit: bool=False, apply_predict: bool=True) -> None:
        if False:
            while True:
                i = 10
        '\n        Create a Rounded postprocessor.\n\n        :param decimals: Number of decimal places after the decimal point.\n        :param apply_fit: True if applied during fitting/training.\n        :param apply_predict: True if applied during predicting.\n        '
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.decimals = decimals
        self._check_params()

    def __call__(self, preds: np.ndarray) -> np.ndarray:
        if False:
            return 10
        '\n        Perform model postprocessing and return postprocessed output.\n\n        :param preds: model output to be postprocessed.\n        :return: Postprocessed model output.\n        '
        return np.around(preds, decimals=self.decimals)

    def _check_params(self) -> None:
        if False:
            return 10
        if not isinstance(self.decimals, int) or self.decimals <= 0:
            raise ValueError('Number of decimal places must be a positive integer.')