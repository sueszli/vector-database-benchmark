"""
This module implements confidence added to the classifier output.
"""
import logging
import numpy as np
from art.defences.postprocessor.postprocessor import Postprocessor
logger = logging.getLogger(__name__)

class HighConfidence(Postprocessor):
    """
    Implementation of a postprocessor based on selecting high confidence predictions to return as classifier output.
    """
    params = ['cutoff']

    def __init__(self, cutoff: float=0.25, apply_fit: bool=False, apply_predict: bool=True) -> None:
        if False:
            return 10
        '\n        Create a HighConfidence postprocessor.\n\n        :param cutoff: Minimal value for returned prediction output.\n        :param apply_fit: True if applied during fitting/training.\n        :param apply_predict: True if applied during predicting.\n        '
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.cutoff = cutoff
        self._check_params()

    def __call__(self, preds: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Perform model postprocessing and return postprocessed output.\n\n        :param preds: model output to be postprocessed.\n        :return: Postprocessed model output.\n        '
        post_preds = preds.copy()
        post_preds[post_preds < self.cutoff] = 0.0
        return post_preds

    def _check_params(self) -> None:
        if False:
            return 10
        if self.cutoff <= 0:
            raise ValueError('Minimal value must be positive.')