"""
This module implements Gaussian noise added to the classifier output.
"""
import logging
import numpy as np
from art.defences.postprocessor.postprocessor import Postprocessor
from art.utils import is_probability
logger = logging.getLogger(__name__)

class GaussianNoise(Postprocessor):
    """
    Implementation of a postprocessor based on adding Gaussian noise to classifier output.
    """
    params = ['scale']

    def __init__(self, scale: float=0.2, apply_fit: bool=False, apply_predict: bool=True) -> None:
        if False:
            print('Hello World!')
        '\n        Create a GaussianNoise postprocessor.\n\n        :param scale: Standard deviation of the distribution.\n        :param apply_fit: True if applied during fitting/training.\n        :param apply_predict: True if applied during predicting.\n        '
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)
        self.scale = scale
        self._check_params()

    def __call__(self, preds: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Perform model postprocessing and return postprocessed output.\n\n        :param preds: model output to be postprocessed.\n        :return: Postprocessed model output.\n        '
        noise = np.random.normal(loc=0.0, scale=self.scale, size=preds.shape)
        post_preds = preds.copy()
        post_preds += noise
        if preds.shape[1] > 1:
            are_probability = [is_probability(x) for x in preds]
            all_probability = np.sum(are_probability) == preds.shape[0]
            if all_probability:
                post_preds[post_preds < 0.0] = 0.0
                sums = np.sum(post_preds, axis=1, keepdims=True)
                post_preds /= sums
        else:
            post_preds[post_preds < 0.0] = 0.0
        return post_preds

    def _check_params(self) -> None:
        if False:
            i = 10
            return i + 15
        if self.scale <= 0:
            raise ValueError('Standard deviation must be positive.')