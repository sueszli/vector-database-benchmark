"""
This module implements class labels added to the classifier output.
"""
import logging
import numpy as np
from art.defences.postprocessor.postprocessor import Postprocessor
logger = logging.getLogger(__name__)

class ClassLabels(Postprocessor):
    """
    Implementation of a postprocessor based on adding class labels to classifier output.
    """

    def __init__(self, apply_fit: bool=False, apply_predict: bool=True) -> None:
        if False:
            return 10
        '\n        Create a ClassLabels postprocessor.\n\n        :param apply_fit: True if applied during fitting/training.\n        :param apply_predict: True if applied during predicting.\n        '
        super().__init__(is_fitted=True, apply_fit=apply_fit, apply_predict=apply_predict)

    def __call__(self, preds: np.ndarray) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Perform model postprocessing and return postprocessed output.\n\n        :param preds: model output to be postprocessed.\n        :return: Postprocessed model output.\n        '
        class_labels = np.zeros_like(preds)
        if preds.shape[1] > 1:
            index_labels = np.argmax(preds, axis=1)
            class_labels[:, index_labels] = 1
        else:
            class_labels[preds > 0.5] = 1
        return class_labels