"""
This module implements the transforming defence mechanism of defensive distillation.

| Paper link: https://arxiv.org/abs/1511.04508
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, TYPE_CHECKING
import numpy as np
from art.defences.transformer.transformer import Transformer
from art.utils import is_probability
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE
logger = logging.getLogger(__name__)

class DefensiveDistillation(Transformer):
    """
    Implement the defensive distillation mechanism.

    | Paper link: https://arxiv.org/abs/1511.04508
    """
    params = ['batch_size', 'nb_epochs']

    def __init__(self, classifier: 'CLASSIFIER_TYPE', batch_size: int=128, nb_epochs: int=10) -> None:
        if False:
            return 10
        '\n        Create an instance of the defensive distillation defence.\n\n        :param classifier: A trained classifier.\n        :param batch_size: Size of batches.\n        :param nb_epochs: Number of epochs to use for training.\n        '
        super().__init__(classifier=classifier)
        self._is_fitted = True
        self.batch_size = batch_size
        self.nb_epochs = nb_epochs
        self._check_params()

    def __call__(self, x: np.ndarray, transformed_classifier: 'CLASSIFIER_TYPE') -> 'CLASSIFIER_TYPE':
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform the defensive distillation defence mechanism and return a robuster classifier.\n\n        :param x: Dataset for training the transformed classifier.\n        :param transformed_classifier: A classifier to be transformed for increased robustness. Note that, the\n            objective loss function used for fitting inside the input transformed_classifier must support soft labels,\n            i.e. probability labels.\n        :return: The transformed classifier.\n        '
        preds = self.classifier.predict(x=x, batch_size=self.batch_size)
        are_probability = [is_probability(y) for y in preds]
        all_probability = np.sum(are_probability) == preds.shape[0]
        if not all_probability:
            raise ValueError('The input trained classifier do not produce probability outputs.')
        transformed_preds = transformed_classifier.predict(x=x, batch_size=self.batch_size)
        are_probability = [is_probability(y) for y in transformed_preds]
        all_probability = np.sum(are_probability) == transformed_preds.shape[0]
        if not all_probability:
            raise ValueError('The input transformed classifier do not produce probability outputs.')
        transformed_classifier.fit(x=x, y=preds, batch_size=self.batch_size, nb_epochs=self.nb_epochs)
        return transformed_classifier

    def fit(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        '\n        No parameters to learn for this method; do nothing.\n        '
        pass

    def _check_params(self) -> None:
        if False:
            i = 10
            return i + 15
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError('The size of batches must be a positive integer.')
        if not isinstance(self.nb_epochs, int) or self.nb_epochs <= 0:
            raise ValueError('The number of epochs must be a positive integer.')