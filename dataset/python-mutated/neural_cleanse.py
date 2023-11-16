"""
This module implements Neural Cleanse (Wang et. al. 2019)

| Paper link: http://people.cs.uchicago.edu/~ravenben/publications/abstracts/backdoor-sp19.html
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, TYPE_CHECKING, Union
import numpy as np
from art.defences.transformer.transformer import Transformer
from art.estimators.poison_mitigation.neural_cleanse import KerasNeuralCleanse
from art.estimators.classification.keras import KerasClassifier
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE
logger = logging.getLogger(__name__)

class NeuralCleanse(Transformer):
    """
    Implementation of methods in Neural Cleanse: Identifying and Mitigating Backdoor Attacks in Neural Networks.
    Wang et al. (2019).

    | Paper link: https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf
    """
    params = ['steps', 'init_cost', 'norm', 'learning_rate', 'attack_success_threshold', 'patience', 'early_stop', 'early_stop_threshold', 'early_stop_patience', 'cost_multiplier', 'batch_size']

    def __init__(self, classifier: 'CLASSIFIER_TYPE') -> None:
        if False:
            i = 10
            return i + 15
        '\n        Create an instance of the neural cleanse defence.\n\n        :param classifier: A trained classifier.\n        '
        super().__init__(classifier=classifier)
        self._is_fitted = False
        self._check_params()

    def __call__(self, transformed_classifier: 'CLASSIFIER_TYPE', steps: int=1000, init_cost: float=0.001, norm: Union[int, float]=2, learning_rate: float=0.1, attack_success_threshold: float=0.99, patience: int=5, early_stop: bool=True, early_stop_threshold: float=0.99, early_stop_patience: int=10, cost_multiplier: float=1.5, batch_size: int=32) -> KerasNeuralCleanse:
        if False:
            i = 10
            return i + 15
        '\n        Returns an new classifier with implementation of methods in Neural Cleanse: Identifying and Mitigating Backdoor\n        Attacks in Neural Networks. Wang et al. (2019).\n\n        Namely, the new classifier has a new method mitigate(). This can also affect the predict() function.\n\n        | Paper link: https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf\n\n        :param transformed_classifier: An ART classifier\n        :param steps: The maximum number of steps to run the Neural Cleanse optimization\n        :param init_cost: The initial value for the cost tensor in the Neural Cleanse optimization\n        :param norm: The norm to use for the Neural Cleanse optimization, can be 1, 2, or np.inf\n        :param learning_rate: The learning rate for the Neural Cleanse optimization\n        :param attack_success_threshold: The threshold at which the generated backdoor is successful enough to stop the\n                                         Neural Cleanse optimization\n        :param patience: How long to wait for changing the cost multiplier in the Neural Cleanse optimization\n        :param early_stop: Whether or not to allow early stopping in the Neural Cleanse optimization\n        :param early_stop_threshold: How close values need to come to max value to start counting early stop\n        :param early_stop_patience: How long to wait to determine early stopping in the Neural Cleanse optimization\n        :param cost_multiplier: How much to change the cost in the Neural Cleanse optimization\n        :param batch_size: The batch size for optimizations in the Neural Cleanse optimization\n        '
        transformed_classifier = KerasNeuralCleanse(model=transformed_classifier.model, steps=steps, init_cost=init_cost, norm=norm, learning_rate=learning_rate, attack_success_threshold=attack_success_threshold, patience=patience, early_stop=early_stop, early_stop_threshold=early_stop_threshold, early_stop_patience=early_stop_patience, cost_multiplier=cost_multiplier, batch_size=batch_size)
        return transformed_classifier

    def fit(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        '\n        No parameters to learn for this method; do nothing.\n        '
        raise NotImplementedError

    def _check_params(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(self.classifier, KerasClassifier):
            raise NotImplementedError('Only Keras classifiers are supported for this defence.')