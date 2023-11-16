from __future__ import annotations
import abc
from . import estimator

class Clusterer(estimator.Estimator):
    """A clustering model."""

    @property
    def _supervised(self):
        if False:
            while True:
                i = 10
        return False

    @abc.abstractmethod
    def learn_one(self, x: dict) -> Clusterer:
        if False:
            print('Hello World!')
        'Update the model with a set of features `x`.\n\n        Parameters\n        ----------\n        x\n            A dictionary of features.\n\n        Returns\n        -------\n        self\n\n        '

    @abc.abstractmethod
    def predict_one(self, x: dict) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Predicts the cluster number for a set of features `x`.\n\n        Parameters\n        ----------\n        x\n            A dictionary of features.\n\n        Returns\n        -------\n        A cluster number.\n\n        '