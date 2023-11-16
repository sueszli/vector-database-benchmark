from __future__ import annotations
import abc
from .estimator import Estimator
from .typing import FeatureName, RegTarget

class MultiLabelClassifier(Estimator, abc.ABC):
    """Multi-label classifier."""

    @abc.abstractmethod
    def learn_one(self, x: dict, y: dict[FeatureName, bool]) -> MultiLabelClassifier:
        if False:
            for i in range(10):
                print('nop')
        'Update the model with a set of features `x` and the labels `y`.\n\n        Parameters\n        ----------\n        x\n            A dictionary of features.\n        y\n            A dictionary of labels.\n\n        Returns\n        -------\n        self\n\n        '

    def predict_proba_one(self, x: dict, **kwargs) -> dict[FeatureName, dict[bool, float]]:
        if False:
            print('Hello World!')
        'Predict the probability of each label appearing given dictionary of features `x`.\n\n        Parameters\n        ----------\n        x\n            A dictionary of features.\n\n        Returns\n        -------\n        A dictionary that associates a probability which each label.\n\n        '
        raise NotImplementedError

    def predict_one(self, x: dict, **kwargs) -> dict[FeatureName, bool]:
        if False:
            for i in range(10):
                print('nop')
        'Predict the labels of a set of features `x`.\n\n        Parameters\n        ----------\n        x\n            A dictionary of features.\n\n        Returns\n        -------\n        The predicted labels.\n\n        '
        probas = self.predict_proba_one(x, **kwargs)
        preds = {}
        for (label_id, label_probas) in probas.items():
            if not label_probas:
                continue
            preds[label_id] = max(label_probas, key=label_probas.get)
        return preds

class MultiTargetRegressor(Estimator, abc.ABC):
    """Multi-target regressor."""

    @abc.abstractmethod
    def learn_one(self, x: dict, y: dict[FeatureName, RegTarget], **kwargs) -> MultiTargetRegressor:
        if False:
            while True:
                i = 10
        'Fits to a set of features `x` and a real-valued target `y`.\n\n        Parameters\n        ----------\n        x\n            A dictionary of features.\n        y\n            A dictionary of numeric targets.\n\n        Returns\n        -------\n        self\n\n        '

    @abc.abstractmethod
    def predict_one(self, x: dict) -> dict[FeatureName, RegTarget]:
        if False:
            return 10
        'Predict the outputs of features `x`.\n\n        Parameters\n        ----------\n        x\n            A dictionary of features.\n\n        Returns\n        -------\n        The predictions.\n\n        '