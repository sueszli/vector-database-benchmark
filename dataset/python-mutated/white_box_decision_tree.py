"""
This module implements attribute inference attacks.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional
import numpy as np
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from art.attacks.attack import AttributeInferenceAttack
logger = logging.getLogger(__name__)

class AttributeInferenceWhiteBoxDecisionTree(AttributeInferenceAttack):
    """
    A variation of the method proposed by of Fredrikson et al. in:
    https://dl.acm.org/doi/10.1145/2810103.2813677

    Assumes the availability of the attacked model's predictions for the samples under attack, in addition to access to
    the model itself and the rest of the feature values. If this is not available, the true class label of the samples
    may be used as a proxy. Also assumes that the attacked feature is discrete or categorical, with limited number of
    possible values. For example: a boolean feature.

    | Paper link: https://dl.acm.org/doi/10.1145/2810103.2813677
    """
    _estimator_requirements = (ScikitlearnDecisionTreeClassifier,)

    def __init__(self, classifier: ScikitlearnDecisionTreeClassifier, attack_feature: int=0):
        if False:
            i = 10
            return i + 15
        '\n        Create an AttributeInferenceWhiteBox attack instance.\n\n        :param classifier: Target classifier.\n        :param attack_feature: The index of the feature to be attacked.\n        '
        super().__init__(estimator=classifier, attack_feature=attack_feature)
        self.attack_feature: int
        self._check_params()

    def infer(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        "\n        Infer the attacked feature.\n\n        If the model's prediction coincides with the real prediction for the sample for a single value, choose it as the\n        predicted value. If not, fall back to the Fredrikson method (without phi)\n\n        :param x: Input to attack. Includes all features except the attacked feature.\n        :param y: Original model's predictions for x.\n        :param values: Possible values for attacked feature.\n        :type values: list\n        :param priors: Prior distributions of attacked feature values. Same size array as `values`.\n        :type priors: list\n        :return: The inferred feature values.\n        "
        if 'priors' not in kwargs:
            raise ValueError('Missing parameter `priors`.')
        if 'values' not in kwargs:
            raise ValueError('Missing parameter `values`.')
        priors: Optional[list] = kwargs.get('priors')
        values: Optional[list] = kwargs.get('values')
        if self.estimator.input_shape[0] != x.shape[1] + 1:
            raise ValueError('Number of features in x + 1 does not match input_shape of classifier')
        if priors is None or values is None:
            raise ValueError('`priors` and `values` are required as inputs.')
        if len(priors) != len(values):
            raise ValueError('Number of priors does not match number of values')
        if y is not None and y.shape[0] != x.shape[0]:
            raise ValueError('Number of rows in x and y do not match')
        if self.attack_feature >= x.shape[1]:
            raise ValueError('attack_feature must be a valid index to a feature in x')
        n_values = len(values)
        n_samples = x.shape[0]
        pred_values = []
        prob_values = []
        for (i, value) in enumerate(values):
            v_full = np.full((n_samples, 1), value).astype(x.dtype)
            x_value = np.concatenate((x[:, :self.attack_feature], v_full), axis=1)
            x_value = np.concatenate((x_value, x[:, self.attack_feature:]), axis=1)
            pred_value = [np.argmax(arr) for arr in self.estimator.predict(x_value)]
            pred_values.append(pred_value)
            prob_value = [self.estimator.get_samples_at_node(self.estimator.get_decision_path([row])[-1]) / n_samples * priors[i] for row in x_value]
            prob_values.append(prob_value)
        pred_rows = zip(*pred_values)
        predicted_pred = []
        for (row_index, row) in enumerate(pred_rows):
            if y is not None:
                matches = [1 if row[value_index] == y[row_index] else 0 for value_index in range(n_values)]
                match_values = [values[value_index] if row[value_index] == y[row_index] else 0 for value_index in range(n_values)]
            else:
                matches = [0 for _ in range(n_values)]
                match_values = [0 for _ in range(n_values)]
            predicted_pred.append(sum(match_values) if sum(matches) == 1 else None)
        predicted_prob = [np.argmax(list(prob)) for prob in zip(*prob_values)]
        return np.array([value if value is not None else values[predicted_prob[index]] for (index, value) in enumerate(predicted_pred)])

    def _check_params(self) -> None:
        if False:
            i = 10
            return i + 15
        super()._check_params()