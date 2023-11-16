"""
This module implements attribute inference attacks.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Union, TYPE_CHECKING
import numpy as np
from art.attacks.attack import AttributeInferenceAttack
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from art.estimators.regression.scikitlearn import ScikitlearnDecisionTreeRegressor
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE, REGRESSOR_TYPE
logger = logging.getLogger(__name__)

class AttributeInferenceWhiteBoxLifestyleDecisionTree(AttributeInferenceAttack):
    """
    Implementation of Fredrikson et al. white box inference attack for decision trees.

    Assumes that the attacked feature is discrete or categorical, with limited number of possible values. For example:
    a boolean feature.

    | Paper link: https://dl.acm.org/doi/10.1145/2810103.2813677
    """
    _estimator_requirements = ((ScikitlearnDecisionTreeClassifier, ScikitlearnDecisionTreeRegressor),)

    def __init__(self, estimator: Union['CLASSIFIER_TYPE', 'REGRESSOR_TYPE'], attack_feature: int=0):
        if False:
            i = 10
            return i + 15
        '\n        Create an AttributeInferenceWhiteBoxLifestyle attack instance.\n\n        :param estimator: Target estimator.\n        :param attack_feature: The index of the feature to be attacked.\n        '
        super().__init__(estimator=estimator, attack_feature=attack_feature)
        self.attack_feature: int
        self._check_params()

    def infer(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            return 10
        '\n        Infer the attacked feature.\n\n        :param x: Input to attack. Includes all features except the attacked feature.\n        :param y: Not used.\n        :param values: Possible values for attacked feature.\n        :type values: list\n        :param priors: Prior distributions of attacked feature values. Same size array as `values`.\n        :type priors: list\n        :return: The inferred feature values.\n        :rtype: `np.ndarray`\n        '
        priors: Optional[list] = kwargs.get('priors')
        values: Optional[list] = kwargs.get('values')
        if self.estimator.input_shape[0] != x.shape[1] + 1:
            raise ValueError('Number of features in x + 1 does not match input_shape of classifier')
        if priors is None or values is None:
            raise ValueError('`priors` and `values` are required as inputs.')
        if len(priors) != len(values):
            raise ValueError('Number of priors does not match number of values')
        if self.attack_feature >= x.shape[1]:
            raise ValueError('attack_feature must be a valid index to a feature in x')
        n_samples = x.shape[0]
        phi = self._calculate_phi(x, values, n_samples)
        prob_values = []
        for (i, value) in enumerate(values):
            v_full = np.full((n_samples, 1), value).astype(x.dtype)
            x_value = np.concatenate((x[:, :self.attack_feature], v_full), axis=1)
            x_value = np.concatenate((x_value, x[:, self.attack_feature:]), axis=1)
            prob_value = [self.estimator.get_samples_at_node(self.estimator.get_decision_path([row])[-1]) / n_samples * priors[i] / phi[i] for row in x_value]
            prob_values.append(prob_value)
        return np.array([values[np.argmax(list(prob))] for prob in zip(*prob_values)])

    def _calculate_phi(self, x, values, n_samples):
        if False:
            return 10
        phi = []
        for value in values:
            v_full = np.full((n_samples, 1), value).astype(x.dtype)
            x_value = np.concatenate((x[:, :self.attack_feature], v_full), axis=1)
            x_value = np.concatenate((x_value, x[:, self.attack_feature:]), axis=1)
            nodes_value = {}
            for row in x_value:
                node_id = self.estimator.get_decision_path([row])[-1]
                nodes_value[node_id] = self.estimator.get_samples_at_node(node_id)
            num_value = sum(nodes_value.values()) / n_samples
            phi.append(num_value)
        return phi

    def _check_params(self) -> None:
        if False:
            print('Hello World!')
        super()._check_params()