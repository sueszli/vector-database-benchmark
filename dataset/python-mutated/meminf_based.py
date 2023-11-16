"""
This module implements attribute inference attacks using membership inference attacks.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Union, List, TYPE_CHECKING
import numpy as np
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.attacks.attack import AttributeInferenceAttack, MembershipInferenceAttack
from art.estimators.regression import RegressorMixin
from art.exceptions import EstimatorError
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE, REGRESSOR_TYPE
logger = logging.getLogger(__name__)

class AttributeInferenceMembership(AttributeInferenceAttack):
    """
    Implementation of a an attribute inference attack that utilizes a membership inference attack.

    The idea is to find the target feature value that causes the membership inference attack to classify the sample
    as a member with the highest confidence.
    """
    _estimator_requirements = (BaseEstimator, (ClassifierMixin, RegressorMixin))

    def __init__(self, estimator: Union['CLASSIFIER_TYPE', 'REGRESSOR_TYPE'], membership_attack: MembershipInferenceAttack, attack_feature: Union[int, slice]=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create an AttributeInferenceMembership attack instance.\n\n        :param estimator: Target estimator.\n        :param membership_attack: The membership inference attack to use. Should be fit/calibrated in advance, and\n                                  should support returning probabilities. Should also support the target estimator.\n        :param attack_feature: The index of the feature to be attacked or a slice representing multiple indexes in\n                               case of a one-hot encoded feature.\n        '
        super().__init__(estimator=estimator, attack_feature=attack_feature)
        if not membership_attack.is_estimator_valid(estimator, estimator_requirements=self.estimator_requirements):
            raise EstimatorError(membership_attack.__class__, membership_attack.estimator_requirements, estimator)
        self.membership_attack = membership_attack
        self._check_params()

    def infer(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Infer the attacked feature.\n\n        :param x: Input to attack. Includes all features except the attacked feature.\n        :param y: The labels expected by the membership attack.\n        :param values: Possible values for attacked feature. For a single column feature this should be a simple list\n                       containing all possible values, in increasing order (the smallest value in the 0 index and so\n                       on). For a multi-column feature (for example 1-hot encoded and then scaled), this should be a\n                       list of lists, where each internal list represents a column (in increasing order) and the values\n                       represent the possible values for that column (in increasing order).\n        :type values: list\n        :return: The inferred feature values.\n        '
        if self.estimator.input_shape is not None:
            if isinstance(self.attack_feature, int) and self.estimator.input_shape[0] != x.shape[1] + 1:
                raise ValueError('Number of features in x + 1 does not match input_shape of the estimator')
        if 'values' not in kwargs:
            raise ValueError('Missing parameter `values`.')
        values: Optional[List] = kwargs.get('values')
        if not values:
            raise ValueError('`values` cannot be None or empty')
        if y is not None:
            if y.shape[0] != x.shape[0]:
                raise ValueError('Number of rows in x and y do not match')
        if isinstance(self.attack_feature, int):
            first = True
            for value in values:
                v_full = np.full((x.shape[0], 1), value).astype(x.dtype)
                x_value = np.concatenate((x[:, :self.attack_feature], v_full), axis=1)
                x_value = np.concatenate((x_value, x[:, self.attack_feature:]), axis=1)
                predicted = self.membership_attack.infer(x_value, y, probabilities=True)
                if first:
                    probabilities = predicted
                    first = False
                else:
                    probabilities = np.hstack((probabilities, predicted))
            value_indexes = np.argmax(probabilities, axis=1).astype(x.dtype)
            pred_values = np.zeros_like(value_indexes)
            for (index, value) in enumerate(values):
                pred_values[value_indexes == index] = value
        else:
            first = True
            for (index, value) in enumerate(values):
                curr_value = np.zeros((x.shape[0], len(values)))
                curr_value[:, index] = value[1]
                for (not_index, not_value) in enumerate(values):
                    if not_index != index:
                        curr_value[:, not_index] = not_value[0]
                x_value = np.concatenate((x[:, :self.attack_feature.start], curr_value), axis=1)
                x_value = np.concatenate((x_value, x[:, self.attack_feature.start:]), axis=1)
                predicted = self.membership_attack.infer(x_value, y, probabilities=True)
                if first:
                    probabilities = predicted
                else:
                    probabilities = np.hstack((probabilities, predicted))
                first = False
            value_indexes = np.argmax(probabilities, axis=1).astype(x.dtype)
            pred_values = np.zeros_like(probabilities)
            for (index, value) in enumerate(values):
                curr_value = np.zeros(len(values))
                curr_value[index] = value[1]
                for (not_index, not_value) in enumerate(values):
                    if not_index != index:
                        curr_value[not_index] = not_value[0]
                pred_values[value_indexes == index] = curr_value
        return pred_values

    def _check_params(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super()._check_params()
        if not isinstance(self.membership_attack, MembershipInferenceAttack):
            raise ValueError('membership_attack should be a sub-class of MembershipInferenceAttack')