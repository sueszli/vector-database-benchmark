"""
This module implements membership inference attacks.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, TYPE_CHECKING
import numpy as np
from art.attacks.attack import MembershipInferenceAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE
logger = logging.getLogger(__name__)

class MembershipInferenceBlackBoxRuleBased(MembershipInferenceAttack):
    """
    Implementation of a simple, rule-based black-box membership inference attack.

    This implementation uses the simple rule: if the model's prediction for a sample is correct, then it is a
    member. Otherwise, it is not a member.
    """
    attack_params = MembershipInferenceAttack.attack_params
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(self, classifier: 'CLASSIFIER_TYPE'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a MembershipInferenceBlackBoxRuleBased attack instance.\n\n        :param classifier: Target classifier.\n        '
        super().__init__(estimator=classifier)

    def infer(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Infer membership in the training set of the target estimator.\n\n        :param x: Input records to attack.\n        :param y: True labels for `x`.\n        :param probabilities: a boolean indicating whether to return the predicted probabilities per class, or just\n                              the predicted class.\n        :return: An array holding the inferred membership status, 1 indicates a member and 0 indicates non-member,\n                 or class probabilities.\n        '
        if y is None:
            raise ValueError('MembershipInferenceBlackBoxRuleBased requires true labels `y`.')
        if self.estimator.input_shape is not None:
            if self.estimator.input_shape[0] != x.shape[1]:
                raise ValueError('Shape of x does not match input_shape of classifier')
        if 'probabilities' in kwargs:
            probabilities = kwargs.get('probabilities')
        else:
            probabilities = False
        y = check_and_transform_label_format(y, nb_classes=len(np.unique(y)), return_one_hot=True)
        if y is None:
            raise ValueError('None value detected.')
        if y.shape[0] != x.shape[0]:
            raise ValueError('Number of rows in x and y do not match')
        y_pred = self.estimator.predict(x=x)
        predicted_class = (np.argmax(y, axis=1) == np.argmax(y_pred, axis=1)).astype(int)
        if probabilities:
            if y_pred.shape[1] == 2:
                pred_prob = np.max(y_pred, axis=1)
                prob = np.zeros((predicted_class.shape[0], 2))
                prob[:, predicted_class] = pred_prob
                prob[:, np.ones_like(predicted_class) - predicted_class] = np.ones_like(pred_prob) - pred_prob
            else:
                prob_none = check_and_transform_label_format(predicted_class, nb_classes=2, return_one_hot=True)
                if prob_none is not None:
                    prob = prob_none
            return prob
        return predicted_class