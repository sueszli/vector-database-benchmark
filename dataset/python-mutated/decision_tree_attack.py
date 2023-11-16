"""
This module implements attacks on Decision Trees.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import List, Optional, Union
import numpy as np
from tqdm.auto import trange
from art.attacks.attack import EvasionAttack
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from art.utils import check_and_transform_label_format
logger = logging.getLogger(__name__)

class DecisionTreeAttack(EvasionAttack):
    """
    Close implementation of Papernot's attack on decision trees following Algorithm 2 and communication with the
    authors.

    | Paper link: https://arxiv.org/abs/1605.07277
    """
    attack_params = ['classifier', 'offset', 'verbose']
    _estimator_requirements = (ScikitlearnDecisionTreeClassifier,)

    def __init__(self, classifier: ScikitlearnDecisionTreeClassifier, offset: float=0.001, verbose: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        :param classifier: A trained scikit-learn decision tree model.\n        :param offset: How much the value is pushed away from tree's threshold.\n        :param verbose: Show progress bars.\n        "
        super().__init__(estimator=classifier)
        self.offset = offset
        self.verbose = verbose
        self._check_params()

    def _df_subtree(self, position: int, original_class: Union[int, np.ndarray], target: Optional[int]=None) -> List[int]:
        if False:
            return 10
        '\n        Search a decision tree for a mis-classifying instance.\n\n        :param position: An array with the original inputs to be attacked.\n        :param original_class: original label for the instances we are searching mis-classification for.\n        :param target: If the provided, specifies which output the leaf has to have to be accepted.\n        :return: An array specifying the path to the leaf where the classification is either != original class or\n                 ==target class if provided.\n        '
        if self.estimator.get_left_child(position) == self.estimator.get_right_child(position):
            if target is None:
                if self.estimator.get_classes_at_node(position) != original_class:
                    path = [position]
                else:
                    path = [-1]
            elif self.estimator.get_classes_at_node(position) == target:
                path = [position]
            else:
                path = [-1]
        else:
            res = self._df_subtree(self.estimator.get_left_child(position), original_class, target)
            if res[0] == -1:
                res = self._df_subtree(self.estimator.get_right_child(position), original_class, target)
                if res[0] == -1:
                    path = [-1]
                else:
                    res.append(position)
                    path = res
            else:
                res.append(position)
                path = res
        return path

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Generate adversarial examples and return them as an array.\n\n        :param x: An array with the original inputs to be attacked.\n        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape\n                  (nb_samples,).\n        :return: An array holding the adversarial examples.\n        '
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes, return_one_hot=False)
        x_adv = x.copy()
        for index in trange(x_adv.shape[0], desc='Decision tree attack', disable=not self.verbose):
            path = self.estimator.get_decision_path(x_adv[index])
            legitimate_class = int(np.argmax(self.estimator.predict(x_adv[index].reshape(1, -1))))
            position = -2
            adv_path = [-1]
            ancestor = path[position]
            while np.abs(position) < len(path) - 1 or adv_path[0] == -1:
                ancestor = path[position]
                current_child = path[position + 1]
                if current_child == self.estimator.get_left_child(ancestor):
                    if y is None:
                        adv_path = self._df_subtree(self.estimator.get_right_child(ancestor), legitimate_class)
                    else:
                        adv_path = self._df_subtree(self.estimator.get_right_child(ancestor), legitimate_class, y[index])
                elif y is None:
                    adv_path = self._df_subtree(self.estimator.get_left_child(ancestor), legitimate_class)
                else:
                    adv_path = self._df_subtree(self.estimator.get_left_child(ancestor), legitimate_class, y[index])
                position = position - 1
            adv_path.append(ancestor)
            for i in range(1, 1 + len(adv_path[1:])):
                go_for = adv_path[i - 1]
                threshold = self.estimator.get_threshold_at_node(adv_path[i])
                feature = self.estimator.get_feature_at_node(adv_path[i])
                if x_adv[index][feature] > threshold and go_for == self.estimator.get_left_child(adv_path[i]):
                    x_adv[index][feature] = threshold - self.offset
                elif x_adv[index][feature] <= threshold and go_for == self.estimator.get_right_child(adv_path[i]):
                    x_adv[index][feature] = threshold + self.offset
        return x_adv

    def _check_params(self) -> None:
        if False:
            while True:
                i = 10
        if self.offset <= 0:
            raise ValueError('The offset parameter must be strictly positive.')
        if not isinstance(self.verbose, bool):
            raise ValueError('The argument `verbose` has to be of type bool.')