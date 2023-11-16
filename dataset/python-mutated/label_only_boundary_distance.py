"""
This module implements the Label-Only Inference Attack based on Decision Boundary.

| Paper link: https://arxiv.org/abs/2007.14321 (Choquette-Choo et al.)
| Paper link: https://arxiv.org/abs/2007.15528 (Li and Zhang)
"""
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

class LabelOnlyDecisionBoundary(MembershipInferenceAttack):
    """
    Implementation of Label-Only Inference Attack based on Decision Boundary.

    You only need to call ONE of the calibrate methods, depending on which attack you want to launch.

    | Paper link: https://arxiv.org/abs/2007.14321 (Choquette-Choo et al.)
    | Paper link: https://arxiv.org/abs/2007.15528 (Li and Zhang)
    """
    attack_params = MembershipInferenceAttack.attack_params + ['distance_threshold_tau']
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(self, estimator: 'CLASSIFIER_TYPE', distance_threshold_tau: Optional[float]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a `LabelOnlyDecisionBoundary` instance for Label-Only Inference Attack based on Decision Boundary.\n\n        :param estimator: A trained classification estimator.\n        :param distance_threshold_tau: Threshold distance for decision boundary. Samples with boundary distances larger\n                                       than threshold are considered members of the training dataset.\n        '
        super().__init__(estimator=estimator)
        self.distance_threshold_tau = distance_threshold_tau
        self.threshold_bins: list = []
        self._check_params()

    def infer(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            return 10
        '\n        Infer membership of input `x` in estimator\'s training data.\n\n        :param x: Input data.\n        :param y: True labels for `x`.\n        :param probabilities: a boolean indicating whether to return the predicted probabilities per class, or just\n                              the predicted class\n\n        :Keyword Arguments for HopSkipJump:\n            * *norm*: Order of the norm. Possible values: "inf", np.inf or 2.\n            * *max_iter*: Maximum number of iterations.\n            * *max_eval*: Maximum number of evaluations for estimating gradient.\n            * *init_eval*: Initial number of evaluations for estimating gradient.\n            * *init_size*: Maximum number of trials for initial generation of adversarial examples.\n            * *verbose*: Show progress bars.\n\n        :return: An array holding the inferred membership status, 1 indicates a member and 0 indicates non-member,\n                 or class probabilities.\n        '
        from art.attacks.evasion.hop_skip_jump import HopSkipJump
        if y is None:
            raise ValueError('Argument `y` is None, but this attack requires true labels `y` to be provided.')
        if self.distance_threshold_tau is None:
            raise ValueError('No value for distance threshold `distance_threshold_tau` provided. Please set`distance_threshold_tau` or run method `calibrate_distance_threshold` on known training and testdataset.')
        if 'probabilities' in kwargs:
            probabilities = kwargs.get('probabilities')
            del kwargs['probabilities']
        else:
            probabilities = False
        if 'classifier' in kwargs:
            raise ValueError('Keyword `classifier` in kwargs is not supported.')
        if 'targeted' in kwargs:
            raise ValueError('Keyword `targeted` in kwargs is not supported.')
        y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        hsj = HopSkipJump(classifier=self.estimator, targeted=False, **kwargs)
        x_adv = hsj.generate(x=x, y=y)
        distance = np.linalg.norm((x_adv - x).reshape((x.shape[0], -1)), ord=2, axis=1)
        y_pred = self.estimator.predict(x=x)
        distance[np.argmax(y_pred, axis=1) != np.argmax(y, axis=1)] = 0
        predicted_class = np.where(distance > self.distance_threshold_tau, 1, 0)
        if probabilities:
            prob_1 = np.zeros_like(distance)
            if self.threshold_bins:
                for t_bin in self.threshold_bins:
                    prob_1[distance > t_bin[0]] = t_bin[1]
            else:
                dist_threshold = distance - self.distance_threshold_tau
                prob_1 = 1 / (1 + np.exp(-dist_threshold))
            prob_0 = np.ones_like(prob_1) - prob_1
            return np.stack((prob_0, prob_1), axis=1)
        return predicted_class

    def calibrate_distance_threshold(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, **kwargs):
        if False:
            print('Hello World!')
        '\n        Calibrate distance threshold maximising the membership inference accuracy on `x_train` and `x_test`.\n\n        | Paper link: https://arxiv.org/abs/2007.14321\n\n        :param x_train: Training data.\n        :param y_train: Labels of training data `x_train`.\n        :param x_test: Test data.\n        :param y_test: Labels of test data `x_test`.\n\n        :Keyword Arguments for HopSkipJump:\n            * *norm*: Order of the norm. Possible values: "inf", np.inf or 2.\n            * *max_iter*: Maximum number of iterations.\n            * *max_eval*: Maximum number of evaluations for estimating gradient.\n            * *init_eval*: Initial number of evaluations for estimating gradient.\n            * *init_size*: Maximum number of trials for initial generation of adversarial examples.\n            * *verbose*: Show progress bars.\n        '
        from art.attacks.evasion.hop_skip_jump import HopSkipJump
        if 'classifier' in kwargs:
            raise ValueError('Keyword `classifier` in kwargs is not supported.')
        if 'targeted' in kwargs:
            raise ValueError('Keyword `targeted` in kwargs is not supported.')
        y_train_onehot = check_and_transform_label_format(y_train, nb_classes=self.estimator.nb_classes)
        if y_train_onehot is None:
            raise ValueError('None value detected.')
        y_test_onehot = check_and_transform_label_format(y_test, nb_classes=self.estimator.nb_classes)
        if y_test_onehot is None:
            raise ValueError('None value detected.')
        hsj = HopSkipJump(classifier=self.estimator, targeted=False, **kwargs)
        x_train_adv = hsj.generate(x=x_train, y=y_train_onehot)
        x_test_adv = hsj.generate(x=x_test, y=y_test_onehot)
        distance_train = np.linalg.norm((x_train_adv - x_train).reshape((x_train.shape[0], -1)), ord=2, axis=1)
        distance_test = np.linalg.norm((x_test_adv - x_test).reshape((x_test.shape[0], -1)), ord=2, axis=1)
        y_train_pred = self.estimator.predict(x=x_train)
        y_test_pred = self.estimator.predict(x=x_test)
        distance_train[np.argmax(y_train_pred, axis=1) != np.argmax(y_train_onehot, axis=1)] = 0
        distance_test[np.argmax(y_test_pred, axis=1) != np.argmax(y_test_onehot, axis=1)] = 0
        num_increments = 100
        tau_increment = np.amax([np.amax(distance_train), np.amax(distance_test)]) / num_increments
        acc_max = 0.0
        distance_threshold_tau = 0.0
        self.threshold_bins = []
        for i_tau in range(1, num_increments):
            is_member_train = np.where(distance_train > i_tau * tau_increment, 1, 0)
            is_member_test = np.where(distance_test > i_tau * tau_increment, 1, 0)
            acc = (np.sum(is_member_train) + (is_member_test.shape[0] - np.sum(is_member_test))) / (is_member_train.shape[0] + is_member_test.shape[0])
            new_threshold_tau = i_tau * tau_increment
            self.threshold_bins.append((new_threshold_tau, acc))
            if acc > acc_max:
                distance_threshold_tau = new_threshold_tau
                acc_max = acc
        self.distance_threshold_tau = distance_threshold_tau

    def calibrate_distance_threshold_unsupervised(self, top_t: int=50, num_samples: int=100, max_queries: int=1, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Calibrate distance threshold on randomly generated samples, choosing the top-t percentile of the noise needed\n        to change the classifier\'s initial prediction. This method requires the model\'s clip_values to be set.\n\n        | Paper link: https://arxiv.org/abs/2007.15528\n\n        :param top_t: Top-t percentile.\n        :param num_samples: Number of random samples to generate.\n        :param max_queries: Maximum number of queries. Maximum number of HSJ iterations on a single sample will be\n                            max_queries * max_iter.\n        :Keyword Arguments for HopSkipJump:\n            * *norm*: Order of the norm. Possible values: "inf", np.inf or 2.\n            * *max_iter*: Maximum number of iterations.\n            * *max_eval*: Maximum number of evaluations for estimating gradient.\n            * *init_eval*: Initial number of evaluations for estimating gradient.\n            * *init_size*: Maximum number of trials for initial generation of adversarial examples.\n            * *verbose*: Show progress bars.\n        '
        from art.attacks.evasion.hop_skip_jump import HopSkipJump
        if self.estimator.clip_values is not None:
            (x_min, x_max) = self.estimator.clip_values
        else:
            raise RuntimeError("You need to set the estimator's clip_values in order to calibrate the distance threshold.")
        x_rand = np.random.rand(*(num_samples,) + self.estimator.input_shape).astype(np.float32)
        x_rand *= x_max - x_min
        x_rand += x_min
        y_rand = self.estimator.predict(x=x_rand)
        y_rand = check_and_transform_label_format(y_rand, nb_classes=self.estimator.nb_classes)
        hsj = HopSkipJump(classifier=self.estimator, targeted=False, **kwargs)
        distances = []
        i = 0
        while len(x_rand) != 0 and i < max_queries:
            x_adv = hsj.generate(x=x_rand, y=y_rand)
            distance = np.linalg.norm((x_adv - x_rand).reshape((x_rand.shape[0], -1)), ord=2, axis=1)
            y_pred = self.estimator.predict(x=x_adv)
            changed_predictions = np.argmax(y_pred, axis=1) != np.argmax(y_rand, axis=1)
            distances.extend(distance[changed_predictions])
            (x_rand, y_rand) = (x_adv[~changed_predictions], y_rand[~changed_predictions])
            i += 1
        if len(distances) == 0:
            raise RuntimeWarning('No successful adversarial examples were generated - no distances were obtained.Distance threshold will not be set.')
        self.distance_threshold_tau = np.percentile(distances, top_t)

    def _check_params(self) -> None:
        if False:
            return 10
        if self.distance_threshold_tau is not None and (not isinstance(self.distance_threshold_tau, (int, float)) or self.distance_threshold_tau <= 0.0):
            raise ValueError('The distance threshold `distance_threshold_tau` needs to be a positive float.')