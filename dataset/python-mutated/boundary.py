"""
This module implements the boundary attack `BoundaryAttack`. This is a black-box attack which only requires class
predictions.

| Paper link: https://arxiv.org/abs/1712.04248
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
from tqdm.auto import tqdm, trange
from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import compute_success, to_categorical, check_and_transform_label_format, get_labels_np_array
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE
logger = logging.getLogger(__name__)

class BoundaryAttack(EvasionAttack):
    """
    Implementation of the boundary attack from Brendel et al. (2018). This is a powerful black-box attack that
    only requires final class prediction.

    | Paper link: https://arxiv.org/abs/1712.04248
    """
    attack_params = EvasionAttack.attack_params + ['targeted', 'delta', 'epsilon', 'step_adapt', 'max_iter', 'num_trial', 'sample_size', 'init_size', 'batch_size', 'verbose']
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(self, estimator: 'CLASSIFIER_TYPE', batch_size: int=64, targeted: bool=True, delta: float=0.01, epsilon: float=0.01, step_adapt: float=0.667, max_iter: int=5000, num_trial: int=25, sample_size: int=20, init_size: int=100, min_epsilon: float=0.0, verbose: bool=True) -> None:
        if False:
            while True:
                i = 10
        '\n        Create a boundary attack instance.\n\n        :param estimator: A trained classifier.\n        :param batch_size: The size of the batch used by the estimator during inference.\n        :param targeted: Should the attack target one specific class.\n        :param delta: Initial step size for the orthogonal step.\n        :param epsilon: Initial step size for the step towards the target.\n        :param step_adapt: Factor by which the step sizes are multiplied or divided, must be in the range (0, 1).\n        :param max_iter: Maximum number of iterations.\n        :param num_trial: Maximum number of trials per iteration.\n        :param sample_size: Number of samples per trial.\n        :param init_size: Maximum number of trials for initial generation of adversarial examples.\n        :param min_epsilon: Stop attack if perturbation is smaller than `min_epsilon`.\n        :param verbose: Show progress bars.\n        '
        super().__init__(estimator=estimator)
        self._targeted = targeted
        self.delta = delta
        self.epsilon = epsilon
        self.step_adapt = step_adapt
        self.max_iter = max_iter
        self.num_trial = num_trial
        self.sample_size = sample_size
        self.init_size = init_size
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()
        self.curr_adv: Optional[np.ndarray] = None

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate adversarial samples and return them in an array.\n\n        :param x: An array with the original inputs to be attacked.\n        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape\n                  (nb_samples,). If `self.targeted` is true, then `y` represents the target labels.\n        :param x_adv_init: Initial array to act as initial adversarial examples. Same shape as `x`.\n        :type x_adv_init: `np.ndarray`\n        :return: An array holding the adversarial examples.\n        '
        if y is None:
            if self.targeted:
                raise ValueError('Target labels `y` need to be provided for a targeted attack.')
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes, return_one_hot=False)
        if self.estimator.clip_values is not None:
            (clip_min, clip_max) = self.estimator.clip_values
        else:
            (clip_min, clip_max) = (np.min(x), np.max(x))
        preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size), axis=1)
        x_adv_init = kwargs.get('x_adv_init')
        if x_adv_init is not None:
            init_preds = np.argmax(self.estimator.predict(x_adv_init, batch_size=self.batch_size), axis=1)
        else:
            init_preds = [None] * len(x)
            x_adv_init = [None] * len(x)
        if self.targeted and y is None:
            raise ValueError('Target labels `y` need to be provided for a targeted attack.')
        x_adv = x.astype(ART_NUMPY_DTYPE)
        for (ind, val) in enumerate(tqdm(x_adv, desc='Boundary attack', disable=not self.verbose)):
            if self.targeted:
                x_adv[ind] = self._perturb(x=val, y=y[ind], y_p=preds[ind], init_pred=init_preds[ind], adv_init=x_adv_init[ind], clip_min=clip_min, clip_max=clip_max)
            else:
                x_adv[ind] = self._perturb(x=val, y=-1, y_p=preds[ind], init_pred=init_preds[ind], adv_init=x_adv_init[ind], clip_min=clip_min, clip_max=clip_max)
        y = to_categorical(y, self.estimator.nb_classes)
        logger.info('Success rate of Boundary attack: %.2f%%', 100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size))
        return x_adv

    def _perturb(self, x: np.ndarray, y: int, y_p: int, init_pred: int, adv_init: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Internal attack function for one example.\n\n        :param x: An array with one original input to be attacked.\n        :param y: If `self.targeted` is true, then `y` represents the target label.\n        :param y_p: The predicted label of x.\n        :param init_pred: The predicted label of the initial image.\n        :param adv_init: Initial array to act as an initial adversarial example.\n        :param clip_min: Minimum value of an example.\n        :param clip_max: Maximum value of an example.\n        :return: An adversarial example.\n        '
        initial_sample = self._init_sample(x, y, y_p, init_pred, adv_init, clip_min, clip_max)
        if initial_sample is None:
            return x
        x_adv = self._attack(initial_sample[0], x, y_p, initial_sample[1], self.delta, self.epsilon, clip_min, clip_max)
        return x_adv

    def _attack(self, initial_sample: np.ndarray, original_sample: np.ndarray, y_p: int, target: int, initial_delta: float, initial_epsilon: float, clip_min: float, clip_max: float) -> np.ndarray:
        if False:
            return 10
        '\n        Main function for the boundary attack.\n\n        :param initial_sample: An initial adversarial example.\n        :param original_sample: The original input.\n        :param y_p: The predicted label of the original input.\n        :param target: The target label.\n        :param initial_delta: Initial step size for the orthogonal step.\n        :param initial_epsilon: Initial step size for the step towards the target.\n        :param clip_min: Minimum value of an example.\n        :param clip_max: Maximum value of an example.\n        :return: an adversarial example.\n        '
        x_adv = initial_sample
        self.curr_delta = initial_delta
        self.curr_epsilon = initial_epsilon
        self.curr_adv = x_adv
        for _ in trange(self.max_iter, desc='Boundary attack - iterations', disable=not self.verbose):
            for _ in range(self.num_trial):
                potential_advs = []
                for _ in range(self.sample_size):
                    potential_adv = x_adv + self._orthogonal_perturb(self.curr_delta, x_adv, original_sample)
                    potential_adv = np.clip(potential_adv, clip_min, clip_max)
                    potential_advs.append(potential_adv)
                preds = np.argmax(self.estimator.predict(np.array(potential_advs), batch_size=self.batch_size), axis=1)
                if self.targeted:
                    satisfied = preds == target
                else:
                    satisfied = preds != y_p
                delta_ratio = np.mean(satisfied)
                if delta_ratio < 0.2:
                    self.curr_delta *= self.step_adapt
                elif delta_ratio > 0.5:
                    self.curr_delta /= self.step_adapt
                if delta_ratio > 0:
                    x_advs = np.array(potential_advs)[np.where(satisfied)[0]]
                    break
            else:
                logger.warning('Adversarial example found but not optimal.')
                return x_adv
            for _ in range(self.num_trial):
                perturb = np.repeat(np.array([original_sample]), len(x_advs), axis=0) - x_advs
                perturb *= self.curr_epsilon
                potential_advs = x_advs + perturb
                potential_advs = np.clip(potential_advs, clip_min, clip_max)
                preds = np.argmax(self.estimator.predict(potential_advs, batch_size=self.batch_size), axis=1)
                if self.targeted:
                    satisfied = preds == target
                else:
                    satisfied = preds != y_p
                epsilon_ratio = np.mean(satisfied)
                if epsilon_ratio < 0.2:
                    self.curr_epsilon *= self.step_adapt
                elif epsilon_ratio > 0.5:
                    self.curr_epsilon /= self.step_adapt
                if epsilon_ratio > 0:
                    x_adv = self._best_adv(original_sample, potential_advs[np.where(satisfied)[0]])
                    self.curr_adv = x_adv
                    break
            else:
                logger.warning('Adversarial example found but not optimal.')
                return self._best_adv(original_sample, x_advs)
            if self.curr_epsilon < self.min_epsilon:
                return x_adv
        return x_adv

    def _orthogonal_perturb(self, delta: float, current_sample: np.ndarray, original_sample: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Create an orthogonal perturbation.\n\n        :param delta: Initial step size for the orthogonal step.\n        :param current_sample: Current adversarial example.\n        :param original_sample: The original input.\n        :return: a possible perturbation.\n        '
        perturb = np.random.randn(*self.estimator.input_shape).astype(ART_NUMPY_DTYPE)
        perturb /= np.linalg.norm(perturb)
        perturb *= delta * np.linalg.norm(original_sample - current_sample)
        direction = original_sample - current_sample
        direction_flat = direction.flatten()
        perturb_flat = perturb.flatten()
        direction_flat /= np.linalg.norm(direction_flat)
        perturb_flat -= np.dot(perturb_flat, direction_flat.T) * direction_flat
        perturb = perturb_flat.reshape(self.estimator.input_shape)
        hypotenuse = np.sqrt(1 + delta ** 2)
        perturb = ((1 - hypotenuse) * (current_sample - original_sample) + perturb) / hypotenuse
        return perturb

    def _init_sample(self, x: np.ndarray, y: int, y_p: int, init_pred: int, adv_init: np.ndarray, clip_min: float, clip_max: float) -> Optional[Tuple[np.ndarray, int]]:
        if False:
            print('Hello World!')
        '\n        Find initial adversarial example for the attack.\n\n        :param x: An array with one original input to be attacked.\n        :param y: If `self.targeted` is true, then `y` represents the target label.\n        :param y_p: The predicted label of x.\n        :param init_pred: The predicted label of the initial image.\n        :param adv_init: Initial array to act as an initial adversarial example.\n        :param clip_min: Minimum value of an example.\n        :param clip_max: Maximum value of an example.\n        :return: an adversarial example.\n        '
        nprd = np.random.RandomState()
        initial_sample = None
        if self.targeted:
            if y == y_p:
                return None
            if adv_init is not None and init_pred == y:
                return (adv_init.astype(ART_NUMPY_DTYPE), init_pred)
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                random_class = np.argmax(self.estimator.predict(np.array([random_img]), batch_size=self.batch_size), axis=1)[0]
                if random_class == y:
                    initial_sample = (random_img, random_class)
                    logger.info('Found initial adversarial image for targeted attack.')
                    break
            else:
                logger.warning('Failed to draw a random image that is adversarial, attack failed.')
        else:
            if adv_init is not None and init_pred != y_p:
                return (adv_init.astype(ART_NUMPY_DTYPE), init_pred)
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                random_class = np.argmax(self.estimator.predict(np.array([random_img]), batch_size=self.batch_size), axis=1)[0]
                if random_class != y_p:
                    initial_sample = (random_img, random_class)
                    logger.info('Found initial adversarial image for untargeted attack.')
                    break
            else:
                logger.warning('Failed to draw a random image that is adversarial, attack failed.')
        return initial_sample

    @staticmethod
    def _best_adv(original_sample: np.ndarray, potential_advs: np.ndarray) -> np.ndarray:
        if False:
            return 10
        '\n        From the potential adversarial examples, find the one that has the minimum L2 distance from the original sample\n\n        :param original_sample: The original input.\n        :param potential_advs: Array containing the potential adversarial examples\n        :return: The adversarial example that has the minimum L2 distance from the original input\n        '
        shape = potential_advs.shape
        min_idx = np.linalg.norm(original_sample.flatten() - potential_advs.reshape(shape[0], -1), axis=1).argmin()
        return potential_advs[min_idx]

    def _check_params(self) -> None:
        if False:
            print('Hello World!')
        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError('The number of iterations must be a non-negative integer.')
        if not isinstance(self.num_trial, int) or self.num_trial < 0:
            raise ValueError('The number of trials must be a non-negative integer.')
        if not isinstance(self.sample_size, int) or self.sample_size <= 0:
            raise ValueError('The number of samples must be a positive integer.')
        if not isinstance(self.init_size, int) or self.init_size <= 0:
            raise ValueError('The number of initial trials must be a positive integer.')
        if self.epsilon <= 0:
            raise ValueError('The initial step size for the step towards the target must be positive.')
        if self.delta <= 0:
            raise ValueError('The initial step size for the orthogonal step must be positive.')
        if self.step_adapt <= 0 or self.step_adapt >= 1:
            raise ValueError('The adaptation factor must be in the range (0, 1).')
        if not isinstance(self.min_epsilon, (float, int)) or self.min_epsilon < 0:
            raise ValueError('The minimum epsilon must be non-negative.')
        if not isinstance(self.verbose, bool):
            raise ValueError('The argument `verbose` has to be of type bool.')