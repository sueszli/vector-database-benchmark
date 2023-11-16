"""
This module implements the HopSkipJump attack `HopSkipJump`. This is a black-box attack that only requires class
predictions. It is an advanced version of the Boundary attack.

| Paper link: https://arxiv.org/abs/1904.02144
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
from tqdm.auto import tqdm
from art.config import ART_NUMPY_DTYPE
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification import ClassifierMixin
from art.utils import compute_success, to_categorical, check_and_transform_label_format, get_labels_np_array
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE
logger = logging.getLogger(__name__)

class HopSkipJump(EvasionAttack):
    """
    Implementation of the HopSkipJump attack from Jianbo et al. (2019). This is a powerful black-box attack that
    only requires final class prediction, and is an advanced version of the boundary attack.

    | Paper link: https://arxiv.org/abs/1904.02144
    """
    attack_params = EvasionAttack.attack_params + ['targeted', 'norm', 'max_iter', 'max_eval', 'init_eval', 'init_size', 'curr_iter', 'batch_size', 'verbose']
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(self, classifier: 'CLASSIFIER_TYPE', batch_size: int=64, targeted: bool=False, norm: Union[int, float, str]=2, max_iter: int=50, max_eval: int=10000, init_eval: int=100, init_size: int=100, verbose: bool=True) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a HopSkipJump attack instance.\n\n        :param classifier: A trained classifier.\n        :param batch_size: The size of the batch used by the estimator during inference.\n        :param targeted: Should the attack target one specific class.\n        :param norm: Order of the norm. Possible values: "inf", np.inf or 2.\n        :param max_iter: Maximum number of iterations.\n        :param max_eval: Maximum number of evaluations for estimating gradient.\n        :param init_eval: Initial number of evaluations for estimating gradient.\n        :param init_size: Maximum number of trials for initial generation of adversarial examples.\n        :param verbose: Show progress bars.\n        '
        super().__init__(estimator=classifier)
        self._targeted = targeted
        self.norm = norm
        self.max_iter = max_iter
        self.max_eval = max_eval
        self.init_eval = init_eval
        self.init_size = init_size
        self.curr_iter = 0
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()
        self.curr_iter = 0
        if norm == 2:
            self.theta = 0.01 / np.sqrt(np.prod(self.estimator.input_shape))
        else:
            self.theta = 0.01 / np.prod(self.estimator.input_shape)

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Generate adversarial samples and return them in an array.\n\n        :param x: An array with the original inputs to be attacked.\n        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape\n                  (nb_samples,).\n        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.\n                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any\n                     features for which the mask is zero will not be adversarially perturbed.\n        :type mask: `np.ndarray`\n        :param x_adv_init: Initial array to act as initial adversarial examples. Same shape as `x`.\n        :type x_adv_init: `np.ndarray`\n        :param resume: Allow users to continue their previous attack.\n        :type resume: `bool`\n        :return: An array holding the adversarial examples.\n        '
        mask = kwargs.get('mask')
        if y is None:
            if self.targeted:
                raise ValueError('Target labels `y` need to be provided for a targeted attack.')
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        if self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError('This attack has not yet been tested for binary classification with a single output classifier.')
        resume = kwargs.get('resume')
        if resume is not None and resume:
            start = self.curr_iter
        else:
            start = 0
        if mask is not None:
            if len(mask.shape) == len(x.shape):
                mask = mask.astype(ART_NUMPY_DTYPE)
            else:
                mask = np.array([mask.astype(ART_NUMPY_DTYPE)] * x.shape[0])
        else:
            mask = np.array([None] * x.shape[0])
        if self.estimator.clip_values is not None:
            (clip_min, clip_max) = self.estimator.clip_values
        else:
            (clip_min, clip_max) = (np.min(x), np.max(x))
        preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size), axis=1)
        x_adv_init = kwargs.get('x_adv_init')
        if x_adv_init is not None:
            for i in range(x.shape[0]):
                if mask[i] is not None:
                    x_adv_init[i] = x_adv_init[i] * mask[i] + x[i] * (1 - mask[i])
            init_preds = np.argmax(self.estimator.predict(x_adv_init, batch_size=self.batch_size), axis=1)
        else:
            init_preds = [None] * len(x)
            x_adv_init = [None] * len(x)
        if self.targeted and y is None:
            raise ValueError('Target labels `y` need to be provided for a targeted attack.')
        x_adv = x.astype(ART_NUMPY_DTYPE)
        y = np.argmax(y, axis=1)
        for (ind, val) in enumerate(tqdm(x_adv, desc='HopSkipJump', disable=not self.verbose)):
            self.curr_iter = start
            if self.targeted:
                x_adv[ind] = self._perturb(x=val, y=y[ind], y_p=preds[ind], init_pred=init_preds[ind], adv_init=x_adv_init[ind], mask=mask[ind], clip_min=clip_min, clip_max=clip_max)
            else:
                x_adv[ind] = self._perturb(x=val, y=-1, y_p=preds[ind], init_pred=init_preds[ind], adv_init=x_adv_init[ind], mask=mask[ind], clip_min=clip_min, clip_max=clip_max)
        y = to_categorical(y, self.estimator.nb_classes)
        logger.info('Success rate of HopSkipJump attack: %.2f%%', 100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size))
        return x_adv

    def _perturb(self, x: np.ndarray, y: int, y_p: int, init_pred: int, adv_init: np.ndarray, mask: Optional[np.ndarray], clip_min: float, clip_max: float) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Internal attack function for one example.\n\n        :param x: An array with one original input to be attacked.\n        :param y: If `self.targeted` is true, then `y` represents the target label.\n        :param y_p: The predicted label of x.\n        :param init_pred: The predicted label of the initial image.\n        :param adv_init: Initial array to act as an initial adversarial example.\n        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be\n                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially\n                     perturbed.\n        :param clip_min: Minimum value of an example.\n        :param clip_max: Maximum value of an example.\n        :return: An adversarial example.\n        '
        initial_sample = self._init_sample(x, y, y_p, init_pred, adv_init, mask, clip_min, clip_max)
        if initial_sample is None:
            return x
        x_adv = self._attack(initial_sample[0], x, initial_sample[1], mask, clip_min, clip_max)
        return x_adv

    def _init_sample(self, x: np.ndarray, y: int, y_p: int, init_pred: int, adv_init: np.ndarray, mask: Optional[np.ndarray], clip_min: float, clip_max: float) -> Optional[Union[np.ndarray, Tuple[np.ndarray, int]]]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Find initial adversarial example for the attack.\n\n        :param x: An array with 1 original input to be attacked.\n        :param y: If `self.targeted` is true, then `y` represents the target label.\n        :param y_p: The predicted label of x.\n        :param init_pred: The predicted label of the initial image.\n        :param adv_init: Initial array to act as an initial adversarial example.\n        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be\n                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially\n                     perturbed.\n        :param clip_min: Minimum value of an example.\n        :param clip_max: Maximum value of an example.\n        :return: An adversarial example.\n        '
        nprd = np.random.RandomState()
        initial_sample = None
        if self.targeted:
            if y == y_p:
                return None
            if adv_init is not None and init_pred == y:
                return (adv_init.astype(ART_NUMPY_DTYPE), init_pred)
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                if mask is not None:
                    random_img = random_img * mask + x * (1 - mask)
                random_class = np.argmax(self.estimator.predict(np.array([random_img]), batch_size=self.batch_size), axis=1)[0]
                if random_class == y:
                    random_img = self._binary_search(current_sample=random_img, original_sample=x, target=y, norm=2, clip_min=clip_min, clip_max=clip_max, threshold=0.001)
                    initial_sample = (random_img, random_class)
                    logger.info('Found initial adversarial image for targeted attack.')
                    break
            else:
                logger.warning('Failed to draw a random image that is adversarial, attack failed.')
        else:
            if adv_init is not None and init_pred != y_p:
                return (adv_init.astype(ART_NUMPY_DTYPE), y_p)
            for _ in range(self.init_size):
                random_img = nprd.uniform(clip_min, clip_max, size=x.shape).astype(x.dtype)
                if mask is not None:
                    random_img = random_img * mask + x * (1 - mask)
                random_class = np.argmax(self.estimator.predict(np.array([random_img]), batch_size=self.batch_size), axis=1)[0]
                if random_class != y_p:
                    random_img = self._binary_search(current_sample=random_img, original_sample=x, target=y_p, norm=2, clip_min=clip_min, clip_max=clip_max, threshold=0.001)
                    initial_sample = (random_img, y_p)
                    logger.info('Found initial adversarial image for untargeted attack.')
                    break
            else:
                logger.warning('Failed to draw a random image that is adversarial, attack failed.')
        return initial_sample

    def _attack(self, initial_sample: np.ndarray, original_sample: np.ndarray, target: int, mask: Optional[np.ndarray], clip_min: float, clip_max: float) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Main function for the boundary attack.\n\n        :param initial_sample: An initial adversarial example.\n        :param original_sample: The original input.\n        :param target: The target label.\n        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be\n                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially\n                     perturbed.\n        :param clip_min: Minimum value of an example.\n        :param clip_max: Maximum value of an example.\n        :return: an adversarial example.\n        '
        current_sample = initial_sample
        for _ in range(self.max_iter):
            delta = self._compute_delta(current_sample=current_sample, original_sample=original_sample, clip_min=clip_min, clip_max=clip_max)
            current_sample = self._binary_search(current_sample=current_sample, original_sample=original_sample, norm=self.norm, target=target, clip_min=clip_min, clip_max=clip_max)
            num_eval = min(int(self.init_eval * np.sqrt(self.curr_iter + 1)), self.max_eval)
            update = self._compute_update(current_sample=current_sample, num_eval=num_eval, delta=delta, target=target, mask=mask, clip_min=clip_min, clip_max=clip_max)
            if self.norm == 2:
                dist = np.linalg.norm(original_sample - current_sample)
            else:
                dist = np.max(abs(original_sample - current_sample))
            epsilon = 2.0 * dist / np.sqrt(self.curr_iter + 1)
            success = False
            while not success:
                epsilon /= 2.0
                potential_sample = current_sample + epsilon * update
                success = self._adversarial_satisfactory(samples=potential_sample[None], target=target, clip_min=clip_min, clip_max=clip_max)
            current_sample = np.clip(potential_sample, clip_min, clip_max)
            self.curr_iter += 1
            if np.isnan(current_sample).any():
                logger.debug('NaN detected in sample, returning original sample.')
                return original_sample
        return current_sample

    def _binary_search(self, current_sample: np.ndarray, original_sample: np.ndarray, target: int, norm: Union[int, float, str], clip_min: float, clip_max: float, threshold: Optional[float]=None) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Binary search to approach the boundary.\n\n        :param current_sample: Current adversarial example.\n        :param original_sample: The original input.\n        :param target: The target label.\n        :param norm: Order of the norm. Possible values: "inf", np.inf or 2.\n        :param clip_min: Minimum value of an example.\n        :param clip_max: Maximum value of an example.\n        :param threshold: The upper threshold in binary search.\n        :return: an adversarial example.\n        '
        if norm == 2:
            (upper_bound, lower_bound) = (1, 0)
            if threshold is None:
                threshold = self.theta
        else:
            (upper_bound, lower_bound) = (np.max(abs(original_sample - current_sample)), 0)
            if threshold is None:
                threshold = np.minimum(upper_bound * self.theta, self.theta)
        while upper_bound - lower_bound > threshold:
            alpha = (upper_bound + lower_bound) / 2.0
            interpolated_sample = self._interpolate(current_sample=current_sample, original_sample=original_sample, alpha=alpha, norm=norm)
            satisfied = self._adversarial_satisfactory(samples=interpolated_sample[None], target=target, clip_min=clip_min, clip_max=clip_max)[0]
            lower_bound = np.where(satisfied == 0, alpha, lower_bound)
            upper_bound = np.where(satisfied == 1, alpha, upper_bound)
        result = self._interpolate(current_sample=current_sample, original_sample=original_sample, alpha=upper_bound, norm=norm)
        return result

    def _compute_delta(self, current_sample: np.ndarray, original_sample: np.ndarray, clip_min: float, clip_max: float) -> float:
        if False:
            return 10
        '\n        Compute the delta parameter.\n\n        :param current_sample: Current adversarial example.\n        :param original_sample: The original input.\n        :param clip_min: Minimum value of an example.\n        :param clip_max: Maximum value of an example.\n        :return: Delta value.\n        '
        if self.curr_iter == 0:
            return 0.1 * (clip_max - clip_min)
        if self.norm == 2:
            dist = np.linalg.norm(original_sample - current_sample)
            delta = np.sqrt(np.prod(self.estimator.input_shape)) * self.theta * dist
        else:
            dist = np.max(abs(original_sample - current_sample))
            delta = np.prod(self.estimator.input_shape) * self.theta * dist
        return delta

    def _compute_update(self, current_sample: np.ndarray, num_eval: int, delta: float, target: int, mask: Optional[np.ndarray], clip_min: float, clip_max: float) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Compute the update in Eq.(14).\n\n        :param current_sample: Current adversarial example.\n        :param num_eval: The number of evaluations for estimating gradient.\n        :param delta: The size of random perturbation.\n        :param target: The target label.\n        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be\n                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially\n                     perturbed.\n        :param clip_min: Minimum value of an example.\n        :param clip_max: Maximum value of an example.\n        :return: an updated perturbation.\n        '
        rnd_noise_shape = [num_eval] + list(self.estimator.input_shape)
        if self.norm == 2:
            rnd_noise = np.random.randn(*rnd_noise_shape).astype(ART_NUMPY_DTYPE)
        else:
            rnd_noise = np.random.uniform(low=-1, high=1, size=rnd_noise_shape).astype(ART_NUMPY_DTYPE)
        if mask is not None:
            rnd_noise = rnd_noise * mask
        rnd_noise = rnd_noise / np.sqrt(np.sum(rnd_noise ** 2, axis=tuple(range(len(rnd_noise_shape)))[1:], keepdims=True))
        eval_samples = np.clip(current_sample + delta * rnd_noise, clip_min, clip_max)
        rnd_noise = (eval_samples - current_sample) / delta
        satisfied = self._adversarial_satisfactory(samples=eval_samples, target=target, clip_min=clip_min, clip_max=clip_max)
        f_val = 2 * satisfied.reshape([num_eval] + [1] * len(self.estimator.input_shape)) - 1.0
        f_val = f_val.astype(ART_NUMPY_DTYPE)
        if np.mean(f_val) == 1.0:
            grad = np.mean(rnd_noise, axis=0)
        elif np.mean(f_val) == -1.0:
            grad = -np.mean(rnd_noise, axis=0)
        else:
            f_val -= np.mean(f_val)
            grad = np.mean(f_val * rnd_noise, axis=0)
        if self.norm == 2:
            result = grad / np.linalg.norm(grad)
        else:
            result = np.sign(grad)
        return result

    def _adversarial_satisfactory(self, samples: np.ndarray, target: int, clip_min: float, clip_max: float) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Check whether an image is adversarial.\n\n        :param samples: A batch of examples.\n        :param target: The target label.\n        :param clip_min: Minimum value of an example.\n        :param clip_max: Maximum value of an example.\n        :return: An array of 0/1.\n        '
        samples = np.clip(samples, clip_min, clip_max)
        preds = np.argmax(self.estimator.predict(samples, batch_size=self.batch_size), axis=1)
        if self.targeted:
            result = preds == target
        else:
            result = preds != target
        return result

    @staticmethod
    def _interpolate(current_sample: np.ndarray, original_sample: np.ndarray, alpha: float, norm: Union[int, float, str]) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Interpolate a new sample based on the original and the current samples.\n\n        :param current_sample: Current adversarial example.\n        :param original_sample: The original input.\n        :param alpha: The coefficient of interpolation.\n        :param norm: Order of the norm. Possible values: "inf", np.inf or 2.\n        :return: An adversarial example.\n        '
        if norm == 2:
            result = (1 - alpha) * original_sample + alpha * current_sample
        else:
            result = np.clip(current_sample, original_sample - alpha, original_sample + alpha)
        return result

    def _check_params(self) -> None:
        if False:
            while True:
                i = 10
        if self.norm not in [2, np.inf, 'inf']:
            raise ValueError('Norm order must be either 2, `np.inf` or "inf".')
        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError('The number of iterations must be a non-negative integer.')
        if not isinstance(self.max_eval, int) or self.max_eval <= 0:
            raise ValueError('The maximum number of evaluations must be a positive integer.')
        if not isinstance(self.init_eval, int) or self.init_eval <= 0:
            raise ValueError('The initial number of evaluations must be a positive integer.')
        if self.init_eval > self.max_eval:
            raise ValueError('The maximum number of evaluations must be larger than the initial number of evaluations.')
        if not isinstance(self.init_size, int) or self.init_size <= 0:
            raise ValueError('The number of initial trials must be a positive integer.')
        if not isinstance(self.verbose, bool):
            raise ValueError('The argument `verbose` has to be of type bool.')