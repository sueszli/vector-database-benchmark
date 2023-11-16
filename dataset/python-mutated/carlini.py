"""
This module implements the L2, LInf and L0 optimized attacks `CarliniL2Method`, `CarliniLInfMethod` and `CarliniL0Method
of Carlini and Wagner (2016). These attacks are among the most effective white-box attacks and should be used among the
primary attacks to evaluate potential defences. A major difference with respect to the original implementation
(https://github.com/carlini/nn_robust_attacks) is that this implementation uses line search in the optimization of the
attack objective.

| Paper link: https://arxiv.org/abs/1608.04644
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Tuple, TYPE_CHECKING
import numpy as np
from tqdm.auto import trange
from art.config import ART_NUMPY_DTYPE
from art.optimizers import Adam
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.attacks.attack import EvasionAttack
from art.utils import compute_success, get_labels_np_array, tanh_to_original, original_to_tanh
from art.utils import check_and_transform_label_format
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE
logger = logging.getLogger(__name__)

class CarliniL2Method(EvasionAttack):
    """
    The L_2 optimized attack of Carlini and Wagner (2016). This attack is among the most effective and should be used
    among the primary attacks to evaluate potential defences. A major difference wrt to the original implementation
    (https://github.com/carlini/nn_robust_attacks) is that we use line search in the optimization of the attack
    objective.

    | Paper link: https://arxiv.org/abs/1608.04644
    """
    attack_params = EvasionAttack.attack_params + ['confidence', 'targeted', 'learning_rate', 'max_iter', 'binary_search_steps', 'initial_const', 'max_halving', 'max_doubling', 'batch_size', 'verbose']
    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(self, classifier: 'CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE', confidence: float=0.0, targeted: bool=False, learning_rate: float=0.01, binary_search_steps: int=10, max_iter: int=10, initial_const: float=0.01, max_halving: int=5, max_doubling: int=5, batch_size: int=1, verbose: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Create a Carlini&Wagner L_2 attack instance.\n\n        :param classifier: A trained classifier.\n        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther away,\n               from the original input, but classified with higher confidence as the target class.\n        :param targeted: Should the attack target one specific class.\n        :param learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better results\n               but are slower to converge.\n        :param binary_search_steps: Number of times to adjust constant with binary search (positive value). If\n                                    `binary_search_steps` is large, then the algorithm is not very sensitive to the\n                                    value of `initial_const`. Note that the values gamma=0.999999 and c_upper=10e10 are\n                                    hardcoded with the same values used by the authors of the method.\n        :param max_iter: The maximum number of iterations.\n        :param initial_const: The initial trade-off constant `c` to use to tune the relative importance of distance and\n                confidence. If `binary_search_steps` is large, the initial constant is not important, as discussed in\n                Carlini and Wagner (2016).\n        :param max_halving: Maximum number of halving steps in the line search optimization.\n        :param max_doubling: Maximum number of doubling steps in the line search optimization.\n        :param batch_size: Size of the batch on which adversarial samples are generated.\n        :param verbose: Show progress bars.\n        '
        super().__init__(estimator=classifier)
        self.confidence = confidence
        self._targeted = targeted
        self.learning_rate = learning_rate
        self.binary_search_steps = binary_search_steps
        self.max_iter = max_iter
        self.initial_const = initial_const
        self.max_halving = max_halving
        self.max_doubling = max_doubling
        self.batch_size = batch_size
        self.verbose = verbose
        CarliniL2Method._check_params(self)
        self._c_upper_bound = 100000000000.0
        self._tanh_smoother = 0.999999

    def _loss(self, x: np.ndarray, x_adv: np.ndarray, target: np.ndarray, c_weight: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if False:
            print('Hello World!')
        '\n        Compute the objective function value.\n\n        :param x: An array with the original input.\n        :param x_adv: An array with the adversarial input.\n        :param target: An array with the target class (one-hot encoded).\n        :param c_weight: Weight of the loss term aiming for classification as target.\n        :return: A tuple holding the current logits, l2 distance and overall loss.\n        '
        l2dist = np.sum(np.square(x - x_adv).reshape(x.shape[0], -1), axis=1)
        z_predicted = self.estimator.predict(np.array(x_adv, dtype=ART_NUMPY_DTYPE), logits=True, batch_size=self.batch_size)
        z_target = np.sum(z_predicted * target, axis=1)
        z_other = np.max(z_predicted * (1 - target) + (np.min(z_predicted, axis=1) - 1)[:, np.newaxis] * target, axis=1)
        if self.targeted:
            loss = np.maximum(z_other - z_target + self.confidence, np.zeros(x.shape[0]))
        else:
            loss = np.maximum(z_target - z_other + self.confidence, np.zeros(x.shape[0]))
        return (z_predicted, l2dist, c_weight * loss + l2dist)

    def _loss_gradient(self, z_logits: np.ndarray, target: np.ndarray, x: np.ndarray, x_adv: np.ndarray, x_adv_tanh: np.ndarray, c_weight: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Compute the gradient of the loss function.\n\n        :param z_logits: An array with the current logits.\n        :param target: An array with the target class (one-hot encoded).\n        :param x: An array with the original input.\n        :param x_adv: An array with the adversarial input.\n        :param x_adv_tanh: An array with the adversarial input in tanh space.\n        :param c_weight: Weight of the loss term aiming for classification as target.\n        :param clip_min: Minimum clipping value.\n        :param clip_max: Maximum clipping value.\n        :return: An array with the gradient of the loss function.\n        '
        if self.targeted:
            i_sub = np.argmax(target, axis=1)
            i_add = np.argmax(z_logits * (1 - target) + (np.min(z_logits, axis=1) - 1)[:, np.newaxis] * target, axis=1)
        else:
            i_add = np.argmax(target, axis=1)
            i_sub = np.argmax(z_logits * (1 - target) + (np.min(z_logits, axis=1) - 1)[:, np.newaxis] * target, axis=1)
        loss_gradient = self.estimator.class_gradient(x_adv, label=i_add)
        loss_gradient -= self.estimator.class_gradient(x_adv, label=i_sub)
        loss_gradient = loss_gradient.reshape(x.shape)
        c_mult = c_weight
        for _ in range(len(x.shape) - 1):
            c_mult = c_mult[:, np.newaxis]
        loss_gradient *= c_mult
        loss_gradient += 2 * (x_adv - x)
        loss_gradient *= clip_max - clip_min
        loss_gradient *= (1 - np.square(np.tanh(x_adv_tanh))) / (2 * self._tanh_smoother)
        return loss_gradient

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Generate adversarial samples and return them in an array.\n\n        :param x: An array with the original inputs to be attacked.\n        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape\n                  (nb_samples,). If `self.targeted` is true, then `y` represents the target labels. If `self.targeted`\n                  is true, then `y_val` represents the target labels. Otherwise, the targets are the original class\n                  labels.\n        :return: An array holding the adversarial examples.\n        '
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        x_adv = x.astype(ART_NUMPY_DTYPE)
        if self.estimator.clip_values is not None:
            (clip_min, clip_max) = self.estimator.clip_values
        else:
            (clip_min, clip_max) = (np.amin(x), np.amax(x))
        if self.targeted and y is None:
            raise ValueError('Target labels `y` need to be provided for a targeted attack.')
        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        if self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError('This attack has not yet been tested for binary classification with a single output classifier.')
        nb_batches = int(np.ceil(x_adv.shape[0] / float(self.batch_size)))
        for batch_id in trange(nb_batches, desc='C&W L_2', disable=not self.verbose):
            (batch_index_1, batch_index_2) = (batch_id * self.batch_size, (batch_id + 1) * self.batch_size)
            x_batch = x_adv[batch_index_1:batch_index_2]
            y_batch = y[batch_index_1:batch_index_2]
            x_batch_tanh = original_to_tanh(x_batch, clip_min, clip_max, self._tanh_smoother)
            c_current = self.initial_const * np.ones(x_batch.shape[0])
            c_lower_bound = np.zeros(x_batch.shape[0])
            c_double = np.ones(x_batch.shape[0]) > 0
            best_l2dist = np.inf * np.ones(x_batch.shape[0])
            best_x_adv_batch = x_batch.copy()
            for bss in range(self.binary_search_steps):
                logger.debug('Binary search step %i out of %i (c_mean==%f)', bss, self.binary_search_steps, np.mean(c_current))
                nb_active = int(np.sum(c_current < self._c_upper_bound))
                logger.debug('Number of samples with c_current < _c_upper_bound: %i out of %i', nb_active, x_batch.shape[0])
                if nb_active == 0:
                    break
                learning_rate = self.learning_rate * np.ones(x_batch.shape[0])
                x_adv_batch = x_batch.copy()
                x_adv_batch_tanh = x_batch_tanh.copy()
                (z_logits, l2dist, loss) = self._loss(x_batch, x_adv_batch, y_batch, c_current)
                attack_success = loss - l2dist <= 0
                overall_attack_success = attack_success
                for i_iter in range(self.max_iter):
                    logger.debug('Iteration step %i out of %i', i_iter, self.max_iter)
                    logger.debug('Average Loss: %f', np.mean(loss))
                    logger.debug('Average L2Dist: %f', np.mean(l2dist))
                    logger.debug('Average Margin Loss: %f', np.mean(loss - l2dist))
                    logger.debug('Current number of succeeded attacks: %i out of %i', int(np.sum(attack_success)), len(attack_success))
                    improved_adv = attack_success & (l2dist < best_l2dist)
                    logger.debug('Number of improved L2 distances: %i', int(np.sum(improved_adv)))
                    if np.sum(improved_adv) > 0:
                        best_l2dist[improved_adv] = l2dist[improved_adv]
                        best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]
                    active = (c_current < self._c_upper_bound) & (learning_rate > 0)
                    nb_active = int(np.sum(active))
                    logger.debug('Number of samples with c_current < _c_upper_bound and learning_rate > 0: %i out of %i', nb_active, x_batch.shape[0])
                    if nb_active == 0:
                        break
                    logger.debug('Compute loss gradient')
                    perturbation_tanh = -self._loss_gradient(z_logits[active], y_batch[active], x_batch[active], x_adv_batch[active], x_adv_batch_tanh[active], c_current[active], clip_min, clip_max)
                    prev_loss = loss.copy()
                    best_loss = loss.copy()
                    best_lr = np.zeros(x_batch.shape[0])
                    halving = np.zeros(x_batch.shape[0])
                    for i_halve in range(self.max_halving):
                        logger.debug('Perform halving iteration %i out of %i', i_halve, self.max_halving)
                        do_halving = loss[active] >= prev_loss[active]
                        logger.debug('Halving to be performed on %i samples', int(np.sum(do_halving)))
                        if np.sum(do_halving) == 0:
                            break
                        active_and_do_halving = active.copy()
                        active_and_do_halving[active] = do_halving
                        lr_mult = learning_rate[active_and_do_halving]
                        for _ in range(len(x.shape) - 1):
                            lr_mult = lr_mult[:, np.newaxis]
                        x_adv1 = x_adv_batch_tanh[active_and_do_halving]
                        new_x_adv_batch_tanh = x_adv1 + lr_mult * perturbation_tanh[do_halving]
                        new_x_adv_batch = tanh_to_original(new_x_adv_batch_tanh, clip_min, clip_max)
                        (_, l2dist[active_and_do_halving], loss[active_and_do_halving]) = self._loss(x_batch[active_and_do_halving], new_x_adv_batch, y_batch[active_and_do_halving], c_current[active_and_do_halving])
                        logger.debug('New Average Loss: %f', np.mean(loss))
                        logger.debug('New Average L2Dist: %f', np.mean(l2dist))
                        logger.debug('New Average Margin Loss: %f', np.mean(loss - l2dist))
                        best_lr[loss < best_loss] = learning_rate[loss < best_loss]
                        best_loss[loss < best_loss] = loss[loss < best_loss]
                        learning_rate[active_and_do_halving] /= 2
                        halving[active_and_do_halving] += 1
                    learning_rate[active] *= 2
                    for i_double in range(self.max_doubling):
                        logger.debug('Perform doubling iteration %i out of %i', i_double, self.max_doubling)
                        do_doubling = (halving[active] == 1) & (loss[active] <= best_loss[active])
                        logger.debug('Doubling to be performed on %i samples', int(np.sum(do_doubling)))
                        if np.sum(do_doubling) == 0:
                            break
                        active_and_do_doubling = active.copy()
                        active_and_do_doubling[active] = do_doubling
                        learning_rate[active_and_do_doubling] *= 2
                        lr_mult = learning_rate[active_and_do_doubling]
                        for _ in range(len(x.shape) - 1):
                            lr_mult = lr_mult[:, np.newaxis]
                        x_adv2 = x_adv_batch_tanh[active_and_do_doubling]
                        new_x_adv_batch_tanh = x_adv2 + lr_mult * perturbation_tanh[do_doubling]
                        new_x_adv_batch = tanh_to_original(new_x_adv_batch_tanh, clip_min, clip_max)
                        (_, l2dist[active_and_do_doubling], loss[active_and_do_doubling]) = self._loss(x_batch[active_and_do_doubling], new_x_adv_batch, y_batch[active_and_do_doubling], c_current[active_and_do_doubling])
                        logger.debug('New Average Loss: %f', np.mean(loss))
                        logger.debug('New Average L2Dist: %f', np.mean(l2dist))
                        logger.debug('New Average Margin Loss: %f', np.mean(loss - l2dist))
                        best_lr[loss < best_loss] = learning_rate[loss < best_loss]
                        best_loss[loss < best_loss] = loss[loss < best_loss]
                    learning_rate[halving == 1] /= 2
                    update_adv = best_lr[active] > 0
                    logger.debug('Number of adversarial samples to be finally updated: %i', int(np.sum(update_adv)))
                    if np.sum(update_adv) > 0:
                        active_and_update_adv = active.copy()
                        active_and_update_adv[active] = update_adv
                        best_lr_mult = best_lr[active_and_update_adv]
                        for _ in range(len(x.shape) - 1):
                            best_lr_mult = best_lr_mult[:, np.newaxis]
                        x_adv4 = x_adv_batch_tanh[active_and_update_adv]
                        best_lr1 = best_lr_mult * perturbation_tanh[update_adv]
                        x_adv_batch_tanh[active_and_update_adv] = x_adv4 + best_lr1
                        x_adv6 = x_adv_batch_tanh[active_and_update_adv]
                        x_adv_batch[active_and_update_adv] = tanh_to_original(x_adv6, clip_min, clip_max)
                        (z_logits[active_and_update_adv], l2dist[active_and_update_adv], loss[active_and_update_adv]) = self._loss(x_batch[active_and_update_adv], x_adv_batch[active_and_update_adv], y_batch[active_and_update_adv], c_current[active_and_update_adv])
                        attack_success = loss - l2dist <= 0
                        overall_attack_success = overall_attack_success | attack_success
                improved_adv = attack_success & (l2dist < best_l2dist)
                logger.debug('Number of improved L2 distances: %i', int(np.sum(improved_adv)))
                if np.sum(improved_adv) > 0:
                    best_l2dist[improved_adv] = l2dist[improved_adv]
                    best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]
                c_double[overall_attack_success] = False
                c_current[overall_attack_success] = (c_lower_bound + c_current)[overall_attack_success] / 2
                c_old = c_current
                c_current[~overall_attack_success & c_double] *= 2
                c_current1 = (c_current - c_lower_bound)[~overall_attack_success & ~c_double]
                c_current[~overall_attack_success & ~c_double] += c_current1 / 2
                c_lower_bound[~overall_attack_success] = c_old[~overall_attack_success]
            x_adv[batch_index_1:batch_index_2] = best_x_adv_batch
        logger.info('Success rate of C&W L_2 attack: %.2f%%', 100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size))
        return x_adv

    def _check_params(self) -> None:
        if False:
            return 10
        if not isinstance(self.binary_search_steps, int) or self.binary_search_steps < 0:
            raise ValueError('The number of binary search steps must be a non-negative integer.')
        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError('The number of iterations must be a non-negative integer.')
        if not isinstance(self.max_halving, int) or self.max_halving < 1:
            raise ValueError('The number of halving steps must be an integer greater than zero.')
        if not isinstance(self.max_doubling, int) or self.max_doubling < 1:
            raise ValueError('The number of doubling steps must be an integer greater than zero.')
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError('The batch size must be an integer greater than zero.')

class CarliniLInfMethod(EvasionAttack):
    """
    This is a modified version of the L_2 optimized attack of Carlini and Wagner (2016). It controls the L_Inf
    norm, i.e. the maximum perturbation applied to each pixel.
    """
    attack_params = EvasionAttack.attack_params + ['confidence', 'targeted', 'learning_rate', 'max_iter', 'decrease_factor', 'initial_const', 'largest_const', 'const_factor', 'batch_size', 'verbose']
    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(self, classifier: 'CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE', confidence: float=0.0, targeted: bool=False, learning_rate: float=0.01, max_iter: int=10, decrease_factor: float=0.9, initial_const: float=1e-05, largest_const: float=20.0, const_factor: float=2.0, batch_size: int=1, verbose: bool=True) -> None:
        if False:
            print('Hello World!')
        '\n        Create a Carlini&Wagner L_Inf attack instance.\n\n        :param classifier: A trained classifier.\n        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther away,\n                from the original input, but classified with higher confidence as the target class.\n        :param targeted: Should the attack target one specific class.\n        :param learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better\n                results but are slower to converge.\n        :param max_iter: The maximum number of iterations.\n        :param decrease_factor: The rate of shrinking tau, values in `0 < decrease_factor < 1` where larger is more\n                                accurate.\n        :param initial_const: The initial value of constant `c`.\n        :param largest_const: The largest value of constant `c`.\n        :param const_factor: The rate of increasing constant `c` with `const_factor > 1`, where smaller more accurate.\n        :param batch_size: Size of the batch on which adversarial samples are generated.\n        :param verbose: Show progress bars.\n        '
        super().__init__(estimator=classifier)
        self.confidence = confidence
        self._targeted = targeted
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.decrease_factor = decrease_factor
        self.initial_const = initial_const
        self.largest_const = largest_const
        self.const_factor = const_factor
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()
        self._tanh_smoother = 0.999999

    def _loss(self, x_adv: np.ndarray, target: np.ndarray, x, const, tau) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if False:
            while True:
                i = 10
        '\n        Compute the objective function value.\n\n        :param x_adv: An array with the adversarial examples.\n        :param target: An array with the target class (one-hot encoded).\n        :param x: Benign samples.\n        :param  const: Current constant `c`.\n        :param tau: Current limit `tau`.\n        :return: A tuple of current predictions, total loss, logits loss and regularisation loss.\n        '
        z_predicted = self.estimator.predict(np.array(x_adv, dtype=ART_NUMPY_DTYPE), batch_size=self.batch_size)
        z_target = np.sum(z_predicted * target, axis=1)
        z_other = np.max(z_predicted * (1 - target) + (np.min(z_predicted, axis=1) - 1)[:, np.newaxis] * target, axis=1)
        if self.targeted:
            loss_1 = np.maximum(z_other - z_target + self.confidence, np.zeros(x_adv.shape[0]))
        else:
            loss_1 = np.maximum(z_target - z_other + self.confidence, np.zeros(x_adv.shape[0]))
        loss_2 = np.sum(np.maximum(0.0, np.abs(x_adv - x) - tau))
        loss = loss_1 * const + loss_2
        return (z_predicted, loss, loss_1, loss_2)

    def _loss_gradient(self, z_logits: np.ndarray, target: np.ndarray, x_adv: np.ndarray, x_adv_tanh: np.ndarray, clip_min: np.ndarray, clip_max: np.ndarray, x, tau) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Compute the gradient of the loss function.\n\n        :param z_logits: An array with the current predictions.\n        :param target: An array with the target class (one-hot encoded).\n        :param x_adv: An array with the adversarial input.\n        :param x_adv_tanh: An array with the adversarial input in tanh space.\n        :param clip_min: Minimum clipping values.\n        :param clip_max: Maximum clipping values.\n        :param x: Benign samples.\n        :param tau: Current limit `tau`.\n        :return: An array with the gradient of the loss function.\n        '
        if self.targeted:
            i_sub = np.argmax(target, axis=1)
            i_add = np.argmax(z_logits * (1 - target) + (np.min(z_logits, axis=1) - 1)[:, np.newaxis] * target, axis=1)
        else:
            i_add = np.argmax(target, axis=1)
            i_sub = np.argmax(z_logits * (1 - target) + (np.min(z_logits, axis=1) - 1)[:, np.newaxis] * target, axis=1)
        loss_gradient = self.estimator.class_gradient(x_adv, label=i_add)
        loss_gradient -= self.estimator.class_gradient(x_adv, label=i_sub)
        loss_gradient = loss_gradient.reshape(x_adv.shape)
        loss_gradient_2 = np.sign(np.maximum(0.0, np.abs(x_adv - x) - tau)) * np.sign(x_adv - x)
        loss_gradient_2 *= clip_max - clip_min
        loss_gradient_2 *= (1 - np.square(np.tanh(x_adv_tanh))) / (2 * self._tanh_smoother)
        loss_gradient *= clip_max - clip_min
        loss_gradient *= (1 - np.square(np.tanh(x_adv_tanh))) / (2 * self._tanh_smoother)
        loss_gradient = loss_gradient + loss_gradient_2
        return loss_gradient

    def _generate_single(self, x_batch, y_batch, clip_min, clip_max, const, tau):
        if False:
            return 10
        '\n        Generate a single adversarial example.\n\n        :param x_batch: Current benign sample.\n        :param y_batch: Current label.\n        :param clip_min: Minimum clipping values.\n        :param clip_max: Maximum clipping values.\n        :param  const: Current constant `c`.\n        :param tau: Current limit `tau`.\n        '
        x_adv_batch_tanh = original_to_tanh(x_batch, clip_min, clip_max, self._tanh_smoother)

        def func(x_i):
            if False:
                print('Hello World!')
            x_adv_batch_tanh = x_i
            x_adv_batch = tanh_to_original(x_adv_batch_tanh, clip_min, clip_max)
            (_, loss, _, _) = self._loss(x_adv_batch, y_batch, x_batch, const, tau)
            return loss

        def func_der(x_i):
            if False:
                print('Hello World!')
            x_adv_batch_tanh = x_i
            x_adv_batch = tanh_to_original(x_adv_batch_tanh, clip_min, clip_max)
            (z_logits, _, _, _) = self._loss(x_adv_batch, y_batch, x_batch, const, tau)
            perturbation_tanh = self._loss_gradient(z_logits, y_batch, x_adv_batch, x_adv_batch_tanh, clip_min, clip_max, x_batch, tau)
            return perturbation_tanh
        x_0 = x_adv_batch_tanh.copy()
        adam = Adam(alpha=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        x_adv_batch_tanh = adam.optimize(func=func, jac=func_der, x_0=x_0, max_iter=self.max_iter, loss_converged=0.001)
        x_adv_batch = tanh_to_original(x_adv_batch_tanh, clip_min, clip_max)
        return x_adv_batch

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Generate adversarial samples and return them in an array.\n\n        :param x: An array with the original inputs to be attacked.\n        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape\n                  (nb_samples,). If `self.targeted` is true, then `y_val` represents the target labels. Otherwise, the\n                  targets are the original class labels.\n        :return: An array holding the adversarial examples.\n        '
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        x_adv = x.astype(ART_NUMPY_DTYPE)
        if self.estimator.clip_values is not None:
            (clip_min, clip_max) = self.estimator.clip_values
        else:
            (clip_min, clip_max) = (np.amin(x), np.amax(x))
        if self.targeted and y is None:
            raise ValueError('Target labels `y` need to be provided for a targeted attack.')
        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        if self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError('This attack has not yet been tested for binary classification with a single output classifier.')
        for sample_id in trange(x.shape[0], desc='C&W L_inf', disable=not self.verbose):
            sample_done = False
            tau = 1.0
            delta_i_best = 1.0
            while tau > 1.0 / 256.0 and (not sample_done):
                sample_done = True
                const = self.initial_const
                while const < self.largest_const:
                    x_batch = x[[sample_id]]
                    y_batch = y[[sample_id]]
                    x_adv_batch = self._generate_single(x_batch, y_batch, clip_min, clip_max, const=const, tau=tau)
                    (_, loss, loss_1, loss_2) = self._loss(x_adv_batch, y_batch, x_batch, const, tau)
                    delta_i = np.max(np.abs(x_adv_batch - x[sample_id]))
                    logger.debug('tau: %4.3f, const: %4.5f, loss: %4.3f, loss_1: %4.3f, loss_2: %4.3f, delta_i: %4.3f', tau, const, loss, loss_1, loss_2, delta_i)
                    if np.argmax(self.estimator.predict(x_adv_batch), axis=1) != np.argmax(y_batch, axis=1) and delta_i < delta_i_best:
                        x_adv[sample_id] = x_adv_batch
                        delta_i_best = delta_i
                        sample_done = False
                    const *= self.const_factor
                tau_actual = np.max(np.abs(x_adv[sample_id] - x[sample_id]))
                if tau_actual < tau:
                    tau = tau_actual
                tau *= self.decrease_factor
        return x_adv

    def _check_params(self) -> None:
        if False:
            print('Hello World!')
        if not isinstance(self.max_iter, int) or self.max_iter < 0:
            raise ValueError('The number of iterations must be a non-negative integer.')
        if not isinstance(self.decrease_factor, (int, float)) or not 0.0 < self.decrease_factor < 1.0:
            raise ValueError('The decrease factor must be a float between 0 and 1.')
        if not isinstance(self.initial_const, (int, float)) or self.initial_const < 0:
            raise ValueError('The initial constant value must be a positive float.')
        if not isinstance(self.largest_const, (int, float)) or self.largest_const < 0:
            print(self.largest_const)
            raise ValueError('The largest constant value must be a positive float.')
        if not isinstance(self.const_factor, (int, float)) or self.const_factor < 0:
            raise ValueError('The constant factor value must be a float and greater than 1.')
        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError('The batch size must be an integer greater than zero.')

class CarliniL0Method(CarliniL2Method):
    """
    The L_0 distance metric is non-differentiable and therefore is ill-suited for standard gradient descent.
    Instead, we use an iterative algorithm that, in each iteration, identifies some features that don’t have much effect
    on the classifier output and then fixes those features, so their value will never be changed.
    The set of fixed features grows in each iteration until we have, by process of elimination, identified a minimal
    (but possibly not minimum) subset of features that can be modified to generate an adversarial example.
    In each iteration, we use our L_2 attack to identify which features are unimportant [Carlini and Wagner, 2016].

    | Paper link: https://arxiv.org/abs/1608.04644
    """
    attack_params = EvasionAttack.attack_params + ['confidence', 'targeted', 'learning_rate', 'max_iter', 'binary_search_steps', 'initial_const', 'mask', 'warm_start', 'max_halving', 'max_doubling', 'batch_size', 'verbose']
    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(self, classifier: 'CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE', confidence: float=0.0, targeted: bool=False, learning_rate: float=0.01, binary_search_steps: int=10, max_iter: int=10, initial_const: float=0.01, mask: Optional[np.ndarray]=None, warm_start: bool=True, max_halving: int=5, max_doubling: int=5, batch_size: int=1, verbose: bool=True):
        if False:
            print('Hello World!')
        '\n        Create a Carlini&Wagner L_0 attack instance.\n\n        :param classifier: A trained classifier.\n        :param confidence: Confidence of adversarial examples: a higher value produces examples that are farther away,\n                           from the original input, but classified with higher confidence as the target class.\n        :param targeted: Should the attack target one specific class.\n        :param learning_rate: The initial learning rate for the attack algorithm. Smaller values produce better results\n                              but are slower to converge.\n        :param binary_search_steps: Number of times to adjust constant with binary search (positive value). If\n                                    `binary_search_steps` is large, then the algorithm is not very sensitive to the\n                                    value of `initial_const`. Note that the values gamma=0.999999 and c_upper=10e10 are\n                                    hardcoded with the same values used by the authors of the method.\n        :param max_iter: The maximum number of iterations.\n        :param initial_const: The initial trade-off constant `c` to use to tune the relative importance of distance and\n                              confidence. If `binary_search_steps` is large, the initial constant is not important, as\n                              discussed in Carlini and Wagner (2016).\n        :param mask: The initial features that can be modified by the algorithm. If not specified, the\n                     algorithm uses the full feature set.\n        :param warm_start: Instead of starting gradient descent in each iteration from the initial image. we start the\n                           gradient descent from the solution found on the previous iteration.\n        :param max_halving: Maximum number of halving steps in the line search optimization.\n        :param max_doubling: Maximum number of doubling steps in the line search optimization.\n        :param batch_size: Size of the batch on which adversarial samples are generated.\n        :param verbose: Show progress bars.\n        '
        super().__init__(classifier=classifier, confidence=confidence, targeted=targeted, learning_rate=learning_rate, max_iter=max_iter, max_halving=max_halving, max_doubling=max_doubling, batch_size=batch_size, verbose=verbose)
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
        self.mask = mask
        self.warm_start = warm_start
        self._check_params()
        self._c_upper_bound = 100000000000.0
        self._tanh_smoother = 0.999999
        self._perturbation_threshold = 1e-06

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Generate adversarial samples and return them in an array.\n\n        :param x: An array with the original inputs to be attacked.\n        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape\n                  (nb_samples,). If `self.targeted` is true, then `y` represents the target labels. If `self.targeted`\n                  is true, then `y_val` represents the target labels. Otherwise, the targets are the original class\n                  labels.\n        :return: An array holding the adversarial examples.\n        '
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        x_adv = x.astype(ART_NUMPY_DTYPE)
        if self.estimator.clip_values is not None:
            (clip_min, clip_max) = self.estimator.clip_values
        else:
            (clip_min, clip_max) = (np.amin(x), np.amax(x))
        if self.targeted and y is None:
            raise ValueError('Target labels `y` need to be provided for a targeted attack.')
        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        if self.estimator.nb_classes == 2 and y.shape[1] == 1:
            raise ValueError('This attack has not yet been tested for binary classification with a single output classifier.')
        if self.mask is None:
            activation = np.ones(x.shape)
        else:
            if self.mask.shape != x.shape:
                raise ValueError('The mask must have the same dimensions as the input data.')
            activation = np.array(self.mask).astype(float)
        final_adversarial_example = x.astype(ART_NUMPY_DTYPE)
        old_activation = activation.copy()
        c_final = np.ones(x.shape[0])
        best_l0dist = np.inf * np.ones(x.shape[0])
        for _ in range(x.shape[1] + 1):
            nb_batches = int(np.ceil(x_adv.shape[0] / float(self.batch_size)))
            for batch_id in range(nb_batches):
                logger.debug('Processing batch %i out of %i', batch_id, nb_batches)
                (batch_index_1, batch_index_2) = (batch_id * self.batch_size, (batch_id + 1) * self.batch_size)
                if self.warm_start:
                    x_batch = x_adv[batch_index_1:batch_index_2]
                else:
                    x_batch = x[batch_index_1:batch_index_2]
                y_batch = y[batch_index_1:batch_index_2]
                activation_batch = activation[batch_index_1:batch_index_2]
                x_batch_tanh = original_to_tanh(x_batch, clip_min, clip_max, self._tanh_smoother)
                c_current = self.initial_const * np.ones(x_batch.shape[0])
                c_lower_bound = np.zeros(x_batch.shape[0])
                c_double = np.ones(x_batch.shape[0]) > 0
                best_l0dist_batch = np.inf * np.ones(x_batch.shape[0])
                best_x_adv_batch = x_batch.copy()
                for bss in range(self.binary_search_steps):
                    logger.debug('Binary search step %i / %i (c_mean==%f)', bss, self.binary_search_steps, np.mean(c_current))
                    nb_active = int(np.sum(c_current < self._c_upper_bound))
                    logger.debug('Number of samples with c_current < _c_upper_bound: %i out of %i', nb_active, x_batch.shape[0])
                    if nb_active == 0:
                        break
                    learning_rate = self.learning_rate * np.ones(x_batch.shape[0])
                    x_adv_batch = x_batch.copy()
                    x_adv_batch_tanh = x_batch_tanh.copy()
                    (z_logits, l2dist, loss) = self._loss(x_batch, x_adv_batch, y_batch, c_current)
                    attack_success = loss - l2dist <= 0
                    overall_attack_success = attack_success
                    for i_iter in range(self.max_iter):
                        logger.debug('Iteration step %i out of %i', i_iter, self.max_iter)
                        logger.debug('Average Loss: %f', np.mean(loss))
                        logger.debug('Average L2Dist: %f', np.mean(l2dist))
                        logger.debug('Average Margin Loss: %f', np.mean(loss - l2dist))
                        logger.debug('Current number of succeeded attacks: %i out of %i', int(np.sum(attack_success)), len(attack_success))
                        l0dist = np.sum((np.abs(x_batch - x_adv_batch) > self._perturbation_threshold).astype(int), axis=(1, 2, 3))
                        improved_adv = attack_success & (l0dist < best_l0dist_batch)
                        logger.debug('Number of improved L0 distances: %i', int(np.sum(improved_adv)))
                        if np.sum(improved_adv) > 0:
                            best_l0dist_batch[improved_adv] = l0dist[improved_adv]
                            best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]
                        active = (c_current < self._c_upper_bound) & (learning_rate > 0)
                        nb_active = int(np.sum(active))
                        logger.debug('Number of samples with c_current < _c_upper_bound and learning_rate > 0: %i out of %i', nb_active, x_batch.shape[0])
                        if nb_active == 0:
                            break
                        logger.debug('Compute loss gradient')
                        perturbation_tanh = -self._loss_gradient(z_logits[active], y_batch[active], x_batch[active], x_adv_batch[active], x_adv_batch_tanh[active], c_current[active], clip_min, clip_max)
                        prev_loss = loss.copy()
                        best_loss = loss.copy()
                        best_lr = np.zeros(x_batch.shape[0])
                        halving = np.zeros(x_batch.shape[0])
                        for i_halve in range(self.max_halving):
                            logger.debug('Perform halving iteration %i out of %i', i_halve, self.max_halving)
                            do_halving = loss[active] >= prev_loss[active]
                            logger.debug('Halving to be performed on %i samples', int(np.sum(do_halving)))
                            if np.sum(do_halving) == 0:
                                break
                            active_and_do_halving = active.copy()
                            active_and_do_halving[active] = do_halving
                            lr_mult = learning_rate[active_and_do_halving]
                            for _ in range(len(x.shape) - 1):
                                lr_mult = lr_mult[:, np.newaxis]
                            x_adv1 = x_adv_batch_tanh[active_and_do_halving]
                            new_x_adv_batch_tanh = x_adv1 + lr_mult * perturbation_tanh[do_halving] * activation_batch[do_halving]
                            new_x_adv_batch = tanh_to_original(new_x_adv_batch_tanh, clip_min, clip_max)
                            (_, l2dist[active_and_do_halving], loss[active_and_do_halving]) = self._loss(x_batch[active_and_do_halving], new_x_adv_batch, y_batch[active_and_do_halving], c_current[active_and_do_halving])
                            logger.debug('New Average Loss: %f', np.mean(loss))
                            logger.debug('New Average L2Dist: %f', np.mean(l2dist))
                            logger.debug('New Average Margin Loss: %f', np.mean(loss - l2dist))
                            best_lr[loss < best_loss] = learning_rate[loss < best_loss]
                            best_loss[loss < best_loss] = loss[loss < best_loss]
                            learning_rate[active_and_do_halving] /= 2
                            halving[active_and_do_halving] += 1
                        learning_rate[active] *= 2
                        for i_double in range(self.max_doubling):
                            logger.debug('Perform doubling iteration %i out of %i', i_double, self.max_doubling)
                            do_doubling = (halving[active] == 1) & (loss[active] <= best_loss[active])
                            logger.debug('Doubling to be performed on %i samples', int(np.sum(do_doubling)))
                            if np.sum(do_doubling) == 0:
                                break
                            active_and_do_doubling = active.copy()
                            active_and_do_doubling[active] = do_doubling
                            learning_rate[active_and_do_doubling] *= 2
                            lr_mult = learning_rate[active_and_do_doubling]
                            for _ in range(len(x.shape) - 1):
                                lr_mult = lr_mult[:, np.newaxis]
                            x_adv2 = x_adv_batch_tanh[active_and_do_doubling]
                            new_x_adv_batch_tanh = x_adv2 + lr_mult * perturbation_tanh[do_doubling] * activation_batch[do_doubling]
                            new_x_adv_batch = tanh_to_original(new_x_adv_batch_tanh, clip_min, clip_max)
                            (_, l2dist[active_and_do_doubling], loss[active_and_do_doubling]) = self._loss(x_batch[active_and_do_doubling], new_x_adv_batch, y_batch[active_and_do_doubling], c_current[active_and_do_doubling])
                            logger.debug('New Average Loss: %f', np.mean(loss))
                            logger.debug('New Average L2Dist: %f', np.mean(l2dist))
                            logger.debug('New Average Margin Loss: %f', np.mean(loss - l2dist))
                            best_lr[loss < best_loss] = learning_rate[loss < best_loss]
                            best_loss[loss < best_loss] = loss[loss < best_loss]
                        learning_rate[halving == 1] /= 2
                        update_adv = best_lr[active] > 0
                        logger.debug('Number of adversarial samples to be finally updated: %i', int(np.sum(update_adv)))
                        if np.sum(update_adv) > 0:
                            active_and_update_adv = active.copy()
                            active_and_update_adv[active] = update_adv
                            best_lr_mult = best_lr[active_and_update_adv]
                            for _ in range(len(x.shape) - 1):
                                best_lr_mult = best_lr_mult[:, np.newaxis]
                            x_adv4 = x_adv_batch_tanh[active_and_update_adv]
                            best_lr1 = best_lr_mult * perturbation_tanh[update_adv]
                            x_adv_batch_tanh[active_and_update_adv] = x_adv4 + best_lr1 * activation_batch[active_and_update_adv]
                            x_adv6 = x_adv_batch_tanh[active_and_update_adv]
                            x_adv_batch[active_and_update_adv] = tanh_to_original(x_adv6, clip_min, clip_max)
                            (z_logits[active_and_update_adv], l2dist[active_and_update_adv], loss[active_and_update_adv]) = self._loss(x_batch[active_and_update_adv], x_adv_batch[active_and_update_adv], y_batch[active_and_update_adv], c_current[active_and_update_adv])
                            attack_success = loss - l2dist <= 0
                            overall_attack_success = overall_attack_success | attack_success
                    l0dist = np.sum((np.abs(x_batch - x_adv_batch) > self._perturbation_threshold).astype(int), axis=(1, 2, 3))
                    improved_adv = attack_success & (l0dist < best_l0dist_batch)
                    logger.debug('Number of improved L0 distances: %i', int(np.sum(improved_adv)))
                    if np.sum(improved_adv) > 0:
                        best_l0dist_batch[improved_adv] = l0dist[improved_adv]
                        best_x_adv_batch[improved_adv] = x_adv_batch[improved_adv]
                    c_double[overall_attack_success] = False
                    c_current[overall_attack_success] = (c_lower_bound + c_current)[overall_attack_success] / 2
                    c_old = c_current
                    c_current[~overall_attack_success & c_double] *= 2
                    c_current1 = (c_current - c_lower_bound)[~overall_attack_success & ~c_double]
                    c_current[~overall_attack_success & ~c_double] += c_current1 / 2
                    c_lower_bound[~overall_attack_success] = c_old[~overall_attack_success]
                c_final[batch_index_1:batch_index_2] = c_current
                x_adv[batch_index_1:batch_index_2] = best_x_adv_batch
            logger.info('Success rate of C&W L_2 attack: %.2f%%', 100 * compute_success(self.estimator, x, y, x_adv, self.targeted, batch_size=self.batch_size))
            (z_logits, l2dist, loss) = self._loss(x, x_adv, y, c_final)
            attack_success = loss - l2dist <= 0
            l0dist = np.sum((np.abs(x - x_adv) > self._perturbation_threshold).astype(int), axis=(1, 2, 3))
            improved_adv = attack_success & (l0dist < best_l0dist)
            if np.sum(improved_adv) > 0:
                final_adversarial_example[improved_adv] = x_adv[improved_adv]
            else:
                return x * (old_activation == 0).astype(int) + final_adversarial_example * old_activation
            x_adv_tanh = original_to_tanh(x_adv, clip_min, clip_max, self._tanh_smoother)
            objective_loss_gradient = -self._loss_gradient(z_logits, y, x, x_adv, x_adv_tanh, c_final, clip_min, clip_max)
            perturbation_l1_norm = np.abs(x_adv - x)
            objective_reduction = np.abs(objective_loss_gradient) * perturbation_l1_norm
            objective_reduction += np.array(np.where(activation == 0, np.inf, 0))
            fix_feature_index = np.argmin(objective_reduction.reshape(objective_reduction.shape[0], -1), axis=1)
            fix_feature = np.ones(x.shape)
            fix_feature = fix_feature.reshape(fix_feature.shape[0], -1)
            fix_feature[np.arange(fix_feature_index.size), fix_feature_index] = 0
            fix_feature = fix_feature.reshape(x.shape)
            old_activation[improved_adv] = activation.copy()[improved_adv]
            activation[improved_adv] *= fix_feature[improved_adv]
            logger.info('L0 norm before fixing :\n%f\nNumber active features :\n%f\nIndex of fixed feature :\n%d', np.sum((perturbation_l1_norm > self._perturbation_threshold).astype(int), axis=1), np.sum(activation, axis=1), fix_feature_index)
        return x_adv

    def _check_params(self):
        if False:
            print('Hello World!')
        if not isinstance(self.binary_search_steps, int) or self.binary_search_steps < 0:
            raise ValueError('The number of binary search steps must be a non-negative integer.')