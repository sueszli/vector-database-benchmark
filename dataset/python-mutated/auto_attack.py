"""
This module implements the `AutoAttack` attack.

| Paper link: https://arxiv.org/abs/2003.01690
"""
import logging
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import numpy as np
from art.attacks.attack import EvasionAttack
from art.attacks.evasion.auto_projected_gradient_descent import AutoProjectedGradientDescent
from art.attacks.evasion.deepfool import DeepFool
from art.attacks.evasion.square_attack import SquareAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.estimator import BaseEstimator
from art.utils import check_and_transform_label_format, get_labels_np_array
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE
logger = logging.getLogger(__name__)

class AutoAttack(EvasionAttack):
    """
    Implementation of the `AutoAttack` attack.

    | Paper link: https://arxiv.org/abs/2003.01690
    """
    attack_params = EvasionAttack.attack_params + ['norm', 'eps', 'eps_step', 'attacks', 'batch_size', 'estimator_orig', 'targeted', 'parallel']
    _estimator_requirements = (BaseEstimator, ClassifierMixin)
    SAMPLE_DEFAULT = -1
    SAMPLE_MISCLASSIFIED = -2

    def __init__(self, estimator: 'CLASSIFIER_TYPE', norm: Union[int, float, str]=np.inf, eps: float=0.3, eps_step: float=0.1, attacks: Optional[List[EvasionAttack]]=None, batch_size: int=32, estimator_orig: Optional['CLASSIFIER_TYPE']=None, targeted: bool=False, parallel: bool=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a :class:`.AutoAttack` instance.\n\n        :param estimator: An trained estimator.\n        :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf, 1 or 2.\n        :param eps: Maximum perturbation that the attacker can introduce.\n        :param eps_step: Attack step size (input variation) at each iteration.\n        :param attacks: The list of `art.attacks.EvasionAttack` attacks to be used for AutoAttack. If it is `None` or\n                        empty the standard attacks (PGD, APGD-ce, APGD-dlr, DeepFool, Square) will be used.\n        :param batch_size: Size of the batch on which adversarial samples are generated.\n        :param estimator_orig: Original estimator to be attacked by adversarial examples.\n        :param targeted: If False run only untargeted attacks, if True also run targeted attacks against each possible\n                         target.\n        :param parallel: If True run attacks in parallel.\n        '
        super().__init__(estimator=estimator)
        if attacks is None or not attacks:
            attacks = []
            attacks.append(AutoProjectedGradientDescent(estimator=estimator, norm=norm, eps=eps, eps_step=eps_step, max_iter=100, targeted=False, nb_random_init=5, batch_size=batch_size, loss_type='cross_entropy'))
            attacks.append(AutoProjectedGradientDescent(estimator=estimator, norm=norm, eps=eps, eps_step=eps_step, max_iter=100, targeted=False, nb_random_init=5, batch_size=batch_size, loss_type='difference_logits_ratio'))
            attacks.append(DeepFool(classifier=estimator, max_iter=100, epsilon=0.001, nb_grads=10, batch_size=batch_size))
            attacks.append(SquareAttack(estimator=estimator, norm=norm, max_iter=5000, eps=eps, p_init=0.8, nb_restarts=5))
        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self.attacks = attacks
        self.batch_size = batch_size
        if estimator_orig is not None:
            self.estimator_orig = estimator_orig
        else:
            self.estimator_orig = estimator
        self._targeted = targeted
        self.parallel = parallel
        self.best_attacks: np.ndarray = np.array([])
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate adversarial samples and return them in an array.\n\n        :param x: An array with the original inputs.\n        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape\n                  (nb_samples,). Only provide this parameter if you\'d like to use true labels when crafting adversarial\n                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect\n                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.\n        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.\n                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any\n                     features for which the mask is zero will not be adversarially perturbed.\n        :type mask: `np.ndarray`\n        :return: An array holding the adversarial examples.\n        '
        import multiprocess
        x_adv = x.astype(ART_NUMPY_DTYPE)
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        if y is None:
            y = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        y_pred = self.estimator_orig.predict(x.astype(ART_NUMPY_DTYPE))
        sample_is_robust = np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)
        self.best_attacks = np.array([self.SAMPLE_DEFAULT] * len(x))
        self.best_attacks[np.logical_not(sample_is_robust)] = self.SAMPLE_MISCLASSIFIED
        args = []
        for attack in self.attacks:
            if np.sum(sample_is_robust) == 0:
                break
            if attack.targeted:
                attack.set_params(targeted=False)
            if self.parallel:
                args.append((deepcopy(x_adv), deepcopy(y), deepcopy(sample_is_robust), deepcopy(attack), deepcopy(self.estimator), deepcopy(self.norm), deepcopy(self.eps)))
            else:
                (x_adv, sample_is_robust) = run_attack(x=x_adv, y=y, sample_is_robust=sample_is_robust, attack=attack, estimator_orig=self.estimator, norm=self.norm, eps=self.eps, **kwargs)
                atk_mask = np.logical_and(np.array([i == self.SAMPLE_DEFAULT for i in self.best_attacks]), np.logical_not(sample_is_robust))
                self.best_attacks[atk_mask] = self.attacks.index(attack)
        if self.targeted:
            y_t = np.array([range(y.shape[1])] * y.shape[0])
            y_idx = np.argmax(y, axis=1)
            y_idx = np.expand_dims(y_idx, 1)
            y_t = y_t[y_t != y_idx]
            targeted_labels = np.reshape(y_t, (y.shape[0], self.SAMPLE_DEFAULT))
            for attack in self.attacks:
                try:
                    attack.set_params(targeted=True)
                    for i in range(self.estimator.nb_classes - 1):
                        if np.sum(sample_is_robust) == 0:
                            break
                        target = check_and_transform_label_format(targeted_labels[:, i], nb_classes=self.estimator.nb_classes)
                        if self.parallel:
                            args.append((deepcopy(x_adv), deepcopy(target), deepcopy(sample_is_robust), deepcopy(attack), deepcopy(self.estimator), deepcopy(self.norm), deepcopy(self.eps)))
                        else:
                            (x_adv, sample_is_robust) = run_attack(x=x_adv, y=target, sample_is_robust=sample_is_robust, attack=attack, estimator_orig=self.estimator, norm=self.norm, eps=self.eps, **kwargs)
                            atk_mask = np.logical_and(np.array([i == self.SAMPLE_DEFAULT for i in self.best_attacks]), np.logical_not(sample_is_robust))
                            self.best_attacks[atk_mask] = self.attacks.index(attack)
                except ValueError as error:
                    logger.warning('Error completing attack: %s}', str(error))
        if self.parallel:
            with multiprocess.get_context('spawn').Pool() as pool:
                results = pool.starmap(run_attack, args)
            perturbations = []
            is_robust = []
            for img_idx in range(len(x)):
                perturbations.append(np.array([np.linalg.norm(x[img_idx] - i[0][img_idx]) for i in results]))
                is_robust.append([i[1][img_idx] for i in results])
            best_attacks = np.argmin(np.where(np.invert(np.array(is_robust)), np.array(perturbations), np.inf), axis=1)
            x_adv = np.concatenate([results[best_attacks[img]][0][[img]] for img in range(len(x))])
            self.best_attacks = best_attacks
            self.args = args
        return x_adv

    def _check_params(self) -> None:
        if False:
            return 10
        if self.norm not in [1, 2, np.inf, 'inf']:
            raise ValueError('The argument norm has to be either 1, 2, np.inf, "inf".')
        if not isinstance(self.eps, (int, float)) or self.eps <= 0.0:
            raise ValueError('The argument eps has to be either of type int or float and larger than zero.')
        if not isinstance(self.eps_step, (int, float)) or self.eps_step <= 0.0:
            raise ValueError('The argument eps_step has to be either of type int or float and larger than zero.')
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError('The argument batch_size has to be of type int and larger than zero.')

    def __repr__(self) -> str:
        if False:
            return 10
        '\n        This method returns a summary of the best performing (lowest perturbation in the parallel case) attacks\n        per image passed to the AutoAttack class.\n        '
        if self.parallel:
            best_attack_meta = '\n'.join([f'image {i + 1}: {str(self.args[idx][3])}' if idx != 0 else f'image {i + 1}: n/a' for (i, idx) in enumerate(self.best_attacks)])
            auto_attack_meta = f'AutoAttack(targeted={self.targeted}, parallel={self.parallel}, num_attacks={len(self.args)})'
            return f'{auto_attack_meta}\nBestAttacks:\n{best_attack_meta}'
        best_attack_meta = '\n'.join([f'image {i + 1}: {str(self.attacks[idx])}' if idx != -2 else f'image {i + 1}: n/a' for (i, idx) in enumerate(self.best_attacks)])
        auto_attack_meta = f'AutoAttack(targeted={self.targeted}, parallel={self.parallel}, num_attacks={len(self.attacks)})'
        return f'{auto_attack_meta}\nBestAttacks:\n{best_attack_meta}'

def run_attack(x: np.ndarray, y: np.ndarray, sample_is_robust: np.ndarray, attack: EvasionAttack, estimator_orig: 'CLASSIFIER_TYPE', norm: Union[int, float, str]=np.inf, eps: float=0.3, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    if False:
        i = 10
        return i + 15
    '\n    Run attack.\n\n    :param x: An array of the original inputs.\n    :param y: An array of the labels.\n    :param sample_is_robust: Store the initial robustness of examples.\n    :param attack: Evasion attack to run.\n    :param estimator_orig: Original estimator to be attacked by adversarial examples.\n    :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf, 1 or 2.\n    :param eps: Maximum perturbation that the attacker can introduce.\n    :return: An array holding the adversarial examples.\n    '
    x_robust = x[sample_is_robust]
    y_robust = y[sample_is_robust]
    x_robust_adv = attack.generate(x=x_robust, y=y_robust, **kwargs)
    y_pred_robust_adv = estimator_orig.predict(x_robust_adv)
    rel_acc = 0.0001
    order = np.inf if norm == 'inf' else norm
    assert isinstance(order, (int, float))
    norm_is_smaller_eps = (1 - rel_acc) * np.linalg.norm((x_robust_adv - x_robust).reshape((x_robust_adv.shape[0], -1)), axis=1, ord=order) <= eps
    if attack.targeted:
        samples_misclassified = np.argmax(y_pred_robust_adv, axis=1) == np.argmax(y_robust, axis=1)
    elif not attack.targeted:
        samples_misclassified = np.argmax(y_pred_robust_adv, axis=1) != np.argmax(y_robust, axis=1)
    else:
        raise ValueError
    sample_is_not_robust = np.logical_and(samples_misclassified, norm_is_smaller_eps)
    x_robust[sample_is_not_robust] = x_robust_adv[sample_is_not_robust]
    x[sample_is_robust] = x_robust
    sample_is_robust[sample_is_robust] = np.invert(sample_is_not_robust)
    return (x, sample_is_robust)