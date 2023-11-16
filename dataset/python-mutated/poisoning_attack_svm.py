"""
This module implements poisoning attacks on Support Vector Machines.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Tuple
import numpy as np
from tqdm.auto import tqdm
from art.attacks.attack import PoisoningAttackWhiteBox
from art.estimators.classification.scikitlearn import ScikitlearnSVC
from art.utils import compute_success
logger = logging.getLogger(__name__)

class PoisoningAttackSVM(PoisoningAttackWhiteBox):
    """
    Close implementation of poisoning attack on Support Vector Machines (SVM) by Biggio et al.

    | Paper link: https://arxiv.org/pdf/1206.6389.pdf
    """
    attack_params = PoisoningAttackWhiteBox.attack_params + ['classifier', 'step', 'eps', 'x_train', 'y_train', 'x_val', 'y_val', 'verbose']
    _estimator_requirements = (ScikitlearnSVC,)

    def __init__(self, classifier: 'ScikitlearnSVC', step: Optional[float]=None, eps: Optional[float]=None, x_train: Optional[np.ndarray]=None, y_train: Optional[np.ndarray]=None, x_val: Optional[np.ndarray]=None, y_val: Optional[np.ndarray]=None, max_iter: int=100, verbose: bool=True) -> None:
        if False:
            print('Hello World!')
        '\n        Initialize an SVM poisoning attack.\n\n        :param classifier: A trained :class:`.ScikitlearnSVC` classifier.\n        :param step: The step size of the classifier.\n        :param eps: The minimum difference in loss before convergence of the classifier.\n        :param x_train: The training data used for classification.\n        :param y_train: The training labels used for classification.\n        :param x_val: The validation data used to test the attack.\n        :param y_val: The validation labels used to test the attack.\n        :param max_iter: The maximum number of iterations for the attack.\n        :raises `NotImplementedError`, `TypeError`: If the argument classifier has the wrong type.\n        :param verbose: Show progress bars.\n        '
        from sklearn.svm import LinearSVC, SVC
        super().__init__(classifier=classifier)
        if isinstance(self.estimator.model, LinearSVC):
            self._estimator = ScikitlearnSVC(model=SVC(C=self.estimator.model.C, kernel='linear'), clip_values=self.estimator.clip_values)
            self.estimator.fit(x_train, y_train)
        elif not isinstance(self.estimator.model, SVC):
            raise NotImplementedError(f"Model type '{type(self.estimator.model)}' not yet supported")
        self.step = step
        self.eps = eps
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.max_iter = max_iter
        self.verbose = verbose
        self._check_params()

    def poison(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            return 10
        '\n        Iteratively finds optimal attack points starting at values at `x`.\n\n        :param x: An array with the points that initialize attack points.\n        :param y: The target labels for the attack.\n        :return: A tuple holding the `(poisoning_examples, poisoning_labels)`.\n        '
        if y is None:
            raise ValueError('Target labels `y` need to be provided for a targeted attack.')
        y_attack = np.copy(y)
        num_poison = len(x)
        if num_poison == 0:
            raise ValueError('Must input at least one poison point')
        num_features = len(x[0])
        train_data = np.copy(self.x_train)
        train_labels = np.copy(self.y_train)
        all_poison = []
        for (attack_point, attack_label) in tqdm(zip(x, y_attack), desc='SVM poisoning', disable=not self.verbose):
            poison = self.generate_attack_point(attack_point, attack_label)
            all_poison.append(poison)
            train_data = np.vstack([train_data, poison])
            train_labels = np.vstack([train_labels, attack_label])
        x_adv = np.array(all_poison).reshape((num_poison, num_features))
        targeted = y is not None
        logger.info('Success rate of poisoning attack SVM attack: %.2f%%', 100 * compute_success(self.estimator, x, y, x_adv, targeted=targeted))
        return (x_adv, y_attack)

    def generate_attack_point(self, x_attack: np.ndarray, y_attack: np.ndarray) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        "\n        Generate a single poison attack the model, using `x_val` and `y_val` as validation points.\n        The attack begins at the point init_attack. The attack class will be the opposite of the model's\n        classification for `init_attack`.\n\n        :param x_attack: The initial attack point.\n        :param y_attack: The initial attack label.\n        :return: A tuple containing the final attack point and the poisoned model.\n        "
        from sklearn.preprocessing import normalize
        if self.y_train is None or self.x_train is None:
            raise ValueError('`x_train` and `y_train` cannot be None for generating an attack point.')
        poisoned_model = self.estimator.model
        y_t = np.argmax(self.y_train, axis=1)
        poisoned_model.fit(self.x_train, y_t)
        y_a = np.argmax(y_attack)
        attack_point = np.expand_dims(x_attack, axis=0)
        var_g = poisoned_model.decision_function(self.x_val)
        k_values = np.where(-var_g > 0)
        new_p = np.sum(var_g[k_values])
        old_p = np.copy(new_p)
        i = 0
        while new_p - old_p < self.eps and i < self.max_iter:
            old_p = new_p
            poisoned_input = np.vstack([self.x_train, attack_point])
            poisoned_labels = np.append(y_t, y_a)
            poisoned_model.fit(poisoned_input, poisoned_labels)
            unit_grad = normalize(self.attack_gradient(attack_point))
            attack_point += self.step * unit_grad
            (lower, upper) = self.estimator.clip_values
            new_attack = np.clip(attack_point, lower, upper)
            new_g = poisoned_model.decision_function(self.x_val)
            k_values = np.where(-new_g > 0)
            new_p = np.sum(new_g[k_values])
            i += 1
            attack_point = new_attack
        poisoned_input = np.vstack([self.x_train, attack_point])
        poisoned_labels = np.append(y_t, y_a)
        poisoned_model.fit(poisoned_input, poisoned_labels)
        return attack_point

    def predict_sign(self, vec: np.ndarray) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Predicts the inputs by binary classifier and outputs -1 and 1 instead of 0 and 1.\n\n        :param vec: An input array.\n        :return: An array of -1/1 predictions.\n        '
        preds = self.estimator.model.predict(vec)
        return 2 * preds - 1

    def attack_gradient(self, attack_point: np.ndarray, tol: float=0.0001) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Calculates the attack gradient, or dP for this attack.\n        See equation 8 in Biggio et al. Ch. 14\n\n        :param attack_point: The current attack point.\n        :param tol: Tolerance level.\n        :return: The attack gradient.\n        '
        if self.x_val is None or self.y_val is None:
            raise ValueError('The values of `x_val` and `y_val` are required for computing the gradients.')
        art_model = self.estimator
        model = self.estimator.model
        grad = np.zeros((1, self.x_val.shape[1]))
        support_vectors = model.support_vectors_
        num_support = len(support_vectors)
        support_labels = np.expand_dims(self.predict_sign(support_vectors), axis=1)
        c_idx = np.isin(support_vectors, attack_point).all(axis=1)
        if not c_idx.any():
            return grad
        c_idx = np.where(c_idx > 0)[0][0]
        alpha_c = model.dual_coef_[0, c_idx]
        assert support_labels.shape == (num_support, 1)
        qss = art_model.q_submatrix(support_vectors, support_vectors)
        qss_inv = np.linalg.inv(qss + np.random.uniform(0, 0.01 * np.min(qss) + tol, (num_support, num_support)))
        zeta = np.matmul(qss_inv, support_labels)
        zeta = np.matmul(support_labels.T, zeta)
        nu_k = np.matmul(qss_inv, support_labels)
        for (x_k, y_k) in zip(self.x_val, self.y_val):
            y_k = 2 * np.expand_dims(np.argmax(y_k), axis=0) - 1
            q_ks = art_model.q_submatrix(np.array([x_k]), support_vectors)
            m_k = 1.0 / zeta * np.matmul(q_ks, zeta * qss_inv - np.matmul(nu_k, nu_k.T)) + np.matmul(y_k, nu_k.T)
            d_q_sc = np.fromfunction(lambda i: art_model._get_kernel_gradient_sv(i, attack_point), (len(support_vectors),), dtype=int)
            d_q_kc = art_model._kernel_grad(x_k, attack_point)
            grad += (np.matmul(m_k, d_q_sc) + d_q_kc) * alpha_c
        return grad

    def _check_params(self) -> None:
        if False:
            return 10
        if self.step is not None and self.step <= 0:
            raise ValueError('Step size must be strictly positive.')
        if self.eps is not None and self.eps <= 0:
            raise ValueError('Value of eps must be strictly positive.')
        if self.max_iter <= 1:
            raise ValueError('Value of max_iter must be strictly positive.')
        if not isinstance(self.verbose, bool):
            raise ValueError('The argument `verbose` has to be of type bool.')