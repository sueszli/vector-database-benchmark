"""
This module implements ``Wasserstein Adversarial Examples via Projected Sinkhorn Iterations`` as evasion attack.

| Paper link: https://arxiv.org/abs/1902.07906
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, TYPE_CHECKING
import numpy as np
from scipy.special import lambertw
from tqdm.auto import trange
from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.attacks.attack import EvasionAttack
from art.utils import get_labels_np_array, check_and_transform_label_format
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE
logger = logging.getLogger(__name__)
EPS_LOG = 10 ** (-10)

class Wasserstein(EvasionAttack):
    """
    Implements ``Wasserstein Adversarial Examples via Projected Sinkhorn Iterations`` as evasion attack.

    | Paper link: https://arxiv.org/abs/1902.07906
    """
    attack_params = EvasionAttack.attack_params + ['targeted', 'regularization', 'p', 'kernel_size', 'eps_step', 'norm', 'ball', 'eps', 'eps_iter', 'eps_factor', 'max_iter', 'conjugate_sinkhorn_max_iter', 'projected_sinkhorn_max_iter', 'batch_size', 'verbose']
    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)

    def __init__(self, estimator: 'CLASSIFIER_LOSS_GRADIENTS_TYPE', targeted: bool=False, regularization: float=3000.0, p: int=2, kernel_size: int=5, eps_step: float=0.1, norm: str='wasserstein', ball: str='wasserstein', eps: float=0.3, eps_iter: int=10, eps_factor: float=1.1, max_iter: int=400, conjugate_sinkhorn_max_iter: int=400, projected_sinkhorn_max_iter: int=400, batch_size: int=1, verbose: bool=True):
        if False:
            while True:
                i = 10
        '\n        Create a Wasserstein attack instance.\n\n        :param estimator: A trained estimator.\n        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).\n        :param regularization: Entropy regularization.\n        :param p: The p-wasserstein distance.\n        :param kernel_size: Kernel size for computing the cost matrix.\n        :param eps_step: Attack step size (input variation) at each iteration.\n        :param norm: The norm of the adversarial perturbation. Possible values: `inf`, `1`, `2` or `wasserstein`.\n        :param ball: The ball of the adversarial perturbation. Possible values: `inf`, `1`, `2` or `wasserstein`.\n        :param eps: Maximum perturbation that the attacker can introduce.\n        :param eps_iter: Number of iterations to increase the epsilon.\n        :param eps_factor: Factor to increase the epsilon.\n        :param max_iter: The maximum number of iterations.\n        :param conjugate_sinkhorn_max_iter: The maximum number of iterations for the conjugate sinkhorn optimizer.\n        :param projected_sinkhorn_max_iter: The maximum number of iterations for the projected sinkhorn optimizer.\n        :param batch_size: Size of batches.\n        :param verbose: Show progress bars.\n        '
        super().__init__(estimator=estimator)
        self._targeted = targeted
        self.regularization = regularization
        self.p = p
        self.kernel_size = kernel_size
        self.eps_step = eps_step
        self.norm = norm
        self.ball = ball
        self.eps = eps
        self.eps_iter = eps_iter
        self.eps_factor = eps_factor
        self.max_iter = max_iter
        self.conjugate_sinkhorn_max_iter = conjugate_sinkhorn_max_iter
        self.projected_sinkhorn_max_iter = projected_sinkhorn_max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            return 10
        '\n        Generate adversarial samples and return them in an array.\n\n        :param x: An array with the original inputs.\n        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape\n                  (nb_samples,). Only provide this parameter if you\'d like to use true labels when crafting adversarial\n                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect\n                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.\n        :param cost_matrix: A non-negative cost matrix.\n        :type cost_matrix: `np.ndarray`\n        :return: An array holding the adversarial examples.\n        '
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        x_adv = x.copy().astype(ART_NUMPY_DTYPE)
        if y is None:
            if self.targeted:
                raise ValueError('Target labels `y` need to be provided for a targeted attack.')
            targets = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
        else:
            targets = y
        if self.estimator.nb_classes == 2 and targets.shape[1] == 1:
            raise ValueError('This attack has not yet been tested for binary classification with a single output classifier.')
        cost_matrix = kwargs.get('cost_matrix')
        if cost_matrix is None:
            cost_matrix = self._compute_cost_matrix(self.p, self.kernel_size)
        nb_batches = int(np.ceil(x.shape[0] / float(self.batch_size)))
        for batch_id in trange(nb_batches, desc='Wasserstein', disable=not self.verbose):
            logger.debug('Processing batch %i out of %i', batch_id, nb_batches)
            (batch_index_1, batch_index_2) = (batch_id * self.batch_size, (batch_id + 1) * self.batch_size)
            batch = x_adv[batch_index_1:batch_index_2]
            batch_labels = targets[batch_index_1:batch_index_2]
            x_adv[batch_index_1:batch_index_2] = self._generate_batch(batch, batch_labels, cost_matrix)
        return x_adv

    def _generate_batch(self, x: np.ndarray, targets: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Generate a batch of adversarial samples and return them in an array.\n\n        :param x: An array with the original inputs.\n        :param targets: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.\n        :param cost_matrix: A non-negative cost matrix.\n        :return: Adversarial examples.\n        '
        adv_x = x.copy().astype(ART_NUMPY_DTYPE)
        adv_x_best = x.copy().astype(ART_NUMPY_DTYPE)
        if self.targeted:
            err = np.argmax(self.estimator.predict(adv_x, batch_size=x.shape[0]), axis=1) == np.argmax(targets, axis=1)
        else:
            err = np.argmax(self.estimator.predict(adv_x, batch_size=x.shape[0]), axis=1) != np.argmax(targets, axis=1)
        err_best = err
        eps_ = np.ones(x.shape[0]) * self.eps
        for i in range(self.max_iter):
            adv_x = self._compute(adv_x, x, targets, cost_matrix, eps_, err)
            if self.targeted:
                err = np.argmax(self.estimator.predict(adv_x, batch_size=x.shape[0]), axis=1) == np.argmax(targets, axis=1)
            else:
                err = np.argmax(self.estimator.predict(adv_x, batch_size=x.shape[0]), axis=1) != np.argmax(targets, axis=1)
            if np.mean(err) > np.mean(err_best):
                err_best = err
                adv_x_best = adv_x.copy()
            if np.mean(err) == 1:
                break
            if (i + 1) % self.eps_iter == 0:
                eps_[~err] *= self.eps_factor
        return adv_x_best

    def _compute(self, x_adv: np.ndarray, x_init: np.ndarray, y: np.ndarray, cost_matrix: np.ndarray, eps: np.ndarray, err: np.ndarray) -> np.ndarray:
        if False:
            return 10
        '\n        Compute adversarial examples for one iteration.\n\n        :param x_adv: Current adversarial examples.\n        :param x_init: An array with the original inputs.\n        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape\n                  (nb_samples,). Only provide this parameter if you\'d like to use true labels when crafting adversarial\n                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect\n                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.\n        :param cost_matrix: A non-negative cost matrix.\n        :param eps: Maximum perturbation that the attacker can introduce.\n        :param err: Current successful adversarial examples.\n        :return: Adversarial examples.\n        '
        x_adv[~err] = self._compute_apply_perturbation(x_adv, y, cost_matrix)[~err]
        x_adv[~err] = self._apply_projection(x_adv, x_init, cost_matrix, eps)[~err]
        if self.estimator.clip_values is not None:
            (clip_min, clip_max) = self.estimator.clip_values
            x_adv = np.clip(x_adv, clip_min, clip_max)
        return x_adv

    def _compute_apply_perturbation(self, x: np.ndarray, y: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Compute and apply perturbations.\n\n        :param x: Current adversarial examples.\n        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape\n                  (nb_samples,). Only provide this parameter if you\'d like to use true labels when crafting adversarial\n                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect\n                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.\n        :param cost_matrix: A non-negative cost matrix.\n        :return: Adversarial examples.\n        '
        tol = 1e-07
        grad = self.estimator.loss_gradient(x, y) * (1 - 2 * int(self.targeted))
        if self.norm == 'inf':
            grad = np.sign(grad)
            x_adv = x + self.eps_step * grad
        elif self.norm == '1':
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (np.sum(np.abs(grad), axis=ind, keepdims=True) + tol)
            x_adv = x + self.eps_step * grad
        elif self.norm == '2':
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (np.sqrt(np.sum(np.square(grad), axis=ind, keepdims=True)) + tol)
            x_adv = x + self.eps_step * grad
        elif self.norm == 'wasserstein':
            x_adv = self._conjugate_sinkhorn(x, grad, cost_matrix)
        else:
            raise NotImplementedError('Values of `norm` different from `1`, `2`, `inf` and `wasserstein` are currently not supported.')
        return x_adv

    def _apply_projection(self, x: np.ndarray, x_init: np.ndarray, cost_matrix: np.ndarray, eps: np.ndarray) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Apply projection on the ball of size `eps`.\n\n        :param x: Current adversarial examples.\n        :param x_init: An array with the original inputs.\n        :param cost_matrix: A non-negative cost matrix.\n        :param eps: Maximum perturbation that the attacker can introduce.\n        :return: Adversarial examples.\n        '
        tol = 1e-07
        if self.ball == '2':
            values = x - x_init
            values_tmp = values.reshape((values.shape[0], -1))
            values_tmp = values_tmp * np.expand_dims(np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1) + tol)), axis=1)
            values = values_tmp.reshape(values.shape)
            x_adv = values + x_init
        elif self.ball == '1':
            values = x - x_init
            values_tmp = values.reshape((values.shape[0], -1))
            values_tmp = values_tmp * np.expand_dims(np.minimum(1.0, eps / (np.linalg.norm(values_tmp, axis=1, ord=1) + tol)), axis=1)
            values = values_tmp.reshape(values.shape)
            x_adv = values + x_init
        elif self.ball == 'inf':
            values = x - x_init
            values_tmp = values.reshape((values.shape[0], -1))
            values_tmp = np.sign(values_tmp) * np.minimum(abs(values_tmp), np.expand_dims(eps, -1))
            values = values_tmp.reshape(values.shape)
            x_adv = values + x_init
        elif self.ball == 'wasserstein':
            x_adv = self._projected_sinkhorn(x, x_init, cost_matrix, eps)
        else:
            raise NotImplementedError('Values of `ball` different from `1`, `2`, `inf` and `wasserstein` are currently not supported.')
        return x_adv

    def _conjugate_sinkhorn(self, x: np.ndarray, grad: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        The conjugate sinkhorn_optimizer.\n\n        :param x: Current adversarial examples.\n        :param grad: The loss gradients.\n        :param cost_matrix: A non-negative cost matrix.\n        :return: Adversarial examples.\n        '
        normalization = x.reshape(x.shape[0], -1).sum(-1).reshape(x.shape[0], 1, 1, 1)
        x = x.copy() / normalization
        m = np.prod(x.shape[1:])
        alpha = np.log(np.ones(x.shape) / m) + 0.5
        exp_alpha = np.exp(-alpha)
        beta = -self.regularization * grad
        beta = beta.astype(np.float64)
        exp_beta = np.exp(-beta)
        if (exp_beta == np.inf).any():
            raise ValueError('Overflow error in `_conjugate_sinkhorn` for exponential beta.')
        cost_matrix_new = cost_matrix.copy() + 1
        cost_matrix_new = np.expand_dims(np.expand_dims(cost_matrix_new, 0), 0)
        i_nonzero = self._batch_dot(x, self._local_transport(cost_matrix_new, grad, self.kernel_size)) != 0
        i_nonzero_ = np.zeros(alpha.shape).astype(bool)
        i_nonzero_[:, :, :, :] = np.expand_dims(np.expand_dims(np.expand_dims(i_nonzero, -1), -1), -1)
        psi = np.ones(x.shape[0])
        var_k = np.expand_dims(np.expand_dims(np.expand_dims(psi, -1), -1), -1)
        var_k = np.exp(-var_k * cost_matrix - 1)
        convergence = np.array([-np.inf])
        for _ in range(self.conjugate_sinkhorn_max_iter):
            x[x == 0.0] = EPS_LOG
            alpha[i_nonzero_] = (np.log(self._local_transport(var_k, exp_beta, self.kernel_size)) - np.log(x))[i_nonzero_]
            exp_alpha = np.exp(-alpha)
            var_g = -self.eps_step + self._batch_dot(exp_alpha, self._local_transport(cost_matrix * var_k, exp_beta, self.kernel_size))
            var_h = -self._batch_dot(exp_alpha, self._local_transport(cost_matrix * cost_matrix * var_k, exp_beta, self.kernel_size))
            delta = var_g / var_h
            tmp = np.ones(delta.shape)
            neg = psi - tmp * delta < 0
            while neg.any() and np.min(tmp) > 0.01:
                tmp[neg] /= 2
                neg = psi - tmp * delta < 0
            psi[i_nonzero] = np.maximum(psi - tmp * delta, 0)[i_nonzero]
            var_k = np.expand_dims(np.expand_dims(np.expand_dims(psi, -1), -1), -1)
            var_k = np.exp(-var_k * cost_matrix - 1)
            next_convergence = self._conjugated_sinkhorn_evaluation(x, alpha, exp_alpha, exp_beta, psi, var_k)
            if (np.abs(convergence - next_convergence) <= 0.0001 + 0.0001 * np.abs(next_convergence)).all():
                break
            convergence = next_convergence
        result = exp_beta * self._local_transport(var_k, exp_alpha, self.kernel_size)
        result[~i_nonzero] = 0
        result *= normalization
        return result

    def _projected_sinkhorn(self, x: np.ndarray, x_init: np.ndarray, cost_matrix: np.ndarray, eps: np.ndarray) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        The projected sinkhorn_optimizer.\n\n        :param x: Current adversarial examples.\n        :param x_init: An array with the original inputs.\n        :param cost_matrix: A non-negative cost matrix.\n        :param eps: Maximum perturbation that the attacker can introduce.\n        :return: Adversarial examples.\n        '
        normalization = x_init.reshape(x.shape[0], -1).sum(-1).reshape(x.shape[0], 1, 1, 1)
        x = x.copy() / normalization
        x_init = x_init.copy() / normalization
        m = np.prod(x_init.shape[1:])
        beta = np.log(np.ones(x.shape) / m)
        exp_beta = np.exp(-beta)
        psi = np.ones(x.shape[0])
        var_k = np.expand_dims(np.expand_dims(np.expand_dims(psi, -1), -1), -1)
        var_k = np.exp(-var_k * cost_matrix - 1)
        convergence = np.array([-np.inf])
        for _ in range(self.projected_sinkhorn_max_iter):
            x_init[x_init == 0.0] = EPS_LOG
            alpha = np.log(self._local_transport(var_k, exp_beta, self.kernel_size)) - np.log(x_init)
            exp_alpha = np.exp(-alpha)
            beta = self.regularization * np.exp(self.regularization * x) * self._local_transport(var_k, exp_alpha, self.kernel_size)
            beta[beta > 1e-10] = np.real(lambertw(beta[beta > 1e-10]))
            beta -= self.regularization * x
            exp_beta = np.exp(-beta)
            var_g = -eps + self._batch_dot(exp_alpha, self._local_transport(cost_matrix * var_k, exp_beta, self.kernel_size))
            var_h = -self._batch_dot(exp_alpha, self._local_transport(cost_matrix * cost_matrix * var_k, exp_beta, self.kernel_size))
            delta = var_g / var_h
            tmp = np.ones(delta.shape)
            neg = psi - tmp * delta < 0
            while neg.any() and np.min(tmp) > 0.01:
                tmp[neg] /= 2
                neg = psi - tmp * delta < 0
            psi = np.maximum(psi - tmp * delta, 0)
            var_k = np.expand_dims(np.expand_dims(np.expand_dims(psi, -1), -1), -1)
            var_k = np.exp(-var_k * cost_matrix - 1)
            next_convergence = self._projected_sinkhorn_evaluation(x, x_init, alpha, exp_alpha, beta, exp_beta, psi, var_k, eps)
            if (np.abs(convergence - next_convergence) <= 0.0001 + 0.0001 * np.abs(next_convergence)).all():
                break
            convergence = next_convergence
        result = (beta / self.regularization + x) * normalization
        return result

    @staticmethod
    def _compute_cost_matrix(var_p: int, kernel_size: int) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute the default cost matrix.\n\n        :param var_p: The p-wasserstein distance.\n        :param kernel_size: Kernel size for computing the cost matrix.\n        :return: The cost matrix.\n        '
        center = kernel_size // 2
        cost_matrix = np.zeros((kernel_size, kernel_size))
        for i in range(kernel_size):
            for j in range(kernel_size):
                cost_matrix[i, j] = (abs(i - center) ** var_p + abs(j - center) ** var_p) ** (1 / var_p)
        return cost_matrix

    @staticmethod
    def _batch_dot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Compute batch dot product.\n\n        :param x: Sample batch.\n        :param y: Sample batch.\n        :return: Batch dot product.\n        '
        batch_size = x.shape[0]
        assert batch_size == y.shape[0]
        var_x_ = x.reshape(batch_size, 1, -1)
        var_y_ = y.reshape(batch_size, -1, 1)
        result = np.matmul(var_x_, var_y_).reshape(batch_size)
        return result

    @staticmethod
    def _unfold(x: np.ndarray, kernel_size: int, padding: int) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Extract sliding local blocks from a batched input.\n\n        :param x: A batched input of shape `batch x channel x width x height`.\n        :param kernel_size: Kernel size for computing the cost matrix.\n        :param padding: Controls the amount of implicit zero-paddings on both sides for padding number of points\n            for each dimension before reshaping.\n        :return: Sliding local blocks.\n        '
        shape = tuple(np.array(x.shape[2:]) + padding * 2)
        x_pad = np.zeros(x.shape[:2] + shape)
        x_pad[:, :, padding:shape[0] - padding, padding:shape[1] - padding] = x
        res_dim_0 = x.shape[0]
        res_dim_1 = x.shape[1] * kernel_size ** 2
        res_dim_2 = (shape[0] - kernel_size + 1) * (shape[1] - kernel_size + 1)
        result = np.zeros((res_dim_0, res_dim_1, res_dim_2))
        for i in range(shape[0] - kernel_size + 1):
            for j in range(shape[1] - kernel_size + 1):
                patch = x_pad[:, :, i:i + kernel_size, j:j + kernel_size]
                patch = patch.reshape(x.shape[0], -1)
                result[:, :, i * (shape[1] - kernel_size + 1) + j] = patch
        return result

    def _local_transport(self, var_k: np.ndarray, x: np.ndarray, kernel_size: int) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute local transport.\n\n        :param var_k: K parameter in Algorithm 2 of the paper ``Wasserstein Adversarial Examples via Projected\n            Sinkhorn Iterations``.\n        :param x: An array to apply local transport.\n        :param kernel_size: Kernel size for computing the cost matrix.\n        :return: Local transport result.\n        '
        num_channels = x.shape[1 if self.estimator.channels_first else 3]
        var_k = np.repeat(var_k, num_channels, axis=1)
        if not self.estimator.channels_first:
            x = np.swapaxes(x, 1, 3)
        unfold_x = self._unfold(x=x, kernel_size=kernel_size, padding=kernel_size // 2)
        unfold_x = unfold_x.swapaxes(-1, -2)
        unfold_x = unfold_x.reshape(*unfold_x.shape[:-1], num_channels, kernel_size ** 2)
        unfold_x = unfold_x.swapaxes(-2, -3)
        tmp_k = var_k.reshape(var_k.shape[0], num_channels, -1)
        tmp_k = np.expand_dims(tmp_k, -1)
        result = np.matmul(unfold_x, tmp_k)
        result = np.squeeze(result, -1)
        result = result.reshape(*result.shape[:-1], x.shape[-2], x.shape[-1])
        if not self.estimator.channels_first:
            result = np.swapaxes(result, 1, 3)
        return result

    def _projected_sinkhorn_evaluation(self, x: np.ndarray, x_init: np.ndarray, alpha: np.ndarray, exp_alpha: np.ndarray, beta: np.ndarray, exp_beta: np.ndarray, psi: np.ndarray, var_k: np.ndarray, eps: np.ndarray) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Function to evaluate the objective of the projected sinkhorn optimizer.\n\n        :param x: Current adversarial examples.\n        :param x_init: An array with the original inputs.\n        :param alpha: Alpha parameter in Algorithm 2 of the paper ``Wasserstein Adversarial Examples via Projected\n            Sinkhorn Iterations``.\n        :param exp_alpha: Exponential of alpha.\n        :param beta: Beta parameter in Algorithm 2 of the paper ``Wasserstein Adversarial Examples via Projected\n            Sinkhorn Iterations``.\n        :param exp_beta: Exponential of beta.\n        :param psi: Psi parameter in Algorithm 2 of the paper ``Wasserstein Adversarial Examples via Projected\n            Sinkhorn Iterations``.\n        :param var_k: K parameter in Algorithm 2 of the paper ``Wasserstein Adversarial Examples via Projected\n            Sinkhorn Iterations``.\n        :param eps: Maximum perturbation that the attacker can introduce.\n        :return: Evaluation result.\n        '
        return -0.5 / self.regularization * self._batch_dot(beta, beta) - psi * eps - self._batch_dot(np.minimum(alpha, 10000000000.0), x_init) - self._batch_dot(np.minimum(beta, 10000000000.0), x) - self._batch_dot(exp_alpha, self._local_transport(var_k, exp_beta, self.kernel_size))

    def _conjugated_sinkhorn_evaluation(self, x: np.ndarray, alpha: np.ndarray, exp_alpha: np.ndarray, exp_beta: np.ndarray, psi: np.ndarray, var_k: np.ndarray) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Function to evaluate the objective of the conjugated sinkhorn optimizer.\n\n        :param x: Current adversarial examples.\n        :param alpha: Alpha parameter in the conjugated sinkhorn optimizer of the paper ``Wasserstein Adversarial\n            Examples via Projected Sinkhorn Iterations``.\n        :param exp_alpha: Exponential of alpha.\n        :param exp_beta: Exponential of beta parameter in the conjugated sinkhorn optimizer of the paper ``Wasserstein\n            Adversarial Examples via Projected Sinkhorn Iterations``.\n        :param psi: Psi parameter in the conjugated sinkhorn optimizer of the paper ``Wasserstein Adversarial\n            Examples via Projected Sinkhorn Iterations``.\n        :param var_k: K parameter in the conjugated sinkhorn optimizer of the paper ``Wasserstein Adversarial Examples\n            via Projected Sinkhorn Iterations``.\n        :return: Evaluation result.\n        '
        return -psi * self.eps_step - self._batch_dot(np.minimum(alpha, 1e+38), x) - self._batch_dot(exp_alpha, self._local_transport(var_k, exp_beta, self.kernel_size))

    def _check_params(self) -> None:
        if False:
            print('Hello World!')
        if not isinstance(self.targeted, bool):
            raise ValueError('The flag `targeted` has to be of type bool.')
        if self.regularization <= 0:
            raise ValueError('The entropy regularization has to be greater than 0.')
        if not isinstance(self.p, int):
            raise TypeError('The p-wasserstein has to be of type integer.')
        if self.p < 1:
            raise ValueError('The p-wasserstein must be larger or equal to 1.')
        if not isinstance(self.kernel_size, int):
            raise TypeError('The kernel size has to be of type integer.')
        if self.kernel_size % 2 != 1:
            raise ValueError('Need odd kernel size.')
        if self.norm not in ['inf', '1', '2', 'wasserstein']:
            raise ValueError('Norm order must be either `inf`, `1`, `2` or `wasserstein`.')
        if self.ball not in ['inf', '1', '2', 'wasserstein']:
            raise ValueError('Ball order must be either `inf`, `1`, `2` or `wasserstein`.')
        if self.eps <= 0:
            raise ValueError('The perturbation size `eps` has to be positive.')
        if self.eps_step <= 0:
            raise ValueError('The perturbation step-size `eps_step` has to be positive.')
        if self.norm == 'inf' and self.eps_step > self.eps:
            raise ValueError('The iteration step `eps_step` has to be smaller than or equal to the total attack budget `eps`.')
        if self.eps_iter <= 0:
            raise ValueError('The number of epsilon iterations `eps_iter` has to be a positive integer.')
        if self.eps_factor <= 1:
            raise ValueError('The epsilon factor must be larger than 1.')
        if self.max_iter <= 0:
            raise ValueError('The number of iterations `max_iter` has to be a positive integer.')
        if self.conjugate_sinkhorn_max_iter <= 0:
            raise ValueError('The number of iterations `conjugate_sinkhorn_max_iter` has to be a positive integer.')
        if self.projected_sinkhorn_max_iter <= 0:
            raise ValueError('The number of iterations `projected_sinkhorn_max_iter` has to be a positive integer.')
        if self.batch_size <= 0:
            raise ValueError('The batch size `batch_size` has to be positive.')
        if not isinstance(self.verbose, bool):
            raise ValueError('The argument `verbose` has to be of type bool.')