"""
This module implements the white-box attack `NewtonFool`.

| Paper link: http://doi.acm.org/10.1145/3134600.3134635
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, TYPE_CHECKING
import numpy as np
from tqdm.auto import trange
from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.utils import to_categorical
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE
logger = logging.getLogger(__name__)

class NewtonFool(EvasionAttack):
    """
    Implementation of the attack from Uyeong Jang et al. (2017).

    | Paper link: http://doi.acm.org/10.1145/3134600.3134635
    """
    attack_params = EvasionAttack.attack_params + ['max_iter', 'eta', 'batch_size', 'verbose']
    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(self, classifier: 'CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE', max_iter: int=100, eta: float=0.01, batch_size: int=1, verbose: bool=True) -> None:
        if False:
            return 10
        '\n        Create a NewtonFool attack instance.\n\n        :param classifier: A trained classifier.\n        :param max_iter: The maximum number of iterations.\n        :param eta: The eta coefficient.\n        :param batch_size: Size of the batch on which adversarial samples are generated.\n        :param verbose: Show progress bars.\n        '
        super().__init__(estimator=classifier)
        self.max_iter = max_iter
        self.eta = eta
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate adversarial samples and return them in a Numpy array.\n\n        :param x: An array with the original inputs to be attacked.\n        :param y: An array with the original labels to be predicted.\n        :return: An array holding the adversarial examples.\n        '
        x_adv = x.astype(ART_NUMPY_DTYPE)
        y_pred = self.estimator.predict(x, batch_size=self.batch_size)
        pred_class = np.argmax(y_pred, axis=1)
        if self.estimator.nb_classes == 2 and y_pred.shape[1] == 1:
            raise ValueError('This attack has not yet been tested for binary classification with a single output classifier.')
        for batch_id in trange(int(np.ceil(x_adv.shape[0] / float(self.batch_size))), desc='NewtonFool', disable=not self.verbose):
            (batch_index_1, batch_index_2) = (batch_id * self.batch_size, (batch_id + 1) * self.batch_size)
            batch = x_adv[batch_index_1:batch_index_2]
            norm_batch = np.linalg.norm(np.reshape(batch, (batch.shape[0], -1)), axis=1)
            l_batch = pred_class[batch_index_1:batch_index_2]
            l_b = to_categorical(l_batch, self.estimator.nb_classes).astype(bool)
            for _ in range(self.max_iter):
                score = self.estimator.predict(batch)[l_b]
                grads = self.estimator.class_gradient(batch, label=l_batch)
                if grads.shape[1] == 1:
                    grads = np.squeeze(grads, axis=1)
                norm_grad = np.linalg.norm(np.reshape(grads, (batch.shape[0], -1)), axis=1)
                theta = self._compute_theta(norm_batch, score, norm_grad)
                di_batch = self._compute_pert(theta, grads, norm_grad)
                batch += di_batch
            if self.estimator.clip_values is not None:
                (clip_min, clip_max) = self.estimator.clip_values
                x_adv[batch_index_1:batch_index_2] = np.clip(batch, clip_min, clip_max)
            else:
                x_adv[batch_index_1:batch_index_2] = batch
        return x_adv

    def _compute_theta(self, norm_batch: np.ndarray, score: np.ndarray, norm_grad: np.ndarray) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Function to compute the theta at each step.\n\n        :param norm_batch: Norm of a batch.\n        :param score: Softmax value at the attacked class.\n        :param norm_grad: Norm of gradient values at the attacked class.\n        :return: Theta value.\n        '
        equ1 = self.eta * norm_batch * norm_grad
        equ2 = score - 1.0 / self.estimator.nb_classes
        result = np.minimum.reduce([equ1, equ2])
        return result

    @staticmethod
    def _compute_pert(theta: np.ndarray, grads: np.ndarray, norm_grad: np.ndarray) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Function to compute the perturbation at each step.\n\n        :param theta: Theta value at the current step.\n        :param grads: Gradient values at the attacked class.\n        :param norm_grad: Norm of gradient values at the attacked class.\n        :return: Computed perturbation.\n        '
        tol = 1e-07
        nom = -theta.reshape((-1,) + (1,) * (len(grads.shape) - 1)) * grads
        denom = norm_grad ** 2
        denom[denom < tol] = tol
        result = nom / denom.reshape((-1,) + (1,) * (len(grads.shape) - 1))
        return result

    def _check_params(self) -> None:
        if False:
            i = 10
            return i + 15
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError('The number of iterations must be a positive integer.')
        if not isinstance(self.eta, (float, int)) or self.eta <= 0:
            raise ValueError('The eta coefficient must be a positive float.')
        if self.batch_size <= 0:
            raise ValueError('The batch size `batch_size` has to be positive.')
        if not isinstance(self.verbose, bool):
            raise ValueError('The argument `verbose` has to be of type bool.')