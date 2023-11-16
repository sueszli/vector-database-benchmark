"""
This module implements the virtual adversarial attack. It was originally used for virtual adversarial training.

| Paper link: https://arxiv.org/abs/1507.00677
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, TYPE_CHECKING
import numpy as np
from tqdm.auto import trange
from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE
logger = logging.getLogger(__name__)

class VirtualAdversarialMethod(EvasionAttack):
    """
    This attack was originally proposed by Miyato et al. (2016) and was used for virtual adversarial training.

    | Paper link: https://arxiv.org/abs/1507.00677
    """
    attack_params = EvasionAttack.attack_params + ['eps', 'finite_diff', 'max_iter', 'batch_size', 'verbose']
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(self, classifier: 'CLASSIFIER_TYPE', max_iter: int=10, finite_diff: float=1e-06, eps: float=0.1, batch_size: int=1, verbose: bool=True) -> None:
        if False:
            while True:
                i = 10
        '\n        Create a :class:`.VirtualAdversarialMethod` instance.\n\n        :param classifier: A trained classifier.\n        :param eps: Attack step (max input variation).\n        :param finite_diff: The finite difference parameter.\n        :param max_iter: The maximum number of iterations.\n        :param batch_size: Size of the batch on which adversarial samples are generated.\n        :param verbose: Show progress bars.\n        '
        super().__init__(estimator=classifier)
        self.finite_diff = finite_diff
        self.eps = eps
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate adversarial samples and return them in an array.\n\n        :param x: An array with the original inputs to be attacked.\n        :param y: An array with the original labels to be predicted.\n        :return: An array holding the adversarial examples.\n        '
        x_adv = x.astype(ART_NUMPY_DTYPE)
        preds = self.estimator.predict(x_adv, batch_size=self.batch_size)
        if self.estimator.nb_classes == 2 and preds.shape[1] == 1:
            raise ValueError('This attack has not yet been tested for binary classification with a single output classifier.')
        if (preds < 0.0).any() or (preds > 1.0).any():
            raise TypeError('This attack requires a classifier predicting probabilities in the range [0, 1] as output.Values smaller than 0.0 or larger than 1.0 have been detected.')
        preds_rescaled = preds
        for batch_id in trange(int(np.ceil(x_adv.shape[0] / float(self.batch_size))), desc='VAT', disable=not self.verbose):
            (batch_index_1, batch_index_2) = (batch_id * self.batch_size, (batch_id + 1) * self.batch_size)
            batch = x_adv[batch_index_1:batch_index_2]
            batch = batch.reshape((batch.shape[0], -1))
            var_d = np.random.randn(*batch.shape).astype(ART_NUMPY_DTYPE)
            for _ in range(self.max_iter):
                var_d = self._normalize(var_d)
                preds_new = self.estimator.predict((batch + var_d).reshape((-1,) + self.estimator.input_shape))
                if (preds_new < 0.0).any() or (preds_new > 1.0).any():
                    raise TypeError('This attack requires a classifier predicting probabilities in the range [0, 1] as output. Values smaller than 0.0 or larger than 1.0 have been detected.')
                preds_new_rescaled = preds_new
                from scipy.stats import entropy
                kl_div1 = entropy(np.transpose(preds_rescaled[batch_index_1:batch_index_2]), np.transpose(preds_new_rescaled))
                var_d_new = np.zeros(var_d.shape).astype(ART_NUMPY_DTYPE)
                for current_index in range(var_d.shape[1]):
                    var_d[:, current_index] += self.finite_diff
                    preds_new = self.estimator.predict((batch + var_d).reshape((-1,) + self.estimator.input_shape))
                    if (preds_new < 0.0).any() or (preds_new > 1.0).any():
                        raise TypeError('This attack requires a classifier predicting probabilities in the range [0, 1]as output. Values smaller than 0.0 or larger than 1.0 have been detected.')
                    preds_new_rescaled = preds_new
                    kl_div2 = entropy(np.transpose(preds_rescaled[batch_index_1:batch_index_2]), np.transpose(preds_new_rescaled))
                    var_d_new[:, current_index] = (kl_div2 - kl_div1) / self.finite_diff
                    var_d[:, current_index] -= self.finite_diff
                var_d = var_d_new
            if self.estimator.clip_values is not None:
                (clip_min, clip_max) = self.estimator.clip_values
                x_adv[batch_index_1:batch_index_2] = np.clip(batch + self.eps * self._normalize(var_d), clip_min, clip_max).reshape((-1,) + self.estimator.input_shape)
            else:
                x_adv[batch_index_1:batch_index_2] = (batch + self.eps * self._normalize(var_d)).reshape((-1,) + self.estimator.input_shape)
        return x_adv

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Apply L_2 batch normalization on `x`.\n\n        :param x: The input array batch to normalize.\n        :return: The normalized version of `x`.\n        '
        norm = np.atleast_1d(np.linalg.norm(x, axis=1))
        norm[norm == 0] = 1
        normalized_x = x / np.expand_dims(norm, axis=1)
        return normalized_x

    @staticmethod
    def _rescale(x: np.ndarray) -> np.ndarray:
        if False:
            return 10
        '\n        Rescale values of `x` to the range (0, 1]. The interval is open on the left side, using values close to zero\n        instead. This is to avoid values that are invalid for further KL divergence computation.\n\n        :param x: Input array.\n        :return: Rescaled value of `x`.\n        '
        tol = 1e-05
        current_range = np.amax(x, axis=1, keepdims=True) - np.amin(x, axis=1, keepdims=True)
        current_range[current_range == 0] = 1
        res = (x - np.amin(x, axis=1, keepdims=True) + tol) / current_range
        return res

    def _check_params(self) -> None:
        if False:
            print('Hello World!')
        if not isinstance(self.max_iter, int) or self.max_iter <= 0:
            raise ValueError('The number of iterations must be a positive integer.')
        if self.eps <= 0:
            raise ValueError('The attack step must be positive.')
        if not isinstance(self.finite_diff, float) or self.finite_diff <= 0:
            raise ValueError('The finite difference parameter must be a positive float.')
        if self.batch_size <= 0:
            raise ValueError('The batch size `batch_size` has to be positive.')
        if not isinstance(self.verbose, bool):
            raise ValueError('The argument `verbose` has to be of type bool.')