"""
This module implements the Jacobian-based Saliency Map attack `SaliencyMapMethod`. This is a white-box attack.

| Paper link: https://arxiv.org/abs/1511.07528
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Union, TYPE_CHECKING
import numpy as np
from tqdm.auto import trange
from art.attacks.attack import EvasionAttack
from art.config import ART_NUMPY_DTYPE
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassGradientsMixin
from art.utils import check_and_transform_label_format
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE
logger = logging.getLogger(__name__)

class SaliencyMapMethod(EvasionAttack):
    """
    Implementation of the Jacobian-based Saliency Map Attack (Papernot et al. 2016).

    | Paper link: https://arxiv.org/abs/1511.07528
    """
    attack_params = EvasionAttack.attack_params + ['theta', 'gamma', 'batch_size', 'verbose']
    _estimator_requirements = (BaseEstimator, ClassGradientsMixin)

    def __init__(self, classifier: 'CLASSIFIER_CLASS_LOSS_GRADIENTS_TYPE', theta: float=0.1, gamma: float=1.0, batch_size: int=1, verbose: bool=True) -> None:
        if False:
            while True:
                i = 10
        '\n        Create a SaliencyMapMethod instance.\n\n        :param classifier: A trained classifier.\n        :param theta: Amount of Perturbation introduced to each modified feature per step (can be positive or negative).\n        :param gamma: Maximum fraction of features being perturbed (between 0 and 1).\n        :param batch_size: Size of the batch on which adversarial samples are generated.\n        :param verbose: Show progress bars.\n        '
        super().__init__(estimator=classifier)
        self.theta = theta
        self.gamma = gamma
        self.batch_size = batch_size
        self.verbose = verbose
        self._check_params()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Generate adversarial samples and return them in an array.\n\n        :param x: An array with the original inputs to be attacked.\n        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape\n                  `(nb_samples,)`.\n        :return: An array holding the adversarial examples.\n        '
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        dims = list(x.shape[1:])
        self._nb_features = np.product(dims)
        x_adv = np.reshape(x.astype(ART_NUMPY_DTYPE), (-1, self._nb_features))
        preds = np.argmax(self.estimator.predict(x, batch_size=self.batch_size), axis=1)
        if y is None:
            from art.utils import random_targets
            targets = np.argmax(random_targets(preds, self.estimator.nb_classes), axis=1)
        else:
            if self.estimator.nb_classes == 2 and y.shape[1] == 1:
                raise ValueError('This attack has not yet been tested for binary classification with a single output classifier.')
            targets = np.argmax(y, axis=1)
        for batch_id in trange(int(np.ceil(x_adv.shape[0] / float(self.batch_size))), desc='JSMA', disable=not self.verbose):
            (batch_index_1, batch_index_2) = (batch_id * self.batch_size, (batch_id + 1) * self.batch_size)
            batch = x_adv[batch_index_1:batch_index_2]
            if self.estimator.clip_values is not None:
                search_space = np.zeros(batch.shape)
                (clip_min, clip_max) = self.estimator.clip_values
                if self.theta > 0:
                    search_space[batch < clip_max] = 1
                else:
                    search_space[batch > clip_min] = 1
            else:
                search_space = np.ones(batch.shape)
            current_pred = preds[batch_index_1:batch_index_2]
            target = targets[batch_index_1:batch_index_2]
            active_indices = np.where(current_pred != target)[0]
            all_feat = np.zeros_like(batch)
            while active_indices.size != 0:
                feat_ind = self._saliency_map(np.reshape(batch, [batch.shape[0]] + dims)[active_indices], target[active_indices], search_space[active_indices])
                all_feat[active_indices, feat_ind[:, 0]] = 1
                all_feat[active_indices, feat_ind[:, 1]] = 1
                if self.estimator.clip_values is not None:
                    if self.theta > 0:
                        (clip_func, clip_value) = (np.minimum, clip_max)
                    else:
                        (clip_func, clip_value) = (np.maximum, clip_min)
                    tmp_batch = batch[active_indices]
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 0]] = clip_func(clip_value, tmp_batch[np.arange(len(active_indices)), feat_ind[:, 0]] + self.theta)
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 1]] = clip_func(clip_value, tmp_batch[np.arange(len(active_indices)), feat_ind[:, 1]] + self.theta)
                    batch[active_indices] = tmp_batch
                    search_space[batch == clip_value] = 0
                else:
                    tmp_batch = batch[active_indices]
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 0]] += self.theta
                    tmp_batch[np.arange(len(active_indices)), feat_ind[:, 1]] += self.theta
                    batch[active_indices] = tmp_batch
                current_pred = np.argmax(self.estimator.predict(np.reshape(batch, [batch.shape[0]] + dims)), axis=1)
                active_indices = np.where((current_pred != target) * (np.sum(all_feat, axis=1) / self._nb_features <= self.gamma) * (np.sum(search_space, axis=1) > 0))[0]
            x_adv[batch_index_1:batch_index_2] = batch
        x_adv = np.reshape(x_adv, x.shape)
        return x_adv

    def _saliency_map(self, x: np.ndarray, target: Union[np.ndarray, int], search_space: np.ndarray) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Compute the saliency map of `x`. Return the top 2 coefficients in `search_space` that maximize / minimize\n        the saliency map.\n\n        :param x: A batch of input samples.\n        :param target: Target class for `x`.\n        :param search_space: The set of valid pairs of feature indices to search.\n        :return: The top 2 coefficients in `search_space` that maximize / minimize the saliency map.\n        '
        grads = self.estimator.class_gradient(x, label=target)
        grads = np.reshape(grads, (-1, self._nb_features))
        used_features = 1 - search_space
        coeff = 2 * int(self.theta > 0) - 1
        grads[used_features == 1] = -np.inf * coeff
        if self.theta > 0:
            ind = np.argpartition(grads, -2, axis=1)[:, -2:]
        else:
            ind = np.argpartition(-grads, -2, axis=1)[:, -2:]
        return ind

    def _check_params(self) -> None:
        if False:
            print('Hello World!')
        if self.gamma <= 0 or self.gamma > 1:
            raise ValueError('The total perturbation percentage `gamma` must be between 0 and 1.')
        if self.batch_size <= 0:
            raise ValueError('The batch size `batch_size` has to be positive.')
        if not isinstance(self.verbose, bool):
            raise ValueError('The argument `verbose` has to be of type bool.')