"""
This module implements the Projected Gradient Descent attack `ProjectedGradientDescent` as an iterative method in which,
after each iteration, the perturbation is projected on an lp-ball of specified radius (in addition to clipping the
values of the adversarial sample so that it lies in the permitted data range). This is the attack proposed by Madry et
al. for adversarial training.

| Paper link: https://arxiv.org/abs/1706.06083
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from typing import Optional, Union, TYPE_CHECKING
import numpy as np
from scipy.stats import truncnorm
from tqdm.auto import trange
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.config import ART_NUMPY_DTYPE
from art.estimators.classification.classifier import ClassifierMixin
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.utils import compute_success, get_labels_np_array, check_and_transform_label_format, compute_success_array
from art.summary_writer import SummaryWriter
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE, OBJECT_DETECTOR_TYPE
logger = logging.getLogger(__name__)

class ProjectedGradientDescentCommon(FastGradientMethod):
    """
    Common class for different variations of implementation of the Projected Gradient Descent attack. The attack is an
    iterative method in which, after each iteration, the perturbation is projected on an lp-ball of specified radius (in
    addition to clipping the values of the adversarial sample so that it lies in the permitted data range). This is the
    attack proposed by Madry et al. for adversarial training.

    | Paper link: https://arxiv.org/abs/1706.06083
    """
    attack_params = FastGradientMethod.attack_params + ['decay', 'max_iter', 'random_eps', 'verbose']
    _estimator_requirements = (BaseEstimator, LossGradientsMixin)

    def __init__(self, estimator: Union['CLASSIFIER_LOSS_GRADIENTS_TYPE', 'OBJECT_DETECTOR_TYPE'], norm: Union[int, float, str]=np.inf, eps: Union[int, float, np.ndarray]=0.3, eps_step: Union[int, float, np.ndarray]=0.1, decay: Optional[float]=None, max_iter: int=100, targeted: bool=False, num_random_init: int=0, batch_size: int=32, random_eps: bool=False, summary_writer: Union[str, bool, SummaryWriter]=False, verbose: bool=True) -> None:
        if False:
            return 10
        '\n        Create a :class:`.ProjectedGradientDescentCommon` instance.\n\n        :param estimator: A trained classifier.\n        :param norm: The norm of the adversarial perturbation supporting "inf", np.inf, 1 or 2.\n        :param eps: Maximum perturbation that the attacker can introduce.\n        :param eps_step: Attack step size (input variation) at each iteration.\n        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature\n            suggests this for FGSM based training to generalize across different epsilons. eps_step is\n            modified to preserve the ratio of eps / eps_step. The effectiveness of this method with PGD\n            is untested (https://arxiv.org/pdf/1611.01236.pdf).\n        :param decay: Decay factor for accumulating the velocity vector when using momentum.\n        :param max_iter: The maximum number of iterations.\n        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).\n        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0\n            starting at the original input.\n        :param batch_size: Size of the batch on which adversarial samples are generated.\n        :param summary_writer: Activate summary writer for TensorBoard.\n                               Default is `False` and deactivated summary writer.\n                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.\n                               If of type `str` save in path.\n                               If of type `SummaryWriter` apply provided custom summary writer.\n                               Use hierarchical folder structure to compare between runs easily. e.g. pass in\n                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.\n        :param verbose: Show progress bars.\n        '
        super().__init__(estimator=estimator, norm=norm, eps=eps, eps_step=eps_step, targeted=targeted, num_random_init=num_random_init, batch_size=batch_size, minimal=False, summary_writer=summary_writer)
        self.decay = decay
        self.max_iter = max_iter
        self.random_eps = random_eps
        self.verbose = verbose
        ProjectedGradientDescentCommon._check_params(self)
        lower: Union[int, float, np.ndarray]
        upper: Union[int, float, np.ndarray]
        var_mu: Union[int, float, np.ndarray]
        sigma: Union[int, float, np.ndarray]
        if self.random_eps:
            if isinstance(eps, (int, float)):
                (lower, upper) = (0, eps)
                (var_mu, sigma) = (0, eps / 2)
            else:
                (lower, upper) = (np.zeros_like(eps), eps)
                (var_mu, sigma) = (np.zeros_like(eps), eps / 2)
            self.norm_dist = truncnorm((lower - var_mu) / sigma, (upper - var_mu) / sigma, loc=var_mu, scale=sigma)

    def _random_eps(self):
        if False:
            return 10
        '\n        Check whether random eps is enabled, then scale eps and eps_step appropriately.\n        '
        if self.random_eps:
            ratio = self.eps_step / self.eps
            if isinstance(self.eps, (int, float)):
                self.eps = np.round(self.norm_dist.rvs(1)[0], 10)
            else:
                self.eps = np.round(self.norm_dist.rvs(size=self.eps.shape), 10)
            self.eps_step = ratio * self.eps

    def _set_targets(self, x: np.ndarray, y: Optional[np.ndarray], classifier_mixin: bool=True) -> np.ndarray:
        if False:
            print('Hello World!')
        '\n        Check and set up targets.\n\n        :param x: An array with the original inputs.\n        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape\n                  (nb_samples,). Only provide this parameter if you\'d like to use true labels when crafting adversarial\n                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect\n                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.\n        :param classifier_mixin: Whether the estimator is of type `ClassifierMixin`.\n        :return: The targets.\n        '
        if classifier_mixin:
            if y is not None:
                y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes)
        if y is None:
            if self.targeted:
                raise ValueError('Target labels `y` need to be provided for a targeted attack.')
            if classifier_mixin:
                targets = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
            else:
                targets = self.estimator.predict(x, batch_size=self.batch_size)
        else:
            targets = y
        return targets

    def _check_params(self) -> None:
        if False:
            while True:
                i = 10
        if self.norm not in [1, 2, np.inf, 'inf']:
            raise ValueError('Norm order must be either 1, 2, `np.inf` or "inf".')
        if not (isinstance(self.eps, (int, float)) and isinstance(self.eps_step, (int, float)) or (isinstance(self.eps, np.ndarray) and isinstance(self.eps_step, np.ndarray))):
            raise TypeError('The perturbation size `eps` and the perturbation step-size `eps_step` must have the same type of `int`, `float`, or `np.ndarray`.')
        if isinstance(self.eps, (int, float)):
            if self.eps < 0:
                raise ValueError('The perturbation size `eps` has to be non-negative.')
        elif (self.eps < 0).any():
            raise ValueError('The perturbation size `eps` has to be non-negative.')
        if isinstance(self.eps_step, (int, float)):
            if self.eps_step <= 0:
                raise ValueError('The perturbation step-size `eps_step` has to be positive.')
        elif (self.eps_step <= 0).any():
            raise ValueError('The perturbation step-size `eps_step` has to be positive.')
        if isinstance(self.eps, np.ndarray) and isinstance(self.eps_step, np.ndarray):
            if self.eps.shape != self.eps_step.shape:
                raise ValueError('The perturbation size `eps` and the perturbation step-size `eps_step` must have the same shape.')
        if not isinstance(self.targeted, bool):
            raise ValueError('The flag `targeted` has to be of type bool.')
        if not isinstance(self.num_random_init, int):
            raise TypeError('The number of random initialisations has to be of type integer.')
        if self.num_random_init < 0:
            raise ValueError('The number of random initialisations `random_init` has to be greater than or equal to 0.')
        if self.batch_size <= 0:
            raise ValueError('The batch size `batch_size` has to be positive.')
        if self.max_iter < 0:
            raise ValueError('The number of iterations `max_iter` has to be a non-negative integer.')
        if self.decay is not None and self.decay < 0.0:
            raise ValueError('The decay factor `decay` has to be a nonnegative float.')
        if not isinstance(self.verbose, bool):
            raise ValueError('The verbose has to be a Boolean.')

class ProjectedGradientDescentNumpy(ProjectedGradientDescentCommon):
    """
    The Projected Gradient Descent attack is an iterative method in which, after each iteration, the perturbation is
    projected on an lp-ball of specified radius (in addition to clipping the values of the adversarial sample so that it
    lies in the permitted data range). This is the attack proposed by Madry et al. for adversarial training.

    | Paper link: https://arxiv.org/abs/1706.06083
    """

    def __init__(self, estimator: Union['CLASSIFIER_LOSS_GRADIENTS_TYPE', 'OBJECT_DETECTOR_TYPE'], norm: Union[int, float, str]=np.inf, eps: Union[int, float, np.ndarray]=0.3, eps_step: Union[int, float, np.ndarray]=0.1, decay: Optional[float]=None, max_iter: int=100, targeted: bool=False, num_random_init: int=0, batch_size: int=32, random_eps: bool=False, summary_writer: Union[str, bool, SummaryWriter]=False, verbose: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Create a :class:`.ProjectedGradientDescentNumpy` instance.\n\n        :param estimator: An trained estimator.\n        :param norm: The norm of the adversarial perturbation supporting "inf", np.inf, 1 or 2.\n        :param eps: Maximum perturbation that the attacker can introduce.\n        :param eps_step: Attack step size (input variation) at each iteration.\n        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature\n                           suggests this for FGSM based training to generalize across different epsilons. eps_step\n                           is modified to preserve the ratio of eps / eps_step. The effectiveness of this method with\n                           PGD is untested (https://arxiv.org/pdf/1611.01236.pdf).\n        :param max_iter: The maximum number of iterations.\n        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False)\n        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0 starting\n                                at the original input.\n        :param batch_size: Size of the batch on which adversarial samples are generated.\n        :param summary_writer: Activate summary writer for TensorBoard.\n                               Default is `False` and deactivated summary writer.\n                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.\n                               If of type `str` save in path.\n                               If of type `SummaryWriter` apply provided custom summary writer.\n                               Use hierarchical folder structure to compare between runs easily. e.g. pass in\n                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.\n        :param verbose: Show progress bars.\n        '
        if summary_writer and num_random_init > 1:
            raise ValueError('TensorBoard is not yet supported for more than 1 random restart (num_random_init>1).')
        super().__init__(estimator=estimator, norm=norm, eps=eps, eps_step=eps_step, decay=decay, max_iter=max_iter, targeted=targeted, num_random_init=num_random_init, batch_size=batch_size, random_eps=random_eps, summary_writer=summary_writer, verbose=verbose)
        self._project = True

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Generate adversarial samples and return them in an array.\n\n        :param x: An array with the original inputs.\n        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape\n                  (nb_samples,). Only provide this parameter if you\'d like to use true labels when crafting adversarial\n                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect\n                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.\n\n        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.\n                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any\n                     features for which the mask is zero will not be adversarially perturbed.\n        :type mask: `np.ndarray`\n        :return: An array holding the adversarial examples.\n        '
        mask = self._get_mask(x, **kwargs)
        self._check_compatibility_input_and_eps(x=x)
        self._random_eps()
        if isinstance(self.estimator, ClassifierMixin):
            targets = self._set_targets(x, y)
            adv_x = x.astype(ART_NUMPY_DTYPE)
            for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):
                self._batch_id = batch_id
                for rand_init_num in trange(max(1, self.num_random_init), desc='PGD - Random Initializations', disable=not self.verbose):
                    (batch_index_1, batch_index_2) = (batch_id * self.batch_size, (batch_id + 1) * self.batch_size)
                    batch_index_2 = min(batch_index_2, x.shape[0])
                    batch = x[batch_index_1:batch_index_2]
                    batch_labels = targets[batch_index_1:batch_index_2]
                    mask_batch = mask
                    if mask is not None:
                        if len(mask.shape) == len(x.shape):
                            mask_batch = mask[batch_index_1:batch_index_2]
                    momentum = np.zeros(batch.shape)
                    for i_max_iter in trange(self.max_iter, desc='PGD - Iterations', leave=False, disable=not self.verbose):
                        self._i_max_iter = i_max_iter
                        batch = self._compute(batch, x[batch_index_1:batch_index_2], batch_labels, mask_batch, self.eps, self.eps_step, self._project, self.num_random_init > 0 and i_max_iter == 0, self._batch_id, decay=self.decay, momentum=momentum)
                    if rand_init_num == 0:
                        adv_x[batch_index_1:batch_index_2] = np.copy(batch)
                    else:
                        attack_success = compute_success_array(self.estimator, x[batch_index_1:batch_index_2], targets[batch_index_1:batch_index_2], batch, self.targeted, batch_size=self.batch_size)
                        adv_x[batch_index_1:batch_index_2][attack_success] = batch[attack_success]
            logger.info('Success rate of attack: %.2f%%', 100 * compute_success(self.estimator, x, targets, adv_x, self.targeted, batch_size=self.batch_size))
        else:
            if self.num_random_init > 0:
                raise ValueError('Random initialisation is only supported for classification.')
            targets = self._set_targets(x, y, classifier_mixin=False)
            if x.dtype == object:
                adv_x = x.copy()
            else:
                adv_x = x.astype(ART_NUMPY_DTYPE)
            momentum = np.zeros(adv_x.shape)
            for i_max_iter in trange(self.max_iter, desc='PGD - Iterations', disable=not self.verbose):
                self._i_max_iter = i_max_iter
                adv_x = self._compute(adv_x, x, targets, mask, self.eps, self.eps_step, self._project, self.num_random_init > 0 and i_max_iter == 0, decay=self.decay, momentum=momentum)
        if self.summary_writer is not None:
            self.summary_writer.reset()
        return adv_x