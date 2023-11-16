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
from art.estimators.classification.pytorch import PyTorchClassifier
from art.estimators.classification.tensorflow import TensorFlowV2Classifier
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.attacks.attack import EvasionAttack
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import ProjectedGradientDescentNumpy
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_pytorch import ProjectedGradientDescentPyTorch
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_tensorflow_v2 import ProjectedGradientDescentTensorFlowV2
from art.summary_writer import SummaryWriter
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_LOSS_GRADIENTS_TYPE, OBJECT_DETECTOR_TYPE
logger = logging.getLogger(__name__)

class ProjectedGradientDescent(EvasionAttack):
    """
    The Projected Gradient Descent attack is an iterative method in which, after each iteration, the perturbation is
    projected on an lp-ball of specified radius (in addition to clipping the values of the adversarial sample so that it
    lies in the permitted data range). This is the attack proposed by Madry et al. for adversarial training.

    | Paper link: https://arxiv.org/abs/1706.06083
    """
    attack_params = EvasionAttack.attack_params + ['norm', 'eps', 'eps_step', 'decay', 'targeted', 'num_random_init', 'batch_size', 'max_iter', 'random_eps', 'summary_writer', 'verbose']
    _estimator_requirements = (BaseEstimator, LossGradientsMixin)

    def __init__(self, estimator: Union['CLASSIFIER_LOSS_GRADIENTS_TYPE', 'OBJECT_DETECTOR_TYPE'], norm: Union[int, float, str]=np.inf, eps: Union[int, float, np.ndarray]=0.3, eps_step: Union[int, float, np.ndarray]=0.1, decay: Optional[float]=None, max_iter: int=100, targeted: bool=False, num_random_init: int=0, batch_size: int=32, random_eps: bool=False, summary_writer: Union[str, bool, SummaryWriter]=False, verbose: bool=True):
        if False:
            print('Hello World!')
        '\n        Create a :class:`.ProjectedGradientDescent` instance.\n\n        :param estimator: An trained estimator.\n        :param norm: The norm of the adversarial perturbation supporting "inf", np.inf, 1 or 2.\n        :param eps: Maximum perturbation that the attacker can introduce.\n        :param eps_step: Attack step size (input variation) at each iteration.\n        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature\n                           suggests this for FGSM based training to generalize across different epsilons. eps_step\n                           is modified to preserve the ratio of eps / eps_step. The effectiveness of this\n                           method with PGD is untested (https://arxiv.org/pdf/1611.01236.pdf).\n        :param decay: Decay factor for accumulating the velocity vector when using momentum.\n        :param max_iter: The maximum number of iterations.\n        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).\n        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0 starting\n                                at the original input.\n        :param batch_size: Size of the batch on which adversarial samples are generated.\n        :param summary_writer: Activate summary writer for TensorBoard.\n                               Default is `False` and deactivated summary writer.\n                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.\n                               If of type `str` save in path.\n                               If of type `SummaryWriter` apply provided custom summary writer.\n                               Use hierarchical folder structure to compare between runs easily. e.g. pass in\n                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.\n        :param verbose: Show progress bars.\n        '
        super().__init__(estimator=estimator, summary_writer=False)
        self.norm = norm
        self.eps = eps
        self.eps_step = eps_step
        self.max_iter = max_iter
        self.targeted = targeted
        self.num_random_init = num_random_init
        self.batch_size = batch_size
        self.random_eps = random_eps
        self.verbose = verbose
        ProjectedGradientDescent._check_params(self)
        self._attack: Union[ProjectedGradientDescentPyTorch, ProjectedGradientDescentTensorFlowV2, ProjectedGradientDescentNumpy]
        if isinstance(self.estimator, PyTorchClassifier) and self.estimator.all_framework_preprocessing:
            self._attack = ProjectedGradientDescentPyTorch(estimator=estimator, norm=norm, eps=eps, eps_step=eps_step, decay=decay, max_iter=max_iter, targeted=targeted, num_random_init=num_random_init, batch_size=batch_size, random_eps=random_eps, summary_writer=summary_writer, verbose=verbose)
        elif isinstance(self.estimator, TensorFlowV2Classifier) and self.estimator.all_framework_preprocessing:
            self._attack = ProjectedGradientDescentTensorFlowV2(estimator=estimator, norm=norm, eps=eps, eps_step=eps_step, decay=decay, max_iter=max_iter, targeted=targeted, num_random_init=num_random_init, batch_size=batch_size, random_eps=random_eps, summary_writer=summary_writer, verbose=verbose)
        else:
            self._attack = ProjectedGradientDescentNumpy(estimator=estimator, norm=norm, eps=eps, eps_step=eps_step, decay=decay, max_iter=max_iter, targeted=targeted, num_random_init=num_random_init, batch_size=batch_size, random_eps=random_eps, summary_writer=summary_writer, verbose=verbose)

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            i = 10
            return i + 15
        '\n        Generate adversarial samples and return them in an array.\n\n        :param x: An array with the original inputs.\n        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape\n                  (nb_samples,). Only provide this parameter if you\'d like to use true labels when crafting adversarial\n                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect\n                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.\n        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.\n                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any\n                     features for which the mask is zero will not be adversarially perturbed.\n        :type mask: `np.ndarray`\n        :return: An array holding the adversarial examples.\n        '
        logger.info('Creating adversarial samples.')
        return self._attack.generate(x=x, y=y, **kwargs)

    @property
    def summary_writer(self):
        if False:
            for i in range(10):
                print('nop')
        'The summary writer.'
        return self._attack.summary_writer

    def set_params(self, **kwargs) -> None:
        if False:
            i = 10
            return i + 15
        super().set_params(**kwargs)
        self._attack.set_params(**kwargs)

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
                raise ValueError('The perturbation size `eps` has to be nonnegative.')
        elif (self.eps < 0).any():
            raise ValueError('The perturbation size `eps` has to be nonnegative.')
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
            raise ValueError('The number of iterations `max_iter` has to be a nonnegative integer.')
        if not isinstance(self.verbose, bool):
            raise ValueError('The verbose has to be a Boolean.')