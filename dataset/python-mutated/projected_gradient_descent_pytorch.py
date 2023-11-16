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
from tqdm.auto import tqdm
from art.config import ART_NUMPY_DTYPE
from art.summary_writer import SummaryWriter
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification.classifier import ClassifierMixin
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent_numpy import ProjectedGradientDescentCommon
from art.utils import compute_success, random_sphere, compute_success_array
if TYPE_CHECKING:
    import torch
    from art.estimators.classification.pytorch import PyTorchClassifier
logger = logging.getLogger(__name__)

class ProjectedGradientDescentPyTorch(ProjectedGradientDescentCommon):
    """
    The Projected Gradient Descent attack is an iterative method in which, after each iteration, the perturbation is
    projected on an lp-ball of specified radius (in addition to clipping the values of the adversarial sample so that it
    lies in the permitted data range). This is the attack proposed by Madry et al. for adversarial training.

    | Paper link: https://arxiv.org/abs/1706.06083
    """
    _estimator_requirements = (BaseEstimator, LossGradientsMixin, ClassifierMixin)

    def __init__(self, estimator: Union['PyTorchClassifier'], norm: Union[int, float, str]=np.inf, eps: Union[int, float, np.ndarray]=0.3, eps_step: Union[int, float, np.ndarray]=0.1, decay: Optional[float]=None, max_iter: int=100, targeted: bool=False, num_random_init: int=0, batch_size: int=32, random_eps: bool=False, summary_writer: Union[str, bool, SummaryWriter]=False, verbose: bool=True):
        if False:
            print('Hello World!')
        '\n        Create a :class:`.ProjectedGradientDescentPyTorch` instance.\n\n        :param estimator: An trained estimator.\n        :param norm: The norm of the adversarial perturbation. Possible values: "inf", np.inf, 1 or 2.\n        :param eps: Maximum perturbation that the attacker can introduce.\n        :param eps_step: Attack step size (input variation) at each iteration.\n        :param random_eps: When True, epsilon is drawn randomly from truncated normal distribution. The literature\n                           suggests this for FGSM based training to generalize across different epsilons. eps_step is\n                           modified to preserve the ratio of eps / eps_step. The effectiveness of this method with PGD\n                           is untested (https://arxiv.org/pdf/1611.01236.pdf).\n        :param max_iter: The maximum number of iterations.\n        :param targeted: Indicates whether the attack is targeted (True) or untargeted (False).\n        :param num_random_init: Number of random initialisations within the epsilon ball. For num_random_init=0 starting\n                                at the original input.\n        :param batch_size: Size of the batch on which adversarial samples are generated.\n        :param summary_writer: Activate summary writer for TensorBoard.\n                               Default is `False` and deactivated summary writer.\n                               If `True` save runs/CURRENT_DATETIME_HOSTNAME in current directory.\n                               If of type `str` save in path.\n                               If of type `SummaryWriter` apply provided custom summary writer.\n                               Use hierarchical folder structure to compare between runs easily. e.g. pass in\n                               ‘runs/exp1’, ‘runs/exp2’, etc. for each new experiment to compare across them.\n        :param verbose: Show progress bars.\n        '
        if not estimator.all_framework_preprocessing:
            raise NotImplementedError('The framework-specific implementation only supports framework-specific preprocessing.')
        if summary_writer and num_random_init > 1:
            raise ValueError('TensorBoard is not yet supported for more than 1 random restart (num_random_init>1).')
        super().__init__(estimator=estimator, norm=norm, eps=eps, eps_step=eps_step, decay=decay, max_iter=max_iter, targeted=targeted, num_random_init=num_random_init, batch_size=batch_size, random_eps=random_eps, verbose=verbose, summary_writer=summary_writer)
        self._batch_id = 0
        self._i_max_iter = 0

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Generate adversarial samples and return them in an array.\n\n        :param x: An array with the original inputs.\n        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape\n                  (nb_samples,). Only provide this parameter if you\'d like to use true labels when crafting adversarial\n                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect\n                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.\n        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.\n                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any\n                     features for which the mask is zero will not be adversarially perturbed.\n        :type mask: `np.ndarray`\n        :return: An array holding the adversarial examples.\n        '
        import torch
        mask = self._get_mask(x, **kwargs)
        self._check_compatibility_input_and_eps(x=x)
        self._random_eps()
        targets = self._set_targets(x, y)
        if mask is not None:
            if len(mask.shape) == len(x.shape):
                dataset = torch.utils.data.TensorDataset(torch.from_numpy(x.astype(ART_NUMPY_DTYPE)), torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)), torch.from_numpy(mask.astype(ART_NUMPY_DTYPE)))
            else:
                dataset = torch.utils.data.TensorDataset(torch.from_numpy(x.astype(ART_NUMPY_DTYPE)), torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)), torch.from_numpy(np.array([mask.astype(ART_NUMPY_DTYPE)] * x.shape[0])))
        else:
            dataset = torch.utils.data.TensorDataset(torch.from_numpy(x.astype(ART_NUMPY_DTYPE)), torch.from_numpy(targets.astype(ART_NUMPY_DTYPE)))
        data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)
        adv_x = x.astype(ART_NUMPY_DTYPE)
        for (batch_id, batch_all) in enumerate(tqdm(data_loader, desc='PGD - Batches', leave=False, disable=not self.verbose)):
            self._batch_id = batch_id
            if mask is not None:
                (batch, batch_labels, mask_batch) = (batch_all[0], batch_all[1], batch_all[2])
            else:
                (batch, batch_labels, mask_batch) = (batch_all[0], batch_all[1], None)
            (batch_index_1, batch_index_2) = (batch_id * self.batch_size, (batch_id + 1) * self.batch_size)
            if isinstance(self.eps, np.ndarray) and isinstance(self.eps_step, np.ndarray):
                if len(self.eps.shape) == len(x.shape) and self.eps.shape[0] == x.shape[0]:
                    batch_eps = self.eps[batch_index_1:batch_index_2]
                    batch_eps_step = self.eps_step[batch_index_1:batch_index_2]
                else:
                    batch_eps = self.eps
                    batch_eps_step = self.eps_step
            else:
                batch_eps = self.eps
                batch_eps_step = self.eps_step
            for rand_init_num in range(max(1, self.num_random_init)):
                if rand_init_num == 0:
                    adv_x[batch_index_1:batch_index_2] = self._generate_batch(x=batch, targets=batch_labels, mask=mask_batch, eps=batch_eps, eps_step=batch_eps_step)
                else:
                    adversarial_batch = self._generate_batch(x=batch, targets=batch_labels, mask=mask_batch, eps=batch_eps, eps_step=batch_eps_step)
                    attack_success = compute_success_array(self.estimator, batch, batch_labels, adversarial_batch, self.targeted, batch_size=self.batch_size)
                    adv_x[batch_index_1:batch_index_2][attack_success] = adversarial_batch[attack_success]
        logger.info('Success rate of attack: %.2f%%', 100 * compute_success(self.estimator, x, targets, adv_x, self.targeted, batch_size=self.batch_size))
        if self.summary_writer is not None:
            self.summary_writer.reset()
        return adv_x

    def _generate_batch(self, x: 'torch.Tensor', targets: 'torch.Tensor', mask: 'torch.Tensor', eps: Union[int, float, np.ndarray], eps_step: Union[int, float, np.ndarray]) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Generate a batch of adversarial samples and return them in an array.\n\n        :param x: An array with the original inputs.\n        :param targets: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)`.\n        :param mask: An array with a mask to be applied to the adversarial perturbations. Shape needs to be\n                     broadcastable to the shape of x. Any features for which the mask is zero will not be adversarially\n                     perturbed.\n        :param eps: Maximum perturbation that the attacker can introduce.\n        :param eps_step: Attack step size (input variation) at each iteration.\n        :return: Adversarial examples.\n        '
        import torch
        inputs = x.to(self.estimator.device)
        targets = targets.to(self.estimator.device)
        adv_x = torch.clone(inputs)
        momentum = torch.zeros(inputs.shape).to(self.estimator.device)
        if mask is not None:
            mask = mask.to(self.estimator.device)
        for i_max_iter in range(self.max_iter):
            self._i_max_iter = i_max_iter
            adv_x = self._compute_pytorch(adv_x, inputs, targets, mask, eps, eps_step, self.num_random_init > 0 and i_max_iter == 0, momentum)
        return adv_x.cpu().detach().numpy()

    def _compute_perturbation_pytorch(self, x: 'torch.Tensor', y: 'torch.Tensor', mask: Optional['torch.Tensor'], momentum: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            i = 10
            return i + 15
        '\n        Compute perturbations.\n\n        :param x: Current adversarial examples.\n        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape\n                  (nb_samples,). Only provide this parameter if you\'d like to use true labels when crafting adversarial\n                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect\n                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.\n        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.\n                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any\n                     features for which the mask is zero will not be adversarially perturbed.\n        :return: Perturbations.\n        '
        import torch
        tol = 1e-07
        grad = self.estimator.loss_gradient(x=x, y=y) * (1 - 2 * int(self.targeted))
        if self.summary_writer is not None:
            self.summary_writer.update(batch_id=self._batch_id, global_step=self._i_max_iter, grad=grad.cpu().detach().numpy(), patch=None, estimator=self.estimator, x=x.cpu().detach().numpy(), y=y.cpu().detach().numpy(), targeted=self.targeted)
        if torch.any(grad.isnan()):
            logger.warning('Elements of the loss gradient are NaN and have been replaced with 0.0.')
            grad[grad.isnan()] = 0.0
        if mask is not None:
            grad = torch.where(mask == 0.0, torch.tensor(0.0).to(self.estimator.device), grad)
        if self.decay is not None:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sum(grad.abs(), dim=ind, keepdims=True) + tol)
            grad = self.decay * momentum + grad
            momentum += grad
        if self.norm in ['inf', np.inf]:
            grad = grad.sign()
        elif self.norm == 1:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sum(grad.abs(), dim=ind, keepdims=True) + tol)
        elif self.norm == 2:
            ind = tuple(range(1, len(x.shape)))
            grad = grad / (torch.sqrt(torch.sum(grad * grad, axis=ind, keepdims=True)) + tol)
        assert x.shape == grad.shape
        return grad

    def _apply_perturbation_pytorch(self, x: 'torch.Tensor', perturbation: 'torch.Tensor', eps_step: Union[int, float, np.ndarray]) -> 'torch.Tensor':
        if False:
            while True:
                i = 10
        '\n        Apply perturbation on examples.\n\n        :param x: Current adversarial examples.\n        :param perturbation: Current perturbations.\n        :param eps_step: Attack step size (input variation) at each iteration.\n        :return: Adversarial examples.\n        '
        import torch
        eps_step = np.array(eps_step, dtype=ART_NUMPY_DTYPE)
        perturbation_step = torch.tensor(eps_step).to(self.estimator.device) * perturbation
        perturbation_step[torch.isnan(perturbation_step)] = 0
        x = x + perturbation_step
        if self.estimator.clip_values is not None:
            (clip_min, clip_max) = self.estimator.clip_values
            x = torch.max(torch.min(x, torch.tensor(clip_max).to(self.estimator.device)), torch.tensor(clip_min).to(self.estimator.device))
        return x

    def _compute_pytorch(self, x: 'torch.Tensor', x_init: 'torch.Tensor', y: 'torch.Tensor', mask: 'torch.Tensor', eps: Union[int, float, np.ndarray], eps_step: Union[int, float, np.ndarray], random_init: bool, momentum: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            while True:
                i = 10
        '\n        Compute adversarial examples for one iteration.\n\n        :param x: Current adversarial examples.\n        :param x_init: An array with the original inputs.\n        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape\n                  (nb_samples,). Only provide this parameter if you\'d like to use true labels when crafting adversarial\n                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect\n                  (explained in this paper: https://arxiv.org/abs/1611.01236).\n        :param mask: An array with a mask broadcastable to input `x` defining where to apply adversarial perturbations.\n                     Shape needs to be broadcastable to the shape of x and can also be of the same shape as `x`. Any\n                     features for which the mask is zero will not be adversarially perturbed.\n        :param eps: Maximum perturbation that the attacker can introduce.\n        :param eps_step: Attack step size (input variation) at each iteration.\n        :param random_init: Random initialisation within the epsilon ball. For random_init=False starting at the\n                            original input.\n        :return: Adversarial examples.\n        '
        import torch
        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:]).item()
            random_perturbation_array = random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(ART_NUMPY_DTYPE)
            random_perturbation = torch.from_numpy(random_perturbation_array).to(self.estimator.device)
            if mask is not None:
                random_perturbation = random_perturbation * mask
            x_adv = x + random_perturbation
            if self.estimator.clip_values is not None:
                (clip_min, clip_max) = self.estimator.clip_values
                x_adv = torch.max(torch.min(x_adv, torch.tensor(clip_max).to(self.estimator.device)), torch.tensor(clip_min).to(self.estimator.device))
        else:
            x_adv = x
        perturbation = self._compute_perturbation_pytorch(x_adv, y, mask, momentum)
        x_adv = self._apply_perturbation_pytorch(x_adv, perturbation, eps_step)
        perturbation = self._projection(x_adv - x_init, eps, self.norm)
        x_adv = perturbation + x_init
        return x_adv

    def _projection(self, values: 'torch.Tensor', eps: Union[int, float, np.ndarray], norm_p: Union[int, float, str]) -> 'torch.Tensor':
        if False:
            print('Hello World!')
        '\n        Project `values` on the L_p norm ball of size `eps`.\n\n        :param values: Values to clip.\n        :param eps: Maximum norm allowed.\n        :param norm_p: L_p norm to use for clipping supporting 1, 2, `np.Inf` and "inf".\n        :return: Values of `values` after projection.\n        '
        import torch
        tol = 1e-07
        values_tmp = values.reshape(values.shape[0], -1)
        if norm_p == 2:
            if isinstance(eps, np.ndarray):
                raise NotImplementedError('The parameter `eps` of type `np.ndarray` is not supported to use with norm 2.')
            values_tmp = values_tmp * torch.min(torch.tensor([1.0], dtype=torch.float32).to(self.estimator.device), eps / (torch.norm(values_tmp, p=2, dim=1) + tol)).unsqueeze_(-1)
        elif norm_p == 1:
            if isinstance(eps, np.ndarray):
                raise NotImplementedError('The parameter `eps` of type `np.ndarray` is not supported to use with norm 1.')
            values_tmp = values_tmp * torch.min(torch.tensor([1.0], dtype=torch.float32).to(self.estimator.device), eps / (torch.norm(values_tmp, p=1, dim=1) + tol)).unsqueeze_(-1)
        elif norm_p in [np.inf, 'inf']:
            if isinstance(eps, np.ndarray):
                eps = eps * np.ones_like(values.cpu())
                eps = eps.reshape([eps.shape[0], -1])
            values_tmp = values_tmp.sign() * torch.min(values_tmp.abs(), torch.tensor([eps], dtype=torch.float32).to(self.estimator.device))
        else:
            raise NotImplementedError('Values of `norm_p` different from 1, 2 and `np.inf` are currently not supported.')
        values = values_tmp.reshape(values.shape)
        return values