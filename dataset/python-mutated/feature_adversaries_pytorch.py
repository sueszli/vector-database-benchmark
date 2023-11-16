"""
This module implements the Feature Adversaries attack in PyTorch.

| Paper link: https://arxiv.org/abs/1511.05122
"""
import logging
from typing import TYPE_CHECKING, Optional, Tuple, Union
import numpy as np
from tqdm.auto import trange
from art.attacks.attack import EvasionAttack
from art.estimators.estimator import BaseEstimator, NeuralNetworkMixin
if TYPE_CHECKING:
    import torch
    from torch.optim import Optimizer
    from art.utils import PYTORCH_ESTIMATOR_TYPE
logger = logging.getLogger(__name__)

class FeatureAdversariesPyTorch(EvasionAttack):
    """
    This class represent a Feature Adversaries evasion attack in PyTorch.

    | Paper link: https://arxiv.org/abs/1511.05122
    """
    attack_params = EvasionAttack.attack_params + ['delta', 'optimizer', 'optimizer_kwargs', 'lambda_', 'layer', 'max_iter', 'batch_size', 'step_size', 'random_start', 'verbose']
    _estimator_requirements = (BaseEstimator, NeuralNetworkMixin)

    def __init__(self, estimator: 'PYTORCH_ESTIMATOR_TYPE', delta: float, optimizer: Optional['Optimizer']=None, optimizer_kwargs: Optional[dict]=None, lambda_: float=0.0, layer: Union[int, str, Tuple[int, ...], Tuple[str, ...]]=-1, max_iter: int=100, batch_size: int=32, step_size: Optional[Union[int, float]]=None, random_start: bool=False, verbose: bool=True):
        if False:
            i = 10
            return i + 15
        '\n        Create a :class:`.FeatureAdversariesPyTorch` instance.\n\n        :param estimator: A trained estimator.\n        :param delta: The maximum deviation between source and guide images.\n        :param optimizer: Optimizer applied to problem constrained only by clip values if defined, if None the\n                          Projected Gradient Descent (PGD) optimizer is used.\n        :param optimizer_kwargs: Additional optimizer arguments.\n        :param lambda_: Regularization parameter of the L-inf soft constraint.\n        :param layer: Index or tuple of indices of the representation layer(s).\n        :param max_iter: The maximum number of iterations.\n        :param batch_size: Batch size.\n        :param step_size: Step size for PGD optimizer.\n        :param random_start: Randomly initialize perturbations, when using Projected Gradient Descent variant.\n        :param verbose: Show progress bars.\n        '
        super().__init__(estimator=estimator)
        self.delta = delta
        self.optimizer = optimizer
        self._optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs
        self.lambda_ = lambda_
        self.layer = layer if isinstance(layer, tuple) else (layer,)
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.step_size = step_size
        self.random_start = random_start
        self.verbose = verbose
        self._check_params()

    def _generate_batch(self, x: 'torch.Tensor', y: 'torch.Tensor') -> 'torch.Tensor':
        if False:
            i = 10
            return i + 15
        '\n        Generate adversarial batch.\n\n        :param x: Source samples.\n        :param y: Guide samples.\n        :return: Batch of adversarial examples.\n        '
        import torch

        def loss_fn(source_orig, source_adv, guide):
            if False:
                print('Hello World!')
            representation_loss = torch.zeros(size=(source_orig.shape[0],)).to(self.estimator.device)
            for layer_i in self.layer:
                adv_representation = self.estimator.get_activations(source_adv, layer_i, self.batch_size, True)
                guide_representation = self.estimator.get_activations(guide, layer_i, self.batch_size, True)
                dim = tuple(range(1, len(source_adv.shape)))
                soft_constraint = torch.amax(torch.abs(source_adv - source_orig), dim=dim)
                dim = tuple(range(1, len(adv_representation.shape)))
                representation_loss += torch.sum(torch.square(adv_representation - guide_representation), dim=dim)
            loss = torch.mean(representation_loss + self.lambda_ * soft_constraint)
            return loss
        self.estimator.model.eval()
        adv = x.clone().detach().to(self.estimator.device)
        if self.random_start:
            adv = adv + torch.empty_like(adv).uniform_(-self.delta, self.delta)
            if self.estimator.clip_values is not None:
                adv = torch.clamp(adv, *self.estimator.clip_values)
        if self.optimizer is None:
            adv.requires_grad = True
            for _ in trange(self.max_iter, desc='Feature Adversaries PyTorch', disable=not self.verbose):
                loss = loss_fn(x, adv, y)
                loss.backward()
                if adv.grad is not None:
                    adv.data = adv - adv.grad.detach().sign() * self.step_size
                else:
                    raise ValueError('Gradient tensor in PyTorch model is `None`.')
                perturbation = torch.clamp(adv.detach() - x.detach(), -self.delta, self.delta)
                adv.data = x.detach() + perturbation
                if self.estimator.clip_values is not None:
                    adv.data = torch.clamp(adv.detach(), *self.estimator.clip_values)
                adv.grad.zero_()
        else:
            opt = self.optimizer(params=[adv], **self._optimizer_kwargs)
            for _ in trange(self.max_iter, desc='Feature Adversaries PyTorch', disable=not self.verbose):
                adv.requires_grad = True
                adv.data = x.detach() + torch.clamp(adv.detach() - x.detach(), -self.delta, self.delta)

                def closure():
                    if False:
                        print('Hello World!')
                    if torch.is_grad_enabled():
                        opt.zero_grad()
                    loss = loss_fn(x, adv, y)
                    if loss.requires_grad:
                        loss.backward()
                    return loss
                opt.step(closure)
                if self.estimator.clip_values is not None:
                    adv.data = torch.clamp(adv.detach(), *self.estimator.clip_values)
        return adv.detach().cpu()

    def generate(self, x: np.ndarray, y: Optional[np.ndarray]=None, **kwargs) -> np.ndarray:
        if False:
            while True:
                i = 10
        '\n        Generate adversarial samples and return them in an array.\n\n        :param x: Source samples.\n        :param y: Guide samples.\n        :return: Adversarial examples.\n        '
        import torch
        if y is None:
            raise ValueError('The value of guide `y` cannot be None. Please provide a `np.ndarray` of guide inputs.')
        if x.shape != y.shape:
            raise ValueError('The shape of source `x` and guide `y` must be of same shape.')
        if x.shape[1:] != self.estimator.input_shape:
            raise ValueError('Source and guide inputs must match `input_shape` of estimator.')
        nb_samples = x.shape[0]
        x_adversarial = [None] * nb_samples
        nb_batches = int(np.ceil(nb_samples / float(self.batch_size)))
        for m in range(nb_batches):
            (begin, end) = (m * self.batch_size, min((m + 1) * self.batch_size, nb_samples))
            source_batch = torch.tensor(x[begin:end]).to(self.estimator.device)
            guide_batch = torch.tensor(y[begin:end]).to(self.estimator.device)
            x_adversarial[begin:end] = self._generate_batch(source_batch, guide_batch).numpy()
        return np.array(x_adversarial, dtype=x.dtype)

    def _check_params(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Apply attack-specific checks.\n        '
        if not isinstance(self.delta, float):
            raise ValueError('The value of delta must be of type float.')
        if self.delta <= 0:
            raise ValueError('The maximum deviation value delta has to be positive.')
        if not isinstance(self.lambda_, float):
            raise ValueError('The value of lambda_ must be of type float.')
        if self.lambda_ < 0.0:
            raise ValueError('The regularization parameter `lambda_` has to be non-negative.')
        if not isinstance(self.layer[0], (int, str)):
            raise ValueError('The value of the representation layer must be integer or string.')
        if not isinstance(self.max_iter, int):
            raise ValueError('The value of max_iter must be of type int.')
        if self.max_iter <= 0:
            raise ValueError('The maximum number of iterations has to be a positive.')
        if self.batch_size <= 0:
            raise ValueError('The batch size has to be positive.')
        if self.optimizer is None and self.step_size is None:
            raise ValueError('The step size cannot be None if optimizer is None.')
        if self.step_size is not None and (not isinstance(self.step_size, (int, float))):
            raise ValueError('The value of step_size must be of type int or float.')
        if self.step_size is not None and self.step_size <= 0:
            raise ValueError('The step size has to be a positive.')