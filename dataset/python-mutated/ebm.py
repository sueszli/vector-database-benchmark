"""
Vanilla DFO and EBM are adapted from https://github.com/kevinzakka/ibc.
MCMC is adapted from https://github.com/google-research/ibc.
"""
from typing import Callable, Tuple
from functools import wraps
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from ding.utils import MODEL_REGISTRY, STOCHASTIC_OPTIMIZER_REGISTRY
from ding.torch_utils import unsqueeze_repeat
from ding.model.wrapper import IModelWrapper
from ding.model.common import RegressionHead

def create_stochastic_optimizer(device: str, stochastic_optimizer_config: dict):
    if False:
        return 10
    '\n    Overview:\n        Create stochastic optimizer.\n    Arguments:\n        - device (:obj:`str`): Device.\n        - stochastic_optimizer_config (:obj:`dict`): Stochastic optimizer config.\n    '
    return STOCHASTIC_OPTIMIZER_REGISTRY.build(stochastic_optimizer_config.pop('type'), device=device, **stochastic_optimizer_config)

def no_ebm_grad():
    if False:
        i = 10
        return i + 15
    'Wrapper that disables energy based model gradients'

    def ebm_disable_grad_wrapper(func: Callable):
        if False:
            print('Hello World!')

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                return 10
            ebm = args[-1]
            assert isinstance(ebm, (IModelWrapper, nn.Module)), 'Make sure ebm is the last positional arguments.'
            ebm.requires_grad_(False)
            result = func(*args, **kwargs)
            ebm.requires_grad_(True)
            return result
        return wrapper
    return ebm_disable_grad_wrapper

class StochasticOptimizer(ABC):
    """
    Overview:
        Base class for stochastic optimizers.
    Interface:
        ``__init__``, ``_sample``, ``_get_best_action_sample``, ``set_action_bounds``, ``sample``, ``infer``
    """

    def _sample(self, obs: torch.Tensor, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Drawing action samples from the uniform random distribution                 and tiling observations to the same shape as action samples.\n        Arguments:\n            - obs (:obj:`torch.Tensor`): Observation.\n            - num_samples (:obj:`int`): The number of negative samples.\n        Returns:\n            - tiled_obs (:obj:`torch.Tensor`): Observations tiled.\n            - action (:obj:`torch.Tensor`): Action sampled.\n        Shapes:\n            - obs (:obj:`torch.Tensor`): :math:`(B, O)`.\n            - num_samples (:obj:`int`): :math:`N`.\n            - tiled_obs (:obj:`torch.Tensor`): :math:`(B, N, O)`.\n            - action (:obj:`torch.Tensor`): :math:`(B, N, A)`.\n        Examples:\n            >>> obs = torch.randn(2, 4)\n            >>> opt = StochasticOptimizer()\n            >>> opt.set_action_bounds(np.stack([np.zeros(5), np.ones(5)], axis=0))\n            >>> tiled_obs, action = opt._sample(obs, 8)\n        '
        size = (obs.shape[0], num_samples, self.action_bounds.shape[1])
        (low, high) = (self.action_bounds[0, :], self.action_bounds[1, :])
        action_samples = low + (high - low) * torch.rand(size).to(self.device)
        tiled_obs = unsqueeze_repeat(obs, num_samples, 1)
        return (tiled_obs, action_samples)

    @staticmethod
    @torch.no_grad()
    def _get_best_action_sample(obs: torch.Tensor, action_samples: torch.Tensor, ebm: nn.Module):
        if False:
            return 10
        '\n        Overview:\n            Return one action for each batch with highest probability (lowest energy).\n        Arguments:\n            - obs (:obj:`torch.Tensor`): Observation.\n            - action_samples (:obj:`torch.Tensor`): Action from uniform distributions.\n        Returns:\n            - best_action_samples (:obj:`torch.Tensor`): Best action.\n        Shapes:\n            - obs (:obj:`torch.Tensor`): :math:`(B, O)`.\n            - action_samples (:obj:`torch.Tensor`): :math:`(B, N, A)`.\n            - best_action_samples (:obj:`torch.Tensor`): :math:`(B, A)`.\n        Examples:\n            >>> obs = torch.randn(2, 4)\n            >>> action_samples = torch.randn(2, 8, 5)\n            >>> ebm = EBM(4, 5)\n            >>> opt = StochasticOptimizer()\n            >>> opt.set_action_bounds(np.stack([np.zeros(5), np.ones(5)], axis=0))\n            >>> best_action_samples = opt._get_best_action_sample(obs, action_samples, ebm)\n        '
        energies = ebm.forward(obs, action_samples)
        probs = F.softmax(-1.0 * energies, dim=-1)
        best_idxs = probs.argmax(dim=-1)
        return action_samples[torch.arange(action_samples.size(0)), best_idxs]

    def set_action_bounds(self, action_bounds: np.ndarray):
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Set action bounds calculated from the dataset statistics.\n        Arguments:\n            - action_bounds (:obj:`np.ndarray`): Array of shape (2, A),                 where action_bounds[0] is lower bound and action_bounds[1] is upper bound.\n        Returns:\n            - action_bounds (:obj:`torch.Tensor`): Action bounds.\n        Shapes:\n            - action_bounds (:obj:`np.ndarray`): :math:`(2, A)`.\n            - action_bounds (:obj:`torch.Tensor`): :math:`(2, A)`.\n        Examples:\n            >>> opt = StochasticOptimizer()\n            >>> opt.set_action_bounds(np.stack([np.zeros(5), np.ones(5)], axis=0))\n        '
        self.action_bounds = torch.as_tensor(action_bounds, dtype=torch.float32).to(self.device)

    @abstractmethod
    def sample(self, obs: torch.Tensor, ebm: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Create tiled observations and sample counter-negatives for InfoNCE loss.\n        Arguments:\n            - obs (:obj:`torch.Tensor`): Observations.\n            - ebm (:obj:`torch.nn.Module`): Energy based model.\n        Returns:\n            - tiled_obs (:obj:`torch.Tensor`): Tiled observations.\n            - action (:obj:`torch.Tensor`): Actions.\n        Shapes:\n            - obs (:obj:`torch.Tensor`): :math:`(B, O)`.\n            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.\n            - tiled_obs (:obj:`torch.Tensor`): :math:`(B, N, O)`.\n            - action (:obj:`torch.Tensor`): :math:`(B, N, A)`.\n\n        .. note:: In the case of derivative-free optimization, this function will simply call _sample.\n        '
        raise NotImplementedError

    @abstractmethod
    def infer(self, obs: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        if False:
            return 10
        '\n        Overview:\n            Optimize for the best action conditioned on the current observation.\n        Arguments:\n            - obs (:obj:`torch.Tensor`): Observations.\n            - ebm (:obj:`torch.nn.Module`): Energy based model.\n        Returns:\n            - best_action_samples (:obj:`torch.Tensor`): Best actions.\n        Shapes:\n            - obs (:obj:`torch.Tensor`): :math:`(B, O)`.\n            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.\n            - best_action_samples (:obj:`torch.Tensor`): :math:`(B, A)`.\n        '
        raise NotImplementedError

@STOCHASTIC_OPTIMIZER_REGISTRY.register('dfo')
class DFO(StochasticOptimizer):
    """
    Overview:
        Derivative-Free Optimizer in paper Implicit Behavioral Cloning.
        https://arxiv.org/abs/2109.00137
    Interface:
        ``init``, ``sample``, ``infer``
    """

    def __init__(self, noise_scale: float=0.33, noise_shrink: float=0.5, iters: int=3, train_samples: int=8, inference_samples: int=16384, device: str='cpu'):
        if False:
            print('Hello World!')
        '\n        Overview:\n            Initialize the Derivative-Free Optimizer\n        Arguments:\n            - noise_scale (:obj:`float`): Initial noise scale.\n            - noise_shrink (:obj:`float`): Noise scale shrink rate.\n            - iters (:obj:`int`): Number of iterations.\n            - train_samples (:obj:`int`): Number of samples for training.\n            - inference_samples (:obj:`int`): Number of samples for inference.\n            - device (:obj:`str`): Device.\n        '
        self.action_bounds = None
        self.noise_scale = noise_scale
        self.noise_shrink = noise_shrink
        self.iters = iters
        self.train_samples = train_samples
        self.inference_samples = inference_samples
        self.device = device

    def sample(self, obs: torch.Tensor, ebm: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        if False:
            return 10
        '\n        Overview:\n            Drawing action samples from the uniform random distribution                 and tiling observations to the same shape as action samples.\n        Arguments:\n            - obs (:obj:`torch.Tensor`): Observations.\n            - ebm (:obj:`torch.nn.Module`): Energy based model.\n        Returns:\n            - tiled_obs (:obj:`torch.Tensor`): Tiled observation.\n            - action_samples (:obj:`torch.Tensor`): Action samples.\n        Shapes:\n            - obs (:obj:`torch.Tensor`): :math:`(B, O)`.\n            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.\n            - tiled_obs (:obj:`torch.Tensor`): :math:`(B, N, O)`.\n            - action_samples (:obj:`torch.Tensor`): :math:`(B, N, A)`.\n        Examples:\n            >>> obs = torch.randn(2, 4)\n            >>> ebm = EBM(4, 5)\n            >>> opt = DFO()\n            >>> opt.set_action_bounds(np.stack([np.zeros(5), np.ones(5)], axis=0))\n            >>> tiled_obs, action_samples = opt.sample(obs, ebm)\n        '
        return self._sample(obs, self.train_samples)

    @torch.no_grad()
    def infer(self, obs: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Optimize for the best action conditioned on the current observation.\n        Arguments:\n            - obs (:obj:`torch.Tensor`): Observations.\n            - ebm (:obj:`torch.nn.Module`): Energy based model.\n        Returns:\n            - best_action_samples (:obj:`torch.Tensor`): Actions.\n        Shapes:\n            - obs (:obj:`torch.Tensor`): :math:`(B, O)`.\n            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.\n            - best_action_samples (:obj:`torch.Tensor`): :math:`(B, A)`.\n        Examples:\n            >>> obs = torch.randn(2, 4)\n            >>> ebm = EBM(4, 5)\n            >>> opt = DFO()\n            >>> opt.set_action_bounds(np.stack([np.zeros(5), np.ones(5)], axis=0))\n            >>> best_action_samples = opt.infer(obs, ebm)\n        '
        noise_scale = self.noise_scale
        (obs, action_samples) = self._sample(obs, self.inference_samples)
        for i in range(self.iters):
            energies = ebm.forward(obs, action_samples)
            probs = F.softmax(-1.0 * energies, dim=-1)
            idxs = torch.multinomial(probs, self.inference_samples, replacement=True)
            action_samples = action_samples[torch.arange(action_samples.size(0)).unsqueeze(-1), idxs]
            action_samples = action_samples + torch.randn_like(action_samples) * noise_scale
            action_samples = action_samples.clamp(min=self.action_bounds[0, :], max=self.action_bounds[1, :])
            noise_scale *= self.noise_shrink
        return self._get_best_action_sample(obs, action_samples, ebm)

@STOCHASTIC_OPTIMIZER_REGISTRY.register('ardfo')
class AutoRegressiveDFO(DFO):
    """
    Overview:
        AutoRegressive Derivative-Free Optimizer in paper Implicit Behavioral Cloning.
        https://arxiv.org/abs/2109.00137
    Interface:
        ``__init__``, ``infer``
    """

    def __init__(self, noise_scale: float=0.33, noise_shrink: float=0.5, iters: int=3, train_samples: int=8, inference_samples: int=4096, device: str='cpu'):
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Initialize the AutoRegressive Derivative-Free Optimizer\n        Arguments:\n            - noise_scale (:obj:`float`): Initial noise scale.\n            - noise_shrink (:obj:`float`): Noise scale shrink rate.\n            - iters (:obj:`int`): Number of iterations.\n            - train_samples (:obj:`int`): Number of samples for training.\n            - inference_samples (:obj:`int`): Number of samples for inference.\n            - device (:obj:`str`): Device.\n        '
        super().__init__(noise_scale, noise_shrink, iters, train_samples, inference_samples, device)

    @torch.no_grad()
    def infer(self, obs: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Optimize for the best action conditioned on the current observation.\n        Arguments:\n            - obs (:obj:`torch.Tensor`): Observations.\n            - ebm (:obj:`torch.nn.Module`): Energy based model.\n        Returns:\n            - best_action_samples (:obj:`torch.Tensor`): Actions.\n        Shapes:\n            - obs (:obj:`torch.Tensor`): :math:`(B, O)`.\n            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.\n            - best_action_samples (:obj:`torch.Tensor`): :math:`(B, A)`.\n        Examples:\n            >>> obs = torch.randn(2, 4)\n            >>> ebm = EBM(4, 5)\n            >>> opt = AutoRegressiveDFO()\n            >>> opt.set_action_bounds(np.stack([np.zeros(5), np.ones(5)], axis=0))\n            >>> best_action_samples = opt.infer(obs, ebm)\n        '
        noise_scale = self.noise_scale
        (obs, action_samples) = self._sample(obs, self.inference_samples)
        for i in range(self.iters):
            for j in range(action_samples.shape[-1]):
                energies = ebm.forward(obs, action_samples)[..., j]
                probs = F.softmax(-1.0 * energies, dim=-1)
                idxs = torch.multinomial(probs, self.inference_samples, replacement=True)
                action_samples = action_samples[torch.arange(action_samples.size(0)).unsqueeze(-1), idxs]
                action_samples[..., j] = action_samples[..., j] + torch.randn_like(action_samples[..., j]) * noise_scale
                action_samples[..., j] = action_samples[..., j].clamp(min=self.action_bounds[0, j], max=self.action_bounds[1, j])
            noise_scale *= self.noise_shrink
        energies = ebm.forward(obs, action_samples)[..., -1]
        probs = F.softmax(-1.0 * energies, dim=-1)
        best_idxs = probs.argmax(dim=-1)
        return action_samples[torch.arange(action_samples.size(0)), best_idxs]

@STOCHASTIC_OPTIMIZER_REGISTRY.register('mcmc')
class MCMC(StochasticOptimizer):
    """
    Overview:
        MCMC method as stochastic optimizers in paper Implicit Behavioral Cloning.
        https://arxiv.org/abs/2109.00137
    Interface:
        ``__init__``, ``sample``, ``infer``, ``grad_penalty``
    """

    class BaseScheduler(ABC):
        """
        Overview:
            Base class for learning rate scheduler.
        Interface:
            ``get_rate``
        """

        @abstractmethod
        def get_rate(self, index):
            if False:
                print('Hello World!')
            '\n            Overview:\n                Abstract method for getting learning rate.\n            '
            raise NotImplementedError

    class ExponentialScheduler:
        """
        Overview:
            Exponential learning rate schedule for Langevin sampler.
        Interface:
            ``__init__``, ``get_rate``
        """

        def __init__(self, init, decay):
            if False:
                return 10
            '\n            Overview:\n                Initialize the ExponentialScheduler.\n            Arguments:\n                - init (:obj:`float`): Initial learning rate.\n                - decay (:obj:`float`): Decay rate.\n            '
            self._decay = decay
            self._latest_lr = init

        def get_rate(self, index):
            if False:
                i = 10
                return i + 15
            '\n            Overview:\n                Get learning rate. Assumes calling sequentially.\n            Arguments:\n                - index (:obj:`int`): Current iteration.\n            '
            del index
            lr = self._latest_lr
            self._latest_lr *= self._decay
            return lr

    class PolynomialScheduler:
        """
        Overview:
            Polynomial learning rate schedule for Langevin sampler.
        Interface:
            ``__init__``, ``get_rate``
        """

        def __init__(self, init, final, power, num_steps):
            if False:
                while True:
                    i = 10
            '\n            Overview:\n                Initialize the PolynomialScheduler.\n            Arguments:\n                - init (:obj:`float`): Initial learning rate.\n                - final (:obj:`float`): Final learning rate.\n                - power (:obj:`float`): Power of polynomial.\n                - num_steps (:obj:`int`): Number of steps.\n            '
            self._init = init
            self._final = final
            self._power = power
            self._num_steps = num_steps

        def get_rate(self, index):
            if False:
                print('Hello World!')
            '\n            Overview:\n                Get learning rate for index.\n            Arguments:\n                - index (:obj:`int`): Current iteration.\n            '
            if index == -1:
                return self._init
            return (self._init - self._final) * (1 - float(index) / float(self._num_steps - 1)) ** self._power + self._final

    def __init__(self, iters: int=100, use_langevin_negative_samples: bool=True, train_samples: int=8, inference_samples: int=512, stepsize_scheduler: dict=dict(init=0.5, final=1e-05, power=2.0), optimize_again: bool=True, again_stepsize_scheduler: dict=dict(init=1e-05, final=1e-05, power=2.0), device: str='cpu', noise_scale: float=0.5, grad_clip=None, delta_action_clip: float=0.5, add_grad_penalty: bool=True, grad_norm_type: str='inf', grad_margin: float=1.0, grad_loss_weight: float=1.0, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Initialize the MCMC.\n        Arguments:\n            - iters (:obj:`int`): Number of iterations.\n            - use_langevin_negative_samples (:obj:`bool`): Whether to use Langevin sampler.\n            - train_samples (:obj:`int`): Number of samples for training.\n            - inference_samples (:obj:`int`): Number of samples for inference.\n            - stepsize_scheduler (:obj:`dict`): Step size scheduler for Langevin sampler.\n            - optimize_again (:obj:`bool`): Whether to run a second optimization.\n            - again_stepsize_scheduler (:obj:`dict`): Step size scheduler for the second optimization.\n            - device (:obj:`str`): Device.\n            - noise_scale (:obj:`float`): Initial noise scale.\n            - grad_clip (:obj:`float`): Gradient clip.\n            - delta_action_clip (:obj:`float`): Action clip.\n            - add_grad_penalty (:obj:`bool`): Whether to add gradient penalty.\n            - grad_norm_type (:obj:`str`): Gradient norm type.\n            - grad_margin (:obj:`float`): Gradient margin.\n            - grad_loss_weight (:obj:`float`): Gradient loss weight.\n        '
        self.iters = iters
        self.use_langevin_negative_samples = use_langevin_negative_samples
        self.train_samples = train_samples
        self.inference_samples = inference_samples
        self.stepsize_scheduler = stepsize_scheduler
        self.optimize_again = optimize_again
        self.again_stepsize_scheduler = again_stepsize_scheduler
        self.device = device
        self.noise_scale = noise_scale
        self.grad_clip = grad_clip
        self.delta_action_clip = delta_action_clip
        self.add_grad_penalty = add_grad_penalty
        self.grad_norm_type = grad_norm_type
        self.grad_margin = grad_margin
        self.grad_loss_weight = grad_loss_weight

    @staticmethod
    def _gradient_wrt_act(obs: torch.Tensor, action: torch.Tensor, ebm: nn.Module, create_graph: bool=False) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Calculate gradient w.r.t action.\n        Arguments:\n            - obs (:obj:`torch.Tensor`): Observations.\n            - action (:obj:`torch.Tensor`): Actions.\n            - ebm (:obj:`torch.nn.Module`): Energy based model.\n            - create_graph (:obj:`bool`): Whether to create graph.\n        Returns:\n            - grad (:obj:`torch.Tensor`): Gradient w.r.t action.\n        Shapes:\n            - obs (:obj:`torch.Tensor`): :math:`(B, N, O)`.\n            - action (:obj:`torch.Tensor`): :math:`(B, N, A)`.\n            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.\n            - grad (:obj:`torch.Tensor`): :math:`(B, N, A)`.\n        '
        action.requires_grad_(True)
        energy = ebm.forward(obs, action).sum()
        grad = torch.autograd.grad(energy, action, create_graph=create_graph)[0]
        action.requires_grad_(False)
        return grad

    def grad_penalty(self, obs: torch.Tensor, action: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Calculate gradient penalty.\n        Arguments:\n            - obs (:obj:`torch.Tensor`): Observations.\n            - action (:obj:`torch.Tensor`): Actions.\n            - ebm (:obj:`torch.nn.Module`): Energy based model.\n        Returns:\n            - loss (:obj:`torch.Tensor`): Gradient penalty.\n        Shapes:\n            - obs (:obj:`torch.Tensor`): :math:`(B, N+1, O)`.\n            - action (:obj:`torch.Tensor`): :math:`(B, N+1, A)`.\n            - ebm (:obj:`torch.nn.Module`): :math:`(B, N+1, O)`.\n            - loss (:obj:`torch.Tensor`): :math:`(B, )`.\n        '
        if not self.add_grad_penalty:
            return 0.0
        de_dact = MCMC._gradient_wrt_act(obs, action, ebm, create_graph=True)

        def compute_grad_norm(grad_norm_type, de_dact) -> torch.Tensor:
            if False:
                print('Hello World!')
            grad_norm_type_to_ord = {'1': 1, '2': 2, 'inf': float('inf')}
            ord = grad_norm_type_to_ord[grad_norm_type]
            return torch.linalg.norm(de_dact, ord, dim=-1)
        grad_norms = compute_grad_norm(self.grad_norm_type, de_dact)
        grad_norms = grad_norms - self.grad_margin
        grad_norms = grad_norms.clamp(min=0.0, max=10000000000.0)
        grad_norms = grad_norms.pow(2)
        grad_loss = grad_norms.mean()
        return grad_loss * self.grad_loss_weight

    @no_ebm_grad()
    def _langevin_step(self, obs: torch.Tensor, action: torch.Tensor, stepsize: float, ebm: nn.Module) -> torch.Tensor:
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Run one langevin MCMC step.\n        Arguments:\n            - obs (:obj:`torch.Tensor`): Observations.\n            - action (:obj:`torch.Tensor`): Actions.\n            - stepsize (:obj:`float`): Step size.\n            - ebm (:obj:`torch.nn.Module`): Energy based model.\n        Returns:\n            - action (:obj:`torch.Tensor`): Actions.\n        Shapes:\n            - obs (:obj:`torch.Tensor`): :math:`(B, N, O)`.\n            - action (:obj:`torch.Tensor`): :math:`(B, N, A)`.\n            - stepsize (:obj:`float`): :math:`(B, )`.\n            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.\n        '
        l_lambda = 1.0
        de_dact = MCMC._gradient_wrt_act(obs, action, ebm)
        if self.grad_clip:
            de_dact = de_dact.clamp(min=-self.grad_clip, max=self.grad_clip)
        gradient_scale = 0.5
        de_dact = gradient_scale * l_lambda * de_dact + torch.randn_like(de_dact) * l_lambda * self.noise_scale
        delta_action = stepsize * de_dact
        delta_action_clip = self.delta_action_clip * 0.5 * (self.action_bounds[1] - self.action_bounds[0])
        delta_action = delta_action.clamp(min=-delta_action_clip, max=delta_action_clip)
        action = action - delta_action
        action = action.clamp(min=self.action_bounds[0], max=self.action_bounds[1])
        return action

    @no_ebm_grad()
    def _langevin_action_given_obs(self, obs: torch.Tensor, action: torch.Tensor, ebm: nn.Module, scheduler: BaseScheduler=None) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Run langevin MCMC for `self.iters` steps.\n        Arguments:\n            - obs (:obj:`torch.Tensor`): Observations.\n            - action (:obj:`torch.Tensor`): Actions.\n            - ebm (:obj:`torch.nn.Module`): Energy based model.\n            - scheduler (:obj:`BaseScheduler`): Learning rate scheduler.\n        Returns:\n            - action (:obj:`torch.Tensor`): Actions.\n        Shapes:\n            - obs (:obj:`torch.Tensor`): :math:`(B, N, O)`.\n            - action (:obj:`torch.Tensor`): :math:`(B, N, A)`.\n            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.\n        '
        if not scheduler:
            self.stepsize_scheduler['num_steps'] = self.iters
            scheduler = MCMC.PolynomialScheduler(**self.stepsize_scheduler)
        stepsize = scheduler.get_rate(-1)
        for i in range(self.iters):
            action = self._langevin_step(obs, action, stepsize, ebm)
            stepsize = scheduler.get_rate(i)
        return action

    @no_ebm_grad()
    def sample(self, obs: torch.Tensor, ebm: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        if False:
            i = 10
            return i + 15
        '\n        Overview:\n            Create tiled observations and sample counter-negatives for InfoNCE loss.\n        Arguments:\n            - obs (:obj:`torch.Tensor`): Observations.\n            - ebm (:obj:`torch.nn.Module`): Energy based model.\n        Returns:\n            - tiled_obs (:obj:`torch.Tensor`): Tiled observations.\n            - action_samples (:obj:`torch.Tensor`): Action samples.\n        Shapes:\n            - obs (:obj:`torch.Tensor`): :math:`(B, O)`.\n            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.\n            - tiled_obs (:obj:`torch.Tensor`): :math:`(B, N, O)`.\n            - action_samples (:obj:`torch.Tensor`): :math:`(B, N, A)`.\n        Examples:\n            >>> obs = torch.randn(2, 4)\n            >>> ebm = EBM(4, 5)\n            >>> opt = MCMC()\n            >>> opt.set_action_bounds(np.stack([np.zeros(5), np.ones(5)], axis=0))\n            >>> tiled_obs, action_samples = opt.sample(obs, ebm)\n        '
        (obs, uniform_action_samples) = self._sample(obs, self.train_samples)
        if not self.use_langevin_negative_samples:
            return (obs, uniform_action_samples)
        langevin_action_samples = self._langevin_action_given_obs(obs, uniform_action_samples, ebm)
        return (obs, langevin_action_samples)

    @no_ebm_grad()
    def infer(self, obs: torch.Tensor, ebm: nn.Module) -> torch.Tensor:
        if False:
            print('Hello World!')
        '\n        Overview:\n            Optimize for the best action conditioned on the current observation.\n        Arguments:\n            - obs (:obj:`torch.Tensor`): Observations.\n            - ebm (:obj:`torch.nn.Module`): Energy based model.\n        Returns:\n            - best_action_samples (:obj:`torch.Tensor`): Actions.\n        Shapes:\n            - obs (:obj:`torch.Tensor`): :math:`(B, O)`.\n            - ebm (:obj:`torch.nn.Module`): :math:`(B, N, O)`.\n            - best_action_samples (:obj:`torch.Tensor`): :math:`(B, A)`.\n        Examples:\n            >>> obs = torch.randn(2, 4)\n            >>> ebm = EBM(4, 5)\n            >>> opt = MCMC()\n            >>> opt.set_action_bounds(np.stack([np.zeros(5), np.ones(5)], axis=0))\n            >>> best_action_samples = opt.infer(obs, ebm)\n        '
        (obs, uniform_action_samples) = self._sample(obs, self.inference_samples)
        action_samples = self._langevin_action_given_obs(obs, uniform_action_samples, ebm)
        if self.optimize_again:
            self.again_stepsize_scheduler['num_steps'] = self.iters
            action_samples = self._langevin_action_given_obs(obs, action_samples, ebm, scheduler=MCMC.PolynomialScheduler(**self.again_stepsize_scheduler))
        return self._get_best_action_sample(obs, action_samples, ebm)

@MODEL_REGISTRY.register('ebm')
class EBM(nn.Module):
    """
    Overview:
        Energy based model.
    Interface:
        ``__init__``, ``forward``
    """

    def __init__(self, obs_shape: int, action_shape: int, hidden_size: int=512, hidden_layer_num: int=4, **kwargs):
        if False:
            print('Hello World!')
        '\n        Overview:\n            Initialize the EBM.\n        Arguments:\n            - obs_shape (:obj:`int`): Observation shape.\n            - action_shape (:obj:`int`): Action shape.\n            - hidden_size (:obj:`int`): Hidden size.\n            - hidden_layer_num (:obj:`int`): Number of hidden layers.\n        '
        super().__init__()
        input_size = obs_shape + action_shape
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(), RegressionHead(hidden_size, 1, hidden_layer_num, final_tanh=False))

    def forward(self, obs, action):
        if False:
            print('Hello World!')
        '\n        Overview:\n            Forward computation graph of EBM.\n        Arguments:\n            - obs (:obj:`torch.Tensor`): Observation of shape (B, N, O).\n            - action (:obj:`torch.Tensor`): Action of shape (B, N, A).\n        Returns:\n            - pred (:obj:`torch.Tensor`): Energy of shape (B, N).\n        Examples:\n            >>> obs = torch.randn(2, 3, 4)\n            >>> action = torch.randn(2, 3, 5)\n            >>> ebm = EBM(4, 5)\n            >>> pred = ebm(obs, action)\n        '
        x = torch.cat([obs, action], -1)
        x = self.net(x)
        return x['pred']

@MODEL_REGISTRY.register('arebm')
class AutoregressiveEBM(nn.Module):
    """
    Overview:
        Autoregressive energy based model.
    Interface:
        ``__init__``, ``forward``
    """

    def __init__(self, obs_shape: int, action_shape: int, hidden_size: int=512, hidden_layer_num: int=4):
        if False:
            return 10
        '\n        Overview:\n            Initialize the AutoregressiveEBM.\n        Arguments:\n            - obs_shape (:obj:`int`): Observation shape.\n            - action_shape (:obj:`int`): Action shape.\n            - hidden_size (:obj:`int`): Hidden size.\n            - hidden_layer_num (:obj:`int`): Number of hidden layers.\n        '
        super().__init__()
        self.ebm_list = nn.ModuleList()
        for i in range(action_shape):
            self.ebm_list.append(EBM(obs_shape, i + 1, hidden_size, hidden_layer_num))

    def forward(self, obs, action):
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Forward computation graph of AutoregressiveEBM.\n        Arguments:\n            - obs (:obj:`torch.Tensor`): Observation of shape (B, N, O).\n            - action (:obj:`torch.Tensor`): Action of shape (B, N, A).\n        Returns:\n            - pred (:obj:`torch.Tensor`): Energy of shape (B, N, A).\n        Examples:\n            >>> obs = torch.randn(2, 3, 4)\n            >>> action = torch.randn(2, 3, 5)\n            >>> arebm = AutoregressiveEBM(4, 5)\n            >>> pred = arebm(obs, action)\n        '
        output_list = []
        for (i, ebm) in enumerate(self.ebm_list):
            output_list.append(ebm(obs, action[..., :i + 1]))
        return torch.stack(output_list, axis=-1)