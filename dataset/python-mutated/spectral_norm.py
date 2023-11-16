"""Spectral Normalization from https://arxiv.org/abs/1802.05957."""
import torch
from torch.nn.functional import normalize
from typing import Any, Optional, TypeVar
from ..modules import Module
__all__ = ['SpectralNorm', 'SpectralNormLoadStateDictPreHook', 'SpectralNormStateDictHook', 'spectral_norm', 'remove_spectral_norm']

class SpectralNorm:
    _version: int = 1
    name: str
    dim: int
    n_power_iterations: int
    eps: float

    def __init__(self, name: str='weight', n_power_iterations: int=1, dim: int=0, eps: float=1e-12) -> None:
        if False:
            i = 10
            return i + 15
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError(f'Expected n_power_iterations to be positive, but got n_power_iterations={n_power_iterations}')
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        if False:
            return 10
        weight_mat = weight
        if self.dim != 0:
            weight_mat = weight_mat.permute(self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module: Module, do_power_iteration: bool) -> torch.Tensor:
        if False:
            print('Hello World!')
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)
        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)
        sigma = torch.dot(u, torch.mv(weight_mat, v))
        weight = weight / sigma
        return weight

    def remove(self, module: Module) -> None:
        if False:
            while True:
                i = 10
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + '_u')
        delattr(module, self.name + '_v')
        delattr(module, self.name + '_orig')
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module: Module, inputs: Any) -> None:
        if False:
            return 10
        setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        if False:
            print('Hello World!')
        v = torch.linalg.multi_dot([weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)]).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(module: Module, name: str, n_power_iterations: int, dim: int, eps: float) -> 'SpectralNorm':
        if False:
            return 10
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(f'Cannot register two spectral_norm hooks on the same parameter {name}')
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        if weight is None:
            raise ValueError(f'`SpectralNorm` cannot be applied as parameter `{name}` is None')
        if isinstance(weight, torch.nn.parameter.UninitializedParameter):
            raise ValueError("The module passed to `SpectralNorm` can't have uninitialized parameters. Make sure to run the dummy forward before applying spectral normalization")
        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)
            (h, w) = weight_mat.size()
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)
        delattr(module, fn.name)
        module.register_parameter(fn.name + '_orig', weight)
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + '_u', u)
        module.register_buffer(fn.name + '_v', v)
        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn

class SpectralNormLoadStateDictPreHook:

    def __init__(self, fn) -> None:
        if False:
            print('Hello World!')
        self.fn = fn

    def __call__(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None:
        if False:
            i = 10
            return i + 15
        fn = self.fn
        version = local_metadata.get('spectral_norm', {}).get(fn.name + '.version', None)
        if version is None or version < 1:
            weight_key = prefix + fn.name
            if version is None and all((weight_key + s in state_dict for s in ('_orig', '_u', '_v'))) and (weight_key not in state_dict):
                return
            has_missing_keys = False
            for suffix in ('_orig', '', '_u'):
                key = weight_key + suffix
                if key not in state_dict:
                    has_missing_keys = True
                    if strict:
                        missing_keys.append(key)
            if has_missing_keys:
                return
            with torch.no_grad():
                weight_orig = state_dict[weight_key + '_orig']
                weight = state_dict.pop(weight_key)
                sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[weight_key + '_u']
                v = fn._solve_v_and_rescale(weight_mat, u, sigma)
                state_dict[weight_key + '_v'] = v

class SpectralNormStateDictHook:

    def __init__(self, fn) -> None:
        if False:
            i = 10
            return i + 15
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata) -> None:
        if False:
            for i in range(10):
                print('nop')
        if 'spectral_norm' not in local_metadata:
            local_metadata['spectral_norm'] = {}
        key = self.fn.name + '.version'
        if key in local_metadata['spectral_norm']:
            raise RuntimeError(f"Unexpected key in metadata['spectral_norm']: {key}")
        local_metadata['spectral_norm'][key] = self.fn._version
T_module = TypeVar('T_module', bound=Module)

def spectral_norm(module: T_module, name: str='weight', n_power_iterations: int=1, eps: float=1e-12, dim: Optional[int]=None) -> T_module:
    if False:
        for i in range(10):
            print('nop')
    'Apply spectral normalization to a parameter in the given module.\n\n    .. math::\n        \\mathbf{W}_{SN} = \\dfrac{\\mathbf{W}}{\\sigma(\\mathbf{W})},\n        \\sigma(\\mathbf{W}) = \\max_{\\mathbf{h}: \\mathbf{h} \\ne 0} \\dfrac{\\|\\mathbf{W} \\mathbf{h}\\|_2}{\\|\\mathbf{h}\\|_2}\n\n    Spectral normalization stabilizes the training of discriminators (critics)\n    in Generative Adversarial Networks (GANs) by rescaling the weight tensor\n    with spectral norm :math:`\\sigma` of the weight matrix calculated using\n    power iteration method. If the dimension of the weight tensor is greater\n    than 2, it is reshaped to 2D in power iteration method to get spectral\n    norm. This is implemented via a hook that calculates spectral norm and\n    rescales weight before every :meth:`~Module.forward` call.\n\n    See `Spectral Normalization for Generative Adversarial Networks`_ .\n\n    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957\n\n    Args:\n        module (nn.Module): containing module\n        name (str, optional): name of weight parameter\n        n_power_iterations (int, optional): number of power iterations to\n            calculate spectral norm\n        eps (float, optional): epsilon for numerical stability in\n            calculating norms\n        dim (int, optional): dimension corresponding to number of outputs,\n            the default is ``0``, except for modules that are instances of\n            ConvTranspose{1,2,3}d, when it is ``1``\n\n    Returns:\n        The original module with the spectral norm hook\n\n    .. note::\n        This function has been reimplemented as\n        :func:`torch.nn.utils.parametrizations.spectral_norm` using the new\n        parametrization functionality in\n        :func:`torch.nn.utils.parametrize.register_parametrization`. Please use\n        the newer version. This function will be deprecated in a future version\n        of PyTorch.\n\n    Example::\n\n        >>> m = spectral_norm(nn.Linear(20, 40))\n        >>> m\n        Linear(in_features=20, out_features=40, bias=True)\n        >>> m.weight_u.size()\n        torch.Size([40])\n\n    '
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module

def remove_spectral_norm(module: T_module, name: str='weight') -> T_module:
    if False:
        return 10
    'Remove the spectral normalization reparameterization from a module.\n\n    Args:\n        module (Module): containing module\n        name (str, optional): name of weight parameter\n\n    Example:\n        >>> m = spectral_norm(nn.Linear(40, 10))\n        >>> remove_spectral_norm(m)\n    '
    for (k, hook) in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break
    else:
        raise ValueError(f"spectral_norm of '{name}' not found in {module}")
    for (k, hook) in module._state_dict_hooks.items():
        if isinstance(hook, SpectralNormStateDictHook) and hook.fn.name == name:
            del module._state_dict_hooks[k]
            break
    for (k, hook) in module._load_state_dict_pre_hooks.items():
        if isinstance(hook, SpectralNormLoadStateDictPreHook) and hook.fn.name == name:
            del module._load_state_dict_pre_hooks[k]
            break
    return module