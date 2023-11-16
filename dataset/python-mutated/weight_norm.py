"""Weight Normalization from https://arxiv.org/abs/1602.07868."""
from torch.nn.parameter import Parameter, UninitializedParameter
from torch import _weight_norm, norm_except_dim
from typing import Any, TypeVar
import warnings
from ..modules import Module
__all__ = ['WeightNorm', 'weight_norm', 'remove_weight_norm']

class WeightNorm:
    name: str
    dim: int

    def __init__(self, name: str, dim: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    def compute_weight(self, module: Module) -> Any:
        if False:
            while True:
                i = 10
        g = getattr(module, self.name + '_g')
        v = getattr(module, self.name + '_v')
        return _weight_norm(v, g, self.dim)

    @staticmethod
    def apply(module, name: str, dim: int) -> 'WeightNorm':
        if False:
            return 10
        warnings.warn('torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.')
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError(f'Cannot register two weight_norm hooks on the same parameter {name}')
        if dim is None:
            dim = -1
        fn = WeightNorm(name, dim)
        weight = getattr(module, name)
        if isinstance(weight, UninitializedParameter):
            raise ValueError("The module passed to `WeightNorm` can't have uninitialized parameters. Make sure to run the dummy forward before applying weight normalization")
        del module._parameters[name]
        module.register_parameter(name + '_g', Parameter(norm_except_dim(weight, 2, dim).data))
        module.register_parameter(name + '_v', Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))
        module.register_forward_pre_hook(fn)
        return fn

    def remove(self, module: Module) -> None:
        if False:
            return 10
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_g']
        del module._parameters[self.name + '_v']
        setattr(module, self.name, Parameter(weight.data))

    def __call__(self, module: Module, inputs: Any) -> None:
        if False:
            print('Hello World!')
        setattr(module, self.name, self.compute_weight(module))
T_module = TypeVar('T_module', bound=Module)

def weight_norm(module: T_module, name: str='weight', dim: int=0) -> T_module:
    if False:
        i = 10
        return i + 15
    "Apply weight normalization to a parameter in the given module.\n\n    .. math::\n         \\mathbf{w} = g \\dfrac{\\mathbf{v}}{\\|\\mathbf{v}\\|}\n\n    Weight normalization is a reparameterization that decouples the magnitude\n    of a weight tensor from its direction. This replaces the parameter specified\n    by :attr:`name` (e.g. ``'weight'``) with two parameters: one specifying the magnitude\n    (e.g. ``'weight_g'``) and one specifying the direction (e.g. ``'weight_v'``).\n    Weight normalization is implemented via a hook that recomputes the weight\n    tensor from the magnitude and direction before every :meth:`~Module.forward`\n    call.\n\n    By default, with ``dim=0``, the norm is computed independently per output\n    channel/plane. To compute a norm over the entire weight tensor, use\n    ``dim=None``.\n\n    See https://arxiv.org/abs/1602.07868\n\n    .. warning::\n\n        This function is deprecated.  Use :func:`torch.nn.utils.parametrizations.weight_norm`\n        which uses the modern parametrization API.  The new ``weight_norm`` is compatible\n        with ``state_dict`` generated from old ``weight_norm``.\n\n        Migration guide:\n\n        * The magnitude (``weight_g``) and direction (``weight_v``) are now expressed\n          as ``parametrizations.weight.original0`` and ``parametrizations.weight.original1``\n          respectively.  If this is bothering you, please comment on\n          https://github.com/pytorch/pytorch/issues/102999\n\n        * To remove the weight normalization reparametrization, use\n          :func:`torch.nn.utils.parametrize.remove_parametrizations`.\n\n        * The weight is no longer recomputed once at module forward; instead, it will\n          be recomputed on every access.  To restore the old behavior, use\n          :func:`torch.nn.utils.parametrize.cached` before invoking the module\n          in question.\n\n    Args:\n        module (Module): containing module\n        name (str, optional): name of weight parameter\n        dim (int, optional): dimension over which to compute the norm\n\n    Returns:\n        The original module with the weight norm hook\n\n    Example::\n\n        >>> m = weight_norm(nn.Linear(20, 40), name='weight')\n        >>> m\n        Linear(in_features=20, out_features=40, bias=True)\n        >>> m.weight_g.size()\n        torch.Size([40, 1])\n        >>> m.weight_v.size()\n        torch.Size([40, 20])\n\n    "
    WeightNorm.apply(module, name, dim)
    return module

def remove_weight_norm(module: T_module, name: str='weight') -> T_module:
    if False:
        for i in range(10):
            print('nop')
    'Remove the weight normalization reparameterization from a module.\n\n    Args:\n        module (Module): containing module\n        name (str, optional): name of weight parameter\n\n    Example:\n        >>> m = weight_norm(nn.Linear(20, 40))\n        >>> remove_weight_norm(m)\n    '
    for (k, hook) in module._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module
    raise ValueError(f"weight_norm of '{name}' not found in {module}")