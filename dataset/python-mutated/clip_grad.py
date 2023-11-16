import warnings
from typing import Union, Iterable, List, Dict, Tuple, Optional, cast
import torch
from torch import Tensor, inf
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype, _has_foreach_support
_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]
__all__ = ['clip_grad_norm_', 'clip_grad_norm', 'clip_grad_value_']

def clip_grad_norm_(parameters: _tensor_or_tensors, max_norm: float, norm_type: float=2.0, error_if_nonfinite: bool=False, foreach: Optional[bool]=None) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    "Clip the gradient norm of an iterable of parameters.\n\n    The norm is computed over all gradients together, as if they were\n    concatenated into a single vector. Gradients are modified in-place.\n\n    Args:\n        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a\n            single Tensor that will have gradients normalized\n        max_norm (float): max norm of the gradients\n        norm_type (float): type of the used p-norm. Can be ``'inf'`` for\n            infinity norm.\n        error_if_nonfinite (bool): if True, an error is thrown if the total\n            norm of the gradients from :attr:`parameters` is ``nan``,\n            ``inf``, or ``-inf``. Default: False (will switch to True in the future)\n        foreach (bool): use the faster foreach-based implementation.\n            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently\n            fall back to the slow implementation for other device types.\n            Default: ``None``\n\n    Returns:\n        Total norm of the parameter gradients (viewed as a single vector).\n    "
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(grads) == 0:
        return torch.tensor(0.0)
    first_device = grads[0].device
    grouped_grads: Dict[Tuple[torch.device, torch.dtype], List[List[Tensor]]] = _group_tensors_by_device_and_dtype([[g.detach() for g in grads]])
    if norm_type == inf:
        norms = [torch.linalg.vector_norm(g.detach(), inf).to(first_device) for g in grads]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        norms = []
        for ((device, _), ([grads], _)) in grouped_grads.items():
            if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
                norms.extend(torch._foreach_norm(grads, norm_type))
            elif foreach:
                raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
            else:
                norms.extend([torch.linalg.vector_norm(g, norm_type) for g in grads])
        total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(f'The total norm of order {norm_type} for gradients from `parameters` is non-finite, so it cannot be clipped. To disable this error and scale the gradients by the non-finite norm anyway, set `error_if_nonfinite=False`')
    clip_coef = max_norm / (total_norm + 1e-06)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for ((device, _), ([grads], _)) in grouped_grads.items():
        if (foreach is None or foreach) and _has_foreach_support(grads, device=device):
            torch._foreach_mul_(grads, clip_coef_clamped.to(device))
        elif foreach:
            raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
        else:
            clip_coef_clamped_device = clip_coef_clamped.to(device)
            for g in grads:
                g.detach().mul_(clip_coef_clamped_device)
    return total_norm

def clip_grad_norm(parameters: _tensor_or_tensors, max_norm: float, norm_type: float=2.0, error_if_nonfinite: bool=False, foreach: Optional[bool]=None) -> torch.Tensor:
    if False:
        while True:
            i = 10
    'Clip the gradient norm of an iterable of parameters.\n\n    .. warning::\n        This method is now deprecated in favor of\n        :func:`torch.nn.utils.clip_grad_norm_`.\n    '
    warnings.warn('torch.nn.utils.clip_grad_norm is now deprecated in favor of torch.nn.utils.clip_grad_norm_.', stacklevel=2)
    return clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite, foreach)

def clip_grad_value_(parameters: _tensor_or_tensors, clip_value: float, foreach: Optional[bool]=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Clip the gradients of an iterable of parameters at specified value.\n\n    Gradients are modified in-place.\n\n    Args:\n        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a\n            single Tensor that will have gradients normalized\n        clip_value (float): maximum allowed value of the gradients.\n            The gradients are clipped in the range\n            :math:`\\left[\\text{-clip\\_value}, \\text{clip\\_value}\\right]`\n        foreach (bool): use the faster foreach-based implementation\n            If ``None``, use the foreach implementation for CUDA and CPU native tensors and\n            silently fall back to the slow implementation for other device types.\n            Default: ``None``\n    '
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    grads = [p.grad for p in parameters if p.grad is not None]
    grouped_grads = _group_tensors_by_device_and_dtype([grads])
    for ((device, _), ([grads], _)) in grouped_grads.items():
        if (foreach is None or foreach) and _has_foreach_support(cast(List[Tensor], grads), device=device):
            torch._foreach_clamp_min_(cast(List[Tensor], grads), -clip_value)
            torch._foreach_clamp_max_(cast(List[Tensor], grads), clip_value)
        elif foreach:
            raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
        else:
            with torch.no_grad():
                for grad in grads:
                    cast(Tensor, grad).clamp_(min=-clip_value, max=clip_value)