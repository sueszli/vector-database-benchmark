import os
from collections import namedtuple
from typing import Any
import torch
from .grad_mode import _DecoratorContextManager
__all__ = ['UnpackedDualTensor', 'enter_dual_level', 'exit_dual_level', 'make_dual', 'unpack_dual', 'dual_level']
_current_level = -1

def enter_dual_level():
    if False:
        while True:
            i = 10
    'Enter a new forward grad level.\n\n    This level can be used to make and unpack dual Tensors to compute\n    forward gradients.\n\n    This function also updates the current level that is used by default\n    by the other functions in this API.\n    '
    global _current_level
    new_level = torch._C._enter_dual_level()
    if new_level != _current_level + 1:
        raise RuntimeError('Entering a new forward AD level but the current level is not valid. Make sure you did not modified it directly.')
    _current_level = new_level
    return new_level

def exit_dual_level(*, level=None):
    if False:
        print('Hello World!')
    'Exit a forward grad level.\n\n    This function deletes all the gradients associated with this\n    level. Only deleting the latest entered level is allowed.\n\n    This function also updates the current level that is used by default\n    by the other functions in this API.\n    '
    global _current_level
    if level is None:
        level = _current_level
    if level != _current_level:
        raise RuntimeError('Trying to exit a forward AD level that was not the last one that was created. This is not supported.')
    torch._C._exit_dual_level(level=level)
    _current_level = level - 1

def make_dual(tensor, tangent, *, level=None):
    if False:
        i = 10
        return i + 15
    'Associate a tensor value with its tangent to create a "dual tensor" for forward AD gradient computation.\n\n    The result is a new tensor aliased to :attr:`tensor` with :attr:`tangent` embedded\n    as an attribute as-is if it has the same storage layout or copied otherwise.\n    The tangent attribute can be recovered with :func:`unpack_dual`.\n\n    This function is backward differentiable.\n\n    Given a function `f` whose jacobian is `J`, it allows one to compute the Jacobian-vector product (`jvp`)\n    between `J` and a given vector `v` as follows.\n\n    Example::\n\n        >>> # xdoctest: +SKIP("Undefined variables")\n        >>> with dual_level():\n        ...     inp = make_dual(x, v)\n        ...     out = f(inp)\n        ...     y, jvp = unpack_dual(out)\n\n    Please see the `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__\n    for detailed steps on how to use this API.\n\n    '
    if os.environ.get('PYTORCH_JIT', '1') == '1' and __debug__:
        from torch._decomp import decompositions_for_jvp
    if level is None:
        level = _current_level
    if level < 0:
        raise RuntimeError('Trying to create a dual Tensor for forward AD but no level exists, make sure to enter_dual_level() first.')
    if not (tensor.is_floating_point() or tensor.is_complex()):
        raise ValueError(f'Expected primal to be floating point or complex, but got: {tensor.dtype}')
    if not (tangent.is_floating_point() or tangent.is_complex()):
        raise ValueError(f'Expected tangent to be floating point or complex, but got: {tangent.dtype}')
    return torch._VF._make_dual(tensor, tangent, level=level)
_UnpackedDualTensor = namedtuple('_UnpackedDualTensor', ['primal', 'tangent'])

class UnpackedDualTensor(_UnpackedDualTensor):
    """Namedtuple returned by :func:`unpack_dual` containing the primal and tangent components of the dual tensor.

    See :func:`unpack_dual` for more details.

    """
    pass

def unpack_dual(tensor, *, level=None):
    if False:
        for i in range(10):
            print('nop')
    'Unpack a "dual tensor" to get both its Tensor value and its forward AD gradient.\n\n    The result is a namedtuple ``(primal, tangent)`` where ``primal`` is a view of\n    :attr:`tensor`\'s primal and ``tangent`` is :attr:`tensor`\'s tangent as-is.\n    Neither of these tensors can be dual tensor of level :attr:`level`.\n\n    This function is backward differentiable.\n\n    Example::\n\n        >>> # xdoctest: +SKIP("Undefined variables")\n        >>> with dual_level():\n        ...     inp = make_dual(x, x_t)\n        ...     out = f(inp)\n        ...     y, jvp = unpack_dual(out)\n        ...     jvp = unpack_dual(out).tangent\n\n    Please see the `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__\n    for detailed steps on how to use this API.\n    '
    if level is None:
        level = _current_level
    if level < 0:
        return UnpackedDualTensor(tensor, None)
    (primal, dual) = torch._VF._unpack_dual(tensor, level=level)
    return UnpackedDualTensor(primal, dual)

class dual_level(_DecoratorContextManager):
    """Context-manager for forward AD, where all forward AD computation must occur within the ``dual_level`` context.

    .. Note::

        The ``dual_level`` context appropriately enters and exit the dual level to
        controls the current forward AD level, which is used by default by the other
        functions in this API.

        We currently don't plan to support nested ``dual_level`` contexts, however, so
        only a single forward AD level is supported. To compute higher-order
        forward grads, one can use :func:`torch.func.jvp`.

    Example::

        >>> # xdoctest: +SKIP("Undefined variables")
        >>> x = torch.tensor([1])
        >>> x_t = torch.tensor([1])
        >>> with dual_level():
        ...     inp = make_dual(x, x_t)
        ...     # Do computations with inp
        ...     out = your_fn(inp)
        ...     _, grad = unpack_dual(out)
        >>> grad is None
        False
        >>> # After exiting the level, the grad is deleted
        >>> _, grad_after = unpack_dual(out)
        >>> grad is None
        True

    Please see the `forward-mode AD tutorial <https://pytorch.org/tutorials/intermediate/forward_ad_usage.html>`__
    for detailed steps on how to use this API.
    """

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return enter_dual_level()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if False:
            while True:
                i = 10
        exit_dual_level()
_is_fwd_grad_enabled = torch._C._is_fwd_grad_enabled

class _set_fwd_grad_enabled(_DecoratorContextManager):

    def __init__(self, mode: bool) -> None:
        if False:
            while True:
                i = 10
        self.prev = _is_fwd_grad_enabled()
        torch._C._set_fwd_grad_enabled(mode)

    def __enter__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if False:
            print('Hello World!')
        torch._C._set_fwd_grad_enabled(self.prev)