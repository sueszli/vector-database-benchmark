"""Gradient support for Composite Tensors."""
import abc
import sys
from tensorflow.python.framework import composite_tensor
from tensorflow.python.util import nest
if sys.version_info >= (3, 8):
    from typing import Protocol
    from typing import runtime_checkable
else:
    from typing_extensions import Protocol
    from typing_extensions import runtime_checkable

class CompositeTensorGradient(object, metaclass=abc.ABCMeta):
    """Class used to help compute gradients for CompositeTensors.

  This abstract base class defines two methods: `get_gradient_components`, which
  returns the components of a value that should be included in gradients; and
  `replace_gradient_components`, which replaces the gradient components in a
  value.  These methods can be used to compute the gradient of a `y` with
  respect to `x` (`grad(y, x)`) as follows:

  * If `y` is a `CompositeTensor` with `CompositeTensorGradient` `cg` =
    `y.__composite_gradient__`, then `grad(y, x)` =
    `grad(cg.get_gradient_components(y), x)`.

  * If `x` is a `CompositeTensor` with `CompositeTensorGradient` `cg` =
    'x.__composite_gradient__', then `grad(y, x)` =
    `cg.replace_gradient_components(x, grad(y, cg.get_gradient_components(x))`.
  """

    @abc.abstractmethod
    def get_gradient_components(self, value):
        if False:
            return 10
        'Returns the components of `value` that should be included in gradients.\n\n    This method may not call TensorFlow ops, since any new ops added to the\n    graph would not be propertly tracked by the gradient mechanisms.\n\n    Args:\n      value: A `CompositeTensor` value.\n\n    Returns:\n      A nested structure of `Tensor` or `IndexedSlices`.\n    '
        raise NotImplementedError(f'{type(self).__name__}.get_gradient_components()')

    @abc.abstractmethod
    def replace_gradient_components(self, value, component_grads):
        if False:
            return 10
        'Replaces the gradient components in `value` with `component_grads`.\n\n    Args:\n      value: A value with its gradient components compatible with\n        `component_grads`.\n      component_grads: A nested structure of `Tensor` or `IndexedSlices` or\n        `None` (for unconnected gradients).\n\n    Returns:\n      A copy of `value`, where the components that should be included in\n      gradients have been replaced by `component_grads`; or `None` (if\n      `component_grads` includes `None`).\n    '
        raise NotImplementedError(f'{type(self).__name__}.replace_gradient_components()')

@runtime_checkable
class CompositeTensorGradientProtocol(Protocol):
    """Protocol for adding gradient support to CompositeTensors."""
    __composite_gradient__: CompositeTensorGradient

class WithValuesCompositeTensorGradient(CompositeTensorGradient):
    """CompositeTensorGradient based on `T.values` and `T.with_values`."""

    def get_gradient_components(self, value):
        if False:
            for i in range(10):
                print('nop')
        return value.values

    def replace_gradient_components(self, value, component_grads):
        if False:
            for i in range(10):
                print('nop')
        return value.with_values(component_grads)

def _get_tensors_for_gradient(x):
    if False:
        for i in range(10):
            print('nop')
    'Returns the Tensors in `x` that should be differentiated.\n\n  Args:\n    x: A `Tensor` or `CompositeTensor`.\n\n  Returns:\n    A `Tensor` or a nested structure of `Tensor`.\n  '
    if not isinstance(x, composite_tensor.CompositeTensor):
        return x
    if not isinstance(x, CompositeTensorGradientProtocol):
        raise ValueError(f'Type {type(x).__name__} is not supported as a gradient source or gradient target.')
    composite_gradient = x.__composite_gradient__
    gradient_components = composite_gradient.get_gradient_components(x)
    if gradient_components is x:
        return x
    return nest.map_structure(_get_tensors_for_gradient, gradient_components)

def _replace_tensors_for_gradient(x, grad):
    if False:
        for i in range(10):
            print('nop')
    'Replaces the tensors in `x` that should be differentiated with `grad`.\n\n  Args:\n    x: A `Tensor` or `CompositeTensor`.\n    grad: A nested structure of `Tensor`, with the same structure as the value\n      returned by `_get_tensors_for_gradient(x)`.\n\n  Returns:\n    A `Tensor` or `CompositeTensor`.\n  '
    if not isinstance(x, composite_tensor.CompositeTensor):
        return grad
    if not isinstance(x, CompositeTensorGradientProtocol):
        raise ValueError(f'Type {type(x).__name__} is not supported as a gradient source.')
    composite_gradient = x.__composite_gradient__
    x_components = composite_gradient.get_gradient_components(x)
    if x_components is x:
        grad_components = grad
    else:
        grad_components = nest.map_structure_up_to(x_components, _replace_tensors_for_gradient, x_components, grad)
    if grad_components is None:
        return None
    return composite_gradient.replace_gradient_components(x, grad_components)

def get_flat_tensors_for_gradients(xs):
    if False:
        while True:
            i = 10
    'Returns a flat list of Tensors that should be differentiated for `xs`.\n\n  Args:\n    xs: A list of `Tensor`s or `CompositeTensor`s.\n\n  Returns:\n    A flat list of `Tensor`s constructed from `xs`, where `Tensor` values are\n    left as-is, and `CompositeTensor`s are replaced with\n    `_get_tensors_for_gradient(x)`.\n  '
    return nest.flatten([_get_tensors_for_gradient(x) for x in xs])

def replace_flat_tensors_for_gradients(xs, flat_grads):
    if False:
        for i in range(10):
            print('nop')
    'Replaces Tensors that should be differentiated in `xs` with `flat_grads`.\n\n  Args:\n    xs: A list of `Tensor`s or `CompositeTensor`s.\n    flat_grads: A list of `Tensor`.\n\n  Returns:\n    A list of `Tensor` or `CompositeTensor`.\n  '
    xs_structure = [_get_tensors_for_gradient(x) for x in xs]
    grads = nest.pack_sequence_as(xs_structure, flat_grads)
    return [_replace_tensors_for_gradient(x, grad) for (x, grad) in zip(xs, grads)]