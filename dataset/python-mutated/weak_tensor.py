"""An extension type that represents WeakTensor."""
from typing import Optional
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.types import core
_ALLOWED_WEAK_DTYPES = (dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64, dtypes.complex128)

class WeakTensorGradient(composite_tensor_gradient.CompositeTensorGradient):
    """CompositeTensorGradient for WeakTensor."""

    def get_gradient_components(self, weak_tensor):
        if False:
            print('Hello World!')
        return weak_tensor.tensor

    def replace_gradient_components(self, weak_tensor, component_grads):
        if False:
            return 10
        return weak_tensor._type_spec._from_components([component_grads])

class WeakTensor(extension_type.BatchableExtensionType, core.Tensor):
    """A weakly typed Tensor.

  A simple wrapper class that contains a normal Tensor.

  A "weak" type means that its dtype is temporarily inferred by the system,
  and could defer to other dtypes.

  i.g. weak f64 + f16 => f16

  This information is used for auto dtype conversion.
  """
    __name__ = 'tf.WeakTensor'
    tensor: tensor_lib.Tensor

    def __validate__(self):
        if False:
            while True:
                i = 10
        if self.tensor.dtype not in _ALLOWED_WEAK_DTYPES:
            raise TypeError(f'{self.tensor.dtype} not allowed as a weak type. The allowed types are {_ALLOWED_WEAK_DTYPES}.')

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return self._format_weak_tensor(is_repr=False)

    def __repr__(self):
        if False:
            return 10
        return self._format_weak_tensor(is_repr=True)

    def _format_weak_tensor(self, is_repr):
        if False:
            i = 10
            return i + 15
        tensor_str = self.tensor.__repr__() if is_repr else self.tensor.__str__()
        closing_char = tensor_str[len(tensor_str) - 1]
        last_index = tensor_str.rfind(closing_char)
        return tensor_str[:last_index] + ', weak=True' + closing_char

    def __getattr__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self.tensor, *args, **kwargs)

    def _disallow(self, task):
        if False:
            return 10
        raise errors.OperatorNotAllowedInGraphError(f'{task} is not allowed. You can attempt the following resolutions to the problem: If you are running in Graph mode, use Eager execution mode or decorate this function with @tf.function. If you are using AutoGraph, you can try decorating this function with @tf.function. If that does not work, then you may be using an unsupported feature or your source code may not be visible to AutoGraph. See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code for more information.')

    def _disallow_iteration(self):
        if False:
            return 10
        self._disallow('Iterating over a symbolic `tf.WeakTensor`')

    def _shape_as_list(self):
        if False:
            while True:
                i = 10
        if self.shape.ndims is not None:
            return [dim.value for dim in self.shape.dims]
        else:
            return None

    def __iter__(self):
        if False:
            print('Hello World!')
        if not context.executing_eagerly():
            self._disallow_iteration()
        first_dim = self.tensor._get_first_dim()
        return _WeakTensorIterator(self, first_dim)

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tensor.__hash__()

    def __copy__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __len__(self):
        if False:
            return 10
        return self.tensor.__len__()

    def __bool__(self):
        if False:
            print('Hello World!')
        return self.tensor.__bool__()

    def __tf_tensor__(self, dtype: Optional[dtypes.DType]=None, name: Optional[str]=None):
        if False:
            return 10
        return self.tensor.__tf_tensor__(dtype=dtype, name=name)

    def __deepcopy__(self, memo):
        if False:
            i = 10
            return i + 15
        del memo
        return self

    def to_tensor(self):
        if False:
            return 10
        "Converts this 'WeakTensor' into a 'tf.Tensor'."
        return self.tensor

    def _as_graph_element(self):
        if False:
            print('Hello World!')
        'Convert `self` to a graph element.'
        return self.tensor

    @classmethod
    def from_tensor(cls, tensor):
        if False:
            for i in range(10):
                print('nop')
        "Converts a 'tf.Tensor' into a 'WeakTensor'.\n\n    This should be the standard way of creating a WeakTensor instead\n    of directly calling the WeakTensor constructor.\n\n    Args:\n      tensor: The `tf.Tensor` that should be converted into a 'WeakTensor'.\n\n    Returns:\n      A `EagerWeakTensor` or 'GraphWeakTensor' that holds the `tensor`.\n    "
        if isinstance(tensor, core.Value):
            return EagerWeakTensor(tensor)
        if isinstance(tensor, core.Symbol):
            return GraphWeakTensor(tensor)
        raise errors.InvalidArgumentError(None, None, f'WeakTensor can only be constructed from tf.Tensor or tf.WeakTensor, but {type(tensor)} was given.')

    @property
    def dtype(self):
        if False:
            return 10
        return self.tensor.dtype

    @property
    def shape(self):
        if False:
            return 10
        return self.tensor.shape

    @property
    def is_tensor_like(self):
        if False:
            return 10
        return True
    __composite_gradient__ = WeakTensorGradient()

class EagerWeakTensor(core.Value, WeakTensor):
    """A weakly typed Eager Tensor."""
    __name__ = 'tf.EagerWeakTensor'

    def numpy(self):
        if False:
            i = 10
            return i + 15
        'Copy of the contents of this EagerWeakTensor into a NumPy array or scalar.'
        if not isinstance(self.tensor, ops.EagerTensor):
            raise ValueError('WeakTensor.numpy() is only supported in eager mode.')
        return self.tensor.numpy()

    def __complex__(self):
        if False:
            return 10
        return self.tensor.__complex__()

    def __int__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tensor.__int__()

    def __float__(self):
        if False:
            while True:
                i = 10
        return self.tensor.__float__()

    def __index__(self):
        if False:
            i = 10
            return i + 15
        return self.tensor.__index__()

    def __format__(self, format_spec):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.tensor.__format__(format_spec)} weakly typed'

    def __array__(self, dtype=None):
        if False:
            return 10
        return np.array(self.tensor.__array__(dtype))

class GraphWeakTensor(core.Symbol, WeakTensor):
    """A weakly typed Graph Tensor."""
    __name__ = 'tf.GraphWeakTensor'

class _WeakTensorIterator(object):
    """Iterates over the leading dim of a WeakTensor. Performs no error checks."""
    __slots__ = ['_weak_tensor', '_index', '_limit']

    def __init__(self, weak_tensor, dim0):
        if False:
            return 10
        self._weak_tensor = weak_tensor
        self._index = 0
        self._limit = dim0

    def __iter__(self):
        if False:
            print('Hello World!')
        return self

    def __next__(self):
        if False:
            print('Hello World!')
        if self._index == self._limit:
            raise StopIteration
        result = WeakTensor.from_tensor(self._weak_tensor.tensor[self._index])
        self._index += 1
        return result

def convert_to_weak_tensor_or_tensor(t, to_weak):
    if False:
        for i in range(10):
            print('nop')
    if to_weak:
        return WeakTensor.from_tensor(t)
    if isinstance(t, WeakTensor):
        return t.tensor
    return t

def weak_tensor_conversion_function(t):
    if False:
        print('Hello World!')
    if isinstance(t, WeakTensor):
        return t.tensor
tensor_conversion_registry.register_tensor_conversion_function(WeakTensor, weak_tensor_conversion_function)