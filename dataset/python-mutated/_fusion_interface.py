import numpy
from cupy._core._dtype import get_dtype
import cupy
from cupy._core import _fusion_thread_local
from cupy._core import core
from cupy._core._scalar import get_typename
_thread_local = _fusion_thread_local.thread_local
_dtype_to_astype_dict = None

def _set_dtype_to_astype_dict():
    if False:
        for i in range(10):
            print('nop')
    'Set a dict with dtypes and astype ufuncs to `_dtype_to_astype_dict`.\n\n    Creates a ufunc for type cast operations, and set a dict with keys\n    as the dtype of the output array and values as astype ufuncs.\n    This function is called at most once.\n    '
    global _dtype_to_astype_dict
    _dtype_to_astype_dict = {}
    dtype_list = [numpy.dtype(type_char) for type_char in '?bhilqBHILQefdFD']
    for t in dtype_list:
        name = 'astype_{}'.format(t)
        rules = tuple(['{}->{}'.format(s.char, t.char) for s in dtype_list])
        command = 'out0 = static_cast< {} >(in0)'.format(get_typename(t))
        _dtype_to_astype_dict[t] = core.create_ufunc(name, rules, command)

class _VariableProxy:
    """Abstracted array/scalar object passed to the target function.
    """

    def __init__(self, content):
        if False:
            i = 10
            return i + 15
        assert isinstance(content, cupy._core._fusion_variable._TraceVariable)
        self.content = content

    def __neg__(self):
        if False:
            print('Hello World!')
        return cupy.negative(self)

    def __add__(self, other):
        if False:
            print('Hello World!')
        return cupy.add(self, other)

    def __radd__(self, other):
        if False:
            print('Hello World!')
        return cupy.add(other, self)

    def __sub__(self, other):
        if False:
            print('Hello World!')
        return cupy.subtract(self, other)

    def __rsub__(self, other):
        if False:
            while True:
                i = 10
        return cupy.subtract(other, self)

    def __mul__(self, other):
        if False:
            i = 10
            return i + 15
        return cupy.multiply(self, other)

    def __rmul__(self, other):
        if False:
            return 10
        return cupy.multiply(other, self)

    def __div__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return cupy.divide(self, other)

    def __rdiv__(self, other):
        if False:
            print('Hello World!')
        return cupy.divide(other, self)

    def __truediv__(self, other):
        if False:
            print('Hello World!')
        return cupy.true_divide(self, other)

    def __rtruediv__(self, other):
        if False:
            i = 10
            return i + 15
        return cupy.true_divide(other, self)

    def __floordiv__(self, other):
        if False:
            i = 10
            return i + 15
        return cupy.floor_divide(self, other)

    def __rfloordiv__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return cupy.floor_divide(other, self)

    def __mod__(self, other):
        if False:
            while True:
                i = 10
        return cupy.remainder(self, other)

    def __rmod__(self, other):
        if False:
            while True:
                i = 10
        return cupy.remainder(other, self)

    def __pow__(self, other):
        if False:
            i = 10
            return i + 15
        return cupy.power(self, other)

    def __lshift__(self, other):
        if False:
            while True:
                i = 10
        return cupy.left_shift(self, other)

    def __rlshift__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return cupy.left_shift(other, self)

    def __rshift__(self, other):
        if False:
            i = 10
            return i + 15
        return cupy.right_shift(self, other)

    def __rrshift__(self, other):
        if False:
            i = 10
            return i + 15
        return cupy.right_shift(other, self)

    def __invert__(self):
        if False:
            for i in range(10):
                print('nop')
        return cupy.invert(self)

    def __and__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return cupy.bitwise_and(self, other)

    def __rand__(self, other):
        if False:
            return 10
        return cupy.bitwise_and(other, self)

    def __or__(self, other):
        if False:
            return 10
        return cupy.bitwise_or(self, other)

    def __ror__(self, other):
        if False:
            print('Hello World!')
        return cupy.bitwise_or(other, self)

    def __xor__(self, other):
        if False:
            i = 10
            return i + 15
        return cupy.bitwise_xor(self, other)

    def __rxor__(self, other):
        if False:
            return 10
        return cupy.bitwise_xor(other, self)

    def __lt__(self, other):
        if False:
            print('Hello World!')
        return cupy.less(self, other)

    def __le__(self, other):
        if False:
            print('Hello World!')
        return cupy.less_equal(self, other)

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return cupy.equal(self, other)

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return cupy.not_equal(self, other)

    def __ge__(self, other):
        if False:
            while True:
                i = 10
        return cupy.greater_equal(self, other)

    def __gt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return cupy.greater(self, other)

    def copy(self):
        if False:
            i = 10
            return i + 15
        return cupy.copy(self)

    def astype(self, dtype, order=None, casting=None, subok=None, copy=True):
        if False:
            return 10
        dtype = get_dtype(dtype)
        if order is not None:
            raise TypeError('order is not supported yet')
        if casting is not None:
            raise TypeError('casting is not supported yet')
        if subok is not None:
            raise TypeError('subok is not supported yet')
        if not copy and self.dtype == dtype:
            return self
        if _dtype_to_astype_dict is None:
            _set_dtype_to_astype_dict()
        return _dtype_to_astype_dict[dtype](self)

    def sum(self, axis=None, dtype=None, out=None, keepdims=False):
        if False:
            while True:
                i = 10
        return cupy.sum(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def prod(self, axis=None, dtype=None, out=None, keepdims=False):
        if False:
            i = 10
            return i + 15
        return cupy.prod(self, axis=axis, dtype=dtype, out=out, keepdims=keepdims)

    def max(self, axis=None, out=None, keepdims=False):
        if False:
            print('Hello World!')
        return cupy.max(self, axis=axis, out=out, keepdims=keepdims)

    def min(self, axis=None, out=None, keepdims=False):
        if False:
            print('Hello World!')
        return cupy.min(self, axis=axis, out=out, keepdims=keepdims)

    def all(self, axis=None, out=None, keepdims=False):
        if False:
            for i in range(10):
                print('nop')
        return cupy.all(self, axis=axis, out=out, keepdims=keepdims)

    def any(self, axis=None, out=None, keepdims=False):
        if False:
            print('Hello World!')
        return cupy.any(self, axis=axis, out=out, keepdims=keepdims)

    @property
    def dtype(self):
        if False:
            i = 10
            return i + 15
        return self.content.dtype

    @property
    def ndim(self):
        if False:
            print('Hello World!')
        return self.content.ndim

    @property
    def shape(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError('`shape` is not supported, currently.')

    def __bool__(self):
        if False:
            while True:
                i = 10
        raise TypeError('Cannot convert to Python scalar in cupy.fuse')

    def __int__(self):
        if False:
            i = 10
            return i + 15
        raise TypeError('Cannot convert to Python scalar in cupy.fuse')

    def __float__(self):
        if False:
            i = 10
            return i + 15
        raise TypeError('Cannot convert to Python scalar in cupy.fuse')

    def __complex__(self):
        if False:
            print('Hello World!')
        raise TypeError('Cannot convert to Python scalar in cupy.fuse')

class _ScalarProxy(_VariableProxy):
    """An abstracted scalar object passed to the target function.

    Attributes:
        dtype(dtype): The dtype of the array.
        imag(_ArrayProxy): The imaginary part of the array (Not implemented)
        real(_ArrayProxy): The real part of the array (Not implemented)
        ndim(int): The number of dimensions of the array.
    """

    def __repr__(self):
        if False:
            return 10
        return '_ScalarProxy({}, dtype={})'.format(self._emit_param_name(), self.dtype)

class _ArrayProxy(_VariableProxy):
    """An abstracted array object passed to the target function.

    Attributes:
        dtype(dtype): The dtype of the array.
        imag(_ArrayProxy): The imaginary part of the array (Not implemented)
        real(_ArrayProxy): The real part of the array (Not implemented)
        ndim(int): The number of dimensions of the array.
    """

    def __repr__(self):
        if False:
            while True:
                i = 10
        return "_ArrayProxy([...], dtype='{}', ndim={})".format(self.dtype.char, self.ndim)

    def _inplace_op(self, ufunc, other):
        if False:
            return 10
        return ufunc(self, other, self)

    def __iadd__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._inplace_op(cupy.add, other)

    def __isub__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._inplace_op(cupy.subtract, other)

    def __imul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._inplace_op(cupy.multiply, other)

    def __idiv__(self, other):
        if False:
            return 10
        return self._inplace_op(cupy.divide, other)

    def __itruediv__(self, other):
        if False:
            i = 10
            return i + 15
        return self._inplace_op(cupy.true_divide, other)

    def __ifloordiv__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._inplace_op(cupy.floor_divide, other)

    def __imod__(self, other):
        if False:
            i = 10
            return i + 15
        return self._inplace_op(cupy.remainder, other)

    def __ipow__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return self._inplace_op(cupy.power, other)

    def __ilshift__(self, other):
        if False:
            while True:
                i = 10
        return self._inplace_op(cupy.left_shift, other)

    def __irshift__(self, other):
        if False:
            return 10
        return self._inplace_op(cupy.right_shift, other)

    def __iand__(self, other):
        if False:
            while True:
                i = 10
        return self._inplace_op(cupy.bitwise_and, other)

    def __ior__(self, other):
        if False:
            while True:
                i = 10
        return self._inplace_op(cupy.bitwise_or, other)

    def __ixor__(self, other):
        if False:
            return 10
        return self._inplace_op(cupy.bitwise_xor, other)

    def __getitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        return _fusion_thread_local.call_indexing(self, index)

    def __setitem__(self, slices, value):
        if False:
            while True:
                i = 10
        if slices is Ellipsis or (isinstance(slices, slice) and slices == slice(None)):
            _fusion_thread_local.call_ufunc(core.elementwise_copy, value, out=self)
        else:
            raise ValueError('The fusion supports `[...]` or `[:]`.')