from __future__ import annotations
import builtins
import math
import operator
from typing import Sequence
import torch
from . import _dtypes, _dtypes_impl, _funcs, _ufuncs, _util
from ._normalizations import ArrayLike, normalize_array_like, normalizer, NotImplementedType
newaxis = None
FLAGS = ['C_CONTIGUOUS', 'F_CONTIGUOUS', 'OWNDATA', 'WRITEABLE', 'ALIGNED', 'WRITEBACKIFCOPY', 'FNC', 'FORC', 'BEHAVED', 'CARRAY', 'FARRAY']
SHORTHAND_TO_FLAGS = {'C': 'C_CONTIGUOUS', 'F': 'F_CONTIGUOUS', 'O': 'OWNDATA', 'W': 'WRITEABLE', 'A': 'ALIGNED', 'X': 'WRITEBACKIFCOPY', 'B': 'BEHAVED', 'CA': 'CARRAY', 'FA': 'FARRAY'}

class Flags:

    def __init__(self, flag_to_value: dict):
        if False:
            i = 10
            return i + 15
        assert all((k in FLAGS for k in flag_to_value.keys()))
        self._flag_to_value = flag_to_value

    def __getattr__(self, attr: str):
        if False:
            print('Hello World!')
        if attr.islower() and attr.upper() in FLAGS:
            return self[attr.upper()]
        else:
            raise AttributeError(f"No flag attribute '{attr}'")

    def __getitem__(self, key):
        if False:
            i = 10
            return i + 15
        if key in SHORTHAND_TO_FLAGS.keys():
            key = SHORTHAND_TO_FLAGS[key]
        if key in FLAGS:
            try:
                return self._flag_to_value[key]
            except KeyError as e:
                raise NotImplementedError(f'key={key!r}') from e
        else:
            raise KeyError(f"No flag key '{key}'")

    def __setattr__(self, attr, value):
        if False:
            i = 10
            return i + 15
        if attr.islower() and attr.upper() in FLAGS:
            self[attr.upper()] = value
        else:
            super().__setattr__(attr, value)

    def __setitem__(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        if key in FLAGS or key in SHORTHAND_TO_FLAGS.keys():
            raise NotImplementedError('Modifying flags is not implemented')
        else:
            raise KeyError(f"No flag key '{key}'")

def create_method(fn, name=None):
    if False:
        while True:
            i = 10
    name = name or fn.__name__

    def f(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return fn(*args, **kwargs)
    f.__name__ = name
    f.__qualname__ = f'ndarray.{name}'
    return f
methods = {'clip': None, 'nonzero': None, 'repeat': None, 'round': None, 'squeeze': None, 'swapaxes': None, 'ravel': None, 'diagonal': None, 'dot': None, 'trace': None, 'argsort': None, 'searchsorted': None, 'argmax': None, 'argmin': None, 'any': None, 'all': None, 'max': None, 'min': None, 'ptp': None, 'sum': None, 'prod': None, 'mean': None, 'var': None, 'std': None, 'cumsum': None, 'cumprod': None, 'take': None, 'choose': None}
dunder = {'abs': 'absolute', 'invert': None, 'pos': 'positive', 'neg': 'negative', 'gt': 'greater', 'lt': 'less', 'ge': 'greater_equal', 'le': 'less_equal'}
ri_dunder = {'add': None, 'sub': 'subtract', 'mul': 'multiply', 'truediv': 'divide', 'floordiv': 'floor_divide', 'pow': 'power', 'mod': 'remainder', 'and': 'bitwise_and', 'or': 'bitwise_or', 'xor': 'bitwise_xor', 'lshift': 'left_shift', 'rshift': 'right_shift', 'matmul': None}

def _upcast_int_indices(index):
    if False:
        i = 10
        return i + 15
    if isinstance(index, torch.Tensor):
        if index.dtype in (torch.int8, torch.int16, torch.int32, torch.uint8):
            return index.to(torch.int64)
    elif isinstance(index, tuple):
        return tuple((_upcast_int_indices(i) for i in index))
    return index

class ndarray:

    def __init__(self, t=None):
        if False:
            i = 10
            return i + 15
        if t is None:
            self.tensor = torch.Tensor()
        elif isinstance(t, torch.Tensor):
            self.tensor = t
        else:
            raise ValueError('ndarray constructor is not recommended; prefereither array(...) or zeros/empty(...)')
    for (method, name) in methods.items():
        fn = getattr(_funcs, name or method)
        vars()[method] = create_method(fn, method)
    conj = create_method(_ufuncs.conjugate, 'conj')
    conjugate = create_method(_ufuncs.conjugate)
    for (method, name) in dunder.items():
        fn = getattr(_ufuncs, name or method)
        method = f'__{method}__'
        vars()[method] = create_method(fn, method)
    for (method, name) in ri_dunder.items():
        fn = getattr(_ufuncs, name or method)
        plain = f'__{method}__'
        vars()[plain] = create_method(fn, plain)
        rvar = f'__r{method}__'
        vars()[rvar] = create_method(lambda self, other, fn=fn: fn(other, self), rvar)
        ivar = f'__i{method}__'
        vars()[ivar] = create_method(lambda self, other, fn=fn: fn(self, other, out=self), ivar)
    __divmod__ = create_method(_ufuncs.divmod, '__divmod__')
    __rdivmod__ = create_method(lambda self, other: _ufuncs.divmod(other, self), '__rdivmod__')
    del ivar, rvar, name, plain, fn, method

    @property
    def shape(self):
        if False:
            print('Hello World!')
        return tuple(self.tensor.shape)

    @property
    def size(self):
        if False:
            while True:
                i = 10
        return self.tensor.numel()

    @property
    def ndim(self):
        if False:
            return 10
        return self.tensor.ndim

    @property
    def dtype(self):
        if False:
            print('Hello World!')
        return _dtypes.dtype(self.tensor.dtype)

    @property
    def strides(self):
        if False:
            for i in range(10):
                print('nop')
        elsize = self.tensor.element_size()
        return tuple((stride * elsize for stride in self.tensor.stride()))

    @property
    def itemsize(self):
        if False:
            return 10
        return self.tensor.element_size()

    @property
    def flags(self):
        if False:
            return 10
        return Flags({'C_CONTIGUOUS': self.tensor.is_contiguous(), 'F_CONTIGUOUS': self.T.tensor.is_contiguous(), 'OWNDATA': self.tensor._base is None, 'WRITEABLE': True})

    @property
    def data(self):
        if False:
            i = 10
            return i + 15
        return self.tensor.data_ptr()

    @property
    def nbytes(self):
        if False:
            return 10
        return self.tensor.storage().nbytes()

    @property
    def T(self):
        if False:
            print('Hello World!')
        return self.transpose()

    @property
    def real(self):
        if False:
            for i in range(10):
                print('nop')
        return _funcs.real(self)

    @real.setter
    def real(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.tensor.real = asarray(value).tensor

    @property
    def imag(self):
        if False:
            print('Hello World!')
        return _funcs.imag(self)

    @imag.setter
    def imag(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.tensor.imag = asarray(value).tensor

    def astype(self, dtype):
        if False:
            while True:
                i = 10
        torch_dtype = _dtypes.dtype(dtype).torch_dtype
        t = self.tensor.to(torch_dtype)
        return ndarray(t)

    @normalizer
    def copy(self: ArrayLike, order: NotImplementedType='C'):
        if False:
            return 10
        return self.clone()

    @normalizer
    def flatten(self: ArrayLike, order: NotImplementedType='C'):
        if False:
            for i in range(10):
                print('nop')
        return torch.flatten(self)

    def resize(self, *new_shape, refcheck=False):
        if False:
            return 10
        a = self.tensor
        if refcheck:
            raise NotImplementedError(f'resize(..., refcheck={refcheck} is not implemented.')
        if new_shape in [(), (None,)]:
            return
        if len(new_shape) == 1:
            new_shape = new_shape[0]
        if isinstance(new_shape, int):
            new_shape = (new_shape,)
        a = a.flatten()
        if builtins.any((x < 0 for x in new_shape)):
            raise ValueError('all elements of `new_shape` must be non-negative')
        new_numel = math.prod(new_shape)
        if new_numel < a.numel():
            ret = a[:new_numel].reshape(new_shape)
        else:
            b = torch.zeros(new_numel)
            b[:a.numel()] = a
            ret = b.reshape(new_shape)
        self.tensor = ret

    def view(self, dtype):
        if False:
            return 10
        torch_dtype = _dtypes.dtype(dtype).torch_dtype
        tview = self.tensor.view(torch_dtype)
        return ndarray(tview)

    @normalizer
    def fill(self, value: ArrayLike):
        if False:
            return 10
        self.tensor.fill_(value)

    def tolist(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tensor.tolist()

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return (ndarray(x) for x in self.tensor.__iter__())

    def __str__(self):
        if False:
            while True:
                i = 10
        return str(self.tensor).replace('tensor', 'torch.ndarray').replace('dtype=torch.', 'dtype=')
    __repr__ = create_method(__str__)

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        try:
            return _ufuncs.equal(self, other)
        except (RuntimeError, TypeError):
            falsy = torch.full(self.shape, fill_value=False, dtype=bool)
            return asarray(falsy)

    def __ne__(self, other):
        if False:
            return 10
        return ~(self == other)

    def __index__(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return operator.index(self.tensor.item())
        except Exception as exc:
            raise TypeError('only integer scalar arrays can be converted to a scalar index') from exc

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        return bool(self.tensor)

    def __int__(self):
        if False:
            for i in range(10):
                print('nop')
        return int(self.tensor)

    def __float__(self):
        if False:
            return 10
        return float(self.tensor)

    def __complex__(self):
        if False:
            print('Hello World!')
        return complex(self.tensor)

    def is_integer(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            v = self.tensor.item()
            result = int(v) == v
        except Exception:
            result = False
        return result

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.tensor.shape[0]

    def __contains__(self, x):
        if False:
            print('Hello World!')
        return self.tensor.__contains__(x)

    def transpose(self, *axes):
        if False:
            while True:
                i = 10
        return _funcs.transpose(self, axes)

    def reshape(self, *shape, order='C'):
        if False:
            return 10
        return _funcs.reshape(self, shape, order=order)

    def sort(self, axis=-1, kind=None, order=None):
        if False:
            print('Hello World!')
        _funcs.copyto(self, _funcs.sort(self, axis, kind, order))

    def item(self, *args):
        if False:
            while True:
                i = 10
        if args == ():
            return self.tensor.item()
        elif len(args) == 1:
            return self.ravel()[args[0]]
        else:
            return self.__getitem__(args)

    def __getitem__(self, index):
        if False:
            while True:
                i = 10
        tensor = self.tensor

        def neg_step(i, s):
            if False:
                while True:
                    i = 10
            if not (isinstance(s, slice) and s.step is not None and (s.step < 0)):
                return s
            nonlocal tensor
            tensor = torch.flip(tensor, (i,))
            assert isinstance(s.start, int) or s.start is None
            assert isinstance(s.stop, int) or s.stop is None
            start = s.stop + 1 if s.stop else None
            stop = s.start + 1 if s.start else None
            return slice(start, stop, -s.step)
        if isinstance(index, Sequence):
            index = type(index)((neg_step(i, s) for (i, s) in enumerate(index)))
        else:
            index = neg_step(0, index)
        index = _util.ndarrays_to_tensors(index)
        index = _upcast_int_indices(index)
        return ndarray(tensor.__getitem__(index))

    def __setitem__(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        index = _util.ndarrays_to_tensors(index)
        index = _upcast_int_indices(index)
        if not _dtypes_impl.is_scalar(value):
            value = normalize_array_like(value)
            value = _util.cast_if_needed(value, self.tensor.dtype)
        return self.tensor.__setitem__(index, value)
    take = _funcs.take
    put = _funcs.put

    def __dlpack__(self, *, stream=None):
        if False:
            print('Hello World!')
        return self.tensor.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        if False:
            return 10
        return self.tensor.__dlpack_device__()

def _tolist(obj):
    if False:
        while True:
            i = 10
    'Recursively convert tensors into lists.'
    a1 = []
    for elem in obj:
        if isinstance(elem, (list, tuple)):
            elem = _tolist(elem)
        if isinstance(elem, ndarray):
            a1.append(elem.tensor.tolist())
        else:
            a1.append(elem)
    return a1

def array(obj, dtype=None, *, copy=True, order='K', subok=False, ndmin=0, like=None):
    if False:
        for i in range(10):
            print('nop')
    if subok is not False:
        raise NotImplementedError("'subok' parameter is not supported.")
    if like is not None:
        raise NotImplementedError("'like' parameter is not supported.")
    if order != 'K':
        raise NotImplementedError()
    if isinstance(obj, ndarray) and copy is False and (dtype is None) and (ndmin <= obj.ndim):
        return obj
    if isinstance(obj, (list, tuple)):
        if obj and all((isinstance(x, torch.Tensor) for x in obj)):
            obj = torch.stack(obj)
        else:
            obj = _tolist(obj)
    if isinstance(obj, ndarray):
        obj = obj.tensor
    torch_dtype = None
    if dtype is not None:
        torch_dtype = _dtypes.dtype(dtype).torch_dtype
    tensor = _util._coerce_to_tensor(obj, torch_dtype, copy, ndmin)
    return ndarray(tensor)

def asarray(a, dtype=None, order='K', *, like=None):
    if False:
        return 10
    return array(a, dtype=dtype, order=order, like=like, copy=False, ndmin=0)

def ascontiguousarray(a, dtype=None, *, like=None):
    if False:
        for i in range(10):
            print('nop')
    arr = asarray(a, dtype=dtype, like=like)
    if not arr.tensor.is_contiguous():
        arr.tensor = arr.tensor.contiguous()
    return arr

def from_dlpack(x, /):
    if False:
        print('Hello World!')
    t = torch.from_dlpack(x)
    return ndarray(t)

def _extract_dtype(entry):
    if False:
        return 10
    try:
        dty = _dtypes.dtype(entry)
    except Exception:
        dty = asarray(entry).dtype
    return dty

def can_cast(from_, to, casting='safe'):
    if False:
        return 10
    from_ = _extract_dtype(from_)
    to_ = _extract_dtype(to)
    return _dtypes_impl.can_cast_impl(from_.torch_dtype, to_.torch_dtype, casting)

def result_type(*arrays_and_dtypes):
    if False:
        print('Hello World!')
    tensors = []
    for entry in arrays_and_dtypes:
        try:
            t = asarray(entry).tensor
        except (RuntimeError, ValueError, TypeError):
            dty = _dtypes.dtype(entry)
            t = torch.empty(1, dtype=dty.torch_dtype)
        tensors.append(t)
    torch_dtype = _dtypes_impl.result_type_impl(*tensors)
    return _dtypes.dtype(torch_dtype)