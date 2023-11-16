import torch
from copy import deepcopy
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import LoggingTensor

class WrapperTensor(torch.Tensor):

    @staticmethod
    def __new__(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        (t, kwargs) = cls.get_wrapper_properties(*args, **kwargs)
        if 'size' not in kwargs:
            size = t.size()
        else:
            size = kwargs['size']
            del kwargs['size']
        if 'dtype' not in kwargs:
            kwargs['dtype'] = t.dtype
        if 'layout' not in kwargs:
            kwargs['layout'] = t.layout
        if 'device' not in kwargs:
            kwargs['device'] = t.device
        if 'requires_grad' not in kwargs:
            kwargs['requires_grad'] = False
        wrapper = torch.Tensor._make_wrapper_subclass(cls, size, **kwargs)
        wrapper._validate_methods()
        return wrapper

    @classmethod
    def get_wrapper_properties(cls, *args, **kwargs):
        if False:
            print('Hello World!')
        raise NotImplementedError('You need to implement get_wrapper_properties')

    def _validate_methods(self):
        if False:
            i = 10
            return i + 15
        forbidden_overrides = ['size', 'stride', 'dtype', 'layout', 'device', 'requires_grad']
        for el in forbidden_overrides:
            if getattr(self.__class__, el) is not getattr(torch.Tensor, el):
                raise RuntimeError(f'Subclass {self.__class__.__name__} is overwriting the property {el} but this is not allowed as such change would not be reflected to c++ callers.')

class DiagTensorBelow(WrapperTensor):

    @classmethod
    def get_wrapper_properties(cls, diag, requires_grad=False):
        if False:
            print('Hello World!')
        assert diag.ndim == 1
        return (diag, {'size': diag.size() + diag.size(), 'requires_grad': requires_grad})

    def __init__(self, diag, requires_grad=False):
        if False:
            while True:
                i = 10
        self.diag = diag
    handled_ops = {}
    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if False:
            return 10
        if not all((issubclass(cls, t) for t in types)):
            return NotImplemented
        fn = cls.handled_ops.get(func.__name__, None)
        if fn:
            return fn(*args, **kwargs or {})
        else:

            def unwrap(e):
                if False:
                    i = 10
                    return i + 15
                return e.diag.diag() if isinstance(e, DiagTensorBelow) else e

            def wrap(e):
                if False:
                    while True:
                        i = 10
                if isinstance(e, torch.Tensor) and e.ndim == 1:
                    return DiagTensorBelow(e)
                if isinstance(e, torch.Tensor) and e.ndim == 2 and (e.count_nonzero() == e.diag().count_nonzero()):
                    return DiagTensorBelow(e.diag())
                return e
            rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})))
            return rs

    def __repr__(self):
        if False:
            print('Hello World!')
        return super().__repr__(tensor_contents=f'diag={self.diag}')

class SparseTensor(WrapperTensor):

    @classmethod
    def get_wrapper_properties(cls, size, values, indices, requires_grad=False):
        if False:
            while True:
                i = 10
        assert values.device == indices.device
        return (values, {'size': size, 'requires_grad': requires_grad})

    def __init__(self, size, values, indices, requires_grad=False):
        if False:
            while True:
                i = 10
        self.values = values
        self.indices = indices

    def __repr__(self):
        if False:
            print('Hello World!')
        return super().__repr__(tensor_contents=f'values={self.values}, indices={self.indices}')

    def sparse_to_dense(self):
        if False:
            for i in range(10):
                print('nop')
        res = torch.zeros(self.size(), dtype=self.values.dtype)
        res[self.indices.unbind(1)] = self.values
        return res

    @staticmethod
    def from_dense(t):
        if False:
            return 10
        indices = t.nonzero()
        values = t[indices.unbind(1)]
        return SparseTensor(t.size(), values, indices)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if False:
            while True:
                i = 10
        func_name = f'{func.__module__}.{func.__name__}'
        res = cls._try_call_special_impl(func_name, args, kwargs)
        if res is not NotImplemented:
            return res

        def unwrap(e):
            if False:
                for i in range(10):
                    print('nop')
            return e.sparse_to_dense() if isinstance(e, SparseTensor) else e

        def wrap(e):
            if False:
                return 10
            return SparseTensor.from_dense(e) if isinstance(e, torch.Tensor) else e
        rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})))
        return rs

    def __rmul__(self, other):
        if False:
            while True:
                i = 10
        return super().__rmul__(other)
    _SPECIAL_IMPLS = {}

    @classmethod
    def _try_call_special_impl(cls, func, args, kwargs):
        if False:
            return 10
        if func not in cls._SPECIAL_IMPLS:
            return NotImplemented
        return cls._SPECIAL_IMPLS[func](args, kwargs)

class NonWrapperTensor(torch.Tensor):

    def __new__(cls, data):
        if False:
            print('Hello World!')
        t = torch.Tensor._make_subclass(cls, data)
        t.extra_state = {'last_func_called': None}
        return t

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if False:
            print('Hello World!')
        result = super().__torch_function__(func, types, args, kwargs)
        if isinstance(result, cls):
            if func is torch.Tensor.__deepcopy__:
                result.extra_state = deepcopy(args[0].extra_state)
            else:
                result.extra_state = {'last_func_called': func.__name__}
        return result

    def new_empty(self, shape):
        if False:
            i = 10
            return i + 15
        return type(self)(torch.empty(shape))

class SubclassInfo:
    __slots__ = ['name', 'create_fn', 'closed_under_ops']

    def __init__(self, name, create_fn, closed_under_ops=True):
        if False:
            while True:
                i = 10
        self.name = name
        self.create_fn = create_fn
        self.closed_under_ops = closed_under_ops
subclass_db = {torch.Tensor: SubclassInfo('base_tensor', create_fn=torch.randn), NonWrapperTensor: SubclassInfo('non_wrapper_tensor', create_fn=lambda shape: NonWrapperTensor(torch.randn(shape))), LoggingTensor: SubclassInfo('logging_tensor', create_fn=lambda shape: LoggingTensor(torch.randn(shape))), SparseTensor: SubclassInfo('sparse_tensor', create_fn=lambda shape: SparseTensor.from_dense(torch.randn(shape).relu())), DiagTensorBelow: SubclassInfo('diag_tensor_below', create_fn=lambda shape: DiagTensorBelow(torch.randn(shape)), closed_under_ops=False)}