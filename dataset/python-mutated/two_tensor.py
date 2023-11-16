import torch
import torch.utils._pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing

class TwoTensor(torch.Tensor):

    @staticmethod
    def __new__(cls, a, b):
        if False:
            print('Hello World!')
        assert a.device == b.device and a.layout == b.layout and (a.requires_grad == b.requires_grad) and (a.dtype == b.dtype)
        shape = a.shape
        kwargs = {}
        kwargs['strides'] = a.stride()
        kwargs['storage_offset'] = a.storage_offset()
        kwargs['device'] = a.device
        kwargs['layout'] = a.layout
        kwargs['requires_grad'] = a.requires_grad
        kwargs['dtype'] = a.dtype
        out = torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
        assert a.shape == b.shape
        assert a.stride() == b.stride()
        assert a.storage_offset() == b.storage_offset()
        return out

    def __init__(self, a, b):
        if False:
            i = 10
            return i + 15
        self.a = a
        self.b = b

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        a_repr = repr(self.a)
        b_repr = repr(self.b)
        return f'TwoTensor({a_repr}, {b_repr})'

    def __tensor_flatten__(self):
        if False:
            i = 10
            return i + 15
        return (['a', 'b'], None)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta):
        if False:
            while True:
                i = 10
        assert meta is None
        (a, b) = (inner_tensors['a'], inner_tensors['b'])
        return TwoTensor(a, b)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        if False:
            while True:
                i = 10
        if kwargs is None:
            kwargs = {}
        args_a = pytree.tree_map_only(TwoTensor, lambda x: x.a, args)
        args_b = pytree.tree_map_only(TwoTensor, lambda x: x.b, args)
        kwargs_a = pytree.tree_map_only(TwoTensor, lambda x: x.a, kwargs)
        kwargs_b = pytree.tree_map_only(TwoTensor, lambda x: x.b, kwargs)
        out_a = func(*args_a, **kwargs_a)
        out_b = func(*args_b, **kwargs_b)
        assert type(out_a) == type(out_b)
        (out_a_flat, spec) = pytree.tree_flatten(out_a)
        out_b_flat = pytree.tree_leaves(out_b)
        out_flat = [TwoTensor(o_a, o_b) if isinstance(o_a, torch.Tensor) else o_a for (o_a, o_b) in zip(out_a_flat, out_b_flat)]
        out = pytree.tree_unflatten(out_flat, spec)
        return return_and_correct_aliasing(func, args, kwargs, out)

class TwoTensorMode(torch.utils._python_dispatch.TorchDispatchMode):

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if False:
            while True:
                i = 10
        out = func(*args, **kwargs)
        if torch._subclasses.fake_tensor._is_tensor_constructor(func):
            out = TwoTensor(out, out.clone())
        return out