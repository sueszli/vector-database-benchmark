from typing import Tuple
import torch
from torch._C import DispatchKey, DispatchKeySet
from torch._prims_common import is_expandable_to
from torch.fx.experimental.symbolic_shapes import has_free_symbols
from torch.utils.weak import WeakTensorKeyDictionary
from typing import *
_tensor_id_counter = 0
_tensor_symint_registry = WeakTensorKeyDictionary()

def get_tensor_symint(tensor, *, coeff=1):
    if False:
        return 10
    global _tensor_id_counter
    if tensor not in _tensor_symint_registry:
        _tensor_symint_registry[tensor] = torch._C._get_singleton_int(_tensor_id_counter, coeff)
        _tensor_id_counter += 1
    return _tensor_symint_registry[tensor]

class NestedTensor(torch.Tensor):
    _values: torch.Tensor
    _offsets: torch.Tensor
    _lengths: Optional[torch.Tensor]
    _size: Tuple[int, ...]
    _stride: Tuple[int, ...]
    _ragged_idx: int

    @staticmethod
    def __new__(cls, values, offsets, *, lengths=None, **kwargs):
        if False:
            while True:
                i = 10
        ks = DispatchKeySet(DispatchKey.NestedTensor)
        ks = ks.add(DispatchKey.AutogradNestedTensor)
        r = torch.Tensor._make_wrapper_subclass(cls, (0,), (0,), 0, torch.contiguous_format, values.dtype, torch.jagged, values.device, False, kwargs.get('requires_grad', False), 'sizes', False, True, ks)
        return r

    def __init__(self, values, offsets, *, lengths=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        assert offsets is not None
        assert offsets.ndim == 1
        assert not isinstance(values, NestedTensor)
        ragged_source = offsets if lengths is None else lengths
        ragged_size = get_tensor_symint(ragged_source, coeff=1)
        B = offsets.shape[0] - 1
        Ds = values.shape[1:]
        self._size = (B, ragged_size, *Ds)
        stride = values.stride()
        self._strides = (ragged_size * stride[0], *stride)
        self._ragged_idx = 1
        if values.requires_grad:
            raise ValueError('NestedTensor values cannot require grad, please detach before passing to NestedTensor constructor')
        self._values = values
        self._offsets = offsets
        self._lengths = lengths

    def values(self):
        if False:
            i = 10
            return i + 15
        return self._values

    def offsets(self):
        if False:
            i = 10
            return i + 15
        return self._offsets

    def lengths(self):
        if False:
            i = 10
            return i + 15
        return self._lengths

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        grad_fn_str = f', requires_grad={self.requires_grad}' if self.requires_grad else ''
        if self.grad_fn:
            grad_fn_str = f', grad_fn={self.grad_fn}'
        return f'NestedTensor(size={self._size}, offsets={self._offsets}{grad_fn_str}, contiguous={self._lengths is None})'

    def __reduce_ex__(self, proto):
        if False:
            print('Hello World!')
        state = torch._utils._get_obj_state(self)
        assert '_size' in state and '_strides' in state
        state = dict(state)
        del state['_size']
        del state['_strides']
        func = NestedTensor
        args = (self._values, self._offsets)
        return (torch._tensor._rebuild_from_type_v2, (func, type(self), args, state))

    def __tensor_flatten__(self):
        if False:
            i = 10
            return i + 15
        ctx = {'requires_grad': self.requires_grad, 'ragged_size': self._size[self._ragged_idx]}
        inner_tensors = ['_values', '_offsets']
        if self._lengths is not None:
            inner_tensors.append('_lengths')
        return (inner_tensors, ctx)

    @staticmethod
    def __tensor_unflatten__(inner_tensors: Dict, meta):
        if False:
            return 10
        assert len(inner_tensors) >= 2 and len(inner_tensors) <= 3
        values = inner_tensors['_values']
        offsets = inner_tensors['_offsets']
        lengths = inner_tensors.get('_lengths', None)
        ragged_source = offsets if lengths is None else lengths
        if has_free_symbols(ragged_source) or has_free_symbols(values):
            _tensor_symint_registry[ragged_source] = meta['ragged_size']
        return NestedTensor(values, offsets=offsets, lengths=lengths, requires_grad=meta['requires_grad'])

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if False:
            i = 10
            return i + 15
        kwargs = {} if kwargs is None else kwargs
        from .ops import lookup_jagged
        fn = lookup_jagged(func, *args, **kwargs)
        if fn is not None:
            return fn(*args, **kwargs)
        raise NotImplementedError(func)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if False:
            i = 10
            return i + 15
        if kwargs is None:
            kwargs = {}
        from .ops import jagged_torch_function
        try:
            return jagged_torch_function(func, *args, **kwargs)
        except NotImplementedError:
            pass
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

class ViewBufferFromNested(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: NestedTensor):
        if False:
            while True:
                i = 10
        ctx.save_for_backward(x.offsets())
        return x.values()

    @staticmethod
    def backward(ctx, gO: torch.Tensor):
        if False:
            i = 10
            return i + 15
        (offsets,) = ctx.saved_tensors
        return NestedTensor(gO, offsets=offsets)

class ViewNestedFromBuffer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, values: torch.Tensor, offsets: torch.Tensor):
        if False:
            print('Hello World!')
        return NestedTensor(values.detach(), offsets=offsets)

    @staticmethod
    def backward(ctx, gO: NestedTensor):
        if False:
            print('Hello World!')
        return (gO.values(), None, None)

class ViewNonContiguousNestedFromBuffer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, values: torch.Tensor, offsets: torch.Tensor, lengths: torch.Tensor):
        if False:
            print('Hello World!')
        return NestedTensor(values.detach(), offsets=offsets, lengths=lengths)

    @staticmethod
    def backward(ctx, gO: NestedTensor):
        if False:
            return 10
        return (gO.values(), None, None)

def jagged_from_list(tensors: List[torch.Tensor], offsets: Optional[torch.Tensor], dtype=None, device=None) -> Tuple[NestedTensor, torch.Tensor]:
    if False:
        print('Hello World!')
    'Constructs a NestedTensor backed by jagged layout from a list of tensors'
    if not len(set((t.dtype for t in tensors))) == 1:
        raise RuntimeError('When constructing a nested tensor, all tensors in list must have the same dtype')
    if not len(set((t.device for t in tensors))) == 1:
        raise RuntimeError('When constructing a nested tensor, all tensors in list must be on the same device')
    sizes = [t.shape for t in tensors]
    non_first_sizes = [s[1:] for s in sizes]
    at_most_first_ragged = all((s == non_first_sizes[0] for s in non_first_sizes))
    if not at_most_first_ragged:
        raise RuntimeError('Cannot represent given tensor list as a nested tensor with the jagged layout. Note that the jagged layout only represents shapes of the form (B, *, D_0, D_1, ..., D_N), with only * allowed to be ragged.')
    values = torch.cat(tensors, dim=0)
    to_kwargs = {}
    if device is not None:
        to_kwargs['device'] = device
    if dtype is not None:
        to_kwargs['dtype'] = dtype
    values = values.to(**to_kwargs)
    if offsets is None:
        offsets = torch.cat([torch.zeros(1, dtype=torch.int64, device=values.device), torch.tensor([s[0] for s in sizes], device=values.device).cumsum(dim=0)])
    return (ViewNestedFromBuffer.apply(values, offsets), offsets)

def jagged_from_tensor_and_lengths(tensor: torch.Tensor, starts: torch.Tensor, lengths: torch.Tensor) -> Tuple[NestedTensor, torch.Tensor, Optional[torch.Tensor]]:
    if False:
        while True:
            i = 10
    'Constructs a NestedTensor backed by jagged layout from a tensor, starts of sequences, and sequence lengths'
    batch_size = tensor.shape[0]
    if is_expandable_to(starts.shape, (batch_size,)) and is_expandable_to(lengths.shape, (batch_size,)):
        start_list = starts.expand(batch_size)
        length_list = lengths.expand(batch_size)
    else:
        raise RuntimeError('When constructing a jagged nested tensor using narrow(), your start and length must be Tensors that broadcast to input.shape[0]')
    assert len(tensor.shape) >= 2, 'tensor must at least be 2D for the nested narrow op to work'
    max_seq_len = tensor.shape[1]
    offset_lengths = max_seq_len * torch.arange(0, batch_size, dtype=torch.int64, device=tensor.device)
    offsets = torch.cat([start_list + offset_lengths, (start_list[-1] + offset_lengths[-1] + length_list[-1]).unsqueeze(0)])
    if len(tensor.shape) > 2:
        values = tensor.view(-1, *tensor.shape[2:])
    else:
        values = tensor.view(-1)
    is_contiguous = True
    orig_dim = tensor.shape[1]
    if torch.any(length_list[1:-1].ne(orig_dim)):
        is_contiguous = False
    if torch.any(offsets[1:-2].diff().ne(orig_dim)):
        is_contiguous = False
    if offsets[0] + length_list[0] != orig_dim:
        is_contiguous = False
    if is_contiguous:
        return (ViewNestedFromBuffer.apply(values[offsets[0]:offsets[-1]], offsets - offsets[0]), offsets, None)
    return (ViewNonContiguousNestedFromBuffer.apply(values, offsets, length_list), offsets, length_list)

def buffer_from_jagged(jagged):
    if False:
        i = 10
        return i + 15
    return ViewBufferFromNested.apply(jagged)