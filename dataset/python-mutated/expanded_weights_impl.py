from contextlib import contextmanager
import torch
import functools
from torch._decomp import decomposition_table
from typing import Callable, Dict
from torch.utils._pytree import tree_map_only
HANDLED_FUNCTIONS: Dict[Callable, torch.autograd.Function] = {}
aten = torch._ops.ops.aten
expanded_weights_rnn_decomps = {torch.rnn_relu: (decomposition_table[aten.rnn_relu.input], decomposition_table[aten.rnn_relu.data]), torch.rnn_tanh: (decomposition_table[aten.rnn_tanh.input], decomposition_table[aten.rnn_tanh.data]), torch.lstm: (decomposition_table[aten.lstm.input], decomposition_table[aten.lstm.data]), torch.gru: (decomposition_table[aten.gru.input], decomposition_table[aten.gru.data])}

@contextmanager
def batch_second(args, kwargs):
    if False:
        while True:
            i = 10

    def set_batch_second(ew):
        if False:
            while True:
                i = 10
        ew.set_batch_first(False)

    def reset_batch_first(ew):
        if False:
            print('Hello World!')
        ew.set_batch_first(True)
    tree_map_only(ExpandedWeight, set_batch_second, args)
    tree_map_only(ExpandedWeight, set_batch_second, kwargs)
    try:
        yield
    finally:
        tree_map_only(ExpandedWeight, reset_batch_first, args)
        tree_map_only(ExpandedWeight, reset_batch_first, kwargs)

@contextmanager
def allow_smaller_batches(args, kwargs):
    if False:
        i = 10
        return i + 15

    def allow(ew):
        if False:
            print('Hello World!')
        ew.set_allow_smaller_batches(True)

    def reset(ew):
        if False:
            while True:
                i = 10
        ew.set_allow_smaller_batches(False)
    tree_map_only(ExpandedWeight, allow, args)
    tree_map_only(ExpandedWeight, allow, kwargs)
    try:
        yield
    finally:
        tree_map_only(ExpandedWeight, reset, args)
        tree_map_only(ExpandedWeight, reset, kwargs)

@contextmanager
def setup_rnn(use_input_variant, args, kwargs):
    if False:
        while True:
            i = 10
    with batch_second(args, kwargs) if use_input_variant else allow_smaller_batches(args, kwargs):
        yield

def implements_per_sample_grads(torch_function):
    if False:
        print('Hello World!')

    @functools.wraps(torch_function)
    def decorator(autograd_func):
        if False:
            print('Hello World!')
        HANDLED_FUNCTIONS[torch_function] = autograd_func
        return autograd_func
    return decorator

class ExpandedWeight(torch.Tensor):

    def __init__(self, orig_weight, batch_size, loss_reduction):
        if False:
            i = 10
            return i + 15
        self.batch_size = batch_size
        self.batch_first = True
        self.allow_smaller_batches = False
        self.orig_weight = orig_weight
        self.loss_reduction = loss_reduction
    handled_functions = HANDLED_FUNCTIONS

    def __new__(cls, orig_weight, batch_size, loss_reduction):
        if False:
            return 10
        if not isinstance(orig_weight, torch.Tensor):
            raise RuntimeError(f'Can only make Expanded Weights of Tensors, got {type(orig_weight).__name__}')
        if not orig_weight.requires_grad:
            raise RuntimeError('Can only build ExpandedWeights objects of tensors that require_grad')
        ret = torch.Tensor._make_subclass(cls, orig_weight, True)
        return ret

    @classmethod
    def __torch_function__(cls, func, _, args=(), kwargs=None):
        if False:
            for i in range(10):
                print('nop')
        if kwargs is None:
            kwargs = {}
        if func in expanded_weights_rnn_decomps:
            decomp_opts = expanded_weights_rnn_decomps[func]
            use_input_variant = isinstance(args[2], list)
            decomp = decomp_opts[0] if use_input_variant else decomp_opts[1]
            if decomp is not None:
                with setup_rnn(use_input_variant, args, kwargs):
                    return decomp(*args, **kwargs)
        if func == torch._cudnn_rnn_flatten_weight:
            return
        if func in cls.handled_functions:
            return cls.handled_functions[func].apply(tuple(kwargs.keys()), func, *args + tuple(kwargs.values()))
        raise RuntimeError(f'Expanded Weights encountered but cannot handle function {func.__name__}')

    @property
    def dtype(self):
        if False:
            i = 10
            return i + 15
        return self.orig_weight.dtype

    @property
    def data(self):
        if False:
            for i in range(10):
                print('nop')
        return self.orig_weight.data

    @property
    def shape(self):
        if False:
            for i in range(10):
                print('nop')
        return self.orig_weight.shape

    @property
    def device(self):
        if False:
            while True:
                i = 10
        return self.orig_weight.device

    @property
    def is_cuda(self):
        if False:
            for i in range(10):
                print('nop')
        return self.orig_weight.is_cuda

    def data_ptr(self):
        if False:
            return 10
        return self.orig_weight.data_ptr()

    def get_device(self):
        if False:
            print('Hello World!')
        return self.orig_weight.get_device()

    def set_allow_smaller_batches(self, is_allow_smaller_batches):
        if False:
            return 10
        self.allow_smaller_batches = is_allow_smaller_batches

    def set_batch_first(self, is_batch_first=True):
        if False:
            for i in range(10):
                print('nop')
        self.batch_first = is_batch_first