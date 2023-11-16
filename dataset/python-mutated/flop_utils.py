import operator
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum, auto
from functools import partial, reduce
from typing import Any, Callable, List, Union
import torch
from torch.utils._pytree import tree_map
aten = torch.ops.aten

class Phase(Enum):
    FWD = auto()
    BWD = auto()

def _format_flops(flops: float) -> str:
    if False:
        i = 10
        return i + 15
    'Returns a formatted flops string'
    if flops > 1000000000000.0:
        return f'{flops / 1000000000000.0:.2f} TFLOPs'
    elif flops > 1000000000.0:
        return f'{flops / 1000000000.0:.2f} GFLOPs'
    elif flops > 1000000.0:
        return f'{flops / 1000000.0:.2f} MFLOPs'
    elif flops > 1000.0:
        return f'{flops / 1000.0:.2f} kFLOPs'
    return f'{flops} FLOPs'

def flop_count(module: Union[torch.nn.Module, Callable], *args, verbose: bool=False, forward_only: bool=True, **kwargs) -> int:
    if False:
        i = 10
        return i + 15
    '\n    Count the number of floating point operations in a model.\n    Ideas from https://pastebin.com/AkvAyJBw.\n    Parameters\n    ----------\n    module : Union[torch.nn.Module, Callable]\n        The model to count the number of floating point operations.\n    args : Any\n        The positional arguments to pass to the model.\n    verbose : bool\n        Whether to print the number of floating point operations.\n    forward_only : bool\n        Whether to only count the number of floating point operations in the forward pass.\n    kwargs : Any\n        The keyword arguments to pass to the model.\n\n    Returns\n    -------\n    int\n        The number of floating point operations.\n    '
    maybe_inplace = getattr(module, 'inplace', False) or kwargs.get('inplace', False) or getattr(module, '__name__', None) in ('add_', 'mul_', 'div_', 'sub_')

    class DummyModule(torch.nn.Module):

        def __init__(self, func):
            if False:
                for i in range(10):
                    print('nop')
            super().__init__()
            self.func = func
            self.__name__ = func.__name__

        def forward(self, *args, **kwargs):
            if False:
                return 10
            return self.func(*args, **kwargs)
    total_flop_count = {Phase.FWD: 0, Phase.BWD: 0}
    flop_counts = defaultdict(lambda : defaultdict(int))
    parents = ['Global']
    module = module if isinstance(module, torch.nn.Module) else DummyModule(module)

    class FlopTensor(torch.Tensor):
        elem: torch.Tensor
        __slots__ = ['elem']

        @staticmethod
        def __new__(cls, elem):
            if False:
                for i in range(10):
                    print('nop')
            r = torch.Tensor._make_wrapper_subclass(cls, elem.size(), strides=elem.stride(), storage_offset=elem.storage_offset(), dtype=elem.dtype, layout=elem.layout, device=elem.device, requires_grad=elem.requires_grad)
            r.elem = elem
            return r

        def __repr__(self):
            if False:
                for i in range(10):
                    print('nop')
            if self.grad_fn:
                return f'FlopTensor({self.elem}, grad_fn={self.grad_fn})'
            return f'FlopTensor({self.elem})'

        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            if False:
                print('Hello World!')

            def unwrap(e):
                if False:
                    return 10
                return e.elem if isinstance(e, FlopTensor) else e
            rs = func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs))
            outs = normalize_tuple(rs)
            if func in flop_mapping:
                nonlocal flop_counts, total_flop_count, cur_phase
                flop_count = flop_mapping.get(func, zero_flop_jit)(args, outs)
                for par in parents:
                    flop_counts[par][func.__name__] += flop_count
                total_flop_count[cur_phase] += flop_count

            def wrap(e):
                if False:
                    for i in range(10):
                        print('nop')
                return FlopTensor(e) if isinstance(e, torch.Tensor) else e
            rs = tree_map(wrap, rs)
            return rs

    def is_autogradable(x):
        if False:
            print('Hello World!')
        return isinstance(x, torch.Tensor) and x.is_floating_point()

    def normalize_tuple(x):
        if False:
            while True:
                i = 10
        if not isinstance(x, tuple):
            return (x,)
        return x

    def create_backwards_push(name):
        if False:
            i = 10
            return i + 15

        class PushState(torch.autograd.Function):

            @staticmethod
            def forward(ctx, *args):
                if False:
                    print('Hello World!')
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal parents
                parents.append(name)
                return grad_outs
        return PushState.apply

    def create_backwards_pop(name):
        if False:
            while True:
                i = 10

        class PopState(torch.autograd.Function):

            @staticmethod
            def forward(ctx, *args):
                if False:
                    return 10
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal parents
                assert parents[-1] == name
                parents.pop()
                return grad_outs
        return PopState.apply

    def enter_module(name):
        if False:
            return 10

        def f(module, inputs):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal parents
            parents.append(name)
            inputs = normalize_tuple(inputs)
            out = create_backwards_pop(name)(*inputs)
            return out
        return f

    def exit_module(name):
        if False:
            for i in range(10):
                print('nop')

        def f(module, inputs, outputs):
            if False:
                print('Hello World!')
            nonlocal parents
            assert parents[-1] == name
            parents.pop()
            outputs = normalize_tuple(outputs)
            return create_backwards_push(name)(*outputs)
        return f

    @contextmanager
    def instrument_module(mod):
        if False:
            print('Hello World!')
        registered = []
        for (name, module) in dict(mod.named_children()).items():
            registered.append(module.register_forward_pre_hook(enter_module(name)))
            registered.append(module.register_forward_hook(exit_module(name)))
        yield
        for handle in registered:
            handle.remove()

    def display_flops():
        if False:
            return 10
        for mod in flop_counts.keys():
            print(f'Module: ', mod)
            for (k, v) in flop_counts[mod].items():
                print('\t', k, _format_flops(v))
            print()

    def detach_variables(r):
        if False:
            i = 10
            return i + 15
        if isinstance(r, torch.Tensor):
            requires_grad = r.requires_grad
            r = r.detach()
            r.requires_grad = requires_grad
        return r

    def wrap(r):
        if False:
            print('Hello World!')
        if isinstance(r, torch.Tensor):
            r = FlopTensor(detach_variables(r))
            if maybe_inplace:
                r = r + 0
        return r
    with instrument_module(module):
        cur_phase = Phase.FWD
        rst = module(*tree_map(wrap, args), **tree_map(wrap, kwargs))
        rst = tuple((r for r in normalize_tuple(rst) if is_autogradable(r) and r.requires_grad))
        cur_phase = Phase.BWD
        if rst and (not forward_only):
            grad = [torch.zeros_like(t) for t in rst]
            torch.autograd.backward(rst, grad)
    if verbose:
        display_flops()
    if forward_only:
        return total_flop_count[Phase.FWD]
    else:
        return (total_flop_count[Phase.FWD], total_flop_count[Phase.BWD])

def matmul_flop_jit(inputs: List[Any], outputs: List[Any]) -> int:
    if False:
        print('Hello World!')
    '\n    Count flops for matmul.\n    '
    input_shapes = [v.shape for v in inputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    flops = reduce(operator.mul, input_shapes[0]) * input_shapes[-1][-1]
    return flops

def addmm_flop_jit(inputs: List[Any], outputs: List[Any]) -> int:
    if False:
        for i in range(10):
            print('nop')
    '\n    Count flops for fully connected layers.\n    '
    input_shapes = [v.shape for v in inputs[1:3]]
    assert len(input_shapes[0]) == 2, input_shapes[0]
    assert len(input_shapes[1]) == 2, input_shapes[1]
    (batch_size, input_dim) = input_shapes[0]
    output_dim = input_shapes[1][1]
    flops = batch_size * input_dim * output_dim
    return flops

def linear_flop_jit(inputs: List[Any], outputs: List[Any]) -> int:
    if False:
        while True:
            i = 10
    '\n    Count flops for the aten::linear operator.\n    '
    input_shapes = [v.shape for v in inputs[0:2]]
    assert input_shapes[0][-1] == input_shapes[1][-1]
    flops = reduce(operator.mul, input_shapes[0]) * input_shapes[1][0]
    return flops

def bmm_flop_jit(inputs: List[Any], outputs: List[Any]) -> int:
    if False:
        i = 10
        return i + 15
    '\n    Count flops for the bmm operation.\n    '
    assert len(inputs) == 2, len(inputs)
    input_shapes = [v.shape for v in inputs]
    (n, c, t) = input_shapes[0]
    d = input_shapes[-1][-1]
    flops = n * c * t * d
    return flops

def conv_flop_count(x_shape: List[int], w_shape: List[int], out_shape: List[int], transposed: bool=False) -> int:
    if False:
        for i in range(10):
            print('nop')
    '\n    Count flops for convolution. Note only multiplication is\n    counted. Computation for addition and bias is ignored.\n    Flops for a transposed convolution are calculated as\n    flops = (x_shape[2:] * prod(w_shape) * batch_size).\n    Args:\n        x_shape (list(int)): The input shape before convolution.\n        w_shape (list(int)): The filter shape.\n        out_shape (list(int)): The output shape after convolution.\n        transposed (bool): is the convolution transposed\n    Returns:\n        int: the number of flops\n    '
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    flops = batch_size * reduce(operator.mul, w_shape) * reduce(operator.mul, conv_shape)
    return flops

def conv_flop_jit(inputs: List[Any], outputs: List[Any]):
    if False:
        while True:
            i = 10
    '\n    Count flops for convolution.\n    '
    (x, w) = inputs[:2]
    (x_shape, w_shape, out_shape) = (x.shape, w.shape, outputs[0].shape)
    transposed = inputs[6]
    return conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)

def transpose_shape(shape):
    if False:
        return 10
    return [shape[1], shape[0]] + list(shape[2:])

def conv_backward_flop_jit(inputs: List[Any], outputs: List[Any]):
    if False:
        return 10
    (grad_out_shape, x_shape, w_shape) = [i.shape for i in inputs[:3]]
    output_mask = inputs[-1]
    fwd_transposed = inputs[7]
    flop_count = 0
    if output_mask[0]:
        grad_input_shape = outputs[0].shape
        flop_count += conv_flop_count(grad_out_shape, w_shape, grad_input_shape, not fwd_transposed)
    if output_mask[1]:
        grad_weight_shape = outputs[1].shape
        flop_count += conv_flop_count(transpose_shape(x_shape), grad_out_shape, grad_weight_shape, fwd_transposed)
    return flop_count

def norm_flop_counter(affine_arg_index: int, input_arg_index: int) -> Callable:
    if False:
        while True:
            i = 10
    '\n    Args:\n        affine_arg_index: index of the affine argument in inputs\n    '

    def norm_flop_jit(inputs: List[Any], outputs: List[Any]) -> int:
        if False:
            return 10
        '\n        Count flops for norm layers.\n        '
        input_shape = inputs[input_arg_index].shape
        has_affine = inputs[affine_arg_index].shape is not None if hasattr(inputs[affine_arg_index], 'shape') else inputs[affine_arg_index]
        assert 2 <= len(input_shape) <= 5, input_shape
        flop = reduce(operator.mul, input_shape) * (5 if has_affine else 4)
        return flop
    return norm_flop_jit

def batchnorm_flop_jit(inputs: List[Any], outputs: List[Any], training: bool=None) -> int:
    if False:
        i = 10
        return i + 15
    if training is None:
        training = inputs[-3]
    assert isinstance(training, bool), 'Signature of aten::batch_norm has changed!'
    if training:
        return norm_flop_counter(1, 0)(inputs, outputs)
    has_affine = inputs[1].shape is not None
    input_shape = reduce(operator.mul, inputs[0].shape)
    return input_shape * (2 if has_affine else 1)

def ewise_flop_counter(input_scale: float=1, output_scale: float=0) -> Callable:
    if False:
        return 10
    '\n    Count flops by\n        input_tensor.numel() * input_scale + output_tensor.numel() * output_scale\n    Args:\n        input_scale: scale of the input tensor (first argument)\n        output_scale: scale of the output tensor (first element in outputs)\n    '

    def ewise_flop(inputs: List[Any], outputs: List[Any]) -> int:
        if False:
            for i in range(10):
                print('nop')
        ret = 0
        if input_scale != 0:
            shape = inputs[0].shape
            ret += input_scale * reduce(operator.mul, shape) if shape else 0
        if output_scale != 0:
            shape = outputs[0].shape
            ret += output_scale * reduce(operator.mul, shape) if shape else 0
        return ret
    return ewise_flop

def zero_flop_jit(*args):
    if False:
        while True:
            i = 10
    '\n        Count flops for zero flop layers.\n    '
    return 0
flop_mapping = {aten.mm.default: matmul_flop_jit, aten.matmul.default: matmul_flop_jit, aten.addmm.default: addmm_flop_jit, aten.bmm.default: bmm_flop_jit, aten.convolution.default: conv_flop_jit, aten._convolution.default: conv_flop_jit, aten.convolution_backward.default: conv_backward_flop_jit, aten.native_batch_norm.default: batchnorm_flop_jit, aten.native_batch_norm_backward.default: batchnorm_flop_jit, aten.cudnn_batch_norm.default: batchnorm_flop_jit, aten.cudnn_batch_norm_backward.default: partial(batchnorm_flop_jit, training=True), aten.native_layer_norm.default: norm_flop_counter(2, 0), aten.native_layer_norm_backward.default: norm_flop_counter(2, 0), aten.avg_pool1d.default: ewise_flop_counter(1, 0), aten.avg_pool2d.default: ewise_flop_counter(1, 0), aten.avg_pool2d_backward.default: ewise_flop_counter(0, 1), aten.avg_pool3d.default: ewise_flop_counter(1, 0), aten.avg_pool3d_backward.default: ewise_flop_counter(0, 1), aten.max_pool1d.default: ewise_flop_counter(1, 0), aten.max_pool2d.default: ewise_flop_counter(1, 0), aten.max_pool3d.default: ewise_flop_counter(1, 0), aten.max_pool1d_with_indices.default: ewise_flop_counter(1, 0), aten.max_pool2d_with_indices.default: ewise_flop_counter(1, 0), aten.max_pool2d_with_indices_backward.default: ewise_flop_counter(0, 1), aten.max_pool3d_with_indices.default: ewise_flop_counter(1, 0), aten.max_pool3d_with_indices_backward.default: ewise_flop_counter(0, 1), aten._adaptive_avg_pool2d.default: ewise_flop_counter(1, 0), aten._adaptive_avg_pool2d_backward.default: ewise_flop_counter(0, 1), aten._adaptive_avg_pool3d.default: ewise_flop_counter(1, 0), aten._adaptive_avg_pool3d_backward.default: ewise_flop_counter(0, 1), aten.embedding_dense_backward.default: ewise_flop_counter(0, 1), aten.embedding.default: ewise_flop_counter(1, 0)}
ewise_flop_aten = [aten.add.Tensor, aten.add_.Tensor, aten.div.Tensor, aten.div_.Tensor, aten.div.Scalar, aten.div_.Scalar, aten.mul.Tensor, aten.mul.Scalar, aten.mul_.Tensor, aten.neg.default, aten.pow.Tensor_Scalar, aten.rsub.Scalar, aten.sum.default, aten.sum.dim_IntList, aten.mean.dim, aten.hardswish.default, aten.hardswish_.default, aten.hardswish_backward.default, aten.hardtanh.default, aten.hardtanh_.default, aten.hardtanh_backward.default, aten.hardsigmoid_backward.default, aten.hardsigmoid.default, aten.gelu.default, aten.gelu_backward.default, aten.silu.default, aten.silu_.default, aten.silu_backward.default, aten.sigmoid.default, aten.sigmoid_backward.default, aten._softmax.default, aten._softmax_backward_data.default, aten.relu_.default, aten.relu.default, aten.tanh.default, aten.tanh_backward.default, aten.threshold_backward.default, aten.native_dropout.default, aten.native_dropout_backward.default, aten.bernoulli_.float, aten.where.self]
for op in ewise_flop_aten:
    flop_mapping[op] = ewise_flop_counter(1, 0)