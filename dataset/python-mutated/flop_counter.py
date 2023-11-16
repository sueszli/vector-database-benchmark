import torch
import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from typing import List, Any, Dict, Optional, Union, NamedTuple
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle
from torch._decomp import register_decomposition
from math import prod
from functools import wraps
__all__ = ['FlopCounterMode', 'register_flop_formula']
aten = torch.ops.aten

def get_shape(i):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(i, torch.Tensor):
        return i.shape
    return i
flop_registry: Dict[Any, Any] = {}

def shape_wrapper(f):
    if False:
        while True:
            i = 10

    @wraps(f)
    def nf(*args, out=None, **kwargs):
        if False:
            i = 10
            return i + 15
        (args, kwargs, out_shape) = tree_map(get_shape, (args, kwargs, out))
        return f(*args, out_shape=out_shape, **kwargs)
    return nf

def register_flop_formula(targets, get_raw=False):
    if False:
        for i in range(10):
            print('nop')

    def register_fun(flop_formula):
        if False:
            for i in range(10):
                print('nop')
        if not get_raw:
            flop_formula = shape_wrapper(flop_formula)
        register_decomposition(targets, registry=flop_registry, unsafe=True)(flop_formula)
        return flop_formula
    return register_fun

@register_flop_formula(aten.mm)
def mm_flop(a_shape, b_shape, *args, out_shape=None, **kwargs) -> int:
    if False:
        return 10
    'Count flops for matmul.'
    (m, k) = a_shape
    (k2, n) = b_shape
    assert k == k2
    return m * n * 2 * k

@register_flop_formula(aten.addmm)
def addmm_flop(self_shape, a_shape, b_shape, out_shape=None, **kwargs) -> int:
    if False:
        return 10
    'Count flops for addmm.'
    return mm_flop(a_shape, b_shape)

@register_flop_formula(aten.bmm)
def bmm_flop(a_shape, b_shape, out_shape=None, **kwargs) -> int:
    if False:
        i = 10
        return i + 15
    'Count flops for the bmm operation.'
    (b, m, k) = a_shape
    (b2, k2, n) = b_shape
    assert b == b2
    assert k == k2
    flop = b * m * n * 2 * k
    return flop

@register_flop_formula(aten.baddbmm)
def baddbmm_flop(self_shape, a_shape, b_shape, out_shape=None, **kwargs) -> int:
    if False:
        return 10
    'Count flops for the baddbmm operation.'
    return bmm_flop(a_shape, b_shape)

def conv_flop_count(x_shape: List[int], w_shape: List[int], out_shape: List[int], transposed: bool=False) -> int:
    if False:
        return 10
    'Count flops for convolution.\n\n    Note only multiplication is\n    counted. Computation for bias are ignored.\n    Flops for a transposed convolution are calculated as\n    flops = (x_shape[2:] * prod(w_shape) * batch_size).\n    Args:\n        x_shape (list(int)): The input shape before convolution.\n        w_shape (list(int)): The filter shape.\n        out_shape (list(int)): The output shape after convolution.\n        transposed (bool): is the convolution transposed\n    Returns:\n        int: the number of flops\n    '
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    (c_out, c_in, *dims) = w_shape
    flop = batch_size * prod(conv_shape) * c_out * prod(dims) * 2 * c_in
    return flop

@register_flop_formula([aten.convolution, aten._convolution])
def conv_flop(x_shape, w_shape, _bias, _stride, _padding, _dilation, transposed, *args, out_shape=None, **kwargs) -> int:
    if False:
        for i in range(10):
            print('nop')
    'Count flops for convolution.'
    return conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)

def transpose_shape(shape):
    if False:
        while True:
            i = 10
    return [shape[1], shape[0]] + list(shape[2:])

@register_flop_formula(aten.convolution_backward)
def conv_backward_flop(grad_out_shape, x_shape, w_shape, _bias, _stride, _padding, _dilation, transposed, _output_padding, _groups, output_mask, out_shape) -> int:
    if False:
        while True:
            i = 10
    flop_count = 0
    if output_mask[0]:
        grad_input_shape = get_shape(out_shape[0])
        flop_count += conv_flop_count(grad_out_shape, w_shape, grad_input_shape, not transposed)
    if output_mask[1]:
        grad_weight_shape = get_shape(out_shape[1])
        flop_count += conv_flop_count(transpose_shape(x_shape), grad_out_shape, grad_weight_shape, transposed)
    return flop_count

def sdpa_flop_count(query_shape, key_shape, value_shape):
    if False:
        for i in range(10):
            print('nop')
    '\n    Count flops for self-attention.\n\n    NB: We can assume that value_shape == key_shape\n    '
    (b, h, s_q, d_q) = query_shape
    (_b2, _h2, s_k, _d2) = key_shape
    (_b3, _h3, _s3, d_v) = value_shape
    assert b == _b2 == _b3 and h == _h2 == _h3 and (d_q == _d2) and (s_k == _s3) and (d_q == _d2)
    total_flops = 0
    total_flops += bmm_flop((b * h, s_q, d_q), (b * h, d_q, s_k))
    total_flops += bmm_flop((b * h, s_q, s_k), (b * h, s_k, d_v))
    return total_flops

@register_flop_formula([aten._scaled_dot_product_efficient_attention, aten._scaled_dot_product_flash_attention])
def sdpa_flop(query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> int:
    if False:
        return 10
    'Count flops for self-attention.'
    return sdpa_flop_count(query_shape, key_shape, value_shape)

def sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape):
    if False:
        while True:
            i = 10
    total_flops = 0
    (b, h, s_q, d_q) = query_shape
    (_b2, _h2, s_k, _d2) = key_shape
    (_b3, _h3, _s3, d_v) = value_shape
    (_b4, _h4, _s4, _d4) = grad_out_shape
    assert b == _b2 == _b3 == _b4 and h == _h2 == _h3 == _h4 and (d_q == _d2)
    assert d_v == _d4 and s_k == _s3 and (s_q == _s4)
    total_flops = 0
    total_flops += bmm_flop((b * h, s_q, d_q), (b * h, d_q, s_k))
    total_flops += bmm_flop((b * h, s_q, d_v), (b * h, d_v, s_k))
    total_flops += bmm_flop((b * h, s_k, s_q), (b * h, s_q, d_v))
    total_flops += bmm_flop((b * h, s_q, s_k), (b * h, s_k, d_q))
    total_flops += bmm_flop((b * h, d_q, s_q), (b * h, s_q, s_k))
    return total_flops

@register_flop_formula([aten._scaled_dot_product_efficient_attention_backward, aten._scaled_dot_product_flash_attention_backward])
def sdpa_backward_flop(grad_out_shape, query_shape, key_shape, value_shape, *args, out_shape=None, **kwargs) -> int:
    if False:
        i = 10
        return i + 15
    'Count flops for self-attention backward.'
    return sdpa_backward_flop_count(grad_out_shape, query_shape, key_shape, value_shape)
flop_registry = {aten.mm: mm_flop, aten.addmm: addmm_flop, aten.bmm: bmm_flop, aten.baddbmm: baddbmm_flop, aten.convolution: conv_flop, aten._convolution: conv_flop, aten.convolution_backward: conv_backward_flop, aten._scaled_dot_product_efficient_attention: sdpa_flop, aten._scaled_dot_product_flash_attention: sdpa_flop, aten._scaled_dot_product_efficient_attention_backward: sdpa_backward_flop, aten._scaled_dot_product_flash_attention_backward: sdpa_backward_flop}

def normalize_tuple(x):
    if False:
        while True:
            i = 10
    if not isinstance(x, tuple):
        return (x,)
    return x
suffixes = ['', 'K', 'M', 'B', 'T']

def get_suffix_str(number):
    if False:
        i = 10
        return i + 15
    index = max(0, min(len(suffixes) - 1, (len(str(number)) - 3) // 3))
    return suffixes[index]

def convert_num_with_suffix(number, suffix):
    if False:
        i = 10
        return i + 15
    index = suffixes.index(suffix)
    value = f'{number / 1000 ** index:.3f}'
    return value + suffixes[index]

def convert_to_percent_str(num, denom):
    if False:
        return 10
    if denom == 0:
        return '0%'
    return f'{num / denom:.2%}'

def _pytreeify_preserve_structure(f):
    if False:
        i = 10
        return i + 15

    @wraps(f)
    def nf(args):
        if False:
            return 10
        (flat_args, spec) = tree_flatten(args)
        out = f(*flat_args)
        return tree_unflatten(out, spec)
    return nf

class FlopCounterMode(TorchDispatchMode):
    """
    ``FlopCounterMode`` is a context manager that counts the number of flops within its context.

    It does this using a ``TorchDispatchMode``.

    It also supports hierarchical output by passing a module (or list of
    modules) to FlopCounterMode on construction. If you do not need hierarchical
    output, you do not need to use it with a module.

    Example usage

    .. code-block:: python

        mod = ...
        flop_counter = FlopCounterMode(mod)
        with flop_counter:
            mod.sum().backward()

    """

    def __init__(self, mods: Optional[Union[torch.nn.Module, List[torch.nn.Module]]]=None, depth: int=2, display: bool=True, custom_mapping: Optional[Dict[Any, Any]]=None):
        if False:
            while True:
                i = 10
        self.flop_counts: Dict[str, Dict[Any, int]] = defaultdict(lambda : defaultdict(int))
        self.depth = depth
        self.parents = ['Global']
        self.display = display
        if custom_mapping is None:
            custom_mapping = {}
        if isinstance(mods, torch.nn.Module):
            mods = [mods]
        self.mods = mods
        self._module_to_forward_hook_handles: Dict[nn.Module, _ForwardHookHandles] = {}
        self.flop_registry = {**flop_registry, **{k: v if getattr(v, '_get_raw', False) else shape_wrapper(v) for (k, v) in custom_mapping.items()}}

    def _register_forward_hooks(self):
        if False:
            while True:
                i = 10
        if self.mods is None:
            return
        for mod in self.mods:
            prefix = type(mod).__name__
            for (name, module) in dict(mod.named_modules()).items():
                if name == '':
                    name = prefix
                else:
                    name = '.'.join([prefix, name])
                forward_pre_hook_handle = module.register_forward_pre_hook(self._enter_module(name))
                forward_hook_handle = module.register_forward_hook(self._exit_module(name))
                self._module_to_forward_hook_handles[module] = _ForwardHookHandles(forward_pre_hook_handle, forward_hook_handle)

    def _deregister_forward_hooks(self):
        if False:
            while True:
                i = 10
        for forward_hook_handles in self._module_to_forward_hook_handles.values():
            forward_hook_handles[0].remove()
            forward_hook_handles[1].remove()
        self._module_to_forward_hook_handles.clear()

    def _enter_module(self, name):
        if False:
            return 10

        def f(module, inputs):
            if False:
                return 10
            out = _pytreeify_preserve_structure(self._create_pre_module(name))(inputs)
            return out
        return f

    def _exit_module(self, name):
        if False:
            i = 10
            return i + 15

        def f(module, inputs, outputs):
            if False:
                return 10
            outputs = _pytreeify_preserve_structure(self._create_post_module(name))(outputs)
            return outputs
        return f

    def _create_post_module(self, name):
        if False:
            i = 10
            return i + 15

        class PushState(torch.autograd.Function):

            @staticmethod
            def forward(ctx, *args):
                if False:
                    i = 10
                    return i + 15
                assert self.parents[-1] == name
                self.parents.pop()
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                if False:
                    print('Hello World!')
                self.parents.append(name)
                return grad_outs
        return PushState.apply

    def _create_pre_module(self, name):
        if False:
            i = 10
            return i + 15

        class PopState(torch.autograd.Function):

            @staticmethod
            def forward(ctx, *args):
                if False:
                    for i in range(10):
                        print('nop')
                self.parents.append(name)
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                return args

            @staticmethod
            def backward(ctx, *grad_outs):
                if False:
                    for i in range(10):
                        print('nop')
                assert self.parents[-1] == name
                self.parents.pop()
                return grad_outs
        return PopState.apply

    def get_total_flops(self) -> int:
        if False:
            print('Hello World!')
        return sum(self.flop_counts['Global'].values())

    def get_flop_counts(self) -> Dict[str, Dict[Any, int]]:
        if False:
            return 10
        'Return the flop counts as a dictionary of dictionaries.\n\n        The outer\n        dictionary is keyed by module name, and the inner dictionary is keyed by\n        operation name.\n\n        Returns:\n            Dict[str, Dict[Any, int]]: The flop counts as a dictionary.\n        '
        return dict(self.flop_counts)

    def get_table(self, depth=None):
        if False:
            print('Hello World!')
        if depth is None:
            depth = self.depth
        if depth is None:
            depth = 999999
        import tabulate
        tabulate.PRESERVE_WHITESPACE = True
        header = ['Module', 'FLOP', '% Total']
        values = []
        global_flops = self.get_total_flops()
        global_suffix = get_suffix_str(global_flops)
        is_global_subsumed = False

        def process_mod(mod_name, depth):
            if False:
                i = 10
                return i + 15
            nonlocal is_global_subsumed
            total_flops = sum(self.flop_counts[mod_name].values())
            is_global_subsumed |= total_flops >= global_flops
            padding = ' ' * depth
            values = []
            values.append([padding + mod_name, convert_num_with_suffix(total_flops, global_suffix), convert_to_percent_str(total_flops, global_flops)])
            for (k, v) in self.flop_counts[mod_name].items():
                values.append([padding + ' - ' + str(k), convert_num_with_suffix(v, global_suffix), convert_to_percent_str(v, global_flops)])
            return values
        for mod in self.flop_counts.keys():
            if mod == 'Global':
                continue
            mod_depth = mod.count('.') + 1
            if mod_depth > depth:
                continue
            cur_values = process_mod(mod, mod_depth - 1)
            for value in cur_values:
                values.append(value)
        if 'Global' in self.flop_counts and (not is_global_subsumed):
            for (idx, value) in enumerate(values):
                values[idx][0] = ' ' + values[idx][0]
            values = process_mod('Global', 0) + values
        if len(values) == 0:
            values = [['Global', '0', '0%']]
        return tabulate.tabulate(values, headers=header, colalign=('left', 'right', 'right'))

    def __enter__(self):
        if False:
            while True:
                i = 10
        self.flop_counts.clear()
        self._register_forward_hooks()
        super().__enter__()
        return self

    def __exit__(self, *args):
        if False:
            i = 10
            return i + 15
        if self.display:
            print(self.get_table(self.depth))
        self._deregister_forward_hooks()
        super().__exit__(*args)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if False:
            for i in range(10):
                print('nop')
        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)
        func_packet = func._overloadpacket
        if func_packet in self.flop_registry:
            flop_count_func = self.flop_registry[func_packet]
            flop_count = flop_count_func(*args, **kwargs, out=out)
            for par in self.parents:
                self.flop_counts[par][func_packet] += flop_count
        return out

class _ForwardHookHandles(NamedTuple):
    forward_pre_hook_handle: RemovableHandle
    forward_hook_handle: RemovableHandle