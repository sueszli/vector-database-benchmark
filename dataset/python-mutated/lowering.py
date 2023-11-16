import functools
import itertools
import logging
import os
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Dict, List, Optional, Tuple, Union
import sympy
import torch
import torch.fx
import torch.utils._pytree as pytree
from torch._higher_order_ops.triton_kernel_wrap import triton_kernel_wrapper_functional, triton_kernel_wrapper_mutation
from torch._prims_common import canonicalize_dim, canonicalize_dims, check, dtype_to_type, elementwise_dtypes, ELEMENTWISE_TYPE_PROMOTION_KIND, is_boolean_dtype, is_float_dtype, is_integer_dtype, Number
from torch.fx.experimental.sym_node import magic_methods, method_to_operator
from torch.utils._sympy.functions import CeilDiv, FloorDiv, ModularIndexing
from .._dynamo.utils import import_submodule
from . import config, inductor_prims, ir, test_operators
from .decomposition import decompositions, get_decompositions
from .ir import ExpandView, IndexingConstant, is_triton, ops_wrapper, PermuteView, Pointwise, Reduction, SqueezeView, TensorBox, validate_ir, View
from .utils import ceildiv, decode_device, is_dynamic, is_pointwise_use, pad_listlike, sympy_product
from .virtualized import ops, V
log = logging.getLogger(__name__)
lowerings = {}
layout_constraints = {}
fallbacks = set()
aten = torch.ops.aten
tr_c10d = torch.ops.tr_c10d
prims = torch.ops.prims
needs_realized_inputs = set()
foreach_ops = set()

def assert_nyi(cond, msg):
    if False:
        return 10
    if not cond:
        raise NotImplementedError(f'inductor does not support {msg}')

def add_needs_realized_inputs(fn):
    if False:
        while True:
            i = 10
    if isinstance(fn, (list, tuple, set)):
        return [add_needs_realized_inputs(x) for x in fn]
    needs_realized_inputs.add(fn)
    if isinstance(fn, torch._ops.OpOverloadPacket):
        for overload in fn.overloads():
            needs_realized_inputs.add(getattr(fn, overload))

def add_layout_constraint(fn, constraint):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(fn, torch._ops.OpOverloadPacket):
        for overload in fn.overloads():
            layout_constraints[getattr(fn, overload)] = constraint
    else:
        layout_constraints[fn] = constraint
add_needs_realized_inputs([aten.as_strided, aten.avg_pool2d, aten.avg_pool2d_backward, aten.bmm, aten.convolution, aten.convolution_backward, aten.max_pool2d_with_indices, aten.max_pool2d_with_indices_backward, aten.mm, aten.upsample_nearest2d, aten.upsample_bicubic2d, aten._int_mm])
DTYPE_ID_LOOKUP = {0: torch.uint8, 1: torch.int8, 2: torch.int16, 3: torch.int32, 4: torch.int64, 5: torch.float16, 6: torch.float32, 7: torch.float64, 8: torch.complex32, 9: torch.complex64, 10: torch.complex32, 11: torch.bool, 15: torch.bfloat16}

def decode_dtype(dtype: int):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(dtype, int):
        return dtype
    assert dtype in DTYPE_ID_LOOKUP, f'id {dtype} missing from DTYPE_ID_LOOKUP'
    dtype = DTYPE_ID_LOOKUP[dtype]
    return dtype

def is_integer_type(x):
    if False:
        return 10
    if isinstance(x, TensorBox):
        return is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())
    elif isinstance(x, sympy.Symbol):
        return x.is_integer is True
    else:
        return isinstance(x, int)

def is_boolean_type(x):
    if False:
        i = 10
        return i + 15
    if isinstance(x, TensorBox):
        return is_boolean_dtype(x.get_dtype())
    else:
        return isinstance(x, bool)

def get_promoted_dtype(*args, type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND):
    if False:
        return 10

    def construct_input(inp):
        if False:
            print('Hello World!')
        if isinstance(inp, (Number, sympy.Symbol)):
            return inp
        else:
            assert hasattr(inp, 'get_dtype')
            dim = len(inp.get_size())
            return torch.zeros([1] * dim, dtype=inp.get_dtype())
    inps = [construct_input(arg) for arg in args]
    (_, dtype) = elementwise_dtypes(*inps, type_promotion_kind=type_promotion_kind)
    return dtype

def get_overloads(aten_fn):
    if False:
        while True:
            i = 10
    if not isinstance(aten_fn, (list, tuple)):
        aten_fn = [aten_fn]
    else:
        aten_fn = list(aten_fn)
    for fn in list(aten_fn):
        if isinstance(fn, torch._ops.OpOverloadPacket):
            for overload in fn.overloads():
                other_fn = getattr(fn, overload)
                if other_fn not in lowerings:
                    aten_fn.append(other_fn)
    return aten_fn

def transform_args(args, broadcast, type_promotion_kind, convert_input_to_bool):
    if False:
        print('Hello World!')
    indices = [i for (i, x) in enumerate(args) if isinstance(x, TensorBox)]
    if (type_promotion_kind or convert_input_to_bool) and indices:
        if convert_input_to_bool:
            dtype = torch.bool
        else:
            promoting_args = [a for a in args if isinstance(a, Number) or hasattr(a, 'get_dtype')]
            dtype = get_promoted_dtype(*promoting_args, type_promotion_kind=type_promotion_kind)

        def promote(arg):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(arg, TensorBox):
                return to_dtype(arg, dtype)
            elif isinstance(arg, ir.Constant):
                return ir.Constant(arg.value, dtype, args[indices[0]].get_device())
            else:
                return arg
        args = [promote(a) for a in args]
    if broadcast and indices:
        for (i, x) in zip(indices, broadcast_tensors(*[args[i] for i in indices])):
            args[i] = x
        for i in range(len(args)):
            if isinstance(args[i], ir.Constant):
                args[i] = ExpandView.create(args[i], list(args[indices[0]].get_size()))
    return args

def _register_foreach_lowering(aten_fn, decomp_fn):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add a foreach lowering to lowerings dict.\n\n    Arguments:\n        aten_fn: torch.ops.aten.* fn we are lowering\n        decomp_fn: alternate implementation on our IR\n        broadcast: True to apply broadcasting to tensor inputs\n        type_promotion_kind: kind of type promotion applied to tensor inputs, `None` means no type promotion\n        convert_input_to_bool: some logical ops require inputs are converted to bool\n    '

    @functools.wraps(decomp_fn)
    def wrapped(*args, **kwargs):
        if False:
            while True:
                i = 10
        assert len(args) <= 2
        out = decomp_fn(*args, **kwargs)
        validate_ir(out)
        return out
    aten_fns = get_overloads(aten_fn)
    foreach_ops.update(aten_fns)
    lowerings.update({fn: wrapped for fn in aten_fns})
    return wrapped

def _register_lowering(aten_fn, decomp_fn, broadcast, type_promotion_kind, convert_input_to_bool):
    if False:
        i = 10
        return i + 15
    '\n    Add a lowering to lowerings dict\n\n    Arguments:\n        aten_fn: torch.ops.aten.* fn we are lowering\n        decomp_fn: alternate implementation on our IR\n        broadcast: True to apply broadcasting to tensor inputs\n        type_promotion_kind: kind of type promotion applied to tensor inputs, `None` means no type promotion\n        convert_input_to_bool: some logical ops require inputs are converted to bool\n    '

    @functools.wraps(decomp_fn)
    def wrapped(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        args: Union[List[Any], Tuple[Any, ...], Dict[Any, Any]] = list(args)
        unpacked = False
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            unpacked = True
            args = args[0]
        assert not any((x == 'out' for x in kwargs.keys())), "out= ops aren't yet supported"
        assert not any((isinstance(x, TensorBox) for x in kwargs.values())) or all((fn in fallbacks for fn in aten_fn))
        args = transform_args(args, broadcast, type_promotion_kind, convert_input_to_bool)
        if unpacked:
            args = [args]
        out = decomp_fn(*args, **kwargs)
        validate_ir(out)
        return out
    aten_fn = get_overloads(aten_fn)
    lowerings.update({fn: wrapped for fn in aten_fn})
    return wrapped

def register_lowering(aten_fn, broadcast=False, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, convert_input_to_bool=False):
    if False:
        return 10
    '\n    Shim to support decorator syntax.\n    '
    return functools.partial(_register_lowering, aten_fn, broadcast=broadcast, type_promotion_kind=type_promotion_kind, convert_input_to_bool=convert_input_to_bool)

def broadcast_symbolic_shapes(a, b):
    if False:
        for i in range(10):
            print('nop')
    '\n    Broadcasting logic based on symbolic shapes.\n\n    We give the shapes 0 and 1 concrete values, while all other shapes\n    are symbolic sympy formulas.\n    '
    output = []
    for (x, y) in itertools.zip_longest(reversed(a), reversed(b), fillvalue=sympy.Integer(1)):
        if y == 1:
            output.append(x)
        elif x == 1:
            output.append(y)
        else:
            V.graph.sizevars.guard_equals(x, y)
            if len(sympy.expand(y).free_symbols) < len(sympy.expand(x).free_symbols):
                output.append(y)
            else:
                output.append(x)
    return tuple(reversed(output))

def promote_constants(inputs, override_return_dtype=None):
    if False:
        i = 10
        return i + 15
    if not any((isinstance(x, (sympy.Expr, int, float)) for x in inputs)):
        return inputs
    if all((isinstance(x, (int, float, sympy.Symbol)) for x in inputs)):
        dtype = override_return_dtype or get_promoted_dtype(*inputs, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)

        def const_func(x):
            if False:
                while True:
                    i = 10
            if isinstance(x, sympy.Symbol):
                return ir.IndexingConstant(x, dtype, decode_device(None))
            else:
                return ir.Constant(x, dtype, decode_device(None))
        return [const_func(x) for x in inputs]
    ex = next((x for x in inputs if isinstance(x, (TensorBox, ExpandView))))
    out = []
    for x in inputs:
        if isinstance(x, (int, float)):
            out.append(ExpandView.create(ir.Constant(x, ex.get_dtype(), ex.get_device()), list(ex.get_size())))
        elif isinstance(x, sympy.Expr):
            out.append(ExpandView.create(IndexingConstant(x, ex.get_dtype(), ex.get_device()), list(ex.get_size())))
        else:
            out.append(x)
    return out

def make_pointwise(fn, override_return_dtype=None, override_device=None, override_fn_when_input_bool=None, override_fn_when_cuda_float64=None, allow_alpha=False):
    if False:
        return 10

    def inner(*inputs: List[TensorBox], alpha=None):
        if False:
            print('Hello World!')
        inputs = promote_constants(inputs, override_return_dtype)
        if allow_alpha:
            if alpha is not None and alpha != 1:
                inputs = list(inputs)
                inputs[-1] = mul(inputs[-1], alpha)
        else:
            assert alpha is None
        loaders = [x.make_loader() for x in inputs]
        ranges = inputs[0].get_size()
        dtype = override_return_dtype or inputs[0].get_dtype()
        is_cuda = decode_device(inputs[0].get_device()).type == 'cuda'
        for other in inputs[1:]:
            assert isinstance(other, ir.BaseConstant) or len(ranges) == len(other.get_size()), f'ndim mismatch {fn} {ranges} {other.get_size()}'

        def inner_fn(index):
            if False:
                i = 10
                return i + 15
            assert len(index) == len(ranges), f'wrong ndim {index} {ranges}'
            if dtype == torch.bool and override_fn_when_input_bool is not None:
                return override_fn_when_input_bool(*[load(index) for load in loaders])
            elif override_fn_when_cuda_float64 and is_cuda and (dtype == torch.float64):
                return override_fn_when_cuda_float64(*[load(index) for load in loaders])
            else:
                return fn(*[load(index) for load in loaders])
        if not override_device:
            device = None
            for i in inputs:
                if i.get_device().type == 'cuda':
                    device = i.get_device()
                    break
            if not device:
                device = inputs[0].get_device()
        device = override_device or device
        return Pointwise.create(device=device, dtype=dtype, inner_fn=inner_fn, ranges=ranges)
    return inner

def make_foreach_pointwise(pw_fn, allow_alpha=False):
    if False:
        print('Hello World!')

    def inner(*inputs: List[List[TensorBox]], alpha=1):
        if False:
            for i in range(10):
                print('nop')

        def group_args(arg_pairs):
            if False:
                i = 10
                return i + 15
            out = defaultdict(list)
            for (i, args) in enumerate(arg_pairs):
                use_foreach = not is_dynamic(*args)
                device = None
                for t in args:
                    if isinstance(t, TensorBox):
                        device = t.data.get_device()
                        break
                assert device is not None, 'foreach op should have at least one tensor arg'
                out[device, use_foreach].append((i, args))
            return out
        realize_outputs = False
        for node in V.graph.current_node.users:
            for user in node.users:
                if not (user.op == 'call_function' and user.target in foreach_ops):
                    realize_outputs = True
        a_list_input = None
        for input in inputs:
            if isinstance(input, (list, tuple)):
                a_list_input = input
                break
        assert a_list_input is not None, 'at least one input must be a list to a foreach op'
        broadcast_inputs = []
        for input in inputs:
            if not isinstance(input, (list, tuple)):
                broadcast_inputs.append([input] * len(a_list_input))
            else:
                broadcast_inputs.append(input)
        groups = group_args(zip(*broadcast_inputs))
        outputs = [None] * len(a_list_input)
        for ((device, use_foreach), group) in groups.items():
            buffer_list = []
            for (output_ind, args) in group:
                if allow_alpha:
                    output = pw_fn(*args, alpha=alpha)
                else:
                    output = pw_fn(*args)
                outputs[output_ind] = output
                if device.type == 'cuda' and use_foreach and realize_outputs:
                    buffer_list.append(output.realize())
            if buffer_list:
                V.graph.register_list(buffer_list)
        assert all((x is not None for x in outputs))
        return outputs
    return inner

def to_dtype(x: TensorBox, dtype: torch.dtype, copy=False):
    if False:
        i = 10
        return i + 15
    src_dtype = x.get_dtype()
    if src_dtype == dtype:
        return clone(x) if copy else x

    def _to_dtype(x):
        if False:
            while True:
                i = 10
        return ops.to_dtype(x, dtype, src_dtype=src_dtype)
    return make_pointwise(_to_dtype, override_return_dtype=dtype)(x)

@register_lowering(prims.convert_element_type, type_promotion_kind=None)
def _convert_element_type(x: TensorBox, dtype: torch.dtype):
    if False:
        while True:
            i = 10
    return to_dtype(x, dtype, copy=True)

def to_dtype_bitcast(x: TensorBox, dtype: torch.dtype, *, copy=False):
    if False:
        for i in range(10):
            print('nop')
    if x.get_dtype() == dtype:
        return clone(x) if copy else x

    def _get_primitive_bitwidth(dtype):
        if False:
            for i in range(10):
                print('nop')
        if dtype.is_floating_point:
            return torch.finfo(dtype).bits
        else:
            return torch.iinfo(dtype).bits
    src_bits = _get_primitive_bitwidth(x.get_dtype())
    dst_bits = _get_primitive_bitwidth(dtype)
    if src_bits != dst_bits:
        raise NotImplementedError(f'bitcast {x.get_dtype()} to different bitwidth type {dtype} is not supported yet.')

    def _to_dtype_bitcast(x):
        if False:
            return 10
        return ops.to_dtype_bitcast(x, dtype)
    return make_pointwise(_to_dtype_bitcast, override_return_dtype=dtype)(x)

@register_lowering(aten.view.dtype, type_promotion_kind=None)
def _view_dtype(x: TensorBox, dtype: torch.dtype):
    if False:
        return 10
    if dtype.is_complex or x.get_dtype().is_complex:
        return TensorBox.create(ir.ComplexView.create(torch.ops.aten.view.dtype, x, dtype))
    return to_dtype_bitcast(x, dtype, copy=True)

def to_device(x: TensorBox, device: torch.device, *, copy=False):
    if False:
        return 10
    device = decode_device(device)
    if x.get_device() == device:
        return clone(x) if copy else x
    return TensorBox.create(ir.DeviceCopy.create(x, device))

@register_lowering(prims.device_put, type_promotion_kind=None)
def _device_put(x: TensorBox, device: torch.device):
    if False:
        for i in range(10):
            print('nop')
    return to_device(x, device, copy=True)

def register_pointwise(aten_fn, name=None, broadcast=True, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT, convert_input_to_bool=False, override_return_dtype=None, override_fn_when_input_bool=None, allow_alpha=False, use_libdevice_for_f64=False):
    if False:
        while True:
            i = 10
    'A pointwise function that maps ops.{name} to inputs'
    name = name or aten_fn.__name__
    fn = ops_wrapper(name)
    if use_libdevice_for_f64:
        fn_libdevice = ops_wrapper('libdevice_' + name)
    if override_fn_when_input_bool is not None:
        override_fn_when_input_bool = ops_wrapper(override_fn_when_input_bool)
    fn = make_pointwise(fn, override_return_dtype=override_return_dtype, override_fn_when_input_bool=override_fn_when_input_bool, override_fn_when_cuda_float64=fn_libdevice if use_libdevice_for_f64 else None, allow_alpha=allow_alpha)
    fn = register_lowering(aten_fn, broadcast=broadcast, type_promotion_kind=type_promotion_kind, convert_input_to_bool=convert_input_to_bool)(fn)
    if hasattr(prims, name):
        register_lowering(getattr(prims, name), type_promotion_kind=None, convert_input_to_bool=convert_input_to_bool)(fn)
    return fn

def register_foreach_pointwise(aten_fn, pointwise_lowering_fn, allow_alpha=False):
    if False:
        for i in range(10):
            print('nop')
    fn = make_foreach_pointwise(pointwise_lowering_fn, allow_alpha=allow_alpha)
    fn = _register_foreach_lowering(aten_fn, fn)
    return fn

@register_lowering(aten.where, broadcast=False, type_promotion_kind=None)
def where(cond, a, b):
    if False:
        print('Hello World!')

    def fn(*args):
        if False:
            for i in range(10):
                print('nop')
        return ops.where(*args)
    if isinstance(a, (float, int)):
        a = constant_like(a)(b)
    if isinstance(b, (float, int)):
        b = constant_like(b)(a)
    args = [cond, a, b]
    dtype = get_promoted_dtype(args[1], args[2], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
    indices = [i for (i, x) in enumerate(args) if isinstance(x, TensorBox)]
    for (i, x) in zip(indices, broadcast_tensors(*[args[i] for i in indices])):
        args[i] = x
    for i in range(len(args)):
        if isinstance(args[i], ir.Constant):
            args[i] = ExpandView.create(args[i], list(args[indices[0]].get_size()))
    return make_pointwise(fn, override_return_dtype=dtype)(args[0], to_dtype(args[1], dtype), to_dtype(args[2], dtype))

@register_lowering(aten.broadcast_tensors, broadcast=False, type_promotion_kind=None)
def broadcast_tensors(*inputs):
    if False:
        for i in range(10):
            print('nop')
    if len(inputs) == 1 and isinstance(inputs[0], (list, tuple)):
        return broadcast_tensors(*inputs[0])
    target: List[sympy.Expr] = functools.reduce(broadcast_symbolic_shapes, [x.get_size() for x in inputs], [])
    outputs = []
    for x in inputs:
        sizes = x.get_size()
        if len(sizes) != len(target) or any((a == 1 and b != 1 or (a != 1 and b == 1) for (a, b) in zip(sizes, target))):
            x = expand(x, target)
        outputs.append(x)
    return outputs

@register_lowering([aten.alias, aten.detach, aten.detach_, aten.lift, prims.view_of])
def nop(x):
    if False:
        for i in range(10):
            print('nop')
    return x
if hasattr(aten, 'lift_fresh'):
    register_lowering(aten.lift_fresh)(nop)

@register_lowering(aten.squeeze, type_promotion_kind=None)
def squeeze(x, dim=None):
    if False:
        return 10
    assert isinstance(x, TensorBox)
    if dim is None:
        return TensorBox(SqueezeView.create(x.data))
    dim = canonicalize_dims(len(x.get_size()), dim)
    dims = set((dim,) if not isinstance(dim, tuple) else dim)
    new_shape = []
    for (d, s) in enumerate(x.get_size()):
        if not (d in dims and V.graph.sizevars.evaluate_expr(sympy.Eq(s, 1))):
            new_shape.append(s)
    return view(x, new_shape) if new_shape != x.get_size() else x

@register_lowering(aten.squeeze_copy, type_promotion_kind=None)
def squeeze_copy(x, dim=None):
    if False:
        for i in range(10):
            print('nop')
    return clone(squeeze(x, dim))

@register_lowering([aten.squeeze_])
def squeeze_(x, dim=None):
    if False:
        for i in range(10):
            print('nop')
    val = squeeze(x, dim)
    assert isinstance(x, TensorBox)
    assert isinstance(val, TensorBox)
    x.data = val.data
    return x

@register_lowering(aten.isinf)
def isinf(x):
    if False:
        return 10
    if is_integer_type(x):
        return full_like(x, False, dtype=torch.bool)
    fn = ops_wrapper('isinf')
    return make_pointwise(fn, override_return_dtype=torch.bool)(x)

@register_lowering(aten.isnan)
def isnan(x):
    if False:
        print('Hello World!')
    if is_integer_type(x):
        return full_like(x, False, dtype=torch.bool)
    fn = ops_wrapper('isnan')
    return make_pointwise(fn, override_return_dtype=torch.bool)(x)

@register_lowering(aten.ceil)
def ceil(x):
    if False:
        i = 10
        return i + 15
    if is_integer_type(x):
        return clone(x)
    fn = ops_wrapper('ceil')
    return make_pointwise(fn)(x)

@register_lowering(aten.floor)
def floor(x):
    if False:
        for i in range(10):
            print('nop')
    if is_integer_type(x):
        return clone(x)
    fn = ops_wrapper('floor')
    return make_pointwise(fn)(x)

@register_lowering(aten.round)
def round(x):
    if False:
        for i in range(10):
            print('nop')
    if is_integer_type(x):
        return clone(x)
    fn = ops_wrapper('round')
    return make_pointwise(fn)(x)

@register_lowering(aten.trunc)
def trunc(x):
    if False:
        return 10
    if is_integer_type(x):
        return clone(x)
    fn = ops_wrapper('trunc')
    return make_pointwise(fn)(x)

@register_lowering(aten.expand, type_promotion_kind=None)
def expand(x, sizes):
    if False:
        i = 10
        return i + 15
    (x,) = promote_constants([x])
    if isinstance(x, ir.BaseConstant):
        return ExpandView.create(x, tuple(sizes))
    assert isinstance(x, TensorBox)
    assert isinstance(sizes, (list, tuple))
    if tuple(x.get_size()) == tuple(sizes):
        return x
    if not any((V.graph.sizevars.shape_env.is_unbacked_symint(s) for s in x.get_size())):
        x_size_product = V.graph.sizevars.size_hint(sympy_product(x.get_size()))
        if x_size_product > 0 and (not any((V.graph.sizevars.shape_env.is_unbacked_symint(s) for s in sizes))):
            x.mark_reuse(V.graph.sizevars.size_hint(sympy_product(sizes)) // x_size_product)
    return TensorBox(ExpandView.create(x.data, tuple(sizes)))

@register_lowering(prims.broadcast_in_dim, type_promotion_kind=None)
def broadcast_in_dim(a, shape, broadcast_dimensions):
    if False:
        print('Hello World!')
    s = list(shape)
    for broadcast_dimension in broadcast_dimensions:
        s[broadcast_dimension] = -1
    v = a
    for (idx, x) in enumerate(s):
        if x != -1:
            v = unsqueeze(v, idx)
    return expand(v, shape)

@register_lowering(aten.expand_as, type_promotion_kind=None)
def expand_as(x, y):
    if False:
        for i in range(10):
            print('nop')
    return expand(x, y.get_size())

@register_lowering(aten.repeat)
def repeat(x, repeats):
    if False:
        i = 10
        return i + 15
    old_size = list(x.get_size())
    if len(repeats) > len(old_size):
        old_size = [sympy.Integer(1)] * (len(repeats) - len(old_size)) + old_size
        x = view(x, list(old_size))
    assert len(repeats) == len(x.get_size())
    new_size = list(x.get_size())
    zero_tensor = False
    for i in range(len(repeats)):
        if repeats[i] == 0:
            zero_tensor = True
        new_size[i] = new_size[i] * repeats[i]
    if zero_tensor:
        return empty(new_size, dtype=x.get_dtype(), device=x.get_device())
    if all((a == 1 or b == 1 for (a, b) in zip(repeats, old_size))):
        return expand(x, new_size)

    def inner_fn(index):
        if False:
            return 10
        assert len(index) == len(repeats)
        index = list(index)
        for i in range(len(repeats)):
            if repeats[i] != 1:
                if old_size[i] == 1:
                    index[i] = sympy.Integer(0)
                else:
                    index[i] = ModularIndexing(index[i], 1, old_size[i])
        return x_loader(index)
    old_size_product = V.graph.sizevars.size_hint(sympy_product(old_size))
    if old_size_product > 0:
        x.mark_reuse(V.graph.sizevars.size_hint(sympy_product(new_size)) // old_size_product)
    x_loader = x.make_loader()
    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=inner_fn, ranges=list(new_size))

@register_lowering(aten._unsafe_view, type_promotion_kind=None)
@register_lowering(aten.view, type_promotion_kind=None)
@register_lowering(aten.reshape, type_promotion_kind=None)
def view(x, sizes):
    if False:
        while True:
            i = 10
    assert isinstance(x, TensorBox)
    assert isinstance(sizes, (list, tuple))
    return TensorBox(View.create(x.data, sizes))

@register_lowering(aten.permute, type_promotion_kind=None)
def permute(x, dims):
    if False:
        while True:
            i = 10
    assert isinstance(x, TensorBox)
    assert isinstance(dims, (list, tuple))
    return TensorBox(PermuteView.create(x.data, tuple(dims)))

@register_lowering(aten.slice, type_promotion_kind=None)
def slice_(x, dim=0, start=0, end=2 ** 63, step=1):
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(x, TensorBox)
    dim = _validate_dim(x, dim, 0)
    dim_size = x.get_size()[dim]
    if V.graph.sizevars.evaluate_expr(sympy.Lt(start + dim_size, 0)):
        start = 0
    if V.graph.sizevars.evaluate_expr(sympy.Lt(end + dim_size, 0)):
        end = 0
    return TensorBox(ir.SliceView.create(x.data, dim, start, end, step))

@register_lowering(aten.roll, type_promotion_kind=None)
def roll(a, shifts, dims=tuple()):
    if False:
        print('Hello World!')
    "\n    This is based on torch._refs.roll(), but uses ModularIndexing().\n\n    We can't use the ref here because it is based on multiple calls to\n    torch.cat() that this will result in terrible code.\n    "
    if not isinstance(shifts, Iterable):
        shifts = (shifts,)
    if not isinstance(dims, Iterable):
        dims = (dims,)
    dims = [_validate_dim(a, d) for d in dims]
    if sympy_product(a.get_size()) == 0:
        return clone(a)
    len_shifts = len(shifts)
    len_dims = len(dims)
    if len_shifts != 1 or len_dims != 1:
        if len_shifts == 0:
            raise RuntimeError('`shifts` required')
        if len_dims == 0 and len_shifts == 1:
            flat = view(a, [sympy_product(a.get_size())])
            rolled = roll(flat, shifts, 0)
            return view(rolled, list(a.get_size()))
        if len_shifts != len_dims:
            raise RuntimeError(f'shifts and dimensions must align. shifts: {len_shifts}, dims: {len_dims}')
        tail_shifts = shifts[1:]
        tail_dims = dims[1:]
        first_dim_rolled = roll(a, shifts[0], dims[0])
        return roll(first_dim_rolled, tail_shifts, tail_dims)
    (dim,) = dims
    size = V.graph.sizevars.evaluate_static_shape(a.get_size()[dim])
    start = (size - shifts[0]) % size
    a_loader = a.make_loader()

    def fn(index):
        if False:
            return 10
        index = list(index)
        index[dim] = ModularIndexing(index[dim] + start, sympy.Integer(1), sympy.expand(size))
        return a_loader(index)
    return Pointwise.create(device=a.get_device(), dtype=a.get_dtype(), inner_fn=fn, ranges=a.get_size())

@register_lowering(aten.as_strided, type_promotion_kind=None)
def as_strided(x, size, stride, storage_offset=None):
    if False:
        return 10
    if isinstance(x, TensorBox) and isinstance(x.data, ir.BaseView):
        x = x.data.unwrap_view()
    x.realize()
    if not ir.is_storage_and_layout(x):
        raise NotImplementedError(f'unrealized as_strided({x}, ...)')
    (storage, old_layout) = ir.as_storage_and_layout(x)
    new_layout = ir.FixedLayout(old_layout.device, old_layout.dtype, [sympy.expand(s) for s in size], [sympy.expand(s) for s in stride], sympy.expand(storage_offset or 0))
    return TensorBox(ir.ReinterpretView(storage, new_layout))

@register_lowering(aten.as_strided_, type_promotion_kind=None)
def as_strided_(x, size, stride, storage_offset=None):
    if False:
        i = 10
        return i + 15
    assert isinstance(x, TensorBox)
    x.data = as_strided(x, size, stride, storage_offset).data
    return x

@register_lowering(aten.as_strided_copy, type_promotion_kind=None)
def as_strided_copy(x, size, stride, storage_offset=None):
    if False:
        while True:
            i = 10
    result = as_strided(x, size, stride, storage_offset)
    return clone(result)

def pointwise_cat(inputs, dim=0):
    if False:
        i = 10
        return i + 15
    inputs_ranges: List[Tuple[sympy.Expr, sympy.Expr]] = []
    prev_end = 0
    for inp in inputs:
        inputs_ranges.append((prev_end, prev_end + inp.get_size()[dim]))
        prev_end = inputs_ranges[-1][-1]
    inputs_loaders = [inp.make_loader() for inp in inputs]

    def inner_fn(idx):
        if False:
            i = 10
            return i + 15
        idx_dim = ops.index_expr(idx[dim], torch.int64)
        masks = []
        masked_loads = []
        for i in range(len(inputs)):
            start = ops.constant(0, torch.int64) if i == 0 else ops.index_expr(inputs_ranges[i][0], torch.int64)
            end = ops.index_expr(inputs_ranges[i][1], torch.int64)
            start_cond = ops.ge(idx_dim, start)
            end_cond = ops.lt(idx_dim, end)
            if i == 0:
                mask = end_cond
            elif i == len(inputs) - 1:
                mask = start_cond
            else:
                mask = ops.and_(start_cond, end_cond)
            masks.append(mask)
            idx_load = list(idx)
            idx_load[dim] -= inputs_ranges[i][0]
            masked_loads.append(ops.masked(mask, lambda : inputs_loaders[i](idx_load), 0.0))

        def get_masked_val(i):
            if False:
                return 10
            if i != len(inputs) - 1:
                return ops.where(masks[i], masked_loads[i], get_masked_val(i + 1))
            else:
                return masked_loads[-1]
        return get_masked_val(0)
    new_size = list(inputs[0].get_size())
    new_size[dim] = inputs_ranges[-1][-1]
    return Pointwise.create(device=inputs[0].get_device(), dtype=inputs[0].get_dtype(), inner_fn=inner_fn, ranges=new_size)

@register_lowering(aten.cat)
def cat(inputs, dim=0):
    if False:
        return 10
    if all((input.get_dtype() is torch.uint8 for input in inputs)):
        for input in inputs:
            input.realize()
        if all((len(input.get_size()) == 4 for input in inputs)):
            (inputs, _) = require_channels_last(aten.cat, *inputs)
        return fallback_handler(aten.cat.default)(inputs, dim)
    if len(inputs) == 1:
        return clone(inputs[0])
    dim = _validate_dim(inputs[0], dim, 0)
    dtype = get_promoted_dtype(*inputs, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
    inputs = [to_dtype(inp, dtype) for inp in inputs]

    def should_lower_cat_input(x) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if ir.is_storage_and_layout(x):
            (storage, _) = ir.as_storage_and_layout(x, freeze=False)
            return not ir.ConcatKernel.can_realize_into_without_copy(storage)
        if isinstance(x, TensorBox):
            if isinstance(x.data, ir.BaseView):
                return should_lower_cat_input(x.data.unwrap_view())
            else:
                return should_lower_cat_input(x.data)
        if isinstance(x, ir.StorageBox):
            return should_lower_cat_input(x.data)
        if isinstance(x, ir.Pointwise):
            return True
        return False
    if len(inputs) <= config.max_pointwise_cat_inputs:
        pointwise_uses = all((is_pointwise_use(use) for use in V.current_node.users))
        all_pointwise_inputs = all((should_lower_cat_input(inp) for inp in inputs))
        any_pointwise_inputs = any((should_lower_cat_input(inp) for inp in inputs))
        if all_pointwise_inputs or (any_pointwise_inputs and pointwise_uses):
            return pointwise_cat(inputs, dim)
    return TensorBox(ir.ConcatKernel.create(inputs, dim))

@register_lowering(aten.diagonal, type_promotion_kind=None)
def diagonal(input, offset: int=0, dim1: int=0, dim2: int=1):
    if False:
        print('Hello World!')
    original_shape = input.get_size()
    num_dims = len(original_shape)
    dim1 = canonicalize_dim(idx=dim1, rank=num_dims)
    dim2 = canonicalize_dim(idx=dim2, rank=num_dims)
    check(dim1 != dim2, lambda : f'diagonal dimensions cannot be identical {dim1}, {dim2}')
    offset_negative = V.graph.sizevars.evaluate_expr(sympy.Lt(offset, 0))
    if offset_negative:
        diag_size = max(min(original_shape[dim1] + offset, original_shape[dim2]), 0)
    else:
        diag_size = max(min(original_shape[dim1], original_shape[dim2] - offset), 0)
    base_idx = (0, 0)
    if offset_negative:
        base_idx = (-offset, 0)
    else:
        base_idx = (0, offset)
    sizes = [s for (i, s) in enumerate(original_shape) if i not in (dim1, dim2)]
    sizes.append(diag_size)

    def reindexer(idx):
        if False:
            for i in range(10):
                print('nop')
        diag_idx = idx[-1]
        original_idx = [0] * len(original_shape)
        cur_dim = 0
        for d in range(num_dims):
            if d == dim1:
                original_idx[d] = diag_idx + base_idx[0]
            elif d == dim2:
                original_idx[d] = diag_idx + base_idx[1]
            else:
                original_idx[d] = idx[cur_dim]
                cur_dim += 1
        assert cur_dim == len(original_shape) - 2
        return original_idx
    return TensorBox(ir.GenericView.create(input, sizes, reindexer))

@register_lowering(aten.diagonal_copy, type_promotion_kind=None)
def diagonal_copy(input, offset: int=0, dim1: int=0, dim2: int=1):
    if False:
        return 10
    return clone(diagonal(input, offset, dim1, dim2))

@register_lowering(aten.diagonal_scatter, type_promotion_kind=None)
def diagonal_scatter(input, src, offset: int=0, dim1: int=0, dim2: int=1):
    if False:
        print('Hello World!')
    output = clone(input)
    target = diagonal(output, offset, dim1, dim2)
    mutate_to(target, src)
    return output

@register_lowering(aten.select, type_promotion_kind=None)
def select(x, dim, idx):
    if False:
        print('Hello World!')
    idx = View.handle_negative_index(idx, x.get_size()[dim])
    return squeeze(slice_(x, dim, idx, idx + 1), dim)

@register_lowering(aten.split, type_promotion_kind=None)
def split(x, sizes, dim=0):
    if False:
        print('Hello World!')
    dim = _validate_dim(x, dim, 0)
    x_size = V.graph.sizevars.evaluate_static_shape(x.get_size()[dim])
    if isinstance(sizes, sympy.Expr):
        sizes = V.graph.sizevars.evaluate_static_shape(sizes)
    if isinstance(sizes, (int, sympy.Integer)):
        sizes = [sizes] * ((x_size + sizes - 1) // sizes)
    result = []
    start = 0
    for size in sizes:
        end = start + size
        result.append(slice_(x, dim, start, end))
        start = end
    return result

@register_lowering(aten.split_with_sizes, type_promotion_kind=None)
def split_with_sizes(x, sizes, dim=0):
    if False:
        return 10
    return split(x, sizes, dim)

@register_lowering(aten.unbind, type_promotion_kind=None)
def unbind(x, dim=0):
    if False:
        i = 10
        return i + 15
    dim = _validate_dim(x, dim, 0)
    x_size = V.graph.sizevars.evaluate_static_shape(x.get_size()[dim])
    result = []
    for i in range(x_size):
        result.append(select(x, dim, i))
    return result

@register_lowering(aten.unfold, type_promotion_kind=None)
def unfold(x, dimension, size, step):
    if False:
        i = 10
        return i + 15
    sizes = x.get_size()
    ndim = len(sizes)
    dim = canonicalize_dim(ndim, dimension)
    if ndim == 0:
        return slice_(unsqueeze(x, 0), end=size)
    dim_size = sizes[dim]
    sizevars = V.graph.sizevars
    sizevars.guard_leq(size, dim_size)
    sizevars.guard_lt(0, step)
    new_dim_size = FloorDiv(dim_size - size, step) + 1
    if sizevars.size_hint(dim_size) > 0:
        x.mark_reuse(sizevars.size_hint(CeilDiv(new_dim_size * size, dim_size)))
    out_size = [*sizes[:dim], new_dim_size, *sizes[dim + 1:], size]

    def reindexer(idx):
        if False:
            print('Hello World!')
        dim_idx = idx[-1] + idx[dim] * step
        return (*idx[:dim], dim_idx, *idx[dim + 1:-1])
    return TensorBox(ir.GenericView.create(x, out_size, reindexer))

@register_lowering(aten.unsqueeze, type_promotion_kind=None)
def unsqueeze(x, dim):
    if False:
        for i in range(10):
            print('nop')
    dim = _validate_dim(x, dim, 1)
    new_shape = list(x.get_size())
    new_shape.insert(dim, sympy.Integer(1))
    return view(x, new_shape)

@register_lowering(aten.unsqueeze_, type_promotion_kind=None)
def unsqueeze_(x, dim):
    if False:
        print('Hello World!')
    val = unsqueeze(x, dim)
    assert isinstance(x, TensorBox)
    assert isinstance(val, TensorBox)
    x.data = val.data
    return x

def _validate_dim(x, dim, offset=0):
    if False:
        while True:
            i = 10
    assert isinstance(dim, int)
    ndim = len(x.get_size())
    if dim < 0:
        dim += ndim + offset
    assert 0 <= dim < ndim + offset
    return dim

@register_lowering(aten.glu)
def glu(x, dim=-1):
    if False:
        for i in range(10):
            print('nop')
    dim = _validate_dim(x, dim, 0)
    new_len = V.graph.sizevars.evaluate_static_shape(x.get_size()[dim]) // 2
    a = slice_(x, dim, 0, new_len)
    b = slice_(x, dim, new_len, new_len * 2)
    return mul(a, sigmoid(b))

def register_onednn_fusion_ops():
    if False:
        for i in range(10):
            print('nop')
    if torch._C._has_mkldnn:
        cpu_needs_realized_inputs = [torch.ops.mkldnn._convolution_pointwise, torch.ops.mkldnn._convolution_pointwise_, torch.ops.mkldnn._convolution_transpose_pointwise, torch.ops.mkldnn._linear_pointwise, aten.mkldnn_rnn_layer.default, torch.ops.onednn.qconv2d_pointwise]

        @register_lowering(torch.ops.mkldnn._convolution_pointwise)
        def convolution_unary(x: TensorBox, weight: TensorBox, bias: TensorBox, padding, stride, dilation, groups, attr, scalars, algorithm):
            if False:
                print('Hello World!')
            return TensorBox.create(ir.ConvolutionUnary.create(x, weight, bias, padding, stride, dilation, groups, attr, scalars, algorithm))

        @register_lowering(torch.ops.mkldnn._convolution_pointwise.binary)
        def convolution_binary(x: TensorBox, other: TensorBox, weight: TensorBox, bias: TensorBox, padding, stride, dilation, groups, binary_attr, binary_alpha, unary_attr, unary_scalars, unary_algorithm):
            if False:
                for i in range(10):
                    print('nop')
            return TensorBox.create(ir.ConvolutionBinary.create(x, other, weight, bias, padding, stride, dilation, groups, binary_attr, binary_alpha, unary_attr, unary_scalars, unary_algorithm))

        @register_lowering(torch.ops.mkldnn._convolution_pointwise_.binary)
        def convolution_binary_inplace(x: TensorBox, other: TensorBox, weight: TensorBox, bias: TensorBox, padding, stride, dilation, groups, binary_attr, binary_alpha, unary_attr, unary_scalars, unary_algorithm):
            if False:
                while True:
                    i = 10
            return TensorBox.create(ir.ConvolutionBinaryInplace.create(x, other, weight, bias, padding, stride, dilation, groups, binary_attr, binary_alpha, unary_attr, unary_scalars, unary_algorithm))

        @register_lowering(torch.ops.mkldnn._linear_pointwise)
        def linear_unary(x: TensorBox, w: TensorBox, b: TensorBox, attr, scalars, algorithm):
            if False:
                i = 10
                return i + 15
            return TensorBox.create(ir.LinearUnary.create(x, w, b, attr, scalars, algorithm))

        @register_lowering(torch.ops.mkldnn._linear_pointwise.binary)
        def linear_binary(x: TensorBox, y: TensorBox, w: TensorBox, b: TensorBox, attr):
            if False:
                i = 10
                return i + 15
            return TensorBox.create(ir.LinearBinary.create(x, y, w, b, attr))

        @register_lowering(torch.ops.mkldnn._convolution_transpose_pointwise)
        def convolution_transpose_unary(x: TensorBox, weight: TensorBox, bias: TensorBox, padding, output_padding, stride, dilation, groups, attr, scalars, algorithm):
            if False:
                return 10
            return TensorBox.create(ir.ConvolutionTransposeUnary.create(x, weight, bias, padding, output_padding, stride, dilation, groups, attr, scalars, algorithm))

        @register_lowering(aten.mkldnn_rnn_layer.default)
        def mkldnn_rnn_layer(x: TensorBox, w0: TensorBox, w1: TensorBox, w2: TensorBox, w3: TensorBox, hx: TensorBox, cx: TensorBox, reverse: bool, batch_sizes: List[int], mode: int, hidden_size: int, num_layers: int, has_biases: bool, bidirectional: bool, batch_first: bool, train: bool):
            if False:
                i = 10
                return i + 15
            return pytree.tree_map(TensorBox.create, ir.MkldnnRnnLayer.create(x, w0, w1, w2, w3, hx, cx, reverse, batch_sizes, mode, hidden_size, num_layers, has_biases, bidirectional, batch_first, train))

        @register_lowering(torch.ops.onednn.qconv2d_pointwise, type_promotion_kind=None)
        def qconvolution_unary(x: TensorBox, x_scale, x_zp, packed_weight: TensorBox, w_scale: TensorBox, w_zp: TensorBox, bias: TensorBox, stride, padding, dilation, groups, o_inv_scale, o_zero_point, output_dtype, attr, scalars, algorithm):
            if False:
                print('Hello World!')
            return TensorBox.create(ir.QConvPointWisePT2E.create(x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias, stride, padding, dilation, groups, o_inv_scale, o_zero_point, output_dtype, attr, scalars, algorithm))

        @register_lowering(torch.ops.onednn.qconv2d_pointwise.binary, type_promotion_kind=None)
        def qconvolution_binary(x: TensorBox, x_scale, x_zp, accum: TensorBox, accum_scale, accum_zp, packed_weight: TensorBox, w_scale: TensorBox, w_zp: TensorBox, bias: TensorBox, stride, padding, dilation, groups, o_inv_scale, o_zero_point, output_dtype, binary_attr, alpha, unary_attr, unary_scalars, unary_algorithmm):
            if False:
                for i in range(10):
                    print('nop')
            return TensorBox.create(ir.QConvPointWiseBinaryPT2E.create(x, x_scale, x_zp, accum, accum_scale, accum_zp, packed_weight, w_scale, w_zp, bias, stride, padding, dilation, groups, o_inv_scale, o_zero_point, output_dtype, binary_attr, alpha, unary_attr, unary_scalars, unary_algorithmm))

        @register_lowering(torch.ops.onednn.qlinear_pointwise, type_promotion_kind=None)
        def qlinear_unary(x: TensorBox, x_scale, x_zp, packed_weight: TensorBox, w_scale: TensorBox, w_zp: TensorBox, bias: TensorBox, o_inv_scale, o_zero_point, output_dtype, attr, scalars, algorithm):
            if False:
                return 10
            return TensorBox.create(ir.QLinearPointwisePT2E.create(x, x_scale, x_zp, packed_weight, w_scale, w_zp, bias, o_inv_scale, o_zero_point, output_dtype, attr, scalars, algorithm))
        if torch._C.has_mkl:
            cpu_needs_realized_inputs.append(torch.ops.mkl._mkl_linear)

            @register_lowering(torch.ops.mkl._mkl_linear)
            def mkl_packed_linear(x: TensorBox, packed_w: TensorBox, orig_w: TensorBox, b: TensorBox, batch_size):
                if False:
                    print('Hello World!')
                result = TensorBox.create(ir.MKLPackedLinear.create(x, packed_w, orig_w, batch_size))
                if b is not None:
                    result = add(result, b)
                return result
        add_needs_realized_inputs(cpu_needs_realized_inputs)
    else:
        pass
register_onednn_fusion_ops()

def fallback_handler(kernel, add_to_fallback_set=True):
    if False:
        print('Hello World!')
    if add_to_fallback_set:
        fallbacks.add(kernel)

    def handler(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return pytree.tree_map(TensorBox.create, ir.FallbackKernel.create(kernel, *args, **kwargs))
    return handler

@functools.lru_cache(None)
def _warn_complex_not_supported():
    if False:
        print('Hello World!')
    warnings.warn('Torchinductor does not support code generation for complex operators. Performance may be worse than eager.')

def unsupported_input_tensor(t: torch._subclasses.FakeTensor, parent=None):
    if False:
        i = 10
        return i + 15
    'Do not support reading or writing to this tensor'
    if t.is_complex():
        if parent and parent.target == torch.ops.aten.view.dtype:
            return False
        _warn_complex_not_supported()
        return True
    return False

def unsupported_output_tensor(t: torch._subclasses.FakeTensor, parent=None):
    if False:
        i = 10
        return i + 15
    'Do not support writing tensor but can read from it'
    if unsupported_input_tensor(t, parent):
        return True
    return t.is_cpu and config.disable_cpp_codegen

def fallback_node_due_to_unsupported_type(node: torch.fx.Node, allow_cpu_inputs=True):
    if False:
        while True:
            i = 10
    if node.target is aten.view_as_complex.default:
        return False
    if node.target is aten.lift_fresh_copy.default:
        return False

    def check_skip_condition(node, parent, is_output):
        if False:
            i = 10
            return i + 15
        if not isinstance(node, torch.fx.Node):
            return False
        if 'val' not in node.meta:
            return False
        for meta in pytree.tree_leaves(node.meta['val']):
            if not isinstance(meta, torch._subclasses.FakeTensor):
                continue
            if is_output:
                if unsupported_output_tensor(meta, parent):
                    return True
            elif unsupported_input_tensor(meta, parent):
                return True
        return False
    for arg in pytree.arg_tree_leaves(*node.args, **node.kwargs):
        if check_skip_condition(arg, node, is_output=False):
            return True
    return check_skip_condition(node, node, is_output=True)

def make_fallback(op, layout_constraint=None, warn=True):
    if False:
        return 10
    assert op not in decompositions, f'both a fallback and a decomp for same op: {op}'
    if warn and bool(os.getenv('CI')) and get_decompositions([op]) and (not (config.fallback_random and op in torch._decomp.decompositions_for_rng.extra_random_decomps)):
        if torch._dynamo.config.suppress_errors:
            torch._dynamo.config.suppress_errors = False
            log.warning('A make_fallback error occurred in suppress_errors config, and suppress_errors is being disabled to surface it.')
        raise AssertionError(f"make_fallback({op}): a decomposition exists, we should switch to it. To fix this error, either add a decomposition to core_aten_decompositions (preferred) or inductor_decompositions, and delete the corresponding `make_fallback` line. Get help from the inductor team if unsure, don't pick arbitrarily to unblock yourself.")

    def register_fallback(op_overload):
        if False:
            while True:
                i = 10
        add_needs_realized_inputs(op_overload)
        if layout_constraint is not None:
            add_layout_constraint(op_overload, layout_constraint)
        return register_lowering(op_overload, type_promotion_kind=None)(fallback_handler(op_overload))
    if isinstance(op, torch._ops.OpOverloadPacket):
        for ol in op.overloads():
            op_overload = getattr(op, ol)
            register_fallback(op_overload)
    elif isinstance(op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)):
        register_fallback(op)
    else:
        raise RuntimeError(f'Unsupported fallback {op} with type {type(op)}')

def philox_rand_offset(shape):
    if False:
        print('Hello World!')
    '\n    TorchInductor offset calculation differs from PyTorch eager offset\n    calculation for random ops (tl.rand vs torch.rand). In future, we should\n    strive for same impl for tl.rand and torch.rand.\n    '
    numel = 1
    for s in shape:
        numel = numel * s
    return tensor(numel, dtype=torch.int64)

@register_lowering(torch.ops.rngprims.philox_rand, type_promotion_kind=None)
def philox_rand(size, seed, offset, stride, device, dtype):
    if False:
        i = 10
        return i + 15
    random_pos = ir.FixedLayout(device, dtype, size, ir.FlexibleLayout.contiguous_strides(size)).make_indexer()
    seed_loader = seed.make_loader()
    offset_loader = offset.make_loader()

    def inner_fn(index):
        if False:
            return 10
        seed_index_expr = ops.to_dtype(seed_loader([]), torch.int32)
        offset_index_expr = ops.to_dtype(offset_loader([]), torch.int32)
        rand_index_expr = ops.add(ops.index_expr(random_pos(index), torch.int32), offset_index_expr)
        result = ops.rand(seed_index_expr, rand_index_expr)
        return ops.to_dtype(result, dtype)
    random_values_node = Pointwise.create(device=device, dtype=dtype, inner_fn=inner_fn, ranges=list(size))
    offset_node = philox_rand_offset(size)
    return (random_values_node, offset_node)

@register_lowering(aten.native_dropout, type_promotion_kind=None)
def native_dropout(x, p, train):
    if False:
        i = 10
        return i + 15
    if config.fallback_random:
        return pytree.tree_map(TensorBox.create, ir.FallbackKernel.create(aten.native_dropout.default, x, p, train))
    else:
        raise AssertionError('should be handled in replace_random.py')

@register_lowering(aten.bernoulli_, type_promotion_kind=None)
def bernoulli_(x, *args):
    if False:
        print('Hello World!')
    assert config.fallback_random or x.get_device() == torch.device('cpu'), 'this should be handled in decomps unless config.fallback_random or the device is CPU'
    x.realize()
    ir.InplaceBernoulliFallback(x, *args)
    return x

@register_lowering(aten.bernoulli.p, type_promotion_kind=None)
def bernoulli_p(x, *args):
    if False:
        while True:
            i = 10
    assert config.fallback_random or x.get_device() == torch.device('cpu'), 'this should be handled in decomps unless config.fallback_random or the device is CPU'
    return bernoulli_(clone(x), *args)

@register_lowering(aten._foobar)
def _foobar(_):
    if False:
        print('Hello World!')
    raise AssertionError()

@functools.lru_cache(1)
def _warn_triton_random(salt):
    if False:
        i = 10
        return i + 15
    log.info('using triton random, expect difference from eager')

def warn_triton_random():
    if False:
        for i in range(10):
            print('nop')
    _warn_triton_random(V.graph.creation_time)
fallback_rand_default = fallback_handler(aten.rand.default)
fallback_rand_generator = fallback_handler(aten.rand.generator)
fallback_randn_default = fallback_handler(aten.randn.default)
fallback_randn_generator = fallback_handler(aten.randn.generator)
make_fallback(aten.randint)

@register_lowering(aten.rand)
def rand(*args, **kwargs):
    if False:
        while True:
            i = 10
    if kwargs.get('generator', None) is not None:
        return fallback_rand_generator(*args, **kwargs)
    elif config.fallback_random:
        kwargs.pop('generator', None)
        return fallback_rand_default(*args, **kwargs)
    raise AssertionError('should have been handled in replace_random.py')

@register_lowering(aten.randn)
def randn(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    if kwargs.get('generator', None) is not None:
        return fallback_randn_generator(*args, **kwargs)
    elif config.fallback_random:
        kwargs.pop('generator', None)
        return fallback_randn_default(*args, **kwargs)
    raise AssertionError('should have been handled in replace_random.py')

@register_lowering(inductor_prims.force_stride_order, type_promotion_kind=None)
def inductor_force_stride_order(input_tensor, stride):
    if False:
        while True:
            i = 10
    stride_order = ir.get_stride_order(stride)
    return ir.ExternKernel.require_stride_order(input_tensor, stride_order)

@register_lowering(inductor_prims.seed, type_promotion_kind=None)
def inductor_seed(device: torch.device):
    if False:
        for i in range(10):
            print('nop')
    raise AssertionError('should be handled in fuse_seed_creation_pass()')

@register_lowering(inductor_prims.seeds, type_promotion_kind=None)
def inductor_seeds(count, device):
    if False:
        while True:
            i = 10
    warn_triton_random()
    return TensorBox.create(ir.RandomSeeds(count, decode_device(device)))

@register_lowering(inductor_prims.lookup_seed, type_promotion_kind=None)
def inductor_lookup_seed(seeds, index):
    if False:
        print('Hello World!')

    def inner_fn(_):
        if False:
            return 10
        return ops.load_seed(seeds.get_name(), index)
    return Pointwise.create(device=seeds.get_device(), dtype=seeds.get_dtype(), inner_fn=inner_fn, ranges=[])

@register_lowering(inductor_prims.random, type_promotion_kind=None)
def inductor_random(size: List[int], seed: TensorBox, mode: str, *, offset: int=0):
    if False:
        i = 10
        return i + 15
    assert not config.fallback_random
    assert mode in ('rand', 'randn')
    size = [*size]
    dtype = torch.float32
    device = seed.get_device()
    random_pos = ir.FixedLayout(device, dtype, size, ir.FlexibleLayout.contiguous_strides(size), offset=offset).make_indexer()
    seed_loader = seed.make_loader()

    def inner_fn(index):
        if False:
            print('Hello World!')
        return getattr(ops, mode)(seed_loader([]), ops.index_expr(random_pos(index), torch.int32))
    result = Pointwise.create(device=device, dtype=dtype, inner_fn=inner_fn, ranges=[*size])
    result.realize()
    return result

@register_lowering(inductor_prims.randint, type_promotion_kind=None)
def inductor_randint(low: int, high: int, size: List[int], seed: TensorBox, *, offset: int=0):
    if False:
        i = 10
        return i + 15
    assert not config.fallback_random
    size = [*size]
    dtype = torch.int64
    device = seed.get_device()
    random_pos = ir.FixedLayout(device, dtype, size, ir.FlexibleLayout.contiguous_strides(size), offset=offset).make_indexer()
    seed_loader = seed.make_loader()

    def inner_fn(index):
        if False:
            return 10
        return ops.randint64(seed_loader([]), ops.index_expr(random_pos(index), torch.int32), low, high)
    return Pointwise.create(device=device, dtype=dtype, inner_fn=inner_fn, ranges=[*size])

@register_lowering(aten.bucketize, type_promotion_kind=None)
def bucketize(input: TensorBox, boundaries: TensorBox, *, out_int32: bool=False, right: bool=False):
    if False:
        i = 10
        return i + 15
    assert len(boundaries.get_size()) == 1
    if not (is_triton(input) and is_triton(boundaries)):
        return fallback_handler(aten.bucketize.Tensor, add_to_fallback_set=False)(input, boundaries, out_int32=out_int32, right=right)
    boundaries.realize()
    boundaries_size = boundaries.get_size()[0]
    boundaries_loader = boundaries.make_loader()
    device = input.get_device()
    input_loader = input.make_loader()
    index_dtype = torch.int32 if out_int32 else torch.int64

    def inner_fn(index):
        if False:
            return 10
        val = input_loader(index)
        indices = ops.bucketize(val, boundaries.get_name(), boundaries_size, index_dtype, right)
        return indices
    return Pointwise.create(device=device, dtype=index_dtype, inner_fn=inner_fn, ranges=input.get_size())

def require_dense(_, *args, **kwargs):
    if False:
        return 10
    (args, kwargs) = pytree.tree_map_only(ir.IRNode, ir.ExternKernel.require_stride1, (args, kwargs))
    return (args, kwargs)

def require_contiguous(_, *args, **kwargs):
    if False:
        return 10
    (args, kwargs) = pytree.tree_map_only(ir.IRNode, ir.ExternKernel.require_contiguous, (args, kwargs))
    return (args, kwargs)

def require_channels_last(_, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    (args, kwargs) = pytree.tree_map_only(ir.IRNode, ir.ExternKernel.require_channels_last, (args, kwargs))
    return (args, kwargs)

def constrain_to_fx_strides(fx_node, *args, **kwargs):
    if False:
        return 10

    def apply_constraint(arg, fx_arg):
        if False:
            print('Hello World!')
        if isinstance(arg, ir.IRNode):
            stride_order = ir.get_stride_order(fx_arg.meta['val'].stride())
            return ir.ExternKernel.require_stride_order(arg, stride_order)
        return arg
    args = tuple((apply_constraint(arg, fx_arg) for (arg, fx_arg) in zip(args, fx_node.args)))
    kwargs = {k: apply_constraint(v, fx_node.kwargs[k]) for (k, v) in kwargs.items()}
    return (args, kwargs)
FALLBACK_ALLOW_LIST = {'torchvision::roi_align'}
make_fallback(aten._adaptive_avg_pool2d_backward, require_dense)
make_fallback(aten.convolution_backward, constrain_to_fx_strides)
make_fallback(aten._cudnn_rnn, require_dense)
make_fallback(aten._cudnn_rnn_backward, require_contiguous)
make_fallback(aten.cumsum, require_dense, warn=False)
make_fallback(aten.cumprod, require_dense, warn=False)
make_fallback(aten._embedding_bag, require_contiguous)
make_fallback(aten._embedding_bag_forward_only, require_contiguous)
make_fallback(aten._flash_attention_forward)
make_fallback(aten._flash_attention_backward)
make_fallback(aten._fused_moving_avg_obs_fq_helper)
make_fallback(aten._fused_moving_avg_obs_fq_helper_functional)
make_fallback(aten.grid_sampler_2d_backward, require_dense)
make_fallback(aten.randperm)

def sdpa_constraint(fx_node, *args, **kwargs):
    if False:
        return 10

    def apply_constraint(arg, fx_arg):
        if False:
            while True:
                i = 10
        if not isinstance(arg, ir.IRNode):
            return arg
        meta_val = fx_arg.meta['val']
        if not meta_val.is_cuda:
            return arg
        stride_order = ir.get_stride_order(meta_val.stride())
        if stride_order and stride_order[-1] != 0:
            stride_order = list(reversed(range(len(arg.get_size()))))
        ALIGNMENT = 16

        def is_aligned(x):
            if False:
                while True:
                    i = 10
            return V.graph.sizevars.size_hint(x.get_size()[-1]) % ALIGNMENT == 0
        assert isinstance(arg, TensorBox)
        unaligned_input_shape = isinstance(arg.data, ir.SliceView) and (not is_aligned(arg))
        aligned_input_view = unaligned_input_shape and is_aligned(arg.unwrap_view())
        if aligned_input_view:
            return arg
        return ir.ExternKernel.require_stride_order(arg, stride_order)
    args = tuple((apply_constraint(arg, fx_arg) for (arg, fx_arg) in zip(args, fx_node.args)))
    kwargs = {k: apply_constraint(v, fx_node.kwargs[k]) for (k, v) in kwargs.items()}
    return (args, kwargs)
make_fallback(aten._scaled_dot_product_efficient_attention.default, sdpa_constraint, warn=False)
make_fallback(aten._scaled_dot_product_efficient_attention_backward.default, sdpa_constraint, warn=False)
make_fallback(aten._scaled_dot_product_flash_attention.default, sdpa_constraint, warn=False)
make_fallback(aten._scaled_dot_product_flash_attention_backward.default, sdpa_constraint, warn=False)
make_fallback(torch.ops.aten._efficient_attention_forward.default)
make_fallback(torch.ops.aten._efficient_attention_backward.default)
make_fallback(aten.sort)
make_fallback(aten.sort.stable)
make_fallback(aten._sparse_coo_tensor_with_dims_and_tensors)
make_fallback(aten._thnn_fused_lstm_cell, require_dense)
make_fallback(aten.topk)
make_fallback(aten.upsample_bicubic2d_backward, require_contiguous)
make_fallback(aten._scaled_mm.default, constrain_to_fx_strides)
make_fallback(aten.view_as_complex, require_contiguous)
make_fallback(aten.upsample_linear1d)
make_fallback(aten.upsample_trilinear3d)
make_fallback(aten.upsample_linear1d_backward)
make_fallback(aten.upsample_trilinear3d_backward)
make_fallback(aten._adaptive_avg_pool3d)
make_fallback(aten.adaptive_max_pool2d)
make_fallback(aten.adaptive_max_pool3d)
make_fallback(aten.addbmm)
make_fallback(aten.addmv, warn=False)
make_fallback(aten._addmm_activation, warn=False)
make_fallback(aten.avg_pool3d)
make_fallback(aten.block_diag)
make_fallback(aten._cdist_forward)
make_fallback(aten.cummax)
make_fallback(aten.cummin)
make_fallback(aten.cumprod, warn=False)
make_fallback(aten.digamma, warn=False)
make_fallback(aten._efficientzerotensor)
make_fallback(aten._embedding_bag_per_sample_weights_backward)
make_fallback(aten._efficientzerotensor)
make_fallback(aten._embedding_bag_per_sample_weights_backward)
make_fallback(aten.fractional_max_pool2d)
make_fallback(aten.fractional_max_pool3d)
make_fallback(aten.frexp)
make_fallback(aten.geqrf)
make_fallback(aten.histc)
make_fallback(aten.i0)
make_fallback(aten.igamma, warn=False)
make_fallback(aten.igammac, warn=False)
make_fallback(aten.isin)
make_fallback(aten.kthvalue)
make_fallback(aten.linalg_cholesky_ex)
make_fallback(aten.linalg_cross)
make_fallback(aten._linalg_det)
make_fallback(aten.linalg_householder_product)
make_fallback(aten.linalg_inv_ex)
make_fallback(aten.linalg_ldl_factor_ex)
make_fallback(aten.linalg_ldl_solve)
make_fallback(aten.linalg_lu)
make_fallback(aten.linalg_lu_factor_ex)
make_fallback(aten.linalg_lu_solve)
make_fallback(aten.linalg_matrix_exp)
make_fallback(aten.linalg_qr)
make_fallback(aten._linalg_slogdet)
make_fallback(aten._linalg_solve_ex)
make_fallback(aten.linalg_solve_triangular)
make_fallback(aten._linalg_svd)
make_fallback(aten.logcumsumexp)
make_fallback(aten.lu_unpack)
make_fallback(aten.max_pool3d_with_indices)
make_fallback(aten.max_unpool2d)
make_fallback(aten.max_unpool3d)
make_fallback(aten.median)
make_fallback(aten.mode)
make_fallback(aten.nanmedian)
make_fallback(aten.ormqr)
make_fallback(aten._pdist_forward)
make_fallback(aten.pixel_shuffle)
make_fallback(aten.pixel_unshuffle)
make_fallback(aten.polygamma)
make_fallback(aten.put)
make_fallback(aten.reflection_pad1d)
make_fallback(aten.replication_pad1d)
make_fallback(aten.resize)
make_fallback(aten.resize_)
make_fallback(aten.resize_as)
make_fallback(aten.resize_as_)
make_fallback(aten.searchsorted)
make_fallback(aten.special_airy_ai)
make_fallback(aten.special_bessel_j0, warn=False)
make_fallback(aten.special_bessel_j1, warn=False)
make_fallback(aten.special_bessel_y0, warn=False)
make_fallback(aten.special_bessel_y1)
make_fallback(aten.special_chebyshev_polynomial_t)
make_fallback(aten.special_chebyshev_polynomial_u)
make_fallback(aten.special_erfcx, warn=False)
make_fallback(aten.special_hermite_polynomial_h)
make_fallback(aten.special_hermite_polynomial_he)
make_fallback(aten.special_i0e, warn=False)
make_fallback(aten.special_i1, warn=False)
make_fallback(aten.special_i1e, warn=False)
make_fallback(aten.special_laguerre_polynomial_l)
make_fallback(aten.special_modified_bessel_i0)
make_fallback(aten.special_modified_bessel_i1)
make_fallback(aten.special_modified_bessel_k0)
make_fallback(aten.special_modified_bessel_k1)
make_fallback(aten.special_ndtri, warn=False)
make_fallback(aten.special_scaled_modified_bessel_k0)
make_fallback(aten.special_scaled_modified_bessel_k1)
make_fallback(aten.special_spherical_bessel_j0, warn=False)
make_fallback(aten.special_zeta, warn=False)
make_fallback(aten.take)
make_fallback(aten._trilinear)
make_fallback(aten.uniform, warn=False)
make_fallback(aten._adaptive_avg_pool3d_backward)
make_fallback(aten.adaptive_max_pool2d_backward)
make_fallback(aten.adaptive_max_pool3d_backward)
make_fallback(aten.avg_pool3d_backward)
make_fallback(aten._cdist_backward)
make_fallback(aten._embedding_bag_dense_backward)
make_fallback(aten.fractional_max_pool2d_backward)
make_fallback(aten.fractional_max_pool3d_backward)
make_fallback(aten._linalg_check_errors)
make_fallback(aten.max_pool3d_with_indices_backward)
make_fallback(aten._pdist_backward)
make_fallback(aten.reflection_pad1d_backward)
make_fallback(aten.replication_pad1d_backward)
make_fallback(aten.soft_margin_loss_backward, warn=False)
make_fallback(aten.linalg_pinv.atol_rtol_tensor)
make_fallback(aten.segment_reduce.default)
make_fallback(aten._segment_reduce_backward.default)
make_fallback(aten.angle)
make_fallback(aten.cholesky_inverse)
make_fallback(aten.cholesky_solve)
make_fallback(aten._fft_r2c)
make_fallback(aten.histogram.bin_ct)
make_fallback(aten._histogramdd_bin_edges.default)
make_fallback(aten._histogramdd_from_bin_cts.default)
make_fallback(aten.index_reduce)
make_fallback(aten.masked_scatter)
make_fallback(aten.masked_scatter_backward)
make_fallback(aten.to_sparse)
make_fallback(aten._to_sparse)
make_fallback(aten.triangular_solve)
make_fallback(aten.gcd.default, warn=False)
make_fallback(aten._linalg_eigh)
make_fallback(aten.zeros.names)
make_fallback(aten.nonzero.default)
make_fallback(torch._prims.rng_prims.run_and_save_rng_state)
make_fallback(torch._prims.rng_prims.run_with_rng_state)
make_fallback(aten.exponential.default, warn=False)

@register_lowering(aten.copy, type_promotion_kind=None)
def copy(self, src, non_blocking=False):
    if False:
        for i in range(10):
            print('nop')
    x = src
    if self.get_device() != src.get_device():
        x = to_device(x, self.get_device())
    if self.get_dtype() != src.get_dtype():
        x = to_dtype(x, self.get_dtype())
    if self.get_size() != src.get_size():
        out = expand(x, self.get_size())
        return clone(out)
    return clone(x)

@register_lowering(aten.clone)
def clone(x, *, memory_format=None):
    if False:
        print('Hello World!')
    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=x.make_loader(), ranges=list(x.get_size()))
if hasattr(aten, 'lift_fresh_copy'):
    register_lowering(aten.lift_fresh_copy)(clone)

@register_lowering(prims.iota)
def iota(length, *, start, step, dtype, device, requires_grad):
    if False:
        for i in range(10):
            print('nop')

    def fn(index):
        if False:
            print('Hello World!')
        return ops.index_expr(step * index[0] + start, dtype=dtype)
    return Pointwise.create(device=decode_device(device), dtype=dtype, inner_fn=fn, ranges=[length])

@register_lowering(aten.select_scatter, type_promotion_kind=None)
def select_scatter(x, src, dim: int, index: int):
    if False:
        for i in range(10):
            print('nop')
    assert x.get_dtype() == src.get_dtype()
    x_loader = x.make_loader()
    dim = _validate_dim(x, dim, 0)
    if V.graph.sizevars.evaluate_expr(sympy.Lt(index, 0)):
        index = index + x.get_size()[dim]
    V.graph.sizevars.guard_leq(0, index)
    V.graph.sizevars.guard_lt(index, x.get_size()[dim])
    src = expand(unsqueeze(src, dim), x.get_size())
    src_loader = src.make_loader()

    def inner_fn(idx):
        if False:
            i = 10
            return i + 15
        return ops.where(ops.eq(ops.index_expr(idx[dim], torch.int32), ops.index_expr(index, torch.int32)), src_loader(idx), x_loader(idx))
    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=inner_fn, ranges=list(x.get_size()))

@register_lowering(aten.slice_scatter, type_promotion_kind=None)
def slice_scatter(x, src, dim=0, start=None, end=None, step=1):
    if False:
        while True:
            i = 10
    assert x.get_dtype() == src.get_dtype()
    x_loader = x.make_loader()
    dim = _validate_dim(x, dim, 0)
    dim_size = x.get_size()[dim]
    if start is not None and V.graph.sizevars.evaluate_expr(sympy.Lt(start, 0)):
        start = start + dim_size
    if end is not None and V.graph.sizevars.evaluate_expr(sympy.Lt(end, 0)):
        end = end + dim_size
    if start is None:
        start = 0
    if end is None or V.graph.sizevars.statically_known_leq(x.get_size()[dim], end):
        end = dim_size
    src_size = list(x.get_size())
    src_size[dim] = FloorDiv(sympy.expand(end - start), sympy.expand(step))
    src = expand(src, src_size)
    src_loader = src.make_loader()

    def inner_fn(idx):
        if False:
            print('Hello World!')
        if start == 0 and end == dim_size and (step == 1):
            return src_loader(idx)
        idx_dim = ops.index_expr(idx[dim], torch.int64)
        src_idx = list(idx)
        src_idx[dim] = FloorDiv(idx[dim] - start, step)
        mask = []
        if start != 0:
            mask.append(ops.ge(idx_dim, ops.index_expr(sympy.expand(start), torch.int64)))
        if end != dim_size:
            mask.append(ops.lt(idx_dim, ops.index_expr(sympy.expand(end), torch.int64)))
        if step != 1:
            mask.append(ops.eq(ops.index_expr(ModularIndexing(idx[dim] - start, 1, step), torch.int64), ops.constant(0, torch.torch.int64)))
        assert mask
        mask = functools.reduce(ops.and_, mask)
        src_val = ops.masked(mask, lambda : src_loader(src_idx), 0 if is_integer_type(x) else 0.0)
        return ops.where(mask, src_val, x_loader(idx))
    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=inner_fn, ranges=list(x.get_size()))

def _unwrap(x):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(x, (list, tuple)) and len(x) > 0:
        return _unwrap(x[0])
    return x

@register_lowering([torch.tensor, aten.scalar_tensor])
def tensor(data, *, dtype=None, device=None, layout=None, pin_memory=False):
    if False:
        return 10
    assert_nyi(layout in (None, torch.strided), f'layout={layout}')
    assert_nyi(not pin_memory, 'pin_memory')
    if isinstance(_unwrap(data), int):
        dtype = dtype or torch.int64
    else:
        dtype = dtype or torch.get_default_dtype()
    ranges: List[sympy.Expr] = []
    if isinstance(data, sympy.Expr):

        def inner_fn(index):
            if False:
                for i in range(10):
                    print('nop')
            return ops.index_expr(data, dtype)
    elif isinstance(data, (float, int)):

        def inner_fn(index):
            if False:
                i = 10
                return i + 15
            return ops.constant(data, dtype)
    elif len(data) == 0 or (isinstance(data[0], (float, int)) and len(data) <= 8):
        ranges.append(sympy.Integer(len(data)))

        def inner_fn(index):
            if False:
                for i in range(10):
                    print('nop')

            def binary_search(start, end):
                if False:
                    print('Hello World!')
                assert start < end
                if end - start == 1:
                    return ops.constant(data[start], dtype)
                mid = (end - start) // 2 + start
                return ops.where(ops.lt(ops.index_expr(index[0], torch.int64), ops.constant(mid, torch.int64)), binary_search(start, mid), binary_search(mid, end))
            if len(data) == 0:
                return ops.constant(0, dtype)
            return binary_search(0, len(data))
    else:
        return V.graph.add_tensor_constant(torch.tensor(data, dtype=dtype, device=device))
    return Pointwise.create(device=decode_device(device), dtype=dtype, inner_fn=inner_fn, ranges=ranges)

@register_lowering(torch.as_tensor)
def as_tensor(data, dtype=None, device=None):
    if False:
        print('Hello World!')
    if isinstance(data, TensorBox):
        if dtype is not None:
            data = to_dtype(data, dtype)
        if device is not None:
            data = to_device(data, device)
        return data
    return tensor(data, dtype=dtype, device=device)

@register_lowering(torch.LongTensor)
def long_tensor(data):
    if False:
        for i in range(10):
            print('nop')
    return tensor(data, dtype=torch.int64)

@register_lowering(aten._local_scalar_dense)
def _local_scalar_dense(data):
    if False:
        for i in range(10):
            print('nop')
    sym = V.graph.current_node.meta['val'].node.expr
    buffer = ir.DynamicScalar(sym, data)
    buffer.name = V.graph.register_buffer(buffer)
    return sym

def _full(fill_value, device, dtype, size):
    if False:
        while True:
            i = 10
    value = fill_value
    if not isinstance(fill_value, (int, float)) and hasattr(value, 'value'):
        value = value.value
    if isinstance(value, (int, float)):

        def inner_fn(index):
            if False:
                i = 10
                return i + 15
            return ops.constant(value, dtype)
    elif isinstance(value, sympy.Expr):

        def inner_fn(index):
            if False:
                print('Hello World!')
            return ops.index_expr(value, dtype)
    else:
        assert len(value.get_size()) == 0
        value_loader = value.make_loader()

        def inner_fn(index):
            if False:
                i = 10
                return i + 15
            return value_loader([])
    return Pointwise.create(device=device, dtype=dtype, inner_fn=inner_fn, ranges=list(size))

@register_lowering(aten.full_like, type_promotion_kind=None)
def full_like(x, fill_value, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return create_tensor_like(tensor_constructor(fill_value))(x, **kwargs)

def tensor_constructor(fill_value):
    if False:
        return 10

    def inner(*size, names=None, dtype=None, device=None, layout=None, pin_memory=False, memory_format=None):
        if False:
            print('Hello World!')
        assert_nyi(names is None, 'named tensors')
        assert_nyi(layout in (None, torch.strided), f'layout={layout}')
        assert_nyi(not pin_memory, 'pin_memory')
        device = decode_device(device)
        dtype = dtype or torch.get_default_dtype()
        if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)):
            size = tuple(size[0])
        size = [sympy.expand(s) for s in size]
        return _full(fill_value, device, dtype, size)
    return inner

@register_lowering([torch.empty, aten.empty])
def empty(*size, names=None, dtype=None, layout=None, device=None, pin_memory=None, memory_format=None):
    if False:
        print('Hello World!')
    assert_nyi(names is None, 'named tensors')
    device = decode_device(device)
    if len(size) == 1 and isinstance(size[0], (list, tuple, torch.Size)):
        size = tuple(size[0])
    return empty_strided(size, None, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)

def create_tensor_like(creation_fn):
    if False:
        while True:
            i = 10
    '\n    Shim to convert X_like(...) into X(...).  For example zeros_like() into zeros().\n    '

    def _constant_like(x, *, dtype=None, device=None, layout=None, pin_memory=False, memory_format=None):
        if False:
            i = 10
            return i + 15
        assert_nyi(not pin_memory, 'pin_memory')
        assert_nyi(layout in (None, torch.strided), f'layout={layout}')
        if dtype is None:
            dtype = x.get_dtype()
        else:
            dtype = decode_dtype(dtype)
        device = device or x.get_device()
        size = list(x.get_size())
        return creation_fn(size, dtype=dtype, device=device, layout=layout, pin_memory=pin_memory)
    return _constant_like

def constant_like(fill_value):
    if False:
        for i in range(10):
            print('nop')
    return create_tensor_like(tensor_constructor(fill_value))
empty_like = register_lowering(aten.empty_like)(create_tensor_like(empty))
ones_like = create_tensor_like(tensor_constructor(1))
zeros_like = create_tensor_like(tensor_constructor(0))

def new_constant(fill_value):
    if False:
        print('Hello World!')

    def _new_constant(x, size, *, dtype=None, layout=None, device=None, pin_memory=None):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(size, (list, tuple))
        assert_nyi(not pin_memory, 'pin_memory')
        assert_nyi(layout in (None, torch.strided), f'layout={layout}')
        dtype = decode_dtype(dtype) or x.get_dtype()
        device = device or x.get_device()
        size = [sympy.Integer(s) for s in size]
        return _full(fill_value, device, dtype, size)
    return _new_constant

@register_lowering(aten.new_empty)
def new_empty(x, size, *, dtype=None, layout=None, device=None, pin_memory=None):
    if False:
        while True:
            i = 10
    if dtype is None:
        dtype = x.get_dtype()
    if device is None:
        device = x.get_device()
    return empty_strided(size, None, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)

@register_lowering(aten.empty_strided)
def empty_strided(size, stride, *, dtype=None, layout=None, device=None, pin_memory=None):
    if False:
        return 10
    assert isinstance(size, (list, tuple))
    assert isinstance(stride, (list, tuple, type(None)))
    assert_nyi(not pin_memory, 'pin_memory')
    assert_nyi(layout in (None, torch.strided), f'layout={layout}')
    dtype = decode_dtype(dtype) or torch.get_default_dtype()
    device = device or torch.tensor(0.0).device
    pointwise = _full(fill_value=0, device=device, dtype=dtype, size=size)
    pointwise.realize()
    buffer = pointwise.data.data
    buffer.data.ranges = [0] * len(size)
    assert isinstance(buffer, ir.ComputedBuffer)
    size = [sympy.expand(s) for s in size]
    stride = [sympy.expand(s) for s in stride] if stride else ir.FlexibleLayout.contiguous_strides(size)
    buffer.layout = ir.FixedLayout(device=device, dtype=dtype, size=size, stride=stride)
    return pointwise

@register_lowering(aten.new_empty_strided)
def new_empty_strided(x, size, stride, *, dtype=None, layout=None, device=None, pin_memory=None):
    if False:
        i = 10
        return i + 15
    if dtype is None:
        dtype = x.get_dtype()
    if device is None:
        device = x.get_device()
    return empty_strided(size, stride, dtype=dtype, layout=layout, device=device, pin_memory=pin_memory)

@register_lowering(prims.copy_strided.default)
def copy_strided(x, stride):
    if False:
        i = 10
        return i + 15
    stride = [V.graph.sizevars.size_hint(s) for s in stride]
    stride_order = sorted(range(len(stride)), key=stride.__getitem__)
    return ir.ExternKernel.require_stride_order(x, stride_order)

@register_lowering([torch.full, aten.full])
def full(size, fill_value, **kwargs):
    if False:
        while True:
            i = 10
    assert kwargs.get('dtype') is not None, 'dtype should be handled by decomposition'
    return tensor_constructor(fill_value)(size, **kwargs)

@register_lowering(aten.gather, type_promotion_kind=None)
def gather(x, dim, index, sparse_grad=False):
    if False:
        while True:
            i = 10
    assert isinstance(x, TensorBox)
    assert index.get_dtype() == torch.int64
    size = x.get_size()
    offset = len(size) == 0
    dim = _validate_dim(x, dim, offset)
    x_loader = x.make_loader()
    index_loader = index.make_loader()

    def fn(idx):
        if False:
            for i in range(10):
                print('nop')
        idx = list(idx)
        if len(idx) != 0:
            idx[dim] = ops.indirect_indexing(index_loader(idx), size[dim])
        return x_loader(idx)
    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=fn, ranges=index.get_size())

@register_lowering(aten.embedding, type_promotion_kind=None)
def embedding(weight, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False):
    if False:
        while True:
            i = 10
    assert not sparse
    assert isinstance(weight, TensorBox)
    assert isinstance(indices, TensorBox)
    assert 'int' in str(indices.get_dtype())
    weight_loader = weight.make_loader()
    indices_loader = indices.make_loader()
    indices_ndim = len(indices.get_size())
    weight_size = weight.get_size()
    new_size = [*indices.get_size(), *weight_size[1:]]

    def fn(idx):
        if False:
            i = 10
            return i + 15
        assert len(idx) == len(new_size), f'{idx} != {new_size}'
        var_index = indices_loader(idx[:indices_ndim])
        weight_idx = [ops.indirect_indexing(var_index, weight_size[0])] + [*idx[indices_ndim:]]
        return weight_loader(weight_idx)
    return Pointwise.create(device=weight.get_device(), dtype=weight.get_dtype(), inner_fn=fn, ranges=new_size)

def check_and_broadcast_indices(indices, device):
    if False:
        i = 10
        return i + 15
    assert all((i.get_dtype() in (torch.int64, torch.int32, torch.bool, torch.uint8) for i in indices if i is not None)), f'indices must be int64, byte or bool. Got {[i.get_dtype() for i in indices if i is not None]}'
    if any((i.get_dtype() in (torch.bool, torch.uint8) for i in indices if i is not None)):
        raise NotImplementedError('Fallback for bool indices')
    valid_idxs = [i for (i, x) in enumerate(indices) if isinstance(x, TensorBox)]
    assert len(valid_idxs) > 0, 'requires at least 1 non-None index'
    new_indices = [None] * len(indices)
    for (i, x) in zip(valid_idxs, broadcast_tensors(*[indices[i] for i in valid_idxs])):
        if x.get_device() != device:
            raise NotImplementedError('Fallback when indices is on a different device')
        new_indices[i] = x
    return (new_indices, valid_idxs)

def index_output_size_and_inner_fn(x_size, indices, tensor_indices, tensor_size, indices_loaders, indexed_size, x_loader, check):
    if False:
        print('Hello World!')
    non_consecutive_tensors = False
    for (previous, current) in zip(tensor_indices, tensor_indices[1:]):
        if current - previous != 1:
            non_consecutive_tensors = True
    output_size = [x_size[i] for (i, val) in enumerate(indices) if val is None]
    output_size = [*output_size, *x_size[len(output_size) + len(tensor_indices):]]
    first_tensor_index = tensor_indices[0]
    if non_consecutive_tensors:
        output_size = tensor_size + output_size
    else:
        output_size = output_size[:first_tensor_index] + tensor_size + output_size[first_tensor_index:]

    def fn(idx):
        if False:
            for i in range(10):
                print('nop')
        assert len(idx) == len(output_size)
        assert len(indices_loaders) == len(indexed_size)
        rank = len(tensor_size)
        new_index = []
        first_tensor_index = tensor_indices[0]
        start_offset = 0 if non_consecutive_tensors else first_tensor_index
        next_idx = 0
        for i in range(tensor_indices[-1] + 1):
            if i == start_offset:
                next_idx += rank
            if indices[i] is None:
                assert next_idx < len(idx)
                new_index.append(idx[next_idx])
                next_idx += 1
            else:
                loader = indices_loaders[i]
                assert loader is not None
                size = indexed_size[i]
                new_index.append(ops.indirect_indexing(loader(idx[start_offset:start_offset + rank]), size, check=check))
        new_index = [*new_index, *idx[next_idx:]]
        return new_index if x_loader is None else x_loader(new_index)
    return (output_size, fn)

def index_impl(x, indices, check):
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(indices, (list, tuple))
    x_loader = x.make_loader()
    (indices, tensor_indices) = check_and_broadcast_indices(indices, x.get_device())
    assert len(tensor_indices) > 0, 'Must have at least one valid idx'
    indices_loaders = [i.make_loader() if i is not None else None for i in indices]
    tensor_size = list(indices[tensor_indices[0]].get_size())
    x_size = x.get_size()
    indexed_size = [x_size[i] for i in range(len(indices)) if indices[i] is not None]
    if 0 in indexed_size and 0 not in tensor_size:
        raise IndexError('index is out of bounds for dimension with size 0')
    indexed_size = [x_size[i] for i in range(len(indices))]
    (output_size, inner_fn) = index_output_size_and_inner_fn(x_size, indices, tensor_indices, tensor_size, indices_loaders, indexed_size, x_loader, check=check)
    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=inner_fn, ranges=output_size)

@register_lowering(aten.index, type_promotion_kind=None)
def index(x, indices):
    if False:
        return 10
    try:
        return index_impl(x, indices, check=True)
    except NotImplementedError:
        x.realize()
        return fallback_handler(aten.index.Tensor)(x, indices)

@register_lowering(aten._unsafe_index, type_promotion_kind=None)
def _unsafe_index(x, indices):
    if False:
        return 10
    return index_impl(x, indices, check=False)

@register_lowering(aten.index_put)
def index_put(x, indices, values, accumulate=False):
    if False:
        for i in range(10):
            print('nop')
    return index_put_(clone(x), indices, values, accumulate)

@register_lowering(aten._unsafe_index_put)
def _unsafe_index_put(x, indices, values, accumulate=False):
    if False:
        print('Hello World!')
    return index_put_impl_(clone(x), indices, values, accumulate, check=False)

def index_put_as_masked_fill(self, indices, value, accumulate):
    if False:
        return 10
    if value.get_device() != self.get_device():
        value = to_device(value, self.get_device())
    if accumulate:
        value = add(self, value)
    return mutate_to(self, where(indices[0], value, self))

def index_put_fallback(self, indices, values, accumulate):
    if False:
        i = 10
        return i + 15
    if is_triton(values) and (accumulate is True or torch.are_deterministic_algorithms_enabled()):
        V.graph.disable_cudagraphs = True
    ir.IndexPutFallback(self, indices, values, accumulate)
    return self

@register_lowering(aten.index_put_, type_promotion_kind=None)
def index_put_(self, indices, values, accumulate=False):
    if False:
        print('Hello World!')
    return index_put_impl_(self, indices, values, accumulate, check=True)

@register_lowering(inductor_prims._unsafe_index_put_, type_promotion_kind=None)
def _unsafe_index_put_(self, indices, values, accumulate=False):
    if False:
        for i in range(10):
            print('nop')
    return index_put_impl_(self, indices, values, accumulate, check=False)

def needs_fallback_due_to_atomic_add_limitations(dtype):
    if False:
        i = 10
        return i + 15
    return dtype in {torch.int64, torch.bool, torch.bfloat16}

def index_put_impl_(self, indices, values, accumulate, check):
    if False:
        for i in range(10):
            print('nop')
    if values.get_numel() == 1 and len(indices) == 1 and (indices[0].get_dtype() in {torch.bool, torch.uint8}):
        mask = indices[0]
        for _ in range(len(mask.get_size()), len(self.get_size())):
            mask = unsqueeze(mask, -1)
        return index_put_as_masked_fill(self, [mask], values, accumulate)
    if torch.are_deterministic_algorithms_enabled():
        return index_put_fallback(self, indices, values, accumulate)
    for index in indices:
        if index is not None and index.get_dtype() in {torch.bool, torch.uint8}:
            return index_put_fallback(self, indices, values, accumulate)
    x_size = self.get_size()
    x_ndim = len(x_size)
    if needs_fallback_due_to_atomic_add_limitations(self.get_dtype()):
        if x_ndim == 0:
            self = view(self, [1])
        self = index_put_fallback(self, indices, values, accumulate)
        if x_ndim == 0:
            self = view(self, [])
        return self
    values = to_dtype(values, self.get_dtype())
    try:
        (indices, tensor_indices) = check_and_broadcast_indices(indices, self.get_device())
    except NotImplementedError:
        return index_put_fallback(self, indices, values, accumulate)
    indices_loaders = [i.make_loader() if i is not None else None for i in indices]
    assert isinstance(self, TensorBox)
    self.realize()
    if x_ndim == 0:
        self = view(self, [1])
    tensor_size = list(indices[tensor_indices[0]].get_size())
    indexed_size = [x_size[i] for i in range(len(indices))]
    (expected_vals_size, inner_fn) = index_output_size_and_inner_fn(x_size, indices, tensor_indices, tensor_size, indices_loaders, indexed_size, None, check=check)
    values = expand(values, expected_vals_size)
    scatter = ir.Scatter(device=self.get_device(), dtype=self.get_dtype(), inner_fn=values.make_loader(), ranges=expected_vals_size, output_indexer=inner_fn, scatter_mode='atomic_add' if accumulate else None)
    buffer = ir.ComputedBuffer(None, ir.MutationLayout(self), scatter)
    buffer.name = V.graph.register_buffer(buffer)
    if x_ndim == 0:
        self = view(self, [])
    return self

@register_lowering(inductor_prims.masked_scatter_with_index, type_promotion_kind=None, broadcast=False)
def masked_scatter_with_index(self, mask, source_idx, source):
    if False:
        for i in range(10):
            print('nop')
    (self_flat, mask_flat, source_flat) = (view(x, (-1,)) for x in (self, mask, source))
    assert self.get_size() == mask.get_size()
    assert mask.get_dtype() in {torch.bool, torch.uint8}
    self_loader = self_flat.make_loader()
    mask_loader = mask_flat.make_loader()
    source_idx_loader = source_idx.make_loader()
    source_loader = source_flat.make_loader()
    source_numel = source.get_numel()

    def inner_fn(idx):
        if False:
            return 10
        self_val = self_loader(idx)
        mask_val = ops.to_dtype(mask_loader(idx), torch.bool)

        def load_source_val():
            if False:
                return 10
            source_idx_val = source_idx_loader(idx)
            i = ops.indirect_indexing(source_idx_val, source_numel)
            return source_loader([i])
        source_val = ops.masked(mask_val, load_source_val, 0)
        return ops.where(mask_val, source_val, self_val)
    result_flat = Pointwise.create(device=self.get_device(), dtype=self.get_dtype(), inner_fn=inner_fn, ranges=self_flat.get_size())
    return view(result_flat, self.get_size())

@register_lowering(aten.as_strided_scatter, type_promotion_kind=None)
def as_strided_scatter(self, src, size, stride, storage_offset=None):
    if False:
        while True:
            i = 10
    output = clone(self)
    output_view = as_strided(output, size, stride, storage_offset)
    copy_(output_view, src)
    return output

@register_lowering(aten.scatter, type_promotion_kind=None)
def scatter(x, dim: int, index, src, **kwargs):
    if False:
        print('Hello World!')
    return scatter_(clone(x), dim, index, src, **kwargs)

def scatter_fallback(fn, self, dim: int, index, src, *, reduce: Optional[str]=None, include_self: bool=True):
    if False:
        print('Hello World!')
    reduce_ty = 'add' if fn == 'aten.scatter_' else 'sum'
    if reduce not in {None, reduce_ty} or (isinstance(src, TensorBox) and src.get_device().type == torch.device('cuda').type and needs_fallback_due_to_atomic_add_limitations(src.get_dtype())) or (fn == 'aten.scatter_reduce_' and reduce == 'sum' and isinstance(src, TensorBox) and (src.get_device() == torch.device('cpu')) and config.cpp.fallback_scatter_reduce_sum) or (reduce == reduce_ty and self.get_dtype() in {torch.bool, torch.int64}) or torch.are_deterministic_algorithms_enabled():
        ir.ScatterFallback(fn, self, dim, index, src, reduce=reduce, include_self=include_self)
        return self
    return None

@register_lowering(aten.scatter_, type_promotion_kind=None)
def scatter_(self, dim: int, index, src, *, reduce: Optional[str]=None):
    if False:
        print('Hello World!')
    assert reduce in {None, 'add', 'multiply'}
    fallback_result = scatter_fallback('aten.scatter_', self, dim, index, src, reduce=reduce)
    if fallback_result:
        return fallback_result
    if reduce == 'add':
        reduce = 'sum'
    elif reduce == 'multiply':
        reduce = 'prod'
    return scatter_reduce_(self, dim, index, src, reduce)

@register_lowering(aten.scatter_add, type_promotion_kind=None)
def scatter_add(x, dim: int, index, src):
    if False:
        return 10
    return scatter_add_(clone(x), dim, index, src)

@register_lowering(aten.scatter_add_, type_promotion_kind=None)
def scatter_add_(x, dim: int, index, src):
    if False:
        for i in range(10):
            print('nop')
    return scatter_reduce_(x, dim, index, src, 'sum')

@register_lowering(aten.scatter_reduce, type_promotion_kind=None)
def scatter_reduce(x, dim: int, index, src, reduction_type, **kwargs):
    if False:
        print('Hello World!')
    return scatter_reduce_(clone(x), dim, index, src, reduction_type, **kwargs)

@register_lowering(aten.scatter_reduce_, type_promotion_kind=None)
def scatter_reduce_(self, dim: int, index, src, reduce, *, include_self: bool=True):
    if False:
        for i in range(10):
            print('nop')
    assert reduce in {None, 'sum', 'prod', 'mean', 'amax', 'amin'}
    fallback_result = scatter_fallback('aten.scatter_reduce_', self, dim, index, src, reduce=reduce, include_self=include_self)
    if fallback_result:
        return fallback_result
    assert isinstance(self, TensorBox)
    assert 'int' in str(index.get_dtype())
    ndim = len(self.get_size())
    if ndim == 0:
        self = view(self, [1])
    if isinstance(src, TensorBox) and len(src.get_size()) == 0:
        src = view(src, [1])
    if isinstance(index, TensorBox) and len(index.get_size()) == 0:
        index = view(index, [1])
    dim = _validate_dim(self, dim)
    self.realize()
    index_loader = index.make_loader()
    src_loader = src.make_loader() if isinstance(src, TensorBox) else None

    def output_indexer(idx):
        if False:
            return 10
        shape = self.get_size()
        ndim = len(shape)
        indirect_idx = list(idx)
        indirect_idx[dim] = ops.indirect_indexing(index_loader(idx), 1 if ndim == 0 else shape[dim])
        return indirect_idx

    def fn(idx):
        if False:
            i = 10
            return i + 15
        if src_loader:
            return src_loader(idx)
        else:
            return ops.constant(src, self.get_dtype())

    def backend_reduce_str(reduce):
        if False:
            i = 10
            return i + 15
        if reduce == 'sum':
            return 'atomic_add'
        else:
            assert reduce is None
            return None
    if not include_self:
        zero_out = ir.Scatter(device=self.get_device(), dtype=self.get_dtype(), inner_fn=lambda index: ops.constant(0, self.get_dtype()), ranges=index.get_size(), output_indexer=output_indexer, scatter_mode=None)
        buffer = ir.ComputedBuffer(None, ir.MutationLayout(self), zero_out)
        buffer.name = V.graph.register_buffer(buffer)
    scatter = ir.Scatter(device=self.get_device(), dtype=self.get_dtype(), inner_fn=fn, ranges=index.get_size(), output_indexer=output_indexer, scatter_mode=backend_reduce_str(reduce))
    buffer = ir.ComputedBuffer(None, ir.MutationLayout(self), scatter)
    buffer.name = V.graph.register_buffer(buffer)
    if ndim == 0:
        self = view(self, [])
    return self

def upsample_nearestnd(x, output_size, scales_x: Tuple[Optional[float], ...], n: int=2):
    if False:
        while True:
            i = 10
    x.realize_hint()
    x_loader = x.make_loader()
    i_sizes = x.get_size()[-n:]
    batch = x.get_size()[:-n]
    i_sizes = [V.graph.sizevars.evaluate_static_shape(i) for i in i_sizes]
    assert len(scales_x) == n
    o_sizes = output_size
    scales = [i / o for (i, o) in zip(i_sizes, o_sizes)]
    for (i, scale) in enumerate(scales):
        if scale:
            scales[i] = scale

    def scale_fn(x, scale, size):
        if False:
            i = 10
            return i + 15
        x = ops.index_expr(x, torch.float32)
        x = ops.mul(x, ops.constant(scale, torch.float32))
        x = ops.to_dtype(x, torch.int32)
        return ops.indirect_indexing(x, size, check=False)

    def fn(idx):
        if False:
            for i in range(10):
                print('nop')
        x = idx[-n:]
        b = idx[:-n]
        return x_loader([*b, *[scale_fn(i, s, size) for (i, s, size) in zip(x, scales, i_sizes)]])
    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=fn, ranges=[*batch, *o_sizes])

@register_lowering(aten.upsample_nearest1d.default)
def upsample_nearest1d(x, output_size, scales: Optional[float]=None):
    if False:
        i = 10
        return i + 15
    return upsample_nearestnd(x, output_size, (scales,), n=1)

@register_lowering(aten.upsample_nearest2d.default)
def upsample_nearest2d(x, output_size, scales_h: Optional[float]=None, scales_w: Optional[float]=None):
    if False:
        while True:
            i = 10
    return upsample_nearestnd(x, output_size, (scales_h, scales_w), n=2)

@register_lowering(aten.upsample_nearest3d.default)
def upsample_nearest3d(x, output_size, scales_d: Optional[float]=None, scales_h: Optional[float]=None, scales_w: Optional[float]=None):
    if False:
        print('Hello World!')
    return upsample_nearestnd(x, output_size, (scales_d, scales_h, scales_w), n=3)

def _create_constants(*args, dtype):
    if False:
        i = 10
        return i + 15
    return tuple((ops.constant(a, dtype) for a in args))

@register_lowering(aten.upsample_bicubic2d.default)
def upsample_bicubic2d_default(x, output_size, align_corners: bool, scales_h: Optional[float]=None, scales_w: Optional[float]=None):
    if False:
        return 10
    x.realize_hint()
    x_loader = x.make_loader()
    (N, C, iH, iW) = x.get_size()
    (oH, oW) = output_size
    iH = V.graph.sizevars.evaluate_static_shape(iH)
    iW = V.graph.sizevars.evaluate_static_shape(iW)

    def get_int_dtype(maxval):
        if False:
            i = 10
            return i + 15
        if maxval > torch.iinfo(torch.int32).max:
            return torch.int64
        return torch.int32

    def compute_scale(in_size, out_size, align_corners, scale=None):
        if False:
            for i in range(10):
                print('nop')
        if align_corners:
            return (in_size - 1) / (out_size - 1) if out_size > 1 else 0
        else:
            return 1 / scale if scale is not None and scale > 0 else in_size / out_size

    def compute_source_index(scale, dst_index, align_corners):
        if False:
            print('Hello World!')
        dst_index_ie = ops.index_expr(dst_index, torch.float32)
        scale = ops.constant(scale, torch.float32)
        if align_corners:
            return ops.mul(scale, dst_index_ie)
        else:
            half = ops.constant(0.5, torch.float32)
            return scale * (dst_index_ie + half) - half

    def cubic_convolution1(x, A):
        if False:
            for i in range(10):
                print('nop')
        (_Ap2, _Ap3, _1) = _create_constants(A + 2, A + 3, 1, dtype=torch.float32)
        return (_Ap2 * x - _Ap3) * x * x + _1

    def cubic_convolution2(x, A):
        if False:
            return 10
        (_A, _4A, _5A, _8A) = _create_constants(A, 4 * A, 5 * A, 8 * A, dtype=torch.float32)
        return ((_A * x - _5A) * x + _8A) * x - _4A

    def get_cubic_upsample_coefficients(t):
        if False:
            for i in range(10):
                print('nop')
        A = -0.75
        _1 = ops.constant(1.0, torch.float32)
        c0 = cubic_convolution2(ops.add(t, _1), A)
        c1 = cubic_convolution1(t, A)
        x2 = ops.sub(_1, t)
        c2 = cubic_convolution1(x2, A)
        c3 = cubic_convolution2(ops.add(x2, _1), A)
        return (c0, c1, c2, c3)

    def cubic_interp1d(xs, t):
        if False:
            for i in range(10):
                print('nop')
        cs = get_cubic_upsample_coefficients(t)
        return xs[0] * cs[0] + xs[1] * cs[1] + xs[2] * cs[2] + xs[3] * cs[3]
    height_scale = compute_scale(iH, oH, align_corners, scales_h)
    width_scale = compute_scale(iW, oW, align_corners, scales_h)

    def clamp(v, min, max):
        if False:
            print('Hello World!')
        return ops.maximum(min, ops.minimum(max, v))

    def fn(idx):
        if False:
            for i in range(10):
                print('nop')
        (n, c, oy, ox) = idx
        real_x = compute_source_index(width_scale, ox, align_corners)
        in_x = ops.floor(real_x)
        t_x = ops.sub(real_x, in_x)
        real_y = compute_source_index(height_scale, oy, align_corners)
        in_y = ops.floor(real_y)
        t_y = ops.sub(real_y, in_y)

        def load_bounded(fy, fx):
            if False:
                i = 10
                return i + 15
            _0 = ops.constant(0, torch.int32)
            iHm1 = ops.constant(iH - 1, torch.int32)
            iWm1 = ops.constant(iW - 1, torch.int32)
            iy = ops.indirect_indexing(clamp(fy, _0, iHm1), iH, check=False)
            ix = ops.indirect_indexing(clamp(fx, _0, iWm1), iW, check=False)
            return x_loader([n, c, iy, ix])
        iy = ops.to_dtype(in_y, get_int_dtype(iH + 1))
        ix = ops.to_dtype(in_x, get_int_dtype(iW + 1))
        iys_ofs = tuple((ops.add(iy, ofs) for ofs in (-1, 0, 1, 2)))
        ixs_ofs = tuple((ops.add(ix, ofs) for ofs in (-1, 0, 1, 2)))

        def get_x_interp(y):
            if False:
                while True:
                    i = 10
            coeffs_x = tuple((load_bounded(y, x) for x in ixs_ofs))
            return cubic_interp1d(coeffs_x, t_x)
        coeffs_y = tuple((get_x_interp(y) for y in iys_ofs))
        return cubic_interp1d(coeffs_y, t_y)
    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=fn, ranges=[N, C, sympy.Integer(oH), sympy.Integer(oW)])

@register_lowering(aten.reflection_pad2d)
def reflection_pad2d(x, padding):
    if False:
        for i in range(10):
            print('nop')
    assert len(padding) == 4
    (left, right, top, bot) = padding
    x_loader = x.make_loader()
    (*batch, h, w) = x.get_size()

    def reflect(x, size, offset):
        if False:
            i = 10
            return i + 15
        size_num = size
        size = ops.index_expr(size - 1, torch.int32)
        x = ops.index_expr(x, torch.int32)
        x = ops.sub(x, ops.index_expr(offset, torch.int32))
        x = ops.sub(size, ops.abs(ops.sub(size, ops.abs(x))))
        return ops.indirect_indexing(x, size_num, check=False)

    def fn(idx):
        if False:
            return 10
        (*b, x, y) = idx
        x = reflect(x, h, top)
        y = reflect(y, w, left)
        return x_loader([*b, x, y])
    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=fn, ranges=[*batch, h + top + bot, w + left + right])

@register_lowering(aten.reflection_pad2d_backward)
def reflection_pad2d_backward(grad_output, x, padding):
    if False:
        for i in range(10):
            print('nop')
    assert len(padding) == 4
    (left, right, top, bot) = padding
    (*_, h, w) = x.get_size()
    h = V.graph.sizevars.evaluate_static_shape(h) - 1
    w = V.graph.sizevars.evaluate_static_shape(w) - 1
    grad_loader = grad_output.make_loader()
    (*_, h_grad, w_grad) = grad_output.get_size()

    def fn(idx):
        if False:
            i = 10
            return i + 15
        (*b, x, y) = idx

        def load_from_output(x, y):
            if False:
                while True:
                    i = 10
            return grad_loader([*b, x, y])

        def index_range_condition(index_range):
            if False:
                while True:
                    i = 10
            (i, lb, ub) = index_range
            i = ops.index_expr(i, torch.int32)
            lb = ops.index_expr(lb, torch.int64)
            ub = ops.index_expr(ub, torch.int64)
            return ops.and_(ops.ge(i, lb), ops.le(i, ub))
        (center_x, center_y) = (x + top, y + left)
        (top_reflect_x, left_reflect_y) = (top - x, left - y)
        (bot_reflect_x, right_reflect_y) = (2 * h + top - x, 2 * w + left - y)
        range_cx = (center_x, 0, h + top + bot)
        range_cy = (center_y, 0, w + left + right)
        cond = ops.and_(index_range_condition(range_cx), index_range_condition(range_cy))
        grad = ops.masked(cond, lambda : load_from_output(center_x, center_y), 0.0)

        def accumulate(out_x, out_y, index_range1, index_range2=None):
            if False:
                while True:
                    i = 10
            nonlocal grad
            upper_less_than_lower1 = index_range1[2] < index_range1[1]
            if isinstance(upper_less_than_lower1, bool) and upper_less_than_lower1:
                return
            cond = index_range_condition(index_range1)
            if index_range2 is not None:
                upper_less_than_lower2 = index_range2[2] < index_range2[1]
                if isinstance(upper_less_than_lower2, bool) and upper_less_than_lower2:
                    return
                cond = ops.and_(cond, index_range_condition(index_range2))
            g = ops.masked(cond, lambda : load_from_output(out_x, out_y), 0.0)
            grad = ops.add(grad, g)
        accumulate(center_x, left_reflect_y, range_cx, (y, 1, left))
        accumulate(center_x, right_reflect_y, range_cx, (y, w - right, w - 1))
        accumulate(top_reflect_x, center_y, (x, 1, top), range_cy)
        accumulate(bot_reflect_x, center_y, (x, h - bot, h - 1), range_cy)
        accumulate(top_reflect_x, left_reflect_y, (x, 1, top), (y, 1, left))
        accumulate(top_reflect_x, right_reflect_y, (x, 1, top), (y, w - right, w - 1))
        accumulate(bot_reflect_x, left_reflect_y, (x, h - bot, h - 1), (y, 1, left))
        accumulate(bot_reflect_x, right_reflect_y, (x, h - bot, h - 1), (y, w - right, w - 1))
        return grad
    return Pointwise.create(device=grad_output.get_device(), dtype=grad_output.get_dtype(), inner_fn=fn, ranges=list(x.get_size()))

@register_lowering(prims.rev.default)
def rev(x, dims):
    if False:
        for i in range(10):
            print('nop')
    x_loader = x.make_loader()
    sizes = x.get_size()

    def loader(idx):
        if False:
            while True:
                i = 10
        idx = list(idx)
        assert len(idx) == len(sizes)
        for dim in dims:
            idx[dim] = sizes[dim] - 1 - idx[dim]
        return x_loader(idx)
    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=loader, ranges=sizes)

@register_lowering(aten.constant_pad_nd, type_promotion_kind=None)
def constant_pad_nd(x, padding, fill_value=0):
    if False:
        for i in range(10):
            print('nop')
    assert len(padding) % 2 == 0
    if all((p == 0 for p in padding)):
        return clone(x)
    sizes = x.get_size()
    bounds = list(reversed(list(zip(padding[::2], padding[1::2]))))
    n = len(sizes) - len(bounds)
    bounds_precomp = []
    for (l, h) in bounds:
        l_precomp = V.graph.sizevars.lookup_precomputed_size(l) if isinstance(l, sympy.Expr) and (not l.is_number) else l
        bounds_precomp.append((l_precomp, h))
    output_size = list(sizes[:n])
    mask_sizes = []
    for ((low, high), size) in zip(bounds, sizes[n:]):
        mask_sizes.append(size)
        output_size.append(sympy.expand(size + low + high))
    assert len(output_size) == len(sizes)
    fill_value = dtype_to_type(x.get_dtype())(fill_value)

    def mask(index):
        if False:
            i = 10
            return i + 15
        mask = []
        for (idx, (low, high), length) in zip(index[n:], bounds, mask_sizes):
            if low != 0:
                mask.append(range_mask_low(idx, 0))
            if high != 0:
                mask.append(range_mask_high(idx, length))
        mask = functools.reduce(ops.and_, mask)
        return ops.masked(mask, lambda : x_loader(index), fill_value)

    def offset_fn(index):
        if False:
            for i in range(10):
                print('nop')
        new_index = list(index[:n])
        for (idx, (low, high)) in zip(index[n:], bounds_precomp):
            new_index.append(idx - low)
        assert len(new_index) == len(index)
        return mask(new_index)
    x_loader = x.make_loader()
    return Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=offset_fn, ranges=output_size)

def range_mask_low(i: sympy.Expr, low: Union[sympy.Expr, int]):
    if False:
        return 10
    return ops.ge(ops.index_expr(i, torch.int64), ops.index_expr(sympy.Integer(low), torch.int64))

def range_mask_high(i: sympy.Expr, high: sympy.Expr):
    if False:
        for i in range(10):
            print('nop')
    return ops.lt(ops.index_expr(i, torch.int64), ops.index_expr(high, torch.int64))

def range_mask(i: sympy.Expr, high: sympy.Expr, low: sympy.Expr):
    if False:
        while True:
            i = 10
    return ops.and_(range_mask_low(i, low), range_mask_high(i, high))

def constant_boundary_condition_2d(x, fill_value, padding=None, pad_fill_value=1.0):
    if False:
        while True:
            i = 10
    (*_, h, w) = x.get_size()
    x_loader = x.make_loader()
    padding_h = padding[0] if padding else 0
    padding_w = padding[1] if padding else 0

    def load(index):
        if False:
            print('Hello World!')
        (*prefix, ih, iw) = index
        mask = ops.and_(range_mask(ih, h + padding_h, -padding_h), range_mask(iw, w + padding_w, -padding_w))
        return ops.masked(mask, lambda : constant_boundary_condition_2d(x, pad_fill_value)([*prefix, ih, iw]), fill_value) if padding else ops.masked(mask, lambda : x_loader([*prefix, ih, iw]), fill_value)
    return load

def pooling_size(x, i, kernel_size, stride, padding, ceil_mode):
    if False:
        return 10
    x_out = FloorDiv(x + 2 * padding[i] - (kernel_size[i] - 1) + (stride[i] - 1), stride[i])
    if ceil_mode:
        x_alt = FloorDiv(x + 2 * padding[i] - (kernel_size[i] - 1) + 2 * (stride[i] - 1), stride[i])
        if V.graph.sizevars.size_hint((x_alt - 1) * stride[i] - x - padding[i]) >= 0:
            x_alt -= 1
            V.graph.sizevars.guard_leq(0, x_alt * stride[i] - x - padding[i])
        if V.graph.sizevars.size_hint(x_out - x_alt) == 0:
            V.graph.sizevars.guard_equals(x_out, x_alt)
            ceil_mode = False
        else:
            x_out = x_alt
    return (x_out, ceil_mode)
fallback_max_pool2d_with_indices = fallback_handler(aten.max_pool2d_with_indices.default)

@register_lowering(aten.max_pool2d_with_indices, type_promotion_kind=None)
def max_pool2d_with_indices(x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    if False:
        i = 10
        return i + 15
    if padding == 0:
        padding = [0, 0]
    if dilation == 1:
        dilation = [1, 1]
    if not stride:
        stride = kernel_size
    kernel_size = pad_listlike(kernel_size, 2)
    stride = pad_listlike(stride, 2)
    padding = pad_listlike(padding, 2)
    dilation = pad_listlike(dilation, 2)
    assert isinstance(x, TensorBox)
    assert len(kernel_size) == 2
    assert len(stride) == 2
    assert len(padding) == 2
    assert len(dilation) == 2
    assert len(x.get_size()) in (3, 4)
    x.realize_hint()
    (*batch, h, w) = x.get_size()
    (h_out, ceil_mode1) = pooling_size(h, 0, kernel_size, stride, padding, ceil_mode)
    (w_out, ceil_mode2) = pooling_size(w, 1, kernel_size, stride, padding, ceil_mode)
    if padding[0] or padding[1] or ceil_mode1 or ceil_mode2:
        x_loader = constant_boundary_condition_2d(x, float('-inf'))
    else:
        x_loader = x.make_loader()
    new_size = list(batch) + [h_out, w_out]
    window_size = kernel_size[0] * kernel_size[1]
    if window_size > 25 or any((d != 1 for d in dilation)):
        return fallback_max_pool2d_with_indices(x, kernel_size, stride, padding, dilation, ceil_mode)

    def fn(idx, return_index):
        if False:
            i = 10
            return i + 15
        (*prefix, bh, bw) = idx
        maxval = None
        maxindex = None
        for (ih, iw) in itertools.product(range(kernel_size[0]), range(kernel_size[1])):
            ih = bh * stride[0] + ih - padding[0]
            iw = bw * stride[1] + iw - padding[1]
            val = x_loader([*prefix, ih, iw])
            if return_index:
                index = ops.index_expr(ih * w + iw, torch.int64)
                if maxindex is None:
                    maxindex = index
                else:
                    maxindex = ops.where(ops.gt(val, maxval), index, maxindex)
            if maxval is None:
                maxval = val
            else:
                maxval = ops.maximum(val, maxval)
        if return_index:
            return maxindex
        else:
            return maxval
    r1 = Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=functools.partial(fn, return_index=False), ranges=new_size)
    r2 = Pointwise.create(device=x.get_device(), dtype=torch.int64, inner_fn=functools.partial(fn, return_index=True), ranges=new_size)
    return (r1, r2)
fallback_max_pool2d_with_indices_backward = fallback_handler(aten.max_pool2d_with_indices_backward.default)

@register_lowering(aten.max_pool2d_with_indices_backward, type_promotion_kind=None)
def max_pool2d_with_indices_backward(grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices):
    if False:
        i = 10
        return i + 15
    if padding == 0:
        padding = [0, 0]
    if dilation == 1:
        dilation = [1, 1]
    if not stride:
        stride = kernel_size
    assert isinstance(x, TensorBox)
    assert len(kernel_size) == 2
    assert len(stride) == 2
    assert len(padding) == 2
    assert len(dilation) == 2
    assert len(x.get_size()) in (3, 4)
    grad_output.realize_hint()
    try:
        gO_stride = grad_output.get_stride()
    except AttributeError:
        gO_stride = None
    if isinstance(x, TensorBox) and isinstance(x.data.data, Pointwise):
        data = x.data.data
        x_buffer = ir.ComputedBuffer(name=None, layout=ir.FlexibleLayout(device=data.get_device(), dtype=data.get_dtype(), size=data.get_size()), data=data)
        x_buffer.decide_layout()
        x_stride = x_buffer.get_stride()
    else:
        try:
            x_stride = x.get_stride()
        except AttributeError:
            x_stride = None
    is_channels_last = x_stride is not None and x_stride[1] == 1 or (gO_stride is not None and gO_stride[1] == 1)
    autotune = config.coordinate_descent_tuning or config.max_autotune or config.max_autotune_pointwise
    if any((d != 1 for d in dilation)) or (is_channels_last and (not autotune)):
        return fallback_max_pool2d_with_indices_backward(grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices)
    indices.realize_hint()
    (*batch, height, width) = x.get_size()
    (*_, pooled_height, pooled_width) = grad_output.get_size()
    indices_loader = indices.make_loader()
    grad_loader = grad_output.make_loader()
    new_size = list(x.get_size())
    h_window_size = max([max(h // stride[0] - max(0, (h - kernel_size[0]) // stride[0]), 1) for h in range(kernel_size[0] * 2)])
    w_window_size = max([max(w // stride[1] - max(0, (w - kernel_size[1]) // stride[1]), 1) for w in range(kernel_size[1] * 2)])
    window_size = h_window_size * w_window_size
    if window_size > 25:
        return fallback_max_pool2d_with_indices_backward(grad_output, x, kernel_size, stride, padding, dilation, ceil_mode, indices)
    indices_size = indices.get_size()

    def fn(idx):
        if False:
            i = 10
            return i + 15
        (*prefix, h, w) = idx
        index_test = ops.index_expr(h * width + w, torch.int32)
        h = h + padding[0]
        w = w + padding[1]
        phstart = ops.index_expr(FloorDiv(h - kernel_size[0] + stride[0], stride[0]), torch.int32)
        pwstart = ops.index_expr(FloorDiv(w - kernel_size[1] + stride[1], stride[1]), torch.int32)
        phend = ops.index_expr(FloorDiv(h, stride[0]) + 1, torch.int32)
        pwend = ops.index_expr(FloorDiv(w, stride[1]) + 1, torch.int32)
        phstart = ops.maximum(phstart, ops.constant(0, torch.int32))
        pwstart = ops.maximum(pwstart, ops.constant(0, torch.int32))
        phend = ops.minimum(phend, ops.index_expr(pooled_height, torch.int32))
        pwend = ops.minimum(pwend, ops.index_expr(pooled_width, torch.int32))
        gradient = None
        for ph_ in range(h_window_size):
            for pw_ in range(w_window_size):
                ph = ops.add(phstart, ops.constant(ph_, torch.int32))
                pw = ops.add(pwstart, ops.constant(pw_, torch.int32))
                grad_index = [*prefix, ops.indirect_indexing(ops.minimum(ph, ops.sub(phend, ops.constant(1, torch.int32))), indices_size[-2], check=False), ops.indirect_indexing(ops.minimum(pw, ops.sub(pwend, ops.constant(1, torch.int32))), indices_size[-1], check=False)]
                index_actual = indices_loader(grad_index)
                grad_part = grad_loader(grad_index)
                check = ops.eq(index_actual, index_test)
                if gradient is None:
                    gradient = ops.where(check, grad_part, ops.constant(0.0, torch.float32))
                else:
                    mask = ops.and_(ops.and_(ops.lt(ph, phend), ops.lt(pw, pwend)), check)
                    gradient = ops.where(mask, ops.add(gradient, grad_part), gradient)
        assert gradient is not None
        return gradient
    return Pointwise.create(device=grad_output.get_device(), dtype=grad_output.get_dtype(), inner_fn=fn, ranges=new_size)

def pad_adaptive_loader(x):
    if False:
        return 10
    (*_, h, w) = x.get_size()
    x_loader = x.make_loader()

    def load(prefix, increments, start_indices, end_indices):
        if False:
            return 10
        (ih, iw) = increments
        (h_start_index, w_start_index) = start_indices
        (h_end_index, w_end_index) = end_indices
        mask = ops.and_(ops.lt(ops.index_expr(h_start_index + ih, torch.int64), ops.index_expr(h_end_index, torch.int64)), ops.lt(ops.index_expr(w_start_index + iw, torch.int64), ops.index_expr(w_end_index, torch.int64)))
        return ops.masked(mask, lambda : x_loader([*prefix, h_start_index + ih, w_start_index + iw]), 0.0)
    return load

def _adaptive_pooling_idx_sum(kernel_maxes, start_index_fns, end_index_fns):
    if False:
        while True:
            i = 10
    (h_start_index_fn, w_start_index_fn) = start_index_fns
    (h_end_index_fn, w_end_index_fn) = end_index_fns

    def fn_sum(idx, loader):
        if False:
            print('Hello World!')
        (*prefix, bh, bw) = idx
        h_start_index = h_start_index_fn(bh)
        h_end_index = h_end_index_fn(bh)
        w_start_index = w_start_index_fn(bw)
        w_end_index = w_end_index_fn(bw)
        total = None
        for (ih, iw) in itertools.product(range(kernel_maxes[0]), range(kernel_maxes[1])):
            val = loader(prefix, [ih, iw], [h_start_index, w_start_index], [h_end_index, w_end_index])
            if total is None:
                total = val
            else:
                total = ops.add(val, total)
        return total
    return fn_sum
fallback_adaptive_avg_pool2d = fallback_handler(aten._adaptive_avg_pool2d.default)

@register_lowering(aten._adaptive_avg_pool2d)
def _adaptive_avg_pool2d(x, output_size):
    if False:
        print('Hello World!')
    assert isinstance(x, TensorBox)
    assert len(output_size) == 2
    x.realize_hint()
    (*batch, h_in, w_in) = x.get_size()
    h_in = V.graph.sizevars.evaluate_static_shape(h_in)
    w_in = V.graph.sizevars.evaluate_static_shape(w_in)
    (h_out, w_out) = output_size
    if h_in == h_out and w_in == w_out:
        return clone(x)
    if h_out == 0 or w_out == 0:
        o_size = [*batch, h_out, w_out]
        return empty(o_size, dtype=x.get_dtype(), device=x.get_device())
    if h_in % h_out == 0 and w_in % w_out == 0:
        kernel_size = [h_in // h_out, w_in // w_out]
        return avg_pool2d(x, kernel_size)
    h_kernel_max = ceildiv(h_in + h_out - 1, h_out)
    w_kernel_max = ceildiv(w_in + w_out - 1, w_out)
    new_size = list(batch) + [h_out, w_out]
    dtype = x.get_dtype()

    def start_index(index, out_dim, inp_dim):
        if False:
            i = 10
            return i + 15
        return FloorDiv(index * inp_dim, out_dim)

    def end_index(index, out_dim, inp_dim):
        if False:
            i = 10
            return i + 15
        return FloorDiv((index + 1) * inp_dim + out_dim - 1, out_dim)
    h_start_index = functools.partial(start_index, out_dim=h_out, inp_dim=h_in)
    h_end_index = functools.partial(end_index, out_dim=h_out, inp_dim=h_in)
    w_start_index = functools.partial(start_index, out_dim=w_out, inp_dim=w_in)
    w_end_index = functools.partial(end_index, out_dim=w_out, inp_dim=w_in)
    window_size = h_kernel_max * w_kernel_max
    if window_size > 25:
        return fallback_adaptive_avg_pool2d(x, output_size)
    fn_sum = _adaptive_pooling_idx_sum([h_kernel_max, w_kernel_max], [h_start_index, w_start_index], [h_end_index, w_end_index])
    ones_loader = pad_adaptive_loader(ones_like(x))

    def fn(idx):
        if False:
            while True:
                i = 10
        return ops.truediv(fn_sum(idx, pad_adaptive_loader(x)), fn_sum(idx, ones_loader))
    rv = Pointwise.create(device=x.get_device(), dtype=dtype, inner_fn=fn, ranges=new_size)
    return rv

@register_lowering(aten.upsample_nearest2d_backward.default)
def upsample_nearest2d_backward(x, output_size=None, input_size=None, scales_h=None, scales_w=None):
    if False:
        while True:
            i = 10
    x.realize_hint()
    (*batch, inp_h, inp_w) = x.get_size()
    inp_h = V.graph.sizevars.evaluate_static_shape(inp_h)
    inp_w = V.graph.sizevars.evaluate_static_shape(inp_w)
    (*batch, out_h, out_w) = input_size
    if inp_h % out_h == 0 and inp_w % out_w == 0:
        return avg_pool2d(x, [inp_h // out_h, inp_w // out_w], divisor_override=1)
    h_kernel_max = ceildiv(inp_h, out_h)
    w_kernel_max = ceildiv(inp_w, out_w)

    def start_index(index, out_dim, inp_dim):
        if False:
            print('Hello World!')
        return CeilDiv(index * inp_dim, out_dim)

    def end_index(index, out_dim, inp_dim):
        if False:
            while True:
                i = 10
        return start_index(index + 1, out_dim, inp_dim)
    h_start_index = functools.partial(start_index, out_dim=out_h, inp_dim=inp_h)
    h_end_index = functools.partial(end_index, out_dim=out_h, inp_dim=inp_h)
    w_start_index = functools.partial(start_index, out_dim=out_w, inp_dim=inp_w)
    w_end_index = functools.partial(end_index, out_dim=out_w, inp_dim=inp_w)
    fn_sum = _adaptive_pooling_idx_sum([h_kernel_max, w_kernel_max], [h_start_index, w_start_index], [h_end_index, w_end_index])

    def fn(idx):
        if False:
            print('Hello World!')
        return fn_sum(idx, pad_adaptive_loader(x))
    rv = Pointwise.create(device=x.get_device(), dtype=x.get_dtype(), inner_fn=fn, ranges=list(input_size))
    return rv
fallback_avg_pool2d = fallback_handler(aten.avg_pool2d.default)

@register_lowering(aten.avg_pool2d, type_promotion_kind=None)
def avg_pool2d(x, kernel_size, stride=(), padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    if False:
        for i in range(10):
            print('nop')
    if not stride:
        stride = kernel_size
    if not padding:
        padding = [0, 0]
    kernel_size = pad_listlike(kernel_size, 2)
    stride = pad_listlike(stride, 2)
    padding = pad_listlike(padding, 2)
    assert isinstance(x, TensorBox)
    assert len(kernel_size) == 2
    assert len(stride) == 2
    assert len(padding) == 2
    assert len(x.get_size()) in (3, 4)
    x.realize_hint()
    (*batch, h, w) = x.get_size()
    (h_out, ceil_mode1) = pooling_size(h, 0, kernel_size, stride, padding, ceil_mode)
    (w_out, ceil_mode2) = pooling_size(w, 1, kernel_size, stride, padding, ceil_mode)
    if padding[0] or padding[1] or ceil_mode1 or ceil_mode2:
        x_loader = constant_boundary_condition_2d(x, 0.0)
        had_padding = True
    else:
        x_loader = x.make_loader()
        had_padding = False
    new_size = list(batch) + [h_out, w_out]
    dtype = x.get_dtype()
    window_size = kernel_size[0] * kernel_size[1]
    if window_size > 25:
        return fallback_avg_pool2d(x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

    def fn_sum(idx, loader):
        if False:
            return 10
        (*prefix, bh, bw) = idx
        total = None
        for (ih, iw) in itertools.product(range(kernel_size[0]), range(kernel_size[1])):
            ih = bh * stride[0] + ih - padding[0]
            iw = bw * stride[1] + iw - padding[1]
            val = loader([*prefix, ih, iw])
            if total is None:
                total = val
            else:
                total = ops.add(val, total)
        return total
    if not had_padding or divisor_override:
        if divisor_override:
            scale = 1 / divisor_override
        else:
            scale = 1.0 / (kernel_size[0] * kernel_size[1])

        def fn(idx):
            if False:
                while True:
                    i = 10
            return ops.mul(fn_sum(idx, x_loader), ops.constant(scale, dtype))
    else:
        ones_loader = constant_boundary_condition_2d(ones_like(x), 0.0, padding if count_include_pad else None)

        def fn(idx):
            if False:
                for i in range(10):
                    print('nop')
            return ops.truediv(fn_sum(idx, x_loader), fn_sum(idx, ones_loader))
    rv = Pointwise.create(device=x.get_device(), dtype=dtype, inner_fn=fn, ranges=new_size)
    return rv
fallback_avg_pool2d_backward = fallback_handler(aten.avg_pool2d_backward.default)

@register_lowering(aten.avg_pool2d_backward, type_promotion_kind=None)
def avg_pool2d_backward(grad_output, x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override=None):
    if False:
        for i in range(10):
            print('nop')
    assert divisor_override is None or divisor_override != 0, 'divisor must be not zero'
    if not stride:
        stride = kernel_size
    if not padding:
        padding = [0, 0]
    assert isinstance(grad_output, TensorBox)
    assert isinstance(x, TensorBox)
    assert len(kernel_size) == 2
    assert len(stride) == 2
    assert len(padding) == 2
    assert len(x.get_size()) in (3, 4)
    grad_output.realize_hint()
    (*batch, height, width) = x.get_size()
    (h_out, ceil_mode1) = pooling_size(height, 0, kernel_size, stride, padding, ceil_mode)
    (w_out, ceil_mode2) = pooling_size(width, 1, kernel_size, stride, padding, ceil_mode)
    grad_loader = grad_output.make_loader()
    had_padding = padding[0] or padding[1] or ceil_mode1 or ceil_mode2
    (*_, pooled_height, pooled_width) = grad_output.get_size()
    new_size = list(x.get_size())
    dtype = x.get_dtype()
    h_window_size = max([max(h // stride[0] - max(0, (h - kernel_size[0]) // stride[0]), 1) for h in range(kernel_size[0] * 2)])
    w_window_size = max([max(w // stride[1] - max(0, (w - kernel_size[1]) // stride[1]), 1) for w in range(kernel_size[1] * 2)])
    window_size = h_window_size * w_window_size
    if window_size > 25:
        return fallback_avg_pool2d_backward(grad_output, x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)

    def compute_pool_size_without_padding(ph, pw):
        if False:
            i = 10
            return i + 15
        '\n        This computes the scaling factor that we will divide an element\n        by when `count_include_pad=False`\n        '
        stride_h = ops.constant(stride[0], torch.int32)
        stride_w = ops.constant(stride[1], torch.int32)
        pad_h = ops.constant(padding[0], torch.int32)
        pad_w = ops.constant(padding[1], torch.int32)
        kernel_h = ops.constant(kernel_size[0], torch.int32)
        kernel_w = ops.constant(kernel_size[1], torch.int32)
        hstart = ops.sub(ops.mul(ph, stride_h), pad_h)
        wstart = ops.sub(ops.mul(pw, stride_w), pad_w)
        hend = ops.minimum(ops.add(hstart, kernel_h), ops.add(ops.index_expr(height, torch.int32), pad_h))
        wend = ops.minimum(ops.add(wstart, kernel_w), ops.add(ops.index_expr(width, torch.int32), pad_w))
        hstart = ops.maximum(hstart, ops.constant(0, torch.int32))
        wstart = ops.maximum(wstart, ops.constant(0, torch.int32))
        hend = ops.minimum(hend, ops.index_expr(height, torch.int32))
        wend = ops.minimum(wend, ops.index_expr(width, torch.int32))
        divide_factor = ops.mul(ops.sub(hend, hstart), ops.sub(wend, wstart))
        return divide_factor

    def fn(idx):
        if False:
            for i in range(10):
                print('nop')
        (*prefix, h, w) = idx
        h = h + padding[0]
        w = w + padding[1]
        phstart = ops.index_expr(FloorDiv(h - kernel_size[0] + stride[0], stride[0]), torch.int32)
        pwstart = ops.index_expr(FloorDiv(w - kernel_size[1] + stride[1], stride[1]), torch.int32)
        phend = ops.index_expr(FloorDiv(h, stride[0]) + 1, torch.int32)
        pwend = ops.index_expr(FloorDiv(w, stride[1]) + 1, torch.int32)
        phstart = ops.maximum(phstart, ops.constant(0, torch.int32))
        pwstart = ops.maximum(pwstart, ops.constant(0, torch.int32))
        phend = ops.minimum(phend, ops.index_expr(pooled_height, torch.int32))
        pwend = ops.minimum(pwend, ops.index_expr(pooled_width, torch.int32))
        gradient = None
        for ph_ in range(h_window_size):
            for pw_ in range(w_window_size):
                ph = ops.add(phstart, ops.constant(ph_, torch.int32))
                pw = ops.add(pwstart, ops.constant(pw_, torch.int32))
                if divisor_override is not None:
                    scale = divisor_override
                elif count_include_pad or not had_padding:
                    scale = kernel_size[0] * kernel_size[1]
                else:
                    scale = compute_pool_size_without_padding(ph, pw)
                part = ops.truediv(grad_loader([*prefix, ops.indirect_indexing(ops.minimum(ph, ops.sub(phend, ops.constant(1, torch.int32))), pooled_height, check=False), ops.indirect_indexing(ops.minimum(pw, ops.sub(pwend, ops.constant(1, torch.int32))), pooled_width, check=False)]), scale)
                mask = ops.and_(ops.lt(ph, phend), ops.lt(pw, pwend))
                if gradient is None:
                    gradient = ops.where(mask, part, ops.constant(0.0, torch.float32))
                else:
                    gradient = ops.where(mask, ops.add(gradient, part), gradient)
        assert gradient is not None
        return gradient
    rv = Pointwise.create(device=grad_output.get_device(), dtype=dtype, inner_fn=fn, ranges=new_size)
    return rv

def _validate_reduction_axis(x, axis):
    if False:
        for i in range(10):
            print('nop')
    size = x.get_size()
    if isinstance(axis, int):
        axis = [axis]
    elif not axis:
        axis = range(len(size))
    if len(size) == 0:
        assert tuple(axis) in [(), (0,), (-1,)], f'invalid axis: {axis}'
        return []
    axis = list(axis)
    for i in range(len(axis)):
        if axis[i] < 0:
            axis[i] += len(size) if len(size) else 1
        assert 0 <= axis[i] < len(size) or (len(size) == 0 and axis[i] == 0)
    assert len(set(axis)) == len(axis), 'reduction axis not unique'
    return axis

def _make_reduction_inner(x, *, axis, keepdims, dtype, override_return_dtype):
    if False:
        i = 10
        return i + 15
    if dtype is not None:
        x = to_dtype(x, dtype)
    size = x.get_size()
    axis = set(_validate_reduction_axis(x, axis))
    kept_sizes = []
    kept_idx = []
    reduced_sizes = []
    reduced_idx = []
    for i in range(len(size)):
        if i in axis:
            reduced_idx.append(i)
            reduced_sizes.append(size[i])
        else:
            kept_idx.append(i)
            kept_sizes.append(size[i])

    def loader(index, reduction_index):
        if False:
            return 10
        assert len(reduction_index) == len(reduced_idx)
        if keepdims:
            assert len(index) == len(size)
            index = [index[i] for i in kept_idx]
        assert len(index) == len(kept_idx)
        new_index = [None] * (len(index) + len(reduction_index))
        for (idx, var) in itertools.chain(zip(kept_idx, index), zip(reduced_idx, reduction_index)):
            new_index[idx] = var
        return inner_loader(new_index)
    if keepdims:
        new_size = list(size)
        for i in reduced_idx:
            new_size[i] = sympy.Integer(1)
    else:
        new_size = kept_sizes
    inner_loader = x.make_loader()
    return dict(device=x.get_device(), dst_dtype=override_return_dtype or x.get_dtype(), src_dtype=x.get_dtype(), inner_fn=loader, ranges=new_size, reduction_ranges=reduced_sizes)

def make_reduction(reduction_type: str, override_return_dtype=None):
    if False:
        while True:
            i = 10

    def inner(x, axis=None, keepdims=False, *, dtype=None):
        if False:
            print('Hello World!')
        kwargs = _make_reduction_inner(x, axis=axis, keepdims=keepdims, dtype=dtype, override_return_dtype=override_return_dtype)
        result = Reduction.create(reduction_type=reduction_type, input_node=x, **kwargs)
        if isinstance(result.data.data, Reduction):
            result.realize()
        return result
    return inner

@register_lowering(aten.mean)
def mean(x, axis=None, keepdim=False, *, dtype=None):
    if False:
        i = 10
        return i + 15
    if dtype is not None:
        x = to_dtype(x, dtype)
    size = x.get_size()
    axis = _validate_reduction_axis(x, axis)
    output_dtype = x.get_dtype()
    if output_dtype in (torch.float16, torch.bfloat16):
        x = to_dtype(x, torch.float)
    sum_result = sum_(x, axis, keepdim)
    denom = sympy_product((size[i] for i in axis))
    denom = ir.IndexingConstant(denom, x.get_dtype(), x.get_device())
    denom = ExpandView.create(denom, list(sum_result.get_size()))
    return to_dtype(div(sum_result, denom), output_dtype)

def var_mean_sum_(x, axis, correction, keepdim, return_mean):
    if False:
        for i in range(10):
            print('nop')
    if correction is None:
        correction = 1
    size = x.get_size()
    axis = _validate_reduction_axis(x, axis)
    x_mean = mean(x, axis, keepdim=True)
    if return_mean:
        x_mean.realize()
    diffs = square(sub(x, x_mean))
    sum_result = sum_(diffs, axis, keepdim)
    denom = sympy_product((size[i] for i in axis))
    if correction:
        denom = sympy.Max(denom - correction, 0)
    denom = ir.IndexingConstant(denom, x.get_dtype(), x.get_device())
    denom = ExpandView.create(denom, list(sum_result.get_size()))
    x_var = div(sum_result, denom)
    if not return_mean:
        return x_var
    x_mean = x_mean if keepdim else squeeze(x_mean, axis)
    return (x_var, x_mean)

def use_two_step_variance(x, axis, keepdim):
    if False:
        return 10
    axis = _validate_reduction_axis(x, axis)
    kwargs = _make_reduction_inner(x, axis=axis, keepdims=keepdim, dtype=None, override_return_dtype=None)
    ranges = kwargs['ranges']
    reduction_numel = sympy_product(kwargs['reduction_ranges'])
    return isinstance(reduction_numel, sympy.Integer) and int(reduction_numel) < config.unroll_reductions_threshold and (sympy_product(ranges) != 1)

def var_mean_welford_(x, axis, *, correction, keepdim, return_mean):
    if False:
        for i in range(10):
            print('nop')
    if correction is None:
        correction = 1
    kwargs = _make_reduction_inner(x, axis=axis, keepdims=keepdim, dtype=None, override_return_dtype=None)
    loader = kwargs.pop('inner_fn')
    kwargs.pop('dst_dtype')
    kwargs.pop('src_dtype')
    (mean, m2, _) = ir.WelfordReduction.create(inner_fns=(loader,), reduction_type='welford_reduce', dtype=x.get_dtype(), **kwargs)
    m2.realize()
    dtype = x.get_dtype()
    size = x.get_size()
    axis = _validate_reduction_axis(x, axis)
    rnumel = sympy_product((size[i] for i in axis))

    def get_constant_or_index_expr(x, dtype):
        if False:
            print('Hello World!')
        if isinstance(x, sympy.Expr) and (not x.is_constant()):
            return ops.to_dtype(ops.index_expr(x, torch.int64), dtype)
        return ops.constant(x, dtype)

    def scale_fn(data):
        if False:
            for i in range(10):
                print('nop')
        c = get_constant_or_index_expr(correction, dtype)
        N = get_constant_or_index_expr(rnumel, dtype)
        zero = ops.constant(0, dtype)
        return data / ops.maximum(zero, N - c)
    var = make_pointwise(scale_fn)(m2)
    if return_mean:
        mean.realize()
        return (var, mean)
    return var

@register_lowering([aten.var, prims.var])
def var_(x, axis=None, *, correction=None, keepdim=False):
    if False:
        for i in range(10):
            print('nop')
    if use_two_step_variance(x, axis=axis, keepdim=keepdim):
        return var_mean_sum_(x, axis=axis, correction=correction, keepdim=keepdim, return_mean=False)
    return var_mean_welford_(x, axis=axis, correction=correction, keepdim=keepdim, return_mean=False)

@register_lowering(aten.var_mean)
def var_mean(x, axis=None, *, correction=None, keepdim=False):
    if False:
        for i in range(10):
            print('nop')
    if use_two_step_variance(x, axis=axis, keepdim=keepdim):
        return var_mean_sum_(x, axis=axis, correction=correction, keepdim=keepdim, return_mean=True)
    return var_mean_welford_(x, axis=axis, correction=correction, keepdim=keepdim, return_mean=True)

def pow_recursive(x, y, dtype):
    if False:
        i = 10
        return i + 15
    if y < 0:
        return pow_recursive(ops.reciprocal(x), -y, dtype)
    if y == 0:
        return ops.constant(1, dtype)
    if y == 1:
        return x
    result = pow_recursive(x, y // 2, dtype)
    result = ops.mul(result, result)
    if y % 2 == 1:
        result = ops.mul(result, x)
    return result

@make_pointwise
def pow_native(a, b):
    if False:
        while True:
            i = 10
    return ops.pow(a, b)
fallback_pow_tensor_tensor = fallback_handler(aten.pow.Tensor_Tensor)
fallback_pow_scalar = fallback_handler(aten.pow.Scalar)
fallback_pow_tensor_scalar = fallback_handler(aten.pow.Tensor_Scalar)

@register_lowering(aten.pow, broadcast=True)
def pow(a, b):
    if False:
        return 10
    if isinstance(b, float) and b == int(b):
        return pow(a, int(b))
    elif isinstance(b, float) and b == 0.5:
        return sqrt(a)
    elif isinstance(b, int) and b == 1:
        return clone(a)
    dtype = next((x.get_dtype() for x in (a, b) if isinstance(x, ir.TensorBox)))
    is_integer_pow = is_integer_dtype(dtype)
    embed_exponent = isinstance(b, int) and (-32 < b < 32 or (is_integer_pow and b >= 0))
    if embed_exponent:
        loader = a.make_loader()

        def fn(idx):
            if False:
                i = 10
                return i + 15
            return pow_recursive(loader(idx), b, a.get_dtype())
        return Pointwise.create(device=a.get_device(), dtype=a.get_dtype(), inner_fn=fn, ranges=a.get_size())
    if isinstance(a, Number):
        if a == 1:
            return full_like(b, 1)
        if a == 2 and is_float_dtype(b.get_dtype()):
            return exp2(b)
    if is_integer_pow:
        if isinstance(a, Number):
            return fallback_pow_scalar(a, b)
        elif isinstance(b, Number):
            return fallback_pow_tensor_scalar(a, b)
        else:
            return fallback_pow_tensor_tensor(a, b)
    return pow_native(a, b)

def mutate_to(changed, val):
    if False:
        print('Hello World!')
    if isinstance(changed, TensorBox):
        changed_data = changed.data
    else:
        changed_data = changed
    if isinstance(val, TensorBox):
        val = val.data
    if not isinstance(val, ir.StorageBox):
        val = Pointwise.create(device=changed.get_device(), dtype=changed.get_dtype(), inner_fn=val.make_loader(), ranges=changed.get_size()).data
        assert isinstance(val, ir.StorageBox)
    if isinstance(changed_data, ir.StorageBox) and (not (changed_data.is_input_buffer() or isinstance(changed_data.data, ir.NopKernel))):
        val.realize()
        changed_data.data = val.data
        return changed
    ir.MutationLayout.realize_into(val, changed_data)
    return changed

@register_lowering(aten.fill_)
def fill_(x, fill_value):
    if False:
        for i in range(10):
            print('nop')
    return mutate_to(x, full_like(x, fill_value))

@register_lowering(aten.copy_, type_promotion_kind=None)
def copy_(dst, src, non_blocking=False):
    if False:
        return 10
    src = to_device(src, dst.get_device())
    src = to_dtype(src, dst.get_dtype())
    src = expand(src, dst.get_size())
    return mutate_to(dst, src)

@make_pointwise
def floordiv(a, b):
    if False:
        return 10
    return ops.floordiv(a, b)

@make_pointwise
def truncdiv(a, b):
    if False:
        return 10
    return ops.truncdiv(a, b)

@register_lowering(aten.div, broadcast=True)
def div_mode(a, b, rounding_mode=None):
    if False:
        for i in range(10):
            print('nop')
    both_integer = is_integer_type(a) and is_integer_type(b)
    both_boolean = is_boolean_type(a) and is_boolean_type(b)
    if rounding_mode == 'floor':
        assert not both_boolean, 'floordiv operands can not be boolean at the same time'
        return floordiv(a, b) if both_integer else floor(div(a, b))
    if rounding_mode == 'trunc':
        assert not both_boolean, 'truncdiv operands can not be boolean at the same time'
        return truncdiv(a, b) if both_integer else trunc(div(a, b))
    return div(a, b)

@register_lowering([aten.mul], broadcast=True)
def mul(a, b):
    if False:
        while True:
            i = 10
    both_bool = is_boolean_type(a) and is_boolean_type(b)
    if both_bool:
        return logical_and(a, b)
    else:
        fn = ops_wrapper(aten.mul.__name__)
        return make_pointwise(fn)(a, b)

@register_lowering([prims.div], broadcast=True)
def div_prim(a, b):
    if False:
        print('Hello World!')
    is_integral = all((is_boolean_type(x) or is_integer_type(x) for x in [a, b]))
    if is_integral:
        return truncdiv(a, b)

    def fn(*args):
        if False:
            print('Hello World!')
        return ops.truediv(*args)
    return make_pointwise(fn)(a, b)
div = register_lowering([aten.true_divide, aten.div.Tensor], broadcast=True, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)(div_prim)

@register_lowering([aten.fmod, prims.fmod], broadcast=True)
def fmod(a, b):
    if False:
        while True:
            i = 10
    is_integral = is_boolean_type(a) or is_integer_type(a)
    if is_integral:

        def fn(a, b):
            if False:
                while True:
                    i = 10
            return ops.mod(a, b)
    else:

        def fn(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return ops.fmod(a, b)
    return make_pointwise(fn)(a, b)

@register_lowering(aten.rsqrt)
def rsqrt(x):
    if False:
        for i in range(10):
            print('nop')
    dtype = x.get_dtype()
    if is_integer_dtype(dtype) or is_boolean_dtype(dtype):
        x = to_dtype(x, torch.get_default_dtype())

    def _rsqrt(x):
        if False:
            print('Hello World!')
        return ops.rsqrt(x)
    return make_pointwise(_rsqrt)(x)

@register_lowering([aten.sum, prims.sum])
def sum_(x, axis=None, keepdims=False, *, dtype=None):
    if False:
        print('Hello World!')
    if (is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())) and dtype is None:
        dtype = torch.int64
    fn = make_reduction('sum', override_return_dtype=dtype)
    return fn(x, axis, keepdims, dtype=dtype)

@register_lowering(aten.prod)
def prod(x, axis=None, keepdims=False, *, dtype=None):
    if False:
        while True:
            i = 10
    if (is_integer_dtype(x.get_dtype()) or is_boolean_dtype(x.get_dtype())) and dtype is None:
        dtype = torch.int64
    fn = make_reduction('prod', override_return_dtype=dtype)
    return fn(x, axis, keepdims, dtype=dtype)

@register_lowering(aten.any)
def reduce_any(x, dim=None, keepdim=False):
    if False:
        while True:
            i = 10
    x = to_dtype(x, torch.bool)
    return make_reduction('any')(x, axis=dim, keepdims=keepdim)

@register_lowering(aten.max, type_promotion_kind=None)
def reduce_max(x, dim=None, keepdim=False):
    if False:
        while True:
            i = 10
    if dim is not None:
        return (reduce_amax(x, axis=dim, keepdims=keepdim), reduce_argmax(x, axis=dim, keepdims=keepdim))
    return reduce_amax(x, axis=None, keepdims=keepdim)

@register_lowering(aten.min, type_promotion_kind=None)
def reduce_min(x, dim=None, keepdim=False):
    if False:
        i = 10
        return i + 15
    if dim is not None:
        return (reduce_amin(x, axis=dim, keepdims=keepdim), reduce_argmin(x, axis=dim, keepdims=keepdim))
    return reduce_amin(x, axis=None, keepdims=keepdim)
register_lowering(prims.xor_sum)(make_reduction('xor_sum'))
reduce_amax = register_lowering(aten.amax)(make_reduction('max'))
reduce_amin = register_lowering(aten.amin)(make_reduction('min'))
reduce_argmax = register_lowering(aten.argmax)(make_reduction('argmax', override_return_dtype=torch.int64))
reduce_argmin = register_lowering(aten.argmin)(make_reduction('argmin', override_return_dtype=torch.int64))
add = register_pointwise(aten.add, allow_alpha=True, override_fn_when_input_bool='logical_or')

def register_pointwise_numeric(op):
    if False:
        print('Hello World!')
    return register_pointwise(op, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)

def register_pointwise_numeric_ldf64(op):
    if False:
        return 10
    return register_pointwise(op, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT, use_libdevice_for_f64=True)
exp = register_pointwise_numeric_ldf64(aten.exp)
exp2 = register_pointwise_numeric(aten.exp2)
expm1 = register_pointwise_numeric(aten.expm1)
relu = register_pointwise(aten.relu)
sigmoid = register_pointwise_numeric_ldf64(aten.sigmoid)
sqrt = register_pointwise_numeric_ldf64(aten.sqrt)
square = register_pointwise(aten.square)
sub = register_pointwise(aten.sub, allow_alpha=True)
register_pointwise_numeric_ldf64(aten.cos)
register_pointwise_numeric_ldf64(aten.sin)
abs = register_pointwise(aten.abs)
bitwise_and = register_pointwise(aten.bitwise_and)
bitwise_left_shift = register_pointwise(aten.bitwise_left_shift)
bitwise_not = register_pointwise(aten.bitwise_not, override_fn_when_input_bool='logical_not')
bitwise_or = register_pointwise(aten.bitwise_or)
bitwise_right_shift = register_pointwise(aten.bitwise_right_shift)
bitwise_xor = register_pointwise(aten.bitwise_xor)
register_pointwise_numeric(aten.lgamma)
erf = register_pointwise_numeric(aten.erf)
register_lowering(aten.special_erf, type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT)(erf)
register_pointwise_numeric(aten.log1p)
register_pointwise_numeric(aten.tan)
register_pointwise_numeric(aten.tanh)
register_pointwise_numeric_ldf64(aten.log)
logical_and = register_pointwise(aten.logical_and, type_promotion_kind=None, convert_input_to_bool=True, override_return_dtype=torch.bool)
logical_not = register_pointwise(aten.logical_not, type_promotion_kind=None, convert_input_to_bool=True, override_return_dtype=torch.bool)
logical_or = register_pointwise(aten.logical_or, type_promotion_kind=None, convert_input_to_bool=True, override_return_dtype=torch.bool)
logical_xor = register_pointwise(aten.logical_xor, type_promotion_kind=None, convert_input_to_bool=True, override_return_dtype=torch.bool)
maximum = register_pointwise(aten.maximum)
minimum = register_pointwise(aten.minimum)
register_lowering(aten.clamp_min)(maximum)
register_lowering(aten.clamp_max)(minimum)
neg = register_pointwise(aten.neg)
reciprocal = register_pointwise_numeric(aten.reciprocal)
register_pointwise(aten.remainder)
sign = register_pointwise(aten.sign, override_fn_when_input_bool='identity')
register_pointwise(aten.ceil)
register_pointwise(aten.signbit, override_return_dtype=torch.bool)
register_lowering(aten._neg_view)(neg)
register_pointwise(aten.le, override_return_dtype=torch.bool)
register_pointwise(aten.lt, override_return_dtype=torch.bool)
register_pointwise(aten.ge, override_return_dtype=torch.bool)
gt = register_pointwise(aten.gt, override_return_dtype=torch.bool)
register_pointwise(aten.eq, override_return_dtype=torch.bool)
register_pointwise(aten.ne, override_return_dtype=torch.bool)
register_pointwise_numeric(aten.cosh)
register_pointwise_numeric(aten.sinh)
register_pointwise_numeric(aten.acos)
register_pointwise_numeric(aten.acosh)
register_pointwise_numeric(aten.asin)
register_pointwise_numeric(aten.asinh)
register_pointwise_numeric(aten.atan2)
register_pointwise_numeric(aten.atan)
register_pointwise_numeric(aten.atanh)
register_pointwise_numeric(aten.copysign)
register_pointwise_numeric(aten.erfc)
register_pointwise_numeric(aten.erfinv)
register_pointwise_numeric(aten.hypot)
register_pointwise_numeric(aten.log10)
register_pointwise_numeric(aten.nextafter)
register_foreach_pointwise(aten._foreach_add.List, add, allow_alpha=True)
register_foreach_pointwise(aten._foreach_add.Scalar, add, allow_alpha=True)
register_foreach_pointwise(aten._foreach_add.Tensor, add, allow_alpha=True)
register_foreach_pointwise(aten._foreach_mul.List, mul)
register_foreach_pointwise(aten._foreach_mul.Scalar, mul)
register_foreach_pointwise(aten._foreach_sub.List, sub)
register_foreach_pointwise(aten._foreach_sub.Scalar, sub)
register_foreach_pointwise(aten._foreach_neg.default, neg)
register_foreach_pointwise(aten._foreach_abs.default, abs)
register_foreach_pointwise(aten._foreach_pow.Scalar, pow)
register_foreach_pointwise(aten._foreach_pow.ScalarAndTensor, pow)
register_foreach_pointwise(aten._foreach_div.List, div)
register_foreach_pointwise(aten._foreach_div.Scalar, div)
register_foreach_pointwise(aten._foreach_sqrt, sqrt)
register_foreach_pointwise(aten._foreach_maximum.List, maximum)
register_foreach_pointwise(aten._foreach_maximum.Scalar, maximum)
register_foreach_pointwise(aten._foreach_minimum.List, minimum)
register_foreach_pointwise(aten._foreach_minimum.Scalar, minimum)
register_foreach_pointwise(aten._foreach_clamp_min.List, maximum)
register_foreach_pointwise(aten._foreach_clamp_min.Scalar, maximum)
register_foreach_pointwise(aten._foreach_clamp_max.List, minimum)
register_foreach_pointwise(aten._foreach_clamp_max.Scalar, minimum)
register_foreach_pointwise(aten._foreach_reciprocal, reciprocal)
register_foreach_pointwise(aten._foreach_sign, sign)
register_foreach_pointwise(aten._foreach_copy, copy)

def register_inplace(aten_op, outplace_op):
    if False:
        i = 10
        return i + 15

    @register_lowering(aten_op, type_promotion_kind=None)
    def fn(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        result = outplace_op(*args, **kwargs)
        result = to_dtype(result, args[0].get_dtype())
        return mutate_to(args[0], result)
    return fn
register_inplace(aten.add_, add)
register_inplace(aten.bitwise_and_, bitwise_and)
register_inplace(aten.bitwise_left_shift_, bitwise_left_shift)
register_inplace(aten.bitwise_not_, bitwise_not)
register_inplace(aten.bitwise_or_, bitwise_or)
register_inplace(aten.bitwise_right_shift_, bitwise_right_shift)
register_inplace(aten.bitwise_xor_, bitwise_xor)
register_inplace(aten.mul_, mul)
register_inplace(aten.div_.Tensor, div)
register_inplace(aten.div_.Tensor_mode, div_mode)
register_inplace(aten.logical_and_, logical_and)
register_inplace(aten.logical_not_, logical_not)
register_inplace(aten.logical_or_, logical_or)
register_inplace(aten.logical_xor_, logical_xor)
register_inplace(aten.sub_, sub)
register_inplace(aten.relu_, relu)
register_inplace(aten.sigmoid_, sigmoid)
register_lowering(aten.__and__)(bitwise_and)
register_lowering(aten.__lshift__)(bitwise_left_shift)
register_lowering(aten.__or__)(bitwise_or)
register_lowering(aten.__rshift__)(bitwise_right_shift)
register_lowering(aten.__xor__)(bitwise_xor)
register_inplace(aten.__iand__, aten.__and__)
register_inplace(aten.__ilshift__, aten.__lshift__)
register_inplace(aten.__ior__, aten.__or__)
register_inplace(aten.__irshift__, aten.__rshift__)
register_inplace(aten.__ixor__, aten.__xor__)

@register_lowering(aten.sym_constrain_range)
def sym_constrain_range(a, min, max):
    if False:
        for i in range(10):
            print('nop')
    tracing_context = torch._guards.TracingContext.get()
    assert a in tracing_context.fake_mode.shape_env.var_to_range
    return a

@register_lowering(aten.sym_size.int)
def sym_size(a, dim):
    if False:
        i = 10
        return i + 15
    val = V.graph.current_node.meta['val']
    assert isinstance(val, torch.SymInt)
    return val.node.expr

@register_lowering(aten.sym_stride.int)
def sym_stride(a, dim):
    if False:
        print('Hello World!')
    val = V.graph.current_node.meta['val']
    assert isinstance(val, torch.SymInt)
    return val.node.expr

@register_lowering(aten.sym_numel)
def sym_numel(a):
    if False:
        while True:
            i = 10
    return a.get_numel()
for (method, func) in magic_methods.items():
    register_lowering(method_to_operator(method))(func)

@register_lowering(aten._foobar)
def foobar(self, *args, **kwargs):
    if False:
        while True:
            i = 10
    raise NotImplementedError('Helpful for debugging')

@register_lowering(torch.ops._inductor_test.realize)
def _realize(x):
    if False:
        print('Hello World!')
    x.realize()
    return clone(x)

@register_lowering(torch.ops.inductor.accumulate_grad_)
def accumulate_grad_(variable, new_grad):
    if False:
        while True:
            i = 10
    variable.realize()
    new_grad.realize()
    ir.AccumulateGrad(variable, new_grad)
    return variable

@register_lowering(triton_kernel_wrapper_mutation)
def triton_kernel_wrap_(*, kernel_idx, grid, kwargs):
    if False:
        while True:
            i = 10
    ir.UserDefinedTritonKernel(kernel_idx=kernel_idx, grid=grid, kernel_args=kwargs)
    return {key: val for (key, val) in kwargs.items() if isinstance(val, TensorBox)}

@register_lowering(triton_kernel_wrapper_functional)
def triton_kernel_wrap(*, kernel_idx, grid, kwargs, tensors_to_clone):
    if False:
        print('Hello World!')
    kwargs = {key: clone(x) if key in tensors_to_clone else x for (key, x) in kwargs.items()}
    return triton_kernel_wrap_(kernel_idx=kernel_idx, grid=grid, kwargs=kwargs)
try:
    import torch.distributed._functional_collectives
    c10d_functional = torch.ops.c10d_functional

    @register_lowering(c10d_functional.wait_tensor)
    def wait(input):
        if False:
            print('Hello World!')
        return TensorBox.create(ir.Wait.create(input))

    @register_lowering(c10d_functional.broadcast)
    def broadcast(input, src, tag, ranks, group_size):
        if False:
            return 10
        return ir.Broadcast.create(input, src, tag, ranks, group_size)

    @register_lowering(c10d_functional.all_reduce)
    def allreduce(input, reduce_op, tag, ranks, group_size):
        if False:
            i = 10
            return i + 15
        return ir.AllReduce.create(input, reduce_op, tag, ranks, group_size)

    @register_lowering(c10d_functional.all_gather_into_tensor)
    def all_gather_into_tensor(shard, tag, ranks, group_size):
        if False:
            i = 10
            return i + 15
        return TensorBox.create(ir.AllGatherIntoTensor.create(ir.ExternKernel.require_contiguous(shard), tag, ranks, group_size))

    @register_lowering(c10d_functional.reduce_scatter_tensor)
    def reduce_scatter_tensor(input, reduce_op, tag, ranks, group_size):
        if False:
            i = 10
            return i + 15
        return TensorBox.create(ir.ReduceScatterTensor.create(input, reduce_op, tag, ranks, group_size))

    @register_lowering(c10d_functional.all_reduce_coalesced)
    def all_reduce_coalesced(input, reduce_op, tag, ranks, group_size):
        if False:
            return 10
        return ir.AllReduceCoalesced.create(input, reduce_op, tag, ranks, group_size)

    @register_lowering(c10d_functional.all_gather_into_tensor_coalesced)
    def all_gather_into_tensor_coalesced(self, tag, ranks, group_size):
        if False:
            i = 10
            return i + 15
        result = ir.AllGatherIntoTensorCoalesced.create(self, tag, ranks, group_size)
        return list(map(TensorBox.create, result))

    @register_lowering(c10d_functional.reduce_scatter_tensor_coalesced)
    def reduce_scatter_tensor_coalesced(self, reduceOp, tag, ranks, group_size):
        if False:
            i = 10
            return i + 15
        result = ir.ReduceScatterTensorCoalesced.create(self, reduceOp, tag, ranks, group_size)
        return list(map(TensorBox.create, result))

    @register_lowering(c10d_functional.all_to_all_single)
    def all_to_all_single(self, output_split_sizes, input_split_sizes, tag, ranks, group_size):
        if False:
            i = 10
            return i + 15
        return TensorBox.create(ir.AllToAllSingle.create(self, output_split_sizes, input_split_sizes, tag, ranks, group_size))
    _c10d_functional = torch.ops._c10d_functional

    @register_lowering(_c10d_functional.all_reduce)
    def _all_reduce(inp, reduce_op, group_name):
        if False:
            i = 10
            return i + 15
        inp = clone(inp)
        ir._CollectiveKernel.create_inplace(_c10d_functional.all_reduce_.default, inp, reduce_op, group_name)
        return inp

    @register_lowering(_c10d_functional.all_reduce_)
    def _all_reduce_(inp, reduce_op, group_name):
        if False:
            return 10
        ir._CollectiveKernel.create_inplace(_c10d_functional.all_reduce_.default, inp, reduce_op, group_name)
        return inp

    @register_lowering(_c10d_functional.all_reduce_coalesced)
    def _all_reduce_coalesced(inputs, reduce_op, group_name):
        if False:
            print('Hello World!')
        inputs = [clone(inp) for inp in inputs]
        ir._CollectiveKernel.create_inplace(_c10d_functional.all_reduce_coalesced_.default, inputs, reduce_op, group_name)
        return inputs

    @register_lowering(_c10d_functional.all_reduce_coalesced_)
    def _all_reduce_coalesced_(inputs, reduce_op, group_name):
        if False:
            while True:
                i = 10
        ir._CollectiveKernel.create_inplace(_c10d_functional.all_reduce_coalesced_.default, inputs, reduce_op, group_name)
        return inputs

    @register_lowering(_c10d_functional.all_gather_into_tensor)
    def _all_gather_into_tensor(inp, group_size, group_name):
        if False:
            i = 10
            return i + 15
        return ir.TensorBox.create(ir._CollectiveKernel.create_out_of_place(_c10d_functional.all_gather_into_tensor.default, inp, group_size, group_name))

    @register_lowering(_c10d_functional.all_gather_into_tensor_coalesced)
    def _all_gather_into_tensor_coalesced(inputs, group_size, group_name):
        if False:
            for i in range(10):
                print('nop')
        return pytree.tree_map(ir.TensorBox.create, ir._CollectiveKernel.create_out_of_place(_c10d_functional.all_gather_into_tensor_coalesced.default, inputs, group_size, group_name))

    @register_lowering(_c10d_functional.reduce_scatter_tensor)
    def _reduce_scatter_tensor(inp, reduce_op, group_size, group_name):
        if False:
            print('Hello World!')
        return ir.TensorBox.create(ir._CollectiveKernel.create_out_of_place(_c10d_functional.reduce_scatter_tensor.default, inp, reduce_op, group_size, group_name))

    @register_lowering(_c10d_functional.reduce_scatter_tensor_coalesced)
    def _reduce_scatter_tensor_coalesced(inputs, reduce_op, group_size, group_name):
        if False:
            return 10
        return pytree.tree_map(ir.TensorBox.create, ir._CollectiveKernel.create_out_of_place(_c10d_functional.reduce_scatter_tensor_coalesced.default, inputs, reduce_op, group_size, group_name))

    @register_lowering(_c10d_functional.wait_tensor)
    def _wait_tensor(inp):
        if False:
            for i in range(10):
                print('nop')
        ir._WaitKernel.create_wait(_c10d_functional.wait_tensor.default, inp)
        return inp
except ImportError:
    log.info('Inductor support for distributed collectives depends on building torch.distributed')
from . import kernel
import_submodule(kernel)
from . import quantized_lowerings
quantized_lowerings.register_quantized_ops()