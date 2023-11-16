from __future__ import annotations
import functools
import inspect
import sys
import typing
import warnings
from typing import Any, Callable, List, Literal, NoReturn, Optional, Sequence, Set, Tuple, Union
import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, _type_utils, errors
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import _beartype, jit_utils
from torch.types import Number
__all__ = ['args_have_same_dtype', 'cast_pytorch_to_onnx', 'check_training_mode', 'dequantize_helper', 'is_caffe2_aten_fallback', 'is_complex_value', 'parse_args', 'pytorch_name_to_type', 'quantize_helper', 'quantized_args', 'requantize_bias_helper', 'scalar_name_to_pytorch', 'scalar_type_to_onnx', 'scalar_type_to_pytorch_type']
_ValueDescriptor = Literal['v', 'i', 'is', 'f', 'fs', 'b', 's', 't', 'none']

@_beartype.beartype
def _parse_arg(value, desc: _ValueDescriptor, arg_name: Optional[str]=None, node_name: Optional[str]=None):
    if False:
        for i in range(10):
            print('nop')
    if desc == 'none':
        return value
    if desc == 'v' or not _is_value(value):
        return value
    node = value.node()
    if node.mustBeNone():
        return None
    if node.kind() == 'onnx::Constant':
        node_val = _node_get(node, 'value')
        if desc == 'i':
            return int(node_val)
        elif desc == 'f':
            return float(node_val)
        elif desc == 'b':
            return bool(node_val)
        elif desc == 's':
            return str(node_val)
        elif desc == 't':
            return node_val
        elif desc == 'is':
            return [int(v) for v in node_val]
        elif desc == 'fs':
            return [float(v) for v in node_val]
        else:
            raise errors.SymbolicValueError(f"ONNX symbolic does not understand the Constant node '{node}' specified with descriptor '{desc}'.", value)
    elif node.kind() == 'prim::ListConstruct':
        if desc == 'is':
            for v in node.inputs():
                element_node = v.node()
                if element_node.kind() != 'onnx::Constant':
                    raise errors.SymbolicValueError(f"Failed to export a node '{element_node}' (in list node {node}) because it is not constant. Please try to make things (e.g. kernel sizes) static if possible.", value)
            return [int(_node_get(v.node(), 'value')) for v in value.node().inputs()]
        else:
            raise errors.SymbolicValueError(f"ONNX symbolic does not know how to unpack the ListConstruct node that is not a list of integers: '{node}'", value)
    if arg_name is None or node_name is None:
        raise errors.SymbolicValueError(f"Expected node type 'onnx::Constant', got '{node.kind()}'.", value)
    raise errors.SymbolicValueError(f"Expected node type 'onnx::Constant' for argument '{arg_name}' of node '{node_name}', got '{node.kind()}'.", value)

@_beartype.beartype
def _node_get(node: _C.Node, key: str):
    if False:
        while True:
            i = 10
    'Gets attributes of a node which is polymorphic over return type.'
    assert isinstance(node, _C.Node)
    sel = node.kindOf(key)
    return getattr(node, sel)(key)

@_beartype.beartype
def _is_onnx_constant(value: _C.Value):
    if False:
        return 10
    'Whether a Value is an ONNX constant.'
    return value.node().kind() == 'onnx::Constant'

@_beartype.beartype
def _maybe_get_const(value: Optional[Union[_C.Value, torch.Tensor, Number, Sequence]], descriptor: _ValueDescriptor):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(value, _C.Value) and _is_onnx_constant(value):
        return _parse_arg(value, descriptor)
    return value

@_beartype.beartype
def _maybe_get_scalar(value):
    if False:
        return 10
    value_t = _maybe_get_const(value, 't')
    if isinstance(value_t, torch.Tensor) and value_t.shape == ():
        return value_t
    return value

@_beartype.beartype
def _get_const(value, desc, arg_name):
    if False:
        return 10
    if not _is_constant(value):
        raise errors.SymbolicValueError(f"ONNX symbolic expected a constant value of the '{arg_name}' argument, got '{value}'", value)
    return _parse_arg(value, desc)

@_beartype.beartype
def _unpack_list(list_value: _C.Value) -> List[_C.Value]:
    if False:
        i = 10
        return i + 15
    list_node = list_value.node()
    if list_node.kind() != 'prim::ListConstruct':
        raise errors.SymbolicValueError(f"ONNX symbolic expected node type prim::ListConstruct, got '{list_node}'.", list_value)
    return list(list_node.inputs())

@_beartype.beartype
def _unpack_tuple(tuple_value: _C.Value) -> Tuple[_C.Value, ...]:
    if False:
        while True:
            i = 10
    tuple_node = tuple_value.node()
    if not _is_tuple_construct(tuple_value):
        raise errors.SymbolicValueError(f"ONNX symbolic expected node type 'prim::TupleConstruct', got '{tuple_node.kind()}'.", tuple_value)
    return tuple(tuple_node.inputs())

@_beartype.beartype
def _unpack_quantized_tensor(tuple_value: _C.Value) -> Tuple[_C.Value, ...]:
    if False:
        while True:
            i = 10
    'Unpacks a quantized tensor into a tuple of tensor and scale/zero_point.\n    Args:\n        tuple_value: A tuple of tensor, scale, zero_point, and optionally axis.\n    Returns:\n        A tuple of tensor, scale, zero_point, and optionally axis.\n    '
    tuple_node = tuple_value.node()
    if not _is_tuple_construct(tuple_value):
        raise errors.SymbolicValueError(f'ONNX symbolic expected the output of `{tuple_node}` to be a quantized tensor. Is this likely due to missing support for quantized `{tuple_node.kind()}`. Please create an issue on {_constants.PYTORCH_GITHUB_ISSUES_URL}', tuple_value)
    unpacked = tuple(tuple_node.inputs())
    assert len(unpacked) == 3 or len(unpacked) == 4
    return unpacked

@_beartype.beartype
def _is_packed_list(list_value: Any) -> bool:
    if False:
        return 10
    return _is_value(list_value) and list_value.node().kind() == 'prim::ListConstruct'

@_beartype.beartype
def parse_args(*arg_descriptors: _ValueDescriptor):
    if False:
        return 10
    'A decorator which converts args from torch._C.Value to built-in types.\n\n    For example:\n\n    ```\n    @parse_args(\'v\', \'i\', \'fs\')\n    foo(g, a, b, c):\n        assert isinstance(a, torch._C.Value)\n        assert isinstance(b, int)\n        assert isinstance(c, list)\n        assert isinstance(c[0], float)\n    ```\n\n    Args:\n        arg_descriptors: list of str, where each element is\n            a string that specifies the type to convert to. Valid descriptors:\n            "v": no conversion, keep torch._C.Value.\n            "i": int\n            "is": list of int\n            "f": float\n            "fs": list of float\n            "b": bool\n            "s": str\n            "t": torch.Tensor\n            "none": the variable is unused\n    '

    def decorator(fn):
        if False:
            print('Hello World!')
        fn._arg_descriptors = arg_descriptors

        @functools.wraps(fn)
        def wrapper(g, *args, **kwargs):
            if False:
                while True:
                    i = 10
            FILE_BUG_MSG = 'If you believe this is not due to custom symbolic implementation within your code or an external library, please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml to report this bug.'
            assert len(arg_descriptors) >= len(args), f"A mismatch between the number of arguments ({len(args)}) and their descriptors ({len(arg_descriptors)}) was found at symbolic function '{fn.__name__}'. {FILE_BUG_MSG}"
            try:
                sig = inspect.signature(fn)
                arg_names = list(sig.parameters.keys())[1:]
                fn_name = fn.__name__
            except Exception:
                arg_names = [None] * len(args)
                fn_name = None
            args = [_parse_arg(arg, arg_desc, arg_name, fn_name) for (arg, arg_desc, arg_name) in zip(args, arg_descriptors, arg_names)]
            assert len(kwargs) <= 1, f"Symbolic function {fn.__name__}'s '**kwargs' can contain a single key/value entry. {FILE_BUG_MSG}"
            if len(kwargs) == 1:
                assert '_outputs' in kwargs, f"Symbolic function {fn.__name__}'s '**kwargs' can only contain '_outputs' key at '**kwargs'. {FILE_BUG_MSG}"
            return fn(g, *args, **kwargs)
        return wrapper
    return decorator

@_beartype.beartype
def quantized_args(*arg_q_descriptors: bool, scale: Optional[float]=None, zero_point: Optional[int]=None, quantize_output: bool=True):
    if False:
        for i in range(10):
            print('nop')
    'A decorator which extends support for quantized version of the base operator.\n\n    Quantization is detected by examining the arguments that are annotated by\n    `arg_q_descriptors`.\n\n    If quantization is detected, the base operator symbolic function will be wrapped with\n    argument de-quantization and output quantization.\n\n    Otherwise, only the base symbolic function will be invoked.\n\n    For example:\n\n    ```\n    @quantized_args(True, False)\n    def foo(g, x, y):\n        return x + y\n    ```\n\n    is equivalent to\n\n    ```\n    def q_foo(g, x, y):\n        if is_quantized_tensor(x):\n            x = dequantize(x)\n            out = foo(g, x, y)\n            return quantize(out)\n        else:\n            return foo(g, x, y)\n    ```\n\n    Args:\n        arg_q_descriptors: A sequence of bool, where each element represents if the\n          argument is QTensor for quantized version of this operator. It defaults\n          to False for unspecified (variable length) arguments.\n        scale: Quantized output scale. If None, derive from\n          the first quantized input scale.\n        zero_point: Quantized output zero point. If None,\n          derive from the first quantized input zero point.\n        quantize_output: If True, quantize the output of the base operator. Default is True\n    '

    def decorator(fn):
        if False:
            for i in range(10):
                print('nop')

        @functools.wraps(fn)
        def wrapper(g, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            nonlocal scale
            nonlocal zero_point
            if scale is not None:
                _scale = g.op('Constant', value_t=torch.tensor(scale))
            else:
                _scale = None
            if zero_point is not None:
                _zero_point = g.op('Constant', value_t=torch.tensor(zero_point))
            else:
                _zero_point = None
            arg_q_descriptors_extended = arg_q_descriptors + (False,) * (len(args) - len(arg_q_descriptors))
            descriptor_args = tuple(zip(arg_q_descriptors_extended, args))

            def _is_arg_quantized(descriptor, arg):
                if False:
                    for i in range(10):
                        print('nop')
                return descriptor and _is_value(arg) and _is_tuple_construct(arg)
            is_quantized = list()
            for (descriptor, arg) in descriptor_args:
                if _is_packed_list(arg):
                    for arg_input in arg.node().inputs():
                        is_quantized.append(_is_arg_quantized(descriptor, arg_input))
                else:
                    is_quantized.append(_is_arg_quantized(descriptor, arg))
            if not any(is_quantized):
                return fn(g, *args, **kwargs)
            non_quantized_args = []
            for (descriptor, arg) in descriptor_args:
                if _is_arg_quantized(descriptor, arg):
                    (dequantized_arg, arg_scale, arg_zero_point, _) = dequantize_helper(g, arg)
                    non_quantized_args.append(dequantized_arg)
                    if _scale is None:
                        _scale = arg_scale
                    if _zero_point is None:
                        _zero_point = arg_zero_point
                elif _is_packed_list(arg):
                    for arg_input in arg.node().inputs():
                        if _is_arg_quantized(descriptor, arg_input):
                            (dequantized_arg, arg_scale, arg_zero_point, _) = dequantize_helper(g, arg_input)
                            if _scale is None:
                                _scale = arg_scale
                            if _zero_point is None:
                                _zero_point = arg_zero_point
                            arg_input.replaceAllUsesWith(dequantized_arg)
                    non_quantized_args.append(arg)
                else:
                    non_quantized_args.append(arg)
            output = fn(g, *non_quantized_args, **kwargs)
            assert _scale is not None, 'Bug: Scale must be set for quantized operator'
            assert _zero_point is not None, 'Bug: Zero point must be set for quantized operator'
            if quantize_output:
                return quantize_helper(g, output, _scale, _zero_point)
            return output
        return wrapper
    return decorator

@_beartype.beartype
def _scalar(x: Any) -> Optional[Number]:
    if False:
        i = 10
        return i + 15
    'Convert a scalar tensor into a Python value.'
    if isinstance(x, torch.Tensor) and x.shape == ():
        return x.item()
    return None

@_beartype.beartype
def _if_scalar_type_as(self, tensor):
    if False:
        print('Hello World!')
    '\n    Convert self into the same type of tensor, as necessary.\n    We only support implicit casting for scalars, so we never\n    actually need to insert an ONNX cast operator here; just\n    fix up the scalar.\n    '
    if isinstance(self, _C.Value):
        return self
    scalar_type = _type_utils.JitScalarType.from_value(tensor, _type_utils.JitScalarType.UNDEFINED)
    if scalar_type != _type_utils.JitScalarType.UNDEFINED:
        ty = scalar_type.scalar_name().lower()
        return getattr(self, ty)()
    return self

@_beartype.beartype
def _is_none(x: Any) -> bool:
    if False:
        i = 10
        return i + 15
    return x is None or (x.node().mustBeNone() if isinstance(x, _C.Value) else False)

@_beartype.beartype
def _is_value(x: Any) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return isinstance(x, _C.Value)

@_beartype.beartype
def _is_constant(value: Any) -> bool:
    if False:
        return 10
    return not _is_value(value) or value.node().kind() in {'onnx::Constant', 'prim::Constant'}

@_beartype.beartype
def _is_tensor(x: _C.Value) -> bool:
    if False:
        print('Hello World!')
    return x.type().isSubtypeOf(_C.TensorType.get())

def _as_list_type(jit_type: _C.JitType) -> Optional[_C.ListType]:
    if False:
        while True:
            i = 10
    if isinstance(jit_type, _C.ListType):
        return jit_type
    return None

@_beartype.beartype
def _is_list(x: _C.Value) -> bool:
    if False:
        i = 10
        return i + 15
    return _as_list_type(x.type()) is not None

@_beartype.beartype
def _is_tensor_list(x: _C.Value) -> bool:
    if False:
        for i in range(10):
            print('nop')
    x_type = _as_list_type(x.type())
    if x_type is None:
        return False
    return isinstance(x_type.getElementType(), _C.TensorType)

@_beartype.beartype
def _is_scalar_list(x: _C.Value) -> bool:
    if False:
        print('Hello World!')
    'Checks if x is a scalar list, for example: List[float], List[int].\n\n    Besides checking the type is ListType, we also check if the data type is\n    a valid ONNX data type.\n    '
    x_type = _as_list_type(x.type())
    if x_type is None:
        return False
    scalar_type = _type_utils.JitScalarType.from_value(x)
    return scalar_type.onnx_compatible()

@_beartype.beartype
def _is_tuple_construct(x: _C.Value) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return x.node().kind() == 'prim::TupleConstruct'

@_beartype.beartype
def is_complex_value(x: _C.Value) -> bool:
    if False:
        print('Hello World!')
    assert _is_value(x)
    return _type_utils.JitScalarType.from_value(x, _type_utils.JitScalarType.UNDEFINED) in {_type_utils.JitScalarType.COMPLEX32, _type_utils.JitScalarType.COMPLEX64, _type_utils.JitScalarType.COMPLEX128}

@_beartype.beartype
def is_caffe2_aten_fallback() -> bool:
    if False:
        i = 10
        return i + 15
    return GLOBALS.operator_export_type == _C_onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK and _C_onnx._CAFFE2_ATEN_FALLBACK

@_beartype.beartype
def _get_tensor_rank(x: _C.Value) -> Optional[int]:
    if False:
        for i in range(10):
            print('nop')
    if not _is_tensor(x) or x.type() is None:
        return None
    x_type = x.type()
    x_type = typing.cast(_C.TensorType, x_type)
    return x_type.dim()

@_beartype.beartype
def _get_tensor_sizes(x: _C.Value, allow_nonstatic: bool=True):
    if False:
        while True:
            i = 10
    if not _is_tensor(x) or x.type() is None:
        return None
    x_type = x.type()
    x_type = typing.cast(_C.TensorType, x_type)
    if allow_nonstatic:
        return x_type.varyingSizes()
    return x_type.sizes()

@_beartype.beartype
def _get_tensor_dim_size(x: _C.Value, dim: int) -> Optional[int]:
    if False:
        print('Hello World!')
    sizes = _get_tensor_sizes(x)
    return sizes[dim] if sizes else None

@_beartype.beartype
def _get_dim_for_cross(x: _C.Value, dim: Optional[int]):
    if False:
        for i in range(10):
            print('nop')
    if dim == -1:
        tensor_rank = _get_tensor_rank(x)
        assert tensor_rank is not None
        return dim + tensor_rank
    if dim is None:
        sizes = _get_tensor_sizes(x)
        assert sizes is not None
        for (index, size) in enumerate(sizes):
            if size is not None and size == 3:
                return index
    return dim

@_beartype.beartype
def _unimplemented(op: str, msg: str, value: Optional[_C.Value]=None) -> None:
    if False:
        i = 10
        return i + 15
    if _C_onnx._CAFFE2_ATEN_FALLBACK:
        warnings.warn(f'ONNX export failed on {op} because {msg} not supported')
    elif GLOBALS.operator_export_type == _C_onnx.OperatorExportTypes.ONNX:
        _onnx_unsupported(f'{op}, {msg}', value)

@_beartype.beartype
def _onnx_unsupported(op_name: str, value: Optional[_C.Value]=None) -> NoReturn:
    if False:
        i = 10
        return i + 15
    message = f'Unsupported: ONNX export of operator {op_name}. Please feel free to request support or submit a pull request on PyTorch GitHub: {_constants.PYTORCH_GITHUB_ISSUES_URL}'
    if isinstance(value, _C.Value):
        raise errors.SymbolicValueError(message, value)
    raise errors.OnnxExporterError(message)

@_beartype.beartype
def _onnx_opset_unsupported(op_name: str, current_opset: int, supported_opset: int, value: Optional[_C.Value]=None) -> NoReturn:
    if False:
        while True:
            i = 10
    message = f'Unsupported: ONNX export of {op_name} in opset {current_opset}. Please try opset version {supported_opset}.'
    if isinstance(value, _C.Value):
        raise errors.SymbolicValueError(message, value)
    raise errors.OnnxExporterError(message)

@_beartype.beartype
def _onnx_opset_unsupported_detailed(op_name: str, current_opset: int, supported_opset: int, reason: str, value: Optional[_C.Value]=None) -> NoReturn:
    if False:
        i = 10
        return i + 15
    message = f'Unsupported: ONNX export of {op_name} in opset {current_opset}. {reason}. Please try opset version {supported_opset}.'
    if isinstance(value, _C.Value):
        raise errors.SymbolicValueError(message, value)
    raise errors.OnnxExporterError(message)

@_beartype.beartype
def _block_list_in_opset(name: str):
    if False:
        return 10

    def symbolic_fn(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        raise errors.OnnxExporterError(f'ONNX export failed on {name}, which is not implemented for opset {GLOBALS.export_onnx_opset_version}. Try exporting with other opset versions.')
    return symbolic_fn

@_beartype.beartype
def _try_get_scalar_type(*args) -> Optional[_type_utils.JitScalarType]:
    if False:
        while True:
            i = 10
    for arg in args:
        scalar_type = _type_utils.JitScalarType.from_value(arg, _type_utils.JitScalarType.UNDEFINED)
        if scalar_type != _type_utils.JitScalarType.UNDEFINED:
            return scalar_type
    return None

@_beartype.beartype
def _select_helper(g: jit_utils.GraphContext, self, dim, index, apply_reshape=True):
    if False:
        while True:
            i = 10
    index_const = _maybe_get_scalar(index)
    index_dim = _get_tensor_rank(index)
    if not _is_value(index_const):
        index = g.op('Constant', value_t=torch.LongTensor([index_const]))
    elif index_dim is not None and apply_reshape:
        if index_dim == 0:
            index = _reshape_helper(g, index, g.op('Constant', value_t=torch.LongTensor([1])))
    index_scalar_type = _type_utils.JitScalarType.from_value(index, _type_utils.JitScalarType.UNDEFINED)
    if index_scalar_type not in {_type_utils.JitScalarType.INT64, _type_utils.JitScalarType.INT}:
        index = g.op('Cast', index, to_i=_C_onnx.TensorProtoDataType.INT64)
    return g.op('Gather', self, index, axis_i=dim)

@_beartype.beartype
def _slice_helper(g: jit_utils.GraphContext, input, axes, starts, ends, steps=None):
    if False:
        while True:
            i = 10
    if g.opset <= 9:
        from torch.onnx.symbolic_opset9 import _slice as _slice9
        return _slice9(g, input, axes, starts, ends)
    else:
        from torch.onnx.symbolic_opset10 import _slice as _slice10
        return _slice10(g, input, axes, starts, ends, steps)

@_beartype.beartype
def _is_fp(value) -> bool:
    if False:
        i = 10
        return i + 15
    return _type_utils.JitScalarType.from_value(value, _type_utils.JitScalarType.UNDEFINED) in {_type_utils.JitScalarType.FLOAT, _type_utils.JitScalarType.DOUBLE, _type_utils.JitScalarType.HALF, _type_utils.JitScalarType.BFLOAT16}

@_beartype.beartype
def _is_bool(value) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return _type_utils.JitScalarType.from_value(value, _type_utils.JitScalarType.UNDEFINED) in {_type_utils.JitScalarType.BOOL}

@_beartype.beartype
def _generate_wrapped_number(g: jit_utils.GraphContext, scalar):
    if False:
        i = 10
        return i + 15
    'Creates a wrapped number based on https://github.com/pytorch/pytorch/issues/9515.\n\n    A Tensor is a considered a "wrapped number" if it is\n    auto-wrapped from a C++ or Python number type. Integer types are\n    wrapped as 0-dim int64 tensors and floating-point types are\n    wrapped as 0-dim double tensors.\n\n    The input to this function is constant value. If the data type\n    is a floating point type, it is converted to a 0-dim double\n    tensor, else it is converted to a 0-dim tensor of its original type\n    '
    assert not isinstance(scalar, torch.Tensor)
    if isinstance(scalar, float):
        return g.op('Constant', value_t=torch.tensor(scalar, dtype=torch.double))
    return g.op('Constant', value_t=torch.tensor(scalar))

@_beartype.beartype
def _sort_helper(g: jit_utils.GraphContext, input, dim, decending=True, out=None):
    if False:
        i = 10
        return i + 15
    if out is not None:
        _unimplemented('Sort', 'Out parameter is not supported')
    shape_ = g.op('Shape', input)
    dim_size_ = g.op('Gather', shape_, g.op('Constant', value_t=torch.tensor([dim], dtype=torch.int64)))
    if g.opset <= 10:
        if not decending:
            _unimplemented('Sort', 'Ascending is not supported')
        return g.op('TopK', input, dim_size_, axis_i=dim, outputs=2)
    else:
        return g.op('TopK', input, dim_size_, axis_i=dim, largest_i=decending, outputs=2)

@_beartype.beartype
def _topk_helper(g: jit_utils.GraphContext, input, k, dim, largest=True, sorted=False, out=None):
    if False:
        while True:
            i = 10
    if out is not None:
        _unimplemented('TopK', 'Out parameter is not supported')
    if not _is_value(k):
        k = g.op('Constant', value_t=torch.tensor([k], dtype=torch.int64))
    else:
        k = _reshape_helper(g, k, g.op('Constant', value_t=torch.tensor([1])))
        if _try_get_scalar_type(k) != _type_utils.JitScalarType.INT64:
            k = g.op('Cast', k, to_i=_C_onnx.TensorProtoDataType.INT64)
    if g.opset <= 10:
        if not largest:
            _unimplemented('TopK', 'Ascending is not supported')
        return g.op('TopK', input, k, axis_i=dim, outputs=2)
    else:
        return g.op('TopK', input, k, axis_i=dim, largest_i=largest, sorted_i=sorted, outputs=2)

@_beartype.beartype
def _lt_helper(g: jit_utils.GraphContext, input, other):
    if False:
        print('Hello World!')
    if g.opset <= 8:
        from torch.onnx.symbolic_opset8 import lt as _lt8
        return _lt8(g, input, other)
    else:
        from torch.onnx.symbolic_opset9 import lt as _lt9
        return _lt9(g, input, other)

@_beartype.beartype
def _interpolate_warning(interpolate_mode):
    if False:
        while True:
            i = 10
    onnx_op = 'onnx:Resize' if GLOBALS.export_onnx_opset_version >= 10 else 'onnx:Upsample'
    warnings.warn('You are trying to export the model with ' + onnx_op + ' for ONNX opset version ' + str(GLOBALS.export_onnx_opset_version) + ". This operator might cause results to not match the expected results by PyTorch.\nONNX's Upsample/Resize operator did not match Pytorch's Interpolation until opset 11. Attributes to determine how to transform the input were added in onnx:Resize in opset 11 to support Pytorch's behavior (like coordinate_transformation_mode and nearest_mode).\nWe recommend using opset 11 and above for models using this operator.")

@_beartype.beartype
def _unsqueeze_helper(g: jit_utils.GraphContext, input, axes_i):
    if False:
        return 10
    if _is_constant(axes_i[0]):
        if g.opset >= 13:
            axes = g.op('Constant', value_t=torch.tensor(axes_i, dtype=torch.long))
            return g.op('Unsqueeze', input, axes)
        return g.op('Unsqueeze', input, axes_i=axes_i)
    if g.opset < 13:
        raise errors.SymbolicValueError('Opset version must be >= 13 for Unsqueeze with dynamic axes.', input)
    return g.op('Unsqueeze', input, axes_i[0])

@_beartype.beartype
def _squeeze_helper(g: jit_utils.GraphContext, input, axes_i):
    if False:
        while True:
            i = 10
    if _is_constant(axes_i[0]):
        if g.opset >= 13:
            axes = g.op('Constant', value_t=torch.tensor(axes_i, dtype=torch.long))
            return g.op('Squeeze', input, axes)
        return g.op('Squeeze', input, axes_i=axes_i)
    if g.opset < 13:
        raise errors.SymbolicValueError('Opset version must be >= 13 for Squeeze with dynamic axes.', input)
    axes_t = axes_i[0]
    axes_rank = _get_tensor_rank(axes_t)
    assert axes_rank is not None
    if axes_rank > 1:
        raise errors.SymbolicValueError('For Squeeze axses as input, the axes rank must be one in ONNX spec.', input)
    elif axes_rank == 0:
        axes_t = _unsqueeze_helper(g, axes_t, [0])
        return g.op('Squeeze', input, axes_t)
    return g.op('Squeeze', input, axes_t)

@_beartype.beartype
def _reducesum_helper(g: jit_utils.GraphContext, input, axes_i=None, keepdims_i=1, noop_with_empty_axes_i=0):
    if False:
        while True:
            i = 10
    keepdims_i = _maybe_get_const(keepdims_i, 'i')
    if g.opset >= 13:
        if axes_i:
            if not _is_value(axes_i):
                axes_i = g.op('Constant', value_t=torch.tensor(axes_i, dtype=torch.long))
            return g.op('ReduceSum', input, axes_i, keepdims_i=keepdims_i, noop_with_empty_axes_i=noop_with_empty_axes_i)
        return g.op('ReduceSum', input, keepdims_i=keepdims_i, noop_with_empty_axes_i=noop_with_empty_axes_i)
    else:
        return g.op('ReduceSum', input, axes_i=axes_i, keepdims_i=keepdims_i)

@_beartype.beartype
def _interpolate_size_to_scales(g: jit_utils.GraphContext, input, output_size, dim):
    if False:
        print('Hello World!')
    output_size = _maybe_get_const(output_size, 'is')
    if _is_value(output_size):
        offset = 2
        offsets = g.op('Constant', value_t=torch.ones(offset, dtype=torch.float32))
        dividend = g.op('Cast', output_size, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        divisor = _slice_helper(g, g.op('Shape', input), axes=[0], ends=[sys.maxsize], starts=[offset])
        divisor = g.op('Cast', divisor, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        scale_dims = g.op('Div', dividend, divisor)
        scales = g.op('Concat', offsets, scale_dims, axis_i=0)
    else:
        scales_constant = [1.0 if i < 2 else float(output_size[-(dim - i)]) / float(input.type().sizes()[-(dim - i)]) for i in range(0, dim)]
        scales = g.op('Constant', value_t=torch.tensor(scales_constant, dtype=torch.float32))
    return scales

@_beartype.beartype
def _interpolate_get_scales_if_available(g: jit_utils.GraphContext, scales):
    if False:
        return 10
    available_scales = _maybe_get_const(scales[0], 'fs') != -1 and (not _is_none(scales[0]))
    if not available_scales:
        return None
    offsets = g.op('Constant', value_t=torch.ones(2, dtype=torch.float32))
    scales_list = g.op('Constant', value_t=torch.tensor(_maybe_get_const(scales[0], 'fs')))
    scales = g.op('Concat', offsets, scales_list, axis_i=0)
    return scales

@_beartype.beartype
def _get_interpolate_attributes(g: jit_utils.GraphContext, mode, args):
    if False:
        for i in range(10):
            print('nop')
    if mode == 'nearest':
        align_corners = None
        scales = args[0:]
    else:
        align_corners = args[0]
        scales = args[1:]
    scales = _interpolate_get_scales_if_available(g, scales)
    return (scales, align_corners)

@_beartype.beartype
def _interpolate_get_scales(g: jit_utils.GraphContext, scale_factor, dim):
    if False:
        i = 10
        return i + 15
    offsets = g.op('Constant', value_t=torch.ones(2, dtype=torch.float32))
    scale_factor_rank = _get_tensor_rank(scale_factor)
    if isinstance(scale_factor.type(), _C.ListType) or (scale_factor_rank is not None and scale_factor_rank > 0):
        return g.op('Concat', offsets, scale_factor, axis_i=0)
    else:
        scale_factor = _unsqueeze_helper(g, scale_factor, [0])
        scale_factor = g.op('Cast', scale_factor, to_i=_C_onnx.TensorProtoDataType.FLOAT)
        scales = [scale_factor for i in range(dim - 2)]
    scale_factor = g.op('Concat', offsets, *scales, axis_i=0)
    return scale_factor

@_beartype.beartype
def _interpolate_get_scales_and_mode(g: jit_utils.GraphContext, input, size, scale_factor, mode, align_corners):
    if False:
        i = 10
        return i + 15
    mode = _maybe_get_const(mode, 's')
    if 'linear' in mode:
        mode = 'linear'
    if 'cubic' in mode:
        mode = 'cubic'
    _interpolate_warning(mode)
    align_corners = _maybe_get_const(align_corners, 'b')
    if isinstance(align_corners, bool) and align_corners:
        return _unimplemented('interpolate', 'align_corners == True')
    if not input.type().dim():
        return _unimplemented('interpolate', 'missing input shape')
    dim = input.type().dim()
    if not _is_none(scale_factor):
        scale_factor = _interpolate_get_scales(g, scale_factor, dim)
    elif not _is_none(size):
        if not _is_packed_list(size):
            is_scalar = _maybe_get_const(size, 't').dim() == 0
            if is_scalar:
                size = _unsqueeze_helper(g, size, [0])
                size = [size for i in range(dim - 2)]
                size = g.op('Concat', *size, axis_i=0)
        scale_factor = _interpolate_size_to_scales(g, input, size, dim)
    else:
        return _unimplemented('interpolate', 'Both size and scales are None in __interpolate')
    return (scale_factor, mode)

@_beartype.beartype
def _argmin_argmax_helper(g: jit_utils.GraphContext, input: torch._C.Value, dim: torch._C.Value, keepdim: bool, op_name: str):
    if False:
        while True:
            i = 10

    def op_wrapper(input, axis_i, keepdims_i):
        if False:
            for i in range(10):
                print('nop')
        if g.opset >= 12:
            return g.op(op_name, input, axis_i=axis_i, keepdims_i=keepdims_i, select_last_index_i=False)
        return g.op(op_name, input, axis_i=axis_i, keepdims_i=keepdims_i)
    if _is_none(dim):
        flattened = _reshape_helper(g, input, g.op('Constant', value_t=torch.tensor([-1])))
        output = op_wrapper(flattened, axis_i=0, keepdims_i=False)
        if keepdim:
            input_shape = g.op('Shape', input)
            input_shape_shape = g.op('Shape', input_shape)
            new_shape = g.op('ConstantOfShape', input_shape_shape, value_t=torch.tensor([1], dtype=torch.int64))
            output = g.op('Reshape', output, new_shape)
        return output
    dim = _parse_arg(dim, 'i')
    return op_wrapper(input, axis_i=dim, keepdims_i=keepdim)

@_beartype.beartype
def _interpolate_helper(name, dim, interpolate_mode):
    if False:
        print('Hello World!')

    @quantized_args(True, False, False)
    def symbolic_fn(g, input, output_size, *args):
        if False:
            for i in range(10):
                print('nop')
        (scales, align_corners) = _get_interpolate_attributes(g, interpolate_mode, args)
        align_corners = _maybe_get_scalar(align_corners)
        coordinate_transformation_mode = 'asymmetric' if interpolate_mode == 'nearest' else 'align_corners' if align_corners else 'half_pixel'
        if scales is None:
            input_size = g.op('Shape', input)
            input_size_beg = _slice_helper(g, input_size, axes=[0], ends=[2], starts=[0])
            output_size = g.op('Cast', output_size, to_i=_C_onnx.TensorProtoDataType.INT64)
            output_size = g.op('Concat', input_size_beg, output_size, axis_i=0)
            if g.opset >= 13:
                empty_roi = _optional_input_placeholder_tensor(g)
                empty_scales = _optional_input_placeholder_tensor(g)
            else:
                empty_roi = g.op('Constant', value_t=torch.tensor([], dtype=torch.float32))
                empty_scales = g.op('Constant', value_t=torch.tensor([], dtype=torch.float32))
            return g.op('Resize', input, empty_roi, empty_scales, output_size, coordinate_transformation_mode_s=coordinate_transformation_mode, cubic_coeff_a_f=-0.75, mode_s=interpolate_mode, nearest_mode_s='floor')
        else:
            if g.opset >= 13:
                empty_roi = _optional_input_placeholder_tensor(g)
            else:
                empty_roi = g.op('Constant', value_t=torch.tensor([], dtype=torch.float32))
            return g.op('Resize', input, empty_roi, scales, coordinate_transformation_mode_s=coordinate_transformation_mode, cubic_coeff_a_f=-0.75, mode_s=interpolate_mode, nearest_mode_s='floor')
    return symbolic_fn

@_beartype.beartype
def __interpolate_helper(g: jit_utils.GraphContext, input, size, scale_factor, mode, align_corners, recompute_scale_factor):
    if False:
        return 10
    mode = _maybe_get_const(mode, 's')
    if 'linear' in mode:
        mode = 'linear'
    if 'cubic' in mode:
        mode = 'cubic'
    align_corners = _maybe_get_const(align_corners, 'b')
    align_corners = False if not isinstance(align_corners, bool) else align_corners
    coordinate_transformation_mode = 'asymmetric' if mode == 'nearest' else 'align_corners' if align_corners else 'half_pixel'
    if not _is_none(size):
        input_size = g.op('Shape', input)
        input_size = _slice_helper(g, input_size, axes=[0], ends=[2], starts=[0])
        try:
            is_scalar = not _is_packed_list(size) and _maybe_get_const(size, 't').dim() == 0
        except AttributeError:
            is_scalar = not _is_packed_list(size)
            if not is_scalar:
                warnings.warn('Cannot verify if the output_size is a scalar while exporting interpolate. Assuming that it is not a scalar.')
        if is_scalar:
            rank = _get_tensor_rank(input)
            if rank is None:
                return _unimplemented('interpolate (with a scalar output_size)', 'missing input shape (try giving an array of output_size values)')
            size = _unsqueeze_helper(g, size, [0])
            size = [size for i in range(rank - 2)]
            size = g.op('Concat', *size, axis_i=0)
        size = g.op('Cast', size, to_i=_C_onnx.TensorProtoDataType.INT64)
        size = g.op('Concat', input_size, size, axis_i=0)
        if g.opset >= 13:
            empty_roi = _optional_input_placeholder_tensor(g)
            empty_scales = _optional_input_placeholder_tensor(g)
        else:
            empty_roi = g.op('Constant', value_t=torch.tensor([], dtype=torch.float32))
            empty_scales = g.op('Constant', value_t=torch.tensor([], dtype=torch.float32))
        return g.op('Resize', input, empty_roi, empty_scales, size, coordinate_transformation_mode_s=coordinate_transformation_mode, cubic_coeff_a_f=-0.75, mode_s=mode, nearest_mode_s='floor')
    else:
        rank = _get_tensor_rank(input)
        if rank is None:
            return _unimplemented('interpolate (with scales)', 'missing input shape')
        if g.opset >= 13:
            empty_roi = _optional_input_placeholder_tensor(g)
        else:
            empty_roi = g.op('Constant', value_t=torch.tensor([], dtype=torch.float32))
        scales = _interpolate_get_scales(g, scale_factor, rank)
        return g.op('Resize', input, empty_roi, scales, coordinate_transformation_mode_s=coordinate_transformation_mode, cubic_coeff_a_f=-0.75, mode_s=mode, nearest_mode_s='floor')

@_beartype.beartype
def _unbind_helper(g: jit_utils.GraphContext, self, dim, _outputs):
    if False:
        return 10
    if g.opset < 11:
        from torch.onnx.symbolic_opset9 import unbind
    elif g.opset <= 12:
        from torch.onnx.symbolic_opset11 import unbind
    else:
        from torch.onnx.symbolic_opset13 import unbind
    return unbind(g, self, dim, _outputs)

@_beartype.beartype
def _scatter_helper(g: jit_utils.GraphContext, self, dim, index, src):
    if False:
        print('Hello World!')
    if g.opset <= 10:
        from torch.onnx.symbolic_opset9 import scatter
    else:
        from torch.onnx.symbolic_opset11 import scatter
    return scatter(g, self, dim, index, src)

@_beartype.beartype
def _repeat_interleave_split_helper(g: jit_utils.GraphContext, self, reps, dim):
    if False:
        print('Hello World!')
    if g.opset <= 12:
        split_out = g.op('Split', self, split_i=[1] * reps, axis_i=dim, outputs=reps)
    else:
        from torch.onnx.symbolic_opset13 import split
        repeats = g.op('Constant', value_t=torch.tensor([1] * reps))
        split_out = split(g, self, repeats, dim, _outputs=reps)
    return split_out if reps > 1 else [split_out]

@_beartype.beartype
def _repeat_interleave_single_value_repeat_helper(g: jit_utils.GraphContext, self, repeats, dim):
    if False:
        print('Hello World!')
    from torch.onnx.symbolic_opset9 import flatten, unsqueeze
    if not _is_tensor(repeats):
        repeats = g.op('Constant', value_t=torch.LongTensor(repeats))
    const_repeats: bool = _is_constant(repeats)
    reps = _maybe_get_const(repeats, 't')
    if _get_tensor_rank(repeats) == 0:
        repeats = g.op('Reshape', repeats, g.op('Constant', value_t=torch.tensor([1])))
    unsqueezed = unsqueeze(g, self, dim + 1)
    if const_repeats:
        onehot = torch.ones(_get_tensor_rank(unsqueezed), dtype=torch.int64)
        onehot[dim + 1] = reps
        repeats_per_dim = g.op('Constant', value_t=onehot)
    else:
        onehot = g.op('OneHot', unsqueeze(g, dim + 1, 0), g.op('Constant', value_t=torch.tensor(_get_tensor_rank(unsqueezed))), g.op('Concat', g.op('Constant', value_t=torch.tensor([1])), repeats, axis_i=0))
        repeats_per_dim = flatten(g, onehot, 0, 1)
    tiled = g.op('Tile', unsqueezed, repeats_per_dim)
    return flatten(g, tiled, dim, dim + 1)

@_beartype.beartype
def _arange_cast_helper(g: jit_utils.GraphContext, end, start=None, step=None, dtype=None) -> Tuple[_type_utils.JitScalarType, Optional[_C.Value], Optional[_C.Value], Optional[_C.Value]]:
    if False:
        for i in range(10):
            print('nop')

    def _is_all_integral(scalars):
        if False:
            i = 10
            return i + 15
        for scalar in scalars:
            scalar_type = _type_utils.JitScalarType.from_value(scalar, _type_utils.JitScalarType.UNDEFINED)
            if scalar_type != _type_utils.JitScalarType.INT64 and scalar_type != _type_utils.JitScalarType.UNDEFINED:
                return False
        return True
    if dtype is None or (_is_value(dtype) and _is_none(dtype)):
        if _is_all_integral([start, end, step]):
            scalar_type = _type_utils.JitScalarType.INT64
        else:
            scalar_type = _type_utils.JitScalarType.from_dtype(torch.get_default_dtype())
    else:
        assert isinstance(dtype, int)
        scalar_type = _type_utils.JitScalarType(dtype)
    start = g.op('Cast', start, to_i=scalar_type.onnx_type()) if start else None
    end = g.op('Cast', end, to_i=scalar_type.onnx_type()) if end else None
    step = g.op('Cast', step, to_i=scalar_type.onnx_type()) if step else None
    return (scalar_type, end, start, step)

@_beartype.beartype
def _arange_helper(g: jit_utils.GraphContext, *args):
    if False:
        return 10
    if g.opset <= 10:
        from torch.onnx.symbolic_opset9 import arange
    else:
        from torch.onnx.symbolic_opset11 import arange
    return arange(g, *args)

@_beartype.beartype
def _size_helper(g: jit_utils.GraphContext, self, dim):
    if False:
        i = 10
        return i + 15
    full_shape = g.op('Shape', self)
    from torch.onnx.symbolic_opset9 import select
    return select(g, full_shape, g.op('Constant', value_t=torch.tensor([0])), dim)

@_beartype.beartype
def _index_fill_reshape_helper(g: jit_utils.GraphContext, self, dim, index):
    if False:
        return 10
    from torch.onnx.symbolic_opset9 import expand
    if g.opset <= 10:
        from torch.onnx.symbolic_opset9 import scatter
    else:
        from torch.onnx.symbolic_opset11 import scatter
    if self.type().dim() is None:
        return _unimplemented('index_fill', 'input rank not accessible')
    self_dim = self.type().dim()
    dim_value = _parse_arg(dim, 'i')
    unsqueezed_index = _unsqueeze_helper(g, index, [i for i in range(self_dim) if i != dim_value])
    expanded_index_shape = scatter(g, g.op('Shape', self), 0, _unsqueeze_helper(g, dim, [0]), g.op('Shape', index))
    expanded_index = expand(g, unsqueezed_index, expanded_index_shape, None)
    return (expanded_index_shape, expanded_index)

@_beartype.beartype
def _reshape_helper(g: jit_utils.GraphContext, input, shape, allowzero=0):
    if False:
        i = 10
        return i + 15
    shape = _maybe_get_const(shape, 'is')
    if not _is_value(shape):
        shape = g.op('Constant', value_t=torch.LongTensor(shape))
    if g.opset <= 13:
        if allowzero == 1:
            _onnx_opset_unsupported('Reshape with allowzero=1', GLOBALS.export_onnx_opset_version, 14, input)
        return g.op('Reshape', input, shape)
    else:
        return g.op('Reshape', input, shape, allowzero_i=allowzero)

@_beartype.beartype
def _batchnorm_helper(g: jit_utils.GraphContext, input, weight, bias, running_mean, running_var):
    if False:
        i = 10
        return i + 15
    from torch.onnx.symbolic_opset9 import _var_mean
    batch_size = _get_tensor_dim_size(input, 0)
    channel_size = _get_tensor_dim_size(input, 1)
    if weight is None or _is_none(weight):
        if channel_size is None:
            raise errors.SymbolicValueError('Unsupported: ONNX export of batch_norm for unknown channel size.', input)
        weight_value = torch.tensor([1.0] * channel_size, dtype=_type_utils.JitScalarType.from_value(input).dtype())
        weight = g.op('Constant', value_t=weight_value)
    if bias is None or _is_none(bias):
        if channel_size is None:
            raise errors.SymbolicValueError('Unsupported: ONNX export of batch_norm for unknown channel size.', input)
        bias_value = torch.tensor([0.0] * channel_size, dtype=_type_utils.JitScalarType.from_value(input).dtype())
        bias = g.op('Constant', value_t=bias_value)
    if running_mean is None or _is_none(running_mean) or running_var is None or _is_none(running_var):
        assert batch_size is not None and channel_size is not None
        reshape_in = _reshape_helper(g, input, g.op('Constant', value_t=torch.tensor([batch_size, channel_size, -1], dtype=torch.int64)))
        trans_in = g.op('Transpose', reshape_in, perm_i=[0, 2, 1])
        (running_var, running_mean) = _var_mean(g, trans_in, g.op('Constant', value_t=torch.tensor([0, 1], dtype=torch.int64)), False, False)
    return (weight, bias, running_mean, running_var)

@_beartype.beartype
def _avgpool_helper(tuple_fn: Callable[[Any], Sequence[int]], padding: Union[int, Sequence[int]], kernel_size, stride, divisor_override, name) -> Tuple[int, ...]:
    if False:
        i = 10
        return i + 15
    if divisor_override and divisor_override.node().kind() != 'prim::Constant':
        _unimplemented(name, 'divisor_override')
    return tuple(tuple_fn(padding))

@_beartype.beartype
def check_training_mode(op_train_mode: int, op_name: str) -> None:
    if False:
        print('Hello World!')
    "Warns the user if the model's training mode and the export mode do not agree."
    if GLOBALS.training_mode == _C_onnx.TrainingMode.PRESERVE:
        return
    if op_train_mode:
        op_mode_enum = _C_onnx.TrainingMode.TRAINING
    else:
        op_mode_enum = _C_onnx.TrainingMode.EVAL
    if op_mode_enum == GLOBALS.training_mode:
        return
    op_mode_text = f'train={bool(op_train_mode)}'
    warnings.warn(f"ONNX export mode is set to {GLOBALS.training_mode}, but operator '{op_name}' is set to {op_mode_text}. Exporting with {op_mode_text}.")

@_beartype.beartype
def _flatten_helper(g: jit_utils.GraphContext, input, start_dim, end_dim, dim):
    if False:
        i = 10
        return i + 15
    input_size = g.op('Shape', input)
    slice1 = _slice_helper(g, input_size, axes=[0], starts=[0], ends=[start_dim])
    slices = [slice1, g.op('Constant', value_t=torch.tensor([-1], dtype=torch.long))]
    if end_dim < dim - 1:
        slice3 = _slice_helper(g, input_size, axes=[0], starts=[end_dim + 1], ends=[dim])
        slices = [slice1, g.op('Constant', value_t=torch.tensor([-1], dtype=torch.long)), slice3]
    final_shape = g.op('Concat', *slices, axis_i=0)
    from torch.onnx.symbolic_opset9 import _reshape_from_tensor
    return _reshape_from_tensor(g, input, final_shape)

@_beartype.beartype
def _is_split_static(split_size_or_sizes, _outputs):
    if False:
        i = 10
        return i + 15
    if _outputs is None:
        return False
    if _is_value(split_size_or_sizes) and split_size_or_sizes.node().kind() != 'onnx::Constant':
        return False
    return True

@_beartype.beartype
def _optional_input_placeholder_tensor(g):
    if False:
        i = 10
        return i + 15
    n = g.op('prim::Constant')
    n.setType(_C.OptionalType.ofTensor())
    return n

@_beartype.beartype
def _handle_reduce_dim_none(g: jit_utils.GraphContext, self, op_name):
    if False:
        for i in range(10):
            print('nop')
    rank = _get_tensor_rank(self)
    if rank is not None and any((_get_tensor_dim_size(self, i) == 0 for i in range(rank))):
        return g.op(op_name, self, keepdims_i=1)
    return g.op(op_name, self, keepdims_i=0)

@_beartype.beartype
def dequantize_helper(g: jit_utils.GraphContext, qtensor: _C.Value, qdtype: Optional[_C_onnx.TensorProtoDataType]=None) -> Tuple[_C.Value, _C.Value, _C.Value, Optional[_C.Value]]:
    if False:
        print('Hello World!')
    'Appends to graph `g` ONNX nodes that dequantizes `qtensor` into `tensor`.\n\n    Args:\n        g: Graph, the ONNX IR graph that is under construction.\n        qtensor: torch._C.Value, either a tuple of (quantized_tensor, scale, zero_point)\n            for per tensor quantization, or\n            (quantized_tensor, scale, zero_point, axis) for per channel quantization,\n            representing the quantized tensor.\n        qdtype: torch.onnx.TensorProtoDataType default None, if not None, represents the\n            data type of quantized tensor. It must be either\n            torch.onnx.TensorProtoDataType.UINT8 or torch.onnx.TensorProtoDataType.INT8.\n    '
    unpacked_qtensors = _unpack_quantized_tensor(qtensor)
    (tensor, scale, zero_point) = unpacked_qtensors[:3]
    axis = unpacked_qtensors[3] if len(unpacked_qtensors) >= 4 else None
    axis_i = _get_const(axis, 'i', 'axis')
    input_qdtype = _type_utils.JitScalarType.from_value(tensor)
    if qdtype is None:
        if input_qdtype is not None:
            qdtype = input_qdtype.onnx_type()
        else:
            qdtype = _C_onnx.TensorProtoDataType.UINT8
    value = g.op('Cast', tensor, to_i=qdtype)
    scale = g.op('Cast', scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    zero_point = g.op('Cast', zero_point, to_i=qdtype)
    if axis_i is not None and GLOBALS.export_onnx_opset_version < 13:
        _onnx_opset_unsupported_detailed('DequantizeLinear', GLOBALS.export_onnx_opset_version, 13, 'Attribute axis is not supported.', qtensor)
    return (g.op('DequantizeLinear', value, scale, zero_point, axis_i=axis_i), scale, zero_point, axis)

@_beartype.beartype
def quantize_helper(g: jit_utils.GraphContext, tensor: _C.Value, scale: _C.Value, zero_point: _C.Value, axis: Optional[_C.Value]=None) -> _C.Value:
    if False:
        i = 10
        return i + 15
    'Appends to graph `g` ONNX nodes that quantizes `tensor` based on `scale`, `zero_point` and `axis`.\n\n    Args:\n        g: Graph, the ONNX IR graph that is under construction.\n        tensor: torch._C.Value, representing the tensor to be quantized.\n        scale: torch._C.Value, quantized scale.\n        zero_point: torch._C.Value, quantized zero point.\n        axis: Optional[torch._C.Value] default None, if None, represents per tensor quantization.\n            Otherwise, represents per channel quantization, along given axis.\n\n    Returns:\n        A TupleConstruct storing information of the quantized tensor.\n    '
    if axis is not None and (not _is_none(axis)) and (GLOBALS.export_onnx_opset_version < 13):
        _onnx_opset_unsupported_detailed('QuantizeLinear', GLOBALS.export_onnx_opset_version, 13, 'Attribute axis is not supported.', tensor)
    assert scale is not None
    if _type_utils.JitScalarType.from_value(scale, _type_utils.JitScalarType.UNDEFINED) != _type_utils.JitScalarType.FLOAT:
        scale = g.op('Cast', scale, to_i=_C_onnx.TensorProtoDataType.FLOAT)
    assert zero_point is not None
    if _type_utils.JitScalarType.from_value(zero_point, _type_utils.JitScalarType.UNDEFINED) not in {_type_utils.JitScalarType.UINT8, _type_utils.JitScalarType.INT8}:
        zero_point = g.op('Cast', zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
    output = g.op('QuantizeLinear', tensor, scale, zero_point, axis_i=_get_const(axis, 'i', 'axis'))
    args = [output, scale, zero_point]
    if axis is not None and (not _is_none(axis)):
        args.append(axis)
    return g.op('prim::TupleConstruct', *args)

@_beartype.beartype
def requantize_bias_helper(g: jit_utils.GraphContext, bias, input_scale, weight_scale, axis=None):
    if False:
        print('Hello World!')
    'In PyTorch, bias is float and is quantized to int32 implicitly inside the quantized ATen op kernel.\n    In ONNX we need to make the quantization explicit because operators expect all of their inputs to be quantized.\n    Since int32 is not a supported output type by ONNX operator `QuantizeLinear`, quantization is exported using\n    regular operators.\n    '
    bias_scale = g.op('Mul', weight_scale, input_scale)
    bias_scale_shape = g.op('Shape', bias_scale)
    bias_zero_point = g.op('ConstantOfShape', bias_scale_shape, value_t=torch.tensor([0], dtype=torch.int))
    q_bias = g.op('Cast', g.op('Div', bias, bias_scale), to_i=_C_onnx.TensorProtoDataType.INT32)
    axis_args = []
    if axis is not None and (not _is_none(axis)):
        axis_args.append(axis)
    return g.op('prim::TupleConstruct', q_bias, bias_scale, bias_zero_point, *axis_args)

@_beartype.beartype
def args_have_same_dtype(args):
    if False:
        for i in range(10):
            print('nop')
    assert args
    base_dtype = _type_utils.JitScalarType.from_value(args[0])
    has_same_dtype = all((_type_utils.JitScalarType.from_value(elem) == base_dtype for elem in args))
    return has_same_dtype
cast_pytorch_to_onnx = {'Byte': _C_onnx.TensorProtoDataType.UINT8, 'Char': _C_onnx.TensorProtoDataType.INT8, 'Double': _C_onnx.TensorProtoDataType.DOUBLE, 'Float': _C_onnx.TensorProtoDataType.FLOAT, 'Half': _C_onnx.TensorProtoDataType.FLOAT16, 'Int': _C_onnx.TensorProtoDataType.INT32, 'Long': _C_onnx.TensorProtoDataType.INT64, 'Short': _C_onnx.TensorProtoDataType.INT16, 'Bool': _C_onnx.TensorProtoDataType.BOOL, 'ComplexFloat': _C_onnx.TensorProtoDataType.COMPLEX64, 'ComplexDouble': _C_onnx.TensorProtoDataType.COMPLEX128, 'BFloat16': _C_onnx.TensorProtoDataType.BFLOAT16, 'Undefined': _C_onnx.TensorProtoDataType.UNDEFINED}
scalar_name_to_pytorch = {'uint8_t': 'Byte', 'int8_t': 'Char', 'double': 'Double', 'float': 'Float', 'half': 'Half', 'int': 'Int', 'int64_t': 'Long', 'int16_t': 'Short', 'bool': 'Bool', 'complex64': 'ComplexFloat', 'complex128': 'ComplexDouble', 'qint8': 'QInt8', 'quint8': 'QUInt8', 'qint32': 'QInt32', 'bfloat16': 'BFloat16'}
scalar_type_to_pytorch_type = [torch.uint8, torch.int8, torch.short, torch.int, torch.int64, torch.half, torch.float, torch.double, torch.complex32, torch.complex64, torch.complex128, torch.bool, torch.qint8, torch.quint8, torch.qint32, torch.bfloat16]
pytorch_name_to_type = {'Byte': torch.uint8, 'Char': torch.int8, 'Double': torch.double, 'Float': torch.float, 'Half': torch.half, 'Int': torch.int, 'Long': torch.int64, 'Short': torch.short, 'Bool': torch.bool, 'ComplexFloat': torch.complex64, 'ComplexDouble': torch.complex128, 'QInt8': torch.qint8, 'QUInt8': torch.quint8, 'QInt32': torch.qint32, 'BFloat16': torch.bfloat16}
scalar_type_to_onnx = [cast_pytorch_to_onnx['Byte'], cast_pytorch_to_onnx['Char'], cast_pytorch_to_onnx['Short'], cast_pytorch_to_onnx['Int'], cast_pytorch_to_onnx['Long'], cast_pytorch_to_onnx['Half'], cast_pytorch_to_onnx['Float'], cast_pytorch_to_onnx['Double'], cast_pytorch_to_onnx['Undefined'], cast_pytorch_to_onnx['ComplexFloat'], cast_pytorch_to_onnx['ComplexDouble'], cast_pytorch_to_onnx['Bool'], cast_pytorch_to_onnx['Char'], cast_pytorch_to_onnx['Byte'], cast_pytorch_to_onnx['Int'], cast_pytorch_to_onnx['BFloat16']]
_quantized_ops: Set[int] = set()