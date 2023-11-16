"""Utilities to create TensorProtos."""
import typing
from typing import Protocol
import numpy as np
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import tensor_shape
from tensorflow.python.types import core
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
try:
    from tensorflow.python.framework import fast_tensor_util
    _FAST_TENSOR_UTIL_AVAILABLE = True
except ImportError:
    _FAST_TENSOR_UTIL_AVAILABLE = False

def ExtractBitsFromFloat16(x):
    if False:
        while True:
            i = 10
    return np.asarray(x, dtype=np.float16).view(np.uint16).item()

def SlowAppendFloat16ArrayToTensorProto(tensor_proto, proto_values):
    if False:
        print('Hello World!')
    tensor_proto.half_val.extend([ExtractBitsFromFloat16(x) for x in proto_values])

def _MediumAppendFloat16ArrayToTensorProto(tensor_proto, proto_values):
    if False:
        for i in range(10):
            print('nop')
    fast_tensor_util.AppendFloat16ArrayToTensorProto(tensor_proto, np.asarray(proto_values, dtype=np.float16).view(np.uint16))

def ExtractBitsFromBFloat16(x):
    if False:
        for i in range(10):
            print('nop')
    return np.asarray(x, dtype=dtypes.bfloat16.as_numpy_dtype).view(np.uint16).item()

def SlowAppendBFloat16ArrayToTensorProto(tensor_proto, proto_values):
    if False:
        for i in range(10):
            print('nop')
    tensor_proto.half_val.extend([ExtractBitsFromBFloat16(x) for x in proto_values])

def FastAppendBFloat16ArrayToTensorProto(tensor_proto, proto_values):
    if False:
        return 10
    fast_tensor_util.AppendBFloat16ArrayToTensorProto(tensor_proto, np.asarray(proto_values, dtype=dtypes.bfloat16.as_numpy_dtype).view(np.uint16))

def SlowAppendFloat8e5m2ArrayToTensorProto(tensor_proto, proto_values):
    if False:
        print('Hello World!')
    tensor_proto.float8_val += np.asarray(proto_values, dtype=dtypes.float8_e5m2.as_numpy_dtype).view(np.uint8).tobytes()

def FastAppendFloat8e5m2ArrayToTensorProto(tensor_proto, proto_values):
    if False:
        return 10
    fast_tensor_util.AppendFloat8ArrayToTensorProto(tensor_proto, np.asarray(proto_values, dtype=dtypes.float8_e5m2.as_numpy_dtype).view(np.uint8))

def SlowAppendFloat8e4m3fnArrayToTensorProto(tensor_proto, proto_values):
    if False:
        for i in range(10):
            print('nop')
    tensor_proto.float8_val += np.asarray(proto_values, dtype=dtypes.float8_e4m3fn.as_numpy_dtype).view(np.uint8).tobytes()

def FastAppendFloat8e4m3fnArrayToTensorProto(tensor_proto, proto_values):
    if False:
        for i in range(10):
            print('nop')
    fast_tensor_util.AppendFloat8ArrayToTensorProto(tensor_proto, np.asarray(proto_values, dtype=dtypes.float8_e4m3fn.as_numpy_dtype).view(np.uint8))

def SlowAppendInt4ArrayToTensorProto(tensor_proto, proto_values):
    if False:
        i = 10
        return i + 15
    x = np.asarray(proto_values, dtype=dtypes.int4.as_numpy_dtype).astype(np.int8)
    tensor_proto.int_val.extend(x.tolist())

def SlowAppendUInt4ArrayToTensorProto(tensor_proto, proto_values):
    if False:
        print('Hello World!')
    x = np.asarray(proto_values, dtype=dtypes.uint4.as_numpy_dtype).astype(np.int8)
    tensor_proto.int_val.extend(x.tolist())
if _FAST_TENSOR_UTIL_AVAILABLE:
    _NP_TO_APPEND_FN = {np.float16: _MediumAppendFloat16ArrayToTensorProto, np.float32: fast_tensor_util.AppendFloat32ArrayToTensorProto, np.float64: fast_tensor_util.AppendFloat64ArrayToTensorProto, np.int32: fast_tensor_util.AppendInt32ArrayToTensorProto, np.int64: fast_tensor_util.AppendInt64ArrayToTensorProto, np.uint8: fast_tensor_util.AppendUInt8ArrayToTensorProto, np.uint16: fast_tensor_util.AppendUInt16ArrayToTensorProto, np.uint32: fast_tensor_util.AppendUInt32ArrayToTensorProto, np.uint64: fast_tensor_util.AppendUInt64ArrayToTensorProto, np.int8: fast_tensor_util.AppendInt8ArrayToTensorProto, np.int16: fast_tensor_util.AppendInt16ArrayToTensorProto, np.complex64: fast_tensor_util.AppendComplex64ArrayToTensorProto, np.complex128: fast_tensor_util.AppendComplex128ArrayToTensorProto, np.object_: fast_tensor_util.AppendObjectArrayToTensorProto, np.bool_: fast_tensor_util.AppendBoolArrayToTensorProto, dtypes.qint8.as_numpy_dtype: fast_tensor_util.AppendInt8ArrayToTensorProto, dtypes.quint8.as_numpy_dtype: fast_tensor_util.AppendUInt8ArrayToTensorProto, dtypes.qint16.as_numpy_dtype: fast_tensor_util.AppendInt16ArrayToTensorProto, dtypes.quint16.as_numpy_dtype: fast_tensor_util.AppendUInt16ArrayToTensorProto, dtypes.qint32.as_numpy_dtype: fast_tensor_util.AppendInt32ArrayToTensorProto, dtypes.bfloat16.as_numpy_dtype: FastAppendBFloat16ArrayToTensorProto, dtypes.float8_e5m2.as_numpy_dtype: FastAppendFloat8e5m2ArrayToTensorProto, dtypes.float8_e4m3fn.as_numpy_dtype: FastAppendFloat8e4m3fnArrayToTensorProto, dtypes.int4.as_numpy_dtype: SlowAppendInt4ArrayToTensorProto, dtypes.uint4.as_numpy_dtype: SlowAppendUInt4ArrayToTensorProto}
else:

    def SlowAppendFloat32ArrayToTensorProto(tensor_proto, proto_values):
        if False:
            i = 10
            return i + 15
        tensor_proto.float_val.extend([x.item() for x in proto_values])

    def SlowAppendFloat64ArrayToTensorProto(tensor_proto, proto_values):
        if False:
            for i in range(10):
                print('nop')
        tensor_proto.double_val.extend([x.item() for x in proto_values])

    def SlowAppendIntArrayToTensorProto(tensor_proto, proto_values):
        if False:
            print('Hello World!')
        tensor_proto.int_val.extend([x.item() for x in proto_values])

    def SlowAppendInt64ArrayToTensorProto(tensor_proto, proto_values):
        if False:
            i = 10
            return i + 15
        tensor_proto.int64_val.extend([x.item() for x in proto_values])

    def SlowAppendQIntArrayToTensorProto(tensor_proto, proto_values):
        if False:
            while True:
                i = 10
        tensor_proto.int_val.extend([x.item()[0] for x in proto_values])

    def SlowAppendUInt32ArrayToTensorProto(tensor_proto, proto_values):
        if False:
            while True:
                i = 10
        tensor_proto.uint32_val.extend([x.item() for x in proto_values])

    def SlowAppendUInt64ArrayToTensorProto(tensor_proto, proto_values):
        if False:
            return 10
        tensor_proto.uint64_val.extend([x.item() for x in proto_values])

    def SlowAppendComplex64ArrayToTensorProto(tensor_proto, proto_values):
        if False:
            for i in range(10):
                print('nop')
        tensor_proto.scomplex_val.extend([v.item() for x in proto_values for v in [x.real, x.imag]])

    def SlowAppendComplex128ArrayToTensorProto(tensor_proto, proto_values):
        if False:
            i = 10
            return i + 15
        tensor_proto.dcomplex_val.extend([v.item() for x in proto_values for v in [x.real, x.imag]])

    def SlowAppendObjectArrayToTensorProto(tensor_proto, proto_values):
        if False:
            print('Hello World!')
        tensor_proto.string_val.extend([compat.as_bytes(x) for x in proto_values])

    def SlowAppendBoolArrayToTensorProto(tensor_proto, proto_values):
        if False:
            print('Hello World!')
        tensor_proto.bool_val.extend([x.item() for x in proto_values])
    _NP_TO_APPEND_FN = {dtypes.bfloat16.as_numpy_dtype: SlowAppendBFloat16ArrayToTensorProto, dtypes.float8_e5m2.as_numpy_dtype: SlowAppendFloat8e5m2ArrayToTensorProto, dtypes.float8_e4m3fn.as_numpy_dtype: SlowAppendFloat8e4m3fnArrayToTensorProto, np.float16: SlowAppendFloat16ArrayToTensorProto, np.float32: SlowAppendFloat32ArrayToTensorProto, np.float64: SlowAppendFloat64ArrayToTensorProto, np.int32: SlowAppendIntArrayToTensorProto, np.int64: SlowAppendInt64ArrayToTensorProto, np.uint8: SlowAppendIntArrayToTensorProto, np.uint16: SlowAppendIntArrayToTensorProto, np.uint32: SlowAppendUInt32ArrayToTensorProto, np.uint64: SlowAppendUInt64ArrayToTensorProto, np.int8: SlowAppendIntArrayToTensorProto, np.int16: SlowAppendIntArrayToTensorProto, np.complex64: SlowAppendComplex64ArrayToTensorProto, np.complex128: SlowAppendComplex128ArrayToTensorProto, np.object_: SlowAppendObjectArrayToTensorProto, np.bool_: SlowAppendBoolArrayToTensorProto, dtypes.qint8.as_numpy_dtype: SlowAppendQIntArrayToTensorProto, dtypes.quint8.as_numpy_dtype: SlowAppendQIntArrayToTensorProto, dtypes.qint16.as_numpy_dtype: SlowAppendQIntArrayToTensorProto, dtypes.quint16.as_numpy_dtype: SlowAppendQIntArrayToTensorProto, dtypes.qint32.as_numpy_dtype: SlowAppendQIntArrayToTensorProto, dtypes.int4.as_numpy_dtype: SlowAppendInt4ArrayToTensorProto, dtypes.uint4.as_numpy_dtype: SlowAppendUInt4ArrayToTensorProto}

def GetFromNumpyDTypeDict(dtype_dict, dtype):
    if False:
        for i in range(10):
            print('nop')
    for (key, val) in dtype_dict.items():
        if key == dtype:
            return val
    return None

def GetNumpyAppendFn(dtype):
    if False:
        print('Hello World!')
    if dtype.type == np.bytes_ or dtype.type == np.str_:
        if _FAST_TENSOR_UTIL_AVAILABLE:
            return fast_tensor_util.AppendObjectArrayToTensorProto
        else:
            return SlowAppendObjectArrayToTensorProto
    return GetFromNumpyDTypeDict(_NP_TO_APPEND_FN, dtype)

def TensorShapeProtoToList(shape):
    if False:
        return 10
    'Convert a TensorShape to a list.\n\n  Args:\n    shape: A TensorShapeProto.\n\n  Returns:\n    List of integers representing the dimensions of the tensor.\n  '
    return [dim.size for dim in shape.dim]

def _GetDenseDimensions(list_of_lists):
    if False:
        print('Hello World!')
    'Returns the inferred dense dimensions of a list of lists.'
    if not isinstance(list_of_lists, (list, tuple)):
        return []
    elif not list_of_lists:
        return [0]
    else:
        return [len(list_of_lists)] + _GetDenseDimensions(list_of_lists[0])

def _FlattenToStrings(nested_strings):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(nested_strings, (list, tuple)):
        for inner in nested_strings:
            for flattened_string in _FlattenToStrings(inner):
                yield flattened_string
    else:
        yield nested_strings
_TENSOR_CONTENT_TYPES = frozenset([dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32, dtypes.uint8, dtypes.int16, dtypes.int8, dtypes.int64, dtypes.qint8, dtypes.quint8, dtypes.qint16, dtypes.quint16, dtypes.qint32, dtypes.uint32, dtypes.uint64, dtypes.float8_e5m2, dtypes.float8_e4m3fn, dtypes.bfloat16])

def _check_failed(v):
    if False:
        for i in range(10):
            print('nop')
    raise ValueError(v)

def _check_quantized(values):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(values, (list, tuple)):
        _check_failed(values)
    if isinstance(values, tuple):
        _ = [_check_int(v) for v in values]
    else:
        _ = [_check_quantized(v) for v in values]

def _generate_isinstance_check(expected_types):
    if False:
        while True:
            i = 10

    def inner(values):
        if False:
            return 10
        for v in nest.flatten(values):
            if not (isinstance(v, expected_types) or (isinstance(v, np.ndarray) and issubclass(v.dtype.type, expected_types))):
                _check_failed(v)
    return inner
_check_int = _generate_isinstance_check((compat.integral_types, tensor_shape.Dimension))
_check_float = _generate_isinstance_check(compat.real_types)
_check_complex = _generate_isinstance_check(compat.complex_types)
_check_str = _generate_isinstance_check(compat.bytes_or_text_types)
_check_bool = _generate_isinstance_check(bool)

def _check_not_tensor(values):
    if False:
        for i in range(10):
            print('nop')
    _ = [_check_failed(v) for v in nest.flatten(values) if isinstance(v, core.Symbol)]
_TF_TO_IS_OK = {dtypes.bool: _check_bool, dtypes.complex128: _check_complex, dtypes.complex64: _check_complex, dtypes.float16: _check_float, dtypes.float32: _check_float, dtypes.float64: _check_float, dtypes.int16: _check_int, dtypes.int32: _check_int, dtypes.int64: _check_int, dtypes.int8: _check_int, dtypes.qint16: _check_quantized, dtypes.qint32: _check_quantized, dtypes.qint8: _check_quantized, dtypes.quint16: _check_quantized, dtypes.quint8: _check_quantized, dtypes.string: _check_str, dtypes.uint16: _check_int, dtypes.uint8: _check_int, dtypes.uint32: _check_int, dtypes.uint64: _check_int}

def _AssertCompatible(values, dtype):
    if False:
        while True:
            i = 10
    if dtype is None:
        fn = _check_not_tensor
    else:
        try:
            fn = _TF_TO_IS_OK[dtype]
        except KeyError:
            if dtype.is_integer:
                fn = _check_int
            elif dtype.is_floating:
                fn = _check_float
            elif dtype.is_complex:
                fn = _check_complex
            elif dtype.is_quantized:
                fn = _check_quantized
            else:
                fn = _check_not_tensor
    try:
        fn(values)
    except ValueError as e:
        [mismatch] = e.args
        if dtype is None:
            raise TypeError('Expected any non-tensor type, but got a tensor instead.')
        else:
            raise TypeError(f"Expected {dtype.name}, but got {mismatch} of type '{type(mismatch).__name__}'.")

def _is_array_like(obj):
    if False:
        while True:
            i = 10
    'Check if a given object is array-like.'
    if isinstance(obj, core.Symbol) and (not isinstance(obj, core.Value)):
        return False
    if callable(getattr(obj, '__array__', None)) or isinstance(getattr(obj, '__array_interface__', None), dict):
        return True
    try:
        memoryview(obj)
    except TypeError:
        return False
    else:
        return not isinstance(obj, bytes)

@tf_export('make_tensor_proto')
def make_tensor_proto(values, dtype=None, shape=None, verify_shape=False, allow_broadcast=False):
    if False:
        print('Hello World!')
    'Create a TensorProto.\n\n  In TensorFlow 2.0, representing tensors as protos should no longer be a\n  common workflow. That said, this utility function is still useful for\n  generating TF Serving request protos:\n\n  ```python\n    request = tensorflow_serving.apis.predict_pb2.PredictRequest()\n    request.model_spec.name = "my_model"\n    request.model_spec.signature_name = "serving_default"\n    request.inputs["images"].CopyFrom(tf.make_tensor_proto(X_new))\n  ```\n\n  `make_tensor_proto` accepts "values" of a python scalar, a python list, a\n  numpy ndarray, or a numpy scalar.\n\n  If "values" is a python scalar or a python list, make_tensor_proto\n  first convert it to numpy ndarray. If dtype is None, the\n  conversion tries its best to infer the right numpy data\n  type. Otherwise, the resulting numpy array has a compatible data\n  type with the given dtype.\n\n  In either case above, the numpy ndarray (either the caller provided\n  or the auto-converted) must have the compatible type with dtype.\n\n  `make_tensor_proto` then converts the numpy array to a tensor proto.\n\n  If "shape" is None, the resulting tensor proto represents the numpy\n  array precisely.\n\n  Otherwise, "shape" specifies the tensor\'s shape and the numpy array\n  can not have more elements than what "shape" specifies.\n\n  Args:\n    values:         Values to put in the TensorProto.\n    dtype:          Optional tensor_pb2 DataType value.\n    shape:          List of integers representing the dimensions of tensor.\n    verify_shape:   Boolean that enables verification of a shape of values.\n    allow_broadcast:  Boolean that enables allowing scalars and 1 length vector\n        broadcasting. Cannot be true when verify_shape is true.\n\n  Returns:\n    A `TensorProto`. Depending on the type, it may contain data in the\n    "tensor_content" attribute, which is not directly useful to Python programs.\n    To access the values you should convert the proto back to a numpy ndarray\n    with `tf.make_ndarray(proto)`.\n\n    If `values` is a `TensorProto`, it is immediately returned; `dtype` and\n    `shape` are ignored.\n\n  Raises:\n    TypeError:  if unsupported types are provided.\n    ValueError: if arguments have inappropriate values or if verify_shape is\n     True and shape of values is not equals to a shape from the argument.\n\n  '
    if allow_broadcast and verify_shape:
        raise ValueError('allow_broadcast and verify_shape are not both allowed.')
    if isinstance(values, tensor_pb2.TensorProto):
        return values
    if dtype:
        dtype = dtypes.as_dtype(dtype)
    is_quantized = dtype in [dtypes.qint8, dtypes.quint8, dtypes.qint16, dtypes.quint16, dtypes.qint32]
    if _is_array_like(values):
        values = np.asarray(values)
    if isinstance(values, (np.ndarray, np.generic)):
        if dtype and dtype.is_numpy_compatible:
            nparray = values.astype(dtype.as_numpy_dtype)
        else:
            nparray = values
    else:
        if values is None:
            raise ValueError('None values not supported.')
        if dtype and dtype.is_numpy_compatible:
            np_dt = dtype.as_numpy_dtype
        else:
            np_dt = None
        if shape is not None and np.prod(shape, dtype=np.int64) == 0:
            nparray = np.empty(shape, dtype=np_dt)
        else:
            _AssertCompatible(values, dtype)
            nparray = np.array(values, dtype=np_dt)
            if list(nparray.shape) != _GetDenseDimensions(values) and (not is_quantized):
                raise ValueError(f'Expected values {values} to be a dense tensor with shape {_GetDenseDimensions(values)}, but got shape {list(nparray.shape)}.')
        if nparray.dtype == np.float64 and dtype is None:
            nparray = nparray.astype(np.float32)
        elif nparray.dtype == np.int64 and dtype is None:
            downcasted_array = nparray.astype(np.int32)
            if np.array_equal(downcasted_array, nparray):
                nparray = downcasted_array
    numpy_dtype = dtypes.as_dtype(nparray.dtype)
    if numpy_dtype is None:
        raise TypeError(f'Unrecognized data type: {nparray.dtype}.')
    if is_quantized:
        numpy_dtype = dtype
    if dtype is not None and (not hasattr(dtype, 'base_dtype') or dtype.base_dtype != numpy_dtype.base_dtype):
        raise TypeError(f'`dtype` {dtype} is not compatible with {values} of dtype {nparray.dtype}.')
    if shape is None:
        shape = nparray.shape
        is_same_size = True
        shape_size = nparray.size
    else:
        shape = [int(dim) for dim in shape]
        shape_size = np.prod(shape, dtype=np.int64)
        is_same_size = shape_size == nparray.size
        if allow_broadcast:
            if nparray.shape == (1,) or nparray.shape == tuple():
                pass
            elif nparray.size != shape_size:
                raise TypeError(f"Expected Tensor's shape: {tuple(shape)}, but got {nparray.shape}.")
        else:
            if verify_shape and nparray.shape != tuple(shape):
                raise TypeError(f"Expected Tensor's shape: {tuple(shape)}, but got {nparray.shape}.")
            if nparray.size > shape_size:
                raise ValueError(f'Too many elements provided. Takes at most {shape_size:d}, but got {nparray.size:d}.')
    tensor_proto = tensor_pb2.TensorProto(dtype=numpy_dtype.as_datatype_enum, tensor_shape=tensor_shape.as_shape(shape).as_proto())
    if is_same_size and numpy_dtype in _TENSOR_CONTENT_TYPES and (shape_size > 1):
        if nparray.size * nparray.itemsize >= 1 << 31:
            raise ValueError('Cannot create a tensor proto whose content is larger than 2GB.')
        tensor_proto.tensor_content = nparray.tobytes()
        return tensor_proto
    if numpy_dtype == dtypes.string and (not isinstance(values, np.ndarray)):
        proto_values = _FlattenToStrings(values)
        try:
            str_values = [compat.as_bytes(x) for x in proto_values]
        except TypeError:
            raise TypeError(f'Failed to convert elements of {values} to Tensor. Consider casting elements to a supported type. See https://www.tensorflow.org/api_docs/python/tf/dtypes for supported TF dtypes.')
        tensor_proto.string_val.extend(str_values)
        return tensor_proto
    proto_values = nparray.ravel()
    append_fn = GetNumpyAppendFn(proto_values.dtype)
    if append_fn is None:
        raise TypeError(f'Element type not supported in TensorProto: {numpy_dtype.name}.')
    append_fn(tensor_proto, proto_values)
    return tensor_proto

@tf_export('make_ndarray')
def MakeNdarray(tensor):
    if False:
        i = 10
        return i + 15
    'Create a numpy ndarray from a tensor.\n\n  Create a numpy ndarray with the same shape and data as the tensor.\n\n  For example:\n\n  ```python\n  # Tensor a has shape (2,3)\n  a = tf.constant([[1,2,3],[4,5,6]])\n  proto_tensor = tf.make_tensor_proto(a)  # convert `tensor a` to a proto tensor\n  tf.make_ndarray(proto_tensor) # output: array([[1, 2, 3],\n  #                                              [4, 5, 6]], dtype=int32)\n  # output has shape (2,3)\n  ```\n\n  Args:\n    tensor: A TensorProto.\n\n  Returns:\n    A numpy array with the tensor contents.\n\n  Raises:\n    TypeError: if tensor has unsupported type.\n\n  '
    shape = [d.size for d in tensor.tensor_shape.dim]
    num_elements = np.prod(shape, dtype=np.int64)
    tensor_dtype = dtypes.as_dtype(tensor.dtype)
    dtype = tensor_dtype.as_numpy_dtype
    if tensor.tensor_content:
        return np.frombuffer(tensor.tensor_content, dtype=dtype).copy().reshape(shape)
    if tensor_dtype == dtypes.string:
        values = list(tensor.string_val)
        padding = num_elements - len(values)
        if padding > 0:
            last = values[-1] if values else ''
            values.extend([last] * padding)
        return np.array(values, dtype=dtype).reshape(shape)
    if tensor_dtype == dtypes.float16 or tensor_dtype == dtypes.bfloat16:
        values = np.fromiter(tensor.half_val, dtype=np.uint16)
        values.dtype = dtype
    elif tensor_dtype in [dtypes.float8_e5m2, dtypes.float8_e4m3fn]:
        values = np.fromiter(tensor.float8_val, dtype=np.uint8)
        values.dtype = dtype
    elif tensor_dtype == dtypes.float32:
        values = np.fromiter(tensor.float_val, dtype=dtype)
    elif tensor_dtype == dtypes.float64:
        values = np.fromiter(tensor.double_val, dtype=dtype)
    elif tensor_dtype in [dtypes.int32, dtypes.uint8, dtypes.uint16, dtypes.int16, dtypes.int8, dtypes.qint32, dtypes.quint8, dtypes.qint8, dtypes.qint16, dtypes.quint16, dtypes.int4, dtypes.uint4]:
        values = np.fromiter(tensor.int_val, dtype=dtype)
    elif tensor_dtype == dtypes.int64:
        values = np.fromiter(tensor.int64_val, dtype=dtype)
    elif tensor_dtype == dtypes.uint32:
        values = np.fromiter(tensor.uint32_val, dtype=dtype)
    elif tensor_dtype == dtypes.uint64:
        values = np.fromiter(tensor.uint64_val, dtype=dtype)
    elif tensor_dtype == dtypes.complex64:
        it = iter(tensor.scomplex_val)
        values = np.array([complex(x[0], x[1]) for x in zip(it, it)], dtype=dtype)
    elif tensor_dtype == dtypes.complex128:
        it = iter(tensor.dcomplex_val)
        values = np.array([complex(x[0], x[1]) for x in zip(it, it)], dtype=dtype)
    elif tensor_dtype == dtypes.bool:
        values = np.fromiter(tensor.bool_val, dtype=dtype)
    else:
        raise TypeError(f'Unsupported tensor type: {tensor.dtype}. See https://www.tensorflow.org/api_docs/python/tf/dtypes for supported TF dtypes.')
    if values.size == 0:
        return np.zeros(shape, dtype)
    if values.size != num_elements:
        values = np.pad(values, (0, num_elements - values.size), 'edge')
    return values.reshape(shape)

def ShapeEquals(tensor_proto, shape):
    if False:
        i = 10
        return i + 15
    'Returns True if "tensor_proto" has the given "shape".\n\n  Args:\n    tensor_proto: A TensorProto.\n    shape: A tensor shape, expressed as a TensorShape, list, or tuple.\n\n  Returns:\n    True if "tensor_proto" has the given "shape", otherwise False.\n\n  Raises:\n    TypeError: If "tensor_proto" is not a TensorProto, or shape is not a\n      TensorShape, list, or tuple.\n  '
    if not isinstance(tensor_proto, tensor_pb2.TensorProto):
        raise TypeError(f'`tensor_proto` must be a tensor_pb2.TensorProto object, but got type {type(tensor_proto)}.')
    if isinstance(shape, tensor_shape_pb2.TensorShapeProto):
        shape = [d.size for d in shape.dim]
    elif not isinstance(shape, (list, tuple)):
        raise TypeError(f'`shape` must be a list or tuple, but got type {type(shape)}.')
    tensor_shape_list = [d.size for d in tensor_proto.tensor_shape.dim]
    return all((x == y for (x, y) in zip(tensor_shape_list, shape)))

def _ConstantValue(tensor, partial):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(tensor, core.Symbol):
        raise TypeError(f'{tensor!r} must be a Tensor, but got {type(tensor)}.')
    if tensor.op.type == 'Const':
        return MakeNdarray(tensor.op.get_attr('value'))
    elif tensor.op.type == 'Shape':
        input_shape = tensor.op.inputs[0].get_shape()
        if input_shape.is_fully_defined():
            return np.array([dim.value for dim in input_shape.dims], dtype=tensor.dtype.as_numpy_dtype)
        else:
            return None
    elif tensor.op.type == 'Size':
        input_shape = tensor.op.inputs[0].get_shape()
        if input_shape.is_fully_defined():
            return np.prod([dim.value for dim in input_shape.dims], dtype=np.int32)
        else:
            return None
    elif tensor.op.type == 'Rank':
        input_shape = tensor.op.inputs[0].get_shape()
        if input_shape.ndims is not None:
            return np.ndarray(shape=(), buffer=np.array([input_shape.ndims], dtype=np.int32), dtype=np.int32)
        else:
            return None
    elif tensor.op.type == 'Range':
        start = constant_value(tensor.op.inputs[0])
        if start is None:
            return None
        limit = constant_value(tensor.op.inputs[1])
        if limit is None:
            return None
        delta = constant_value(tensor.op.inputs[2])
        if delta is None:
            return None
        return np.arange(start, limit, delta, dtype=tensor.dtype.as_numpy_dtype)
    elif tensor.op.type == 'Cast':
        pre_cast = constant_value(tensor.op.inputs[0])
        if pre_cast is None:
            return None
        cast_dtype = dtypes.as_dtype(tensor.op.get_attr('DstT'))
        return pre_cast.astype(cast_dtype.as_numpy_dtype)
    elif tensor.op.type == 'Concat':
        dim = constant_value(tensor.op.inputs[0])
        if dim is None:
            return None
        values = []
        for x in tensor.op.inputs[1:]:
            value = constant_value(x)
            if value is None:
                return None
            values.append(value)
        return np.concatenate(values, axis=dim)
    elif tensor.op.type == 'ConcatV2':
        dim = constant_value(tensor.op.inputs[-1])
        if dim is None:
            return None
        values = []
        for x in tensor.op.inputs[:-1]:
            value = constant_value(x)
            if value is None:
                return None
            values.append(value)
        return np.concatenate(values, axis=dim)
    elif tensor.op.type == 'Pack':
        values = []
        if not tensor.op.inputs:
            return None
        if tensor.op.get_attr('axis') != 0:
            return None
        for x in tensor.op.inputs:
            value = constant_value(x, partial)
            if value is None and (not partial):
                return None
            values.append(value)
        try:
            return np.array(values)
        except ValueError:
            return np.array(values, dtype=object)
    elif tensor.op.type == 'Unpack':
        if tensor.op.get_attr('axis') != 0:
            return None
        value = constant_value(tensor.op.inputs[0], partial)
        if value is None:
            return None
        return value[tensor.value_index]
    elif tensor.op.type == 'Split':
        dim = constant_value(tensor.op.inputs[0])
        value = constant_value(tensor.op.inputs[1], partial)
        if value is None or dim is None:
            return None
        split = np.split(value, tensor.op.get_attr('num_split'), dim)
        return split[tensor.value_index]
    elif tensor.op.type == 'Fill':
        fill_shape = tensor.shape
        fill_value = constant_value(tensor.op.inputs[1])
        if fill_shape.is_fully_defined() and fill_value is not None:
            return np.full(fill_shape.as_list(), fill_value, dtype=fill_value.dtype)
        else:
            return None
    elif tensor.op.type == 'Equal':
        value1 = constant_value(tensor.op.inputs[0])
        if value1 is None:
            return None
        value2 = constant_value(tensor.op.inputs[1])
        if value2 is None:
            return None
        return np.equal(value1, value2)
    elif tensor.op.type == 'NotEqual':
        value1 = constant_value(tensor.op.inputs[0])
        if value1 is None:
            return None
        value2 = constant_value(tensor.op.inputs[1])
        if value2 is None:
            return None
        return np.not_equal(value1, value2)
    elif tensor.op.type == 'StopGradient':
        return constant_value(tensor.op.inputs[0], partial)
    elif tensor.op.type in ('CheckNumericsV2', 'DebugIdentityV2', 'Identity'):
        return constant_value(tensor.op.inputs[0], partial)
    else:
        return None

@tf_export('get_static_value')
def constant_value(tensor, partial=False):
    if False:
        return 10
    "Returns the constant value of the given tensor, if efficiently calculable.\n\n  This function attempts to partially evaluate the given tensor, and\n  returns its value as a numpy ndarray if this succeeds.\n\n  Example usage:\n\n  >>> a = tf.constant(10)\n  >>> tf.get_static_value(a)\n  10\n  >>> b = tf.constant(20)\n  >>> tf.get_static_value(tf.add(a, b))\n  30\n\n  >>> # `tf.Variable` is not supported.\n  >>> c = tf.Variable(30)\n  >>> print(tf.get_static_value(c))\n  None\n\n  Using `partial` option is most relevant when calling `get_static_value` inside\n  a `tf.function`. Setting it to `True` will return the results but for the\n  values that cannot be evaluated will be `None`. For example:\n\n  ```python\n  class Foo:\n    def __init__(self):\n      self.a = tf.Variable(1)\n      self.b = tf.constant(2)\n\n    @tf.function\n    def bar(self, partial):\n      packed = tf.raw_ops.Pack(values=[self.a, self.b])\n      static_val = tf.get_static_value(packed, partial=partial)\n      tf.print(static_val)\n\n  f = Foo()\n  f.bar(partial=True)  # `array([None, array(2, dtype=int32)], dtype=object)`\n  f.bar(partial=False)  # `None`\n  ```\n\n  Compatibility(V1): If `constant_value(tensor)` returns a non-`None` result, it\n  will no longer be possible to feed a different value for `tensor`. This allows\n  the result of this function to influence the graph that is constructed, and\n  permits static shape optimizations.\n\n  Args:\n    tensor: The Tensor to be evaluated.\n    partial: If True, the returned numpy array is allowed to have partially\n      evaluated values. Values that can't be evaluated will be None.\n\n  Returns:\n    A numpy ndarray containing the constant value of the given `tensor`,\n    or None if it cannot be calculated.\n\n  Raises:\n    TypeError: if tensor is not an tensor.Tensor.\n  "
    if isinstance(tensor, core.Value):
        try:
            return tensor.numpy()
        except errors_impl.UnimplementedError:
            return None
    if not is_tensor(tensor):
        return tensor
    if not isinstance(tensor, core.Symbol):
        return None
    ret = _ConstantValue(tensor, partial)
    if ret is not None:
        tensor.graph.prevent_feeding(tensor)
    return ret

def constant_value_as_shape(tensor):
    if False:
        i = 10
        return i + 15
    'A version of `constant_value()` that returns a `TensorShape`.\n\n  This version should be used when a constant tensor value is\n  interpreted as a (possibly partial) shape, e.g. in the shape\n  function for `tf.reshape()`. By explicitly requesting a\n  `TensorShape` as the return value, it is possible to represent\n  unknown dimensions; by contrast, `constant_value()` is\n  all-or-nothing.\n\n  Args:\n    tensor: The rank-0 or rank-1 Tensor to be evaluated.\n\n  Returns:\n    A `TensorShape` based on the constant value of the given `tensor`.\n\n  Raises:\n    ValueError: If the shape is rank-0 and is not statically known to be -1.\n  '
    if isinstance(tensor, core.Value):
        return tensor_shape.TensorShape([dim if dim != -1 else None for dim in tensor.numpy()])
    if tensor.get_shape().ndims == 0:
        value = constant_value(tensor)
        if value is None:
            raise ValueError("Received a scalar with unknown value as shape; require a statically known scalar with value '-1' to describe an unknown shape.")
        if value != -1:
            raise ValueError(f"Received a scalar value '{value}' as shape; require a statically known scalar with value '-1' to describe an unknown shape.")
        return tensor_shape.unknown_shape()
    shape = tensor.get_shape().with_rank(1)
    if shape == [0]:
        return tensor_shape.TensorShape([])
    elif tensor.op.type == 'Cast':
        pre_cast = constant_value_as_shape(tensor.op.inputs[0])
        if pre_cast.dims is None:
            return pre_cast
        cast_dtype = dtypes.as_dtype(tensor.op.get_attr('DstT'))
        if cast_dtype not in (dtypes.int32, dtypes.int64):
            return tensor_shape.unknown_shape(shape.dims[0].value)
        dest_dtype_shape_array = np.array([x if x is not None else -1 for x in pre_cast.as_list()]).astype(cast_dtype.as_numpy_dtype)
        return tensor_shape.TensorShape([x if x >= 0 else None for x in dest_dtype_shape_array])
    elif tensor.op.type == 'Shape':
        return tensor.op.inputs[0].get_shape()
    elif tensor.op.type == 'Pack':
        ret = tensor_shape.TensorShape([])
        assert tensor.op.get_attr('axis') == 0
        for pack_input in tensor.op.inputs:
            pack_input_val = constant_value(pack_input)
            if pack_input_val is None or pack_input_val < 0:
                new_dim = tensor_shape.Dimension(None)
            else:
                new_dim = tensor_shape.Dimension(pack_input_val)
            ret = ret.concatenate([new_dim])
        return ret
    elif tensor.op.type == 'Concat':
        ret = tensor_shape.TensorShape([])
        for concat_input in tensor.op.inputs[1:]:
            ret = ret.concatenate(constant_value_as_shape(concat_input))
        return ret
    elif tensor.op.type == 'ConcatV2':
        ret = tensor_shape.TensorShape([])
        for concat_input in tensor.op.inputs[:-1]:
            ret = ret.concatenate(constant_value_as_shape(concat_input))
        return ret
    elif tensor.op.type == 'StridedSlice':
        try:
            begin = constant_value(tensor.op.inputs[1])
            end = constant_value(tensor.op.inputs[2])
            strides = constant_value(tensor.op.inputs[3])
            if begin is not None and end is not None and (strides is not None):
                begin = begin[0]
                end = end[0]
                strides = strides[0]
                begin_mask = tensor.op.get_attr('begin_mask')
                if begin_mask == 1:
                    begin = None
                end_mask = tensor.op.get_attr('end_mask')
                if end_mask == 1:
                    end = None
                ellipsis_mask = tensor.op.get_attr('ellipsis_mask')
                new_axis_mask = tensor.op.get_attr('new_axis_mask')
                shrink_axis_mask = tensor.op.get_attr('shrink_axis_mask')
                valid_attributes = not ellipsis_mask and (not new_axis_mask) and (not shrink_axis_mask) and (not begin_mask or begin_mask == 1) and (not end_mask or end_mask == 1)
                if valid_attributes:
                    prev = constant_value_as_shape(tensor.op.inputs[0])
                    prev = prev[begin:end:strides]
                    ret = tensor_shape.TensorShape(prev)
                    return ret
        except ValueError:
            pass
        except TypeError:
            pass
    elif tensor.op.type == 'Placeholder' and tensor.op.graph.building_function and hasattr(tensor.op.graph, 'internal_captures'):
        for (i, capture) in enumerate(tensor.op.graph.internal_captures):
            if capture is tensor:
                external_capture = tensor.op.graph.external_captures[i]
                return constant_value_as_shape(external_capture)
    ret = tensor_shape.unknown_shape(shape.dims[0].value)
    value = constant_value(tensor)
    if value is not None:
        ret = ret.merge_with(tensor_shape.TensorShape([d if d >= 0 else None for d in value]))
    return ret

@typing.runtime_checkable
class IsTensorLike(Protocol):

    def is_tensor_like(self):
        if False:
            while True:
                i = 10
        pass
tf_type_classes = (internal.NativeObject, core.Tensor, IsTensorLike)

@tf_export('is_tensor')
def is_tf_type(x):
    if False:
        for i in range(10):
            print('nop')
    'Checks whether `x` is a TF-native type that can be passed to many TF ops.\n\n  Use `is_tensor` to differentiate types that can ingested by TensorFlow ops\n  without any conversion (e.g., `tf.Tensor`, `tf.SparseTensor`, and\n  `tf.RaggedTensor`) from types that need to be converted into tensors before\n  they are ingested (e.g., numpy `ndarray` and Python scalars).\n\n  For example, in the following code block:\n\n  ```python\n  if not tf.is_tensor(t):\n    t = tf.convert_to_tensor(t)\n  return t.shape, t.dtype\n  ```\n\n  we check to make sure that `t` is a tensor (and convert it if not) before\n  accessing its `shape` and `dtype`.  (But note that not all TensorFlow native\n  types have shapes or dtypes; `tf.data.Dataset` is an example of a TensorFlow\n  native type that has neither shape nor dtype.)\n\n  Args:\n    x: A python object to check.\n\n  Returns:\n    `True` if `x` is a TensorFlow-native type.\n  '
    return isinstance(x, tf_type_classes)
is_tensor = is_tf_type

def try_evaluate_constant(tensor):
    if False:
        for i in range(10):
            print('nop')
    'Evaluates a symbolic tensor as a constant.\n\n  Args:\n    tensor: a symbolic Tensor.\n\n  Returns:\n    ndarray if the evaluation succeeds, or None if it fails.\n  '
    with tensor.graph._c_graph.get() as c_graph:
        return c_api.TF_TryEvaluateConstant_wrapper(c_graph, tensor._as_tf_output())