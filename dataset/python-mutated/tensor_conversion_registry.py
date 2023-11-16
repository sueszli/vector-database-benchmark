"""Registry for tensor conversion functions."""
import collections
import threading
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util.tf_export import tf_export
_tensor_conversion_func_registry = collections.defaultdict(list)
_tensor_conversion_func_cache = {}
_tensor_conversion_func_lock = threading.Lock()
_CONSTANT_OP_CONVERTIBLES = (int, float, np.generic, np.ndarray)

def register_tensor_conversion_function_internal(base_type, conversion_func, priority=100):
    if False:
        return 10
    'Internal version of register_tensor_conversion_function.\n\n  See docstring of `register_tensor_conversion_function` for details.\n\n  The internal version of the function allows registering conversions\n  for types in the _UNCONVERTIBLE_TYPES tuple.\n\n  Args:\n    base_type: The base type or tuple of base types for all objects that\n      `conversion_func` accepts.\n    conversion_func: A function that converts instances of `base_type` to\n      `Tensor`.\n    priority: Optional integer that indicates the priority for applying this\n      conversion function. Conversion functions with smaller priority values run\n      earlier than conversion functions with larger priority values. Defaults to\n      100.\n\n  Raises:\n    TypeError: If the arguments do not have the appropriate type.\n  '
    base_types = base_type if isinstance(base_type, tuple) else (base_type,)
    if any((not isinstance(x, type) for x in base_types)):
        raise TypeError(f'Argument `base_type` must be a type or a tuple of types. Obtained: {base_type}')
    del base_types
    if not callable(conversion_func):
        raise TypeError(f'Argument `conversion_func` must be callable. Received {conversion_func}.')
    with _tensor_conversion_func_lock:
        _tensor_conversion_func_registry[priority].append((base_type, conversion_func))
        _tensor_conversion_func_cache.clear()

@tf_export('register_tensor_conversion_function')
def register_tensor_conversion_function(base_type, conversion_func, priority=100):
    if False:
        print('Hello World!')
    'Registers a function for converting objects of `base_type` to `Tensor`.\n\n  The conversion function must have the following signature:\n\n  ```python\n      def conversion_func(value, dtype=None, name=None, as_ref=False):\n        # ...\n  ```\n\n  It must return a `Tensor` with the given `dtype` if specified. If the\n  conversion function creates a new `Tensor`, it should use the given\n  `name` if specified. All exceptions will be propagated to the caller.\n\n  The conversion function may return `NotImplemented` for some\n  inputs. In this case, the conversion process will continue to try\n  subsequent conversion functions.\n\n  If `as_ref` is true, the function must return a `Tensor` reference,\n  such as a `Variable`.\n\n  NOTE: The conversion functions will execute in order of priority,\n  followed by order of registration. To ensure that a conversion function\n  `F` runs before another conversion function `G`, ensure that `F` is\n  registered with a smaller priority than `G`.\n\n  Args:\n    base_type: The base type or tuple of base types for all objects that\n      `conversion_func` accepts.\n    conversion_func: A function that converts instances of `base_type` to\n      `Tensor`.\n    priority: Optional integer that indicates the priority for applying this\n      conversion function. Conversion functions with smaller priority values run\n      earlier than conversion functions with larger priority values. Defaults to\n      100.\n\n  Raises:\n    TypeError: If the arguments do not have the appropriate type.\n  '
    base_types = base_type if isinstance(base_type, tuple) else (base_type,)
    if any((not isinstance(x, type) for x in base_types)):
        raise TypeError(f'Argument `base_type` must be a type or a tuple of types. Obtained: {base_type}')
    if any((issubclass(x, _CONSTANT_OP_CONVERTIBLES) for x in base_types)):
        raise TypeError('Cannot register conversions for Python numeric types and NumPy scalars and arrays.')
    del base_types
    register_tensor_conversion_function_internal(base_type, conversion_func, priority)

def get(query):
    if False:
        while True:
            i = 10
    'Get conversion function for objects of `cls`.\n\n  Args:\n    query: The type to query for.\n\n  Returns:\n    A list of conversion functions in increasing order of priority.\n  '
    conversion_funcs = _tensor_conversion_func_cache.get(query)
    if conversion_funcs is None:
        with _tensor_conversion_func_lock:
            conversion_funcs = _tensor_conversion_func_cache.get(query)
            if conversion_funcs is None:
                conversion_funcs = []
                for (_, funcs_at_priority) in sorted(_tensor_conversion_func_registry.items()):
                    conversion_funcs.extend(((base_type, conversion_func) for (base_type, conversion_func) in funcs_at_priority if issubclass(query, base_type)))
                _tensor_conversion_func_cache[query] = conversion_funcs
    return conversion_funcs

def _add_error_prefix(msg, *, name=None):
    if False:
        print('Hello World!')
    return msg if name is None else f'{name}: {msg}'

def convert(value, dtype=None, name=None, as_ref=False, preferred_dtype=None, accepted_result_types=(core.Symbol,)):
    if False:
        print('Hello World!')
    'Converts `value` to a `Tensor` using registered conversion functions.\n\n  Args:\n    value: An object whose type has a registered `Tensor` conversion function.\n    dtype: Optional element type for the returned tensor. If missing, the type\n      is inferred from the type of `value`.\n    name: Optional name to use if a new `Tensor` is created.\n    as_ref: Optional boolean specifying if the returned value should be a\n      reference-type `Tensor` (e.g. Variable). Pass-through to the registered\n      conversion function. Defaults to `False`.\n    preferred_dtype: Optional element type for the returned tensor.\n      Used when dtype is None. In some cases, a caller may not have a dtype\n      in mind when converting to a tensor, so `preferred_dtype` can be used\n      as a soft preference. If the conversion to `preferred_dtype` is not\n      possible, this argument has no effect.\n    accepted_result_types: Optional collection of types as an allow-list\n      for the returned value. If a conversion function returns an object\n      which is not an instance of some type in this collection, that value\n      will not be returned.\n\n  Returns:\n    A `Tensor` converted from `value`.\n\n  Raises:\n    ValueError: If `value` is a `Tensor` and conversion is requested\n      to a `Tensor` with an incompatible `dtype`.\n    TypeError: If no conversion function is registered for an element in\n      `values`.\n    RuntimeError: If a registered conversion function returns an invalid\n      value.\n  '
    if dtype is not None:
        dtype = dtypes.as_dtype(dtype)
    if preferred_dtype is not None:
        preferred_dtype = dtypes.as_dtype(preferred_dtype)
    overload = getattr(value, '__tf_tensor__', None)
    if overload is not None:
        return overload(dtype, name)
    for (base_type, conversion_func) in get(type(value)):
        ret = None
        if dtype is None and preferred_dtype is not None:
            try:
                ret = conversion_func(value, dtype=preferred_dtype, name=name, as_ref=as_ref)
            except (TypeError, ValueError):
                pass
            else:
                if ret is not NotImplemented and ret.dtype.base_dtype != preferred_dtype.base_dtype:
                    raise RuntimeError(_add_error_prefix(f'Conversion function {conversion_func!r} for type {base_type} returned incompatible dtype: requested = {preferred_dtype.base_dtype.name}, actual = {ret.dtype.base_dtype.name}', name=name))
        if ret is None:
            ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
        if ret is NotImplemented:
            continue
        if isinstance(ret, core.Tensor):
            to_tensor = getattr(ret, '__tf_tensor__', None)
            ret = to_tensor() if to_tensor is not None else ret
        if not isinstance(ret, accepted_result_types):
            raise RuntimeError(_add_error_prefix(f'Conversion function {conversion_func!r} for type {base_type} returned non-Tensor: {ret!r}', name=name))
        if dtype and (not dtype.is_compatible_with(ret.dtype)):
            raise RuntimeError(_add_error_prefix(f'Conversion function {conversion_func} for type {base_type} returned incompatible dtype: requested = {dtype.name}, actual = {ret.dtype.name}', name=name))
        return ret
    raise TypeError(_add_error_prefix(f'Cannot convert {value!r} with type {type(value)} to Tensor: no conversion function registered.', name=name))