import contextlib
import ivy
import functools
import logging
import weakref
import warnings
import copy as python_copy
from types import FunctionType
from typing import Callable, Literal
import inspect
import numpy as np
from ivy.utils.exceptions import IvyValueError
FN_DECORATORS = ['handle_complex_input', 'handle_device', 'infer_dtype', 'handle_array_function', 'outputs_to_ivy_arrays', 'outputs_to_ivy_shapes', 'outputs_to_native_arrays', 'inputs_to_native_arrays', 'inputs_to_native_shapes', 'inputs_to_ivy_arrays', 'handle_out_argument', 'handle_view_indexing', 'handle_view', 'handle_array_like_without_promotion', 'handle_partial_mixed_function', 'handle_nestable', 'handle_ragged', 'handle_backend_invalid', 'temp_asarray_wrapper', 'handle_exceptions', 'handle_nans']
casting_modes_dict = {'uint': lambda : ivy.valid_uint_dtypes, 'int': lambda : sorted(set(ivy.valid_int_dtypes).difference(set(ivy.valid_uint_dtypes))), 'float': lambda : ivy.valid_float_dtypes, 'complex': lambda : ivy.valid_complex_dtypes}

def caster(dtype, intersect):
    if False:
        return 10
    if hasattr(dtype, 'dtype'):
        dtype = ivy.as_ivy_dtype(dtype.dtype)
    else:
        dtype = ivy.as_ivy_dtype(dtype)
    if str(dtype) in intersect:
        if ivy.cast_dtypes():
            ret_dtype = cross_caster(intersect)
            if ret_dtype:
                return ret_dtype
            ret_dtype = upcaster(dtype, intersect)
            if ret_dtype:
                return ret_dtype
            ret_dtype = downcaster(dtype, intersect)
            if ret_dtype:
                return ret_dtype
        elif ivy.crosscast_dtypes:
            ret_dtype = cross_caster(intersect)
            if ret_dtype:
                return ret_dtype
        elif ivy.upcast_dtypes:
            ret_dtype = upcaster(dtype, intersect)
            if ret_dtype:
                return ret_dtype
        elif ivy.downcast_dtypes:
            ret_dtype = downcaster(dtype, intersect)
            if ret_dtype:
                return ret_dtype

def upcaster(dtype, intersect):
    if False:
        while True:
            i = 10
    if 'uint' in str(dtype):
        index = casting_modes_dict['uint']().index(dtype) + 1
        result = ''
        while index < len(casting_modes_dict['uint']()):
            if casting_modes_dict['uint']()[index] not in intersect:
                result = casting_modes_dict['uint']()[index]
                break
            index += 1
        return result
    if 'int' in dtype:
        index = casting_modes_dict['int']().index(dtype) + 1
        result = ''
        while index < len(casting_modes_dict['int']()):
            if casting_modes_dict['int']()[index] not in intersect:
                result = casting_modes_dict['int']()[index]
                break
            index += 1
        return result
    if 'float' in dtype:
        index = casting_modes_dict['float']().index(dtype) + 1
        result = ''
        while index < len(casting_modes_dict['float']()):
            if casting_modes_dict['float']()[index] not in intersect:
                result = casting_modes_dict['float']()[index]
                break
            index += 1
        return result
    if 'complex' in dtype:
        index = casting_modes_dict['complex']().index(dtype) + 1
        result = ''
        while index < len(casting_modes_dict['complex']()):
            if casting_modes_dict['complex']()[index] not in intersect:
                result = casting_modes_dict['complex']()[index]
                break
            index += 1
        return result

def downcaster(dtype, intersect):
    if False:
        for i in range(10):
            print('nop')
    if 'uint' in str(dtype):
        index = casting_modes_dict['uint']().index(dtype) - 1
        result = ''
        while index >= 0:
            if casting_modes_dict['int']()[index] not in intersect:
                result = casting_modes_dict['uint']()[index]
                break
            index -= 1
        return result
    if 'int' in dtype:
        index = casting_modes_dict['int']().index(dtype) - 1
        result = ''
        while index >= 0:
            if casting_modes_dict['int']()[index] not in intersect:
                result = casting_modes_dict['int']()[index]
                break
            index -= 1
        return result
    if 'float' in dtype:
        index = casting_modes_dict['float']().index(dtype) - 1
        result = ''
        while index >= 0:
            if casting_modes_dict['float']()[index] not in intersect:
                result = casting_modes_dict['float']()[index]
                break
            index -= 1
        return result
    if 'complex' in dtype:
        index = casting_modes_dict['complex']().index(dtype) - 1
        result = ''
        while index >= 0:
            if casting_modes_dict['complex']()[index] not in intersect:
                result = casting_modes_dict['complex']()[index]
                break
            index -= 1
        return result

def cross_caster(intersect):
    if False:
        return 10
    dtype = ''
    valid_float = sorted(ivy.valid_float_dtypes)
    valid_int = sorted(ivy.valid_int_dtypes)
    intersect = sorted(intersect)
    if set(valid_int).issubset(intersect):
        dtype = ivy.default_float_dtype()
    elif set(valid_float).issubset(intersect):
        dtype = ivy.default_int_dtype()
    return str(dtype)

def try_array_function_override(func, overloaded_args, types, args, kwargs):
    if False:
        for i in range(10):
            print('nop')
    if not overloaded_args:
        return (False, None)
    for overloaded_arg in overloaded_args:
        try:
            result = overloaded_arg.__ivy_array_function__(func, types, args, kwargs)
        except Exception:
            raise ivy.utils.exceptions.IvyNotImplementedException
        if result is not NotImplemented:
            return (True, result)
    raise TypeError(f'no implementation found for {func} on types that implement __ivy_array_function__: {list(map(type, overloaded_args))}')

def _get_first_array(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    array_fn = ivy.is_array if 'array_fn' not in kwargs else kwargs['array_fn']
    arr = None
    if args:
        arr_idxs = ivy.nested_argwhere(args, array_fn, stop_after_n_found=1)
        if arr_idxs:
            arr = ivy.index_nest(args, arr_idxs[0])
        else:
            arr_idxs = ivy.nested_argwhere(kwargs, array_fn, stop_after_n_found=1)
            if arr_idxs:
                arr = ivy.index_nest(kwargs, arr_idxs[0])
    elif kwargs:
        arr_idxs = ivy.nested_argwhere(kwargs, array_fn, stop_after_n_found=1)
        if arr_idxs:
            arr = ivy.index_nest(kwargs, arr_idxs[0])
    return arr

def _build_view(original, view, fn, args, kwargs, index=None):
    if False:
        print('Hello World!')
    if ivy.exists(original._base):
        base = original._base
        view._manipulation_stack = python_copy.copy(original._manipulation_stack)
    else:
        base = original
    view._base = base
    base._view_refs.append(weakref.ref(view))
    view._manipulation_stack.append((fn, args[1:], kwargs, index))
    if ivy.exists(original._torch_base):
        view._torch_base = original if ivy.exists(original._torch_manipulation) else original._torch_base
    else:
        view._torch_base = base
    if fn in _torch_non_native_view_functions:
        view._torch_manipulation = (original, (fn, args[1:], kwargs))
        view._torch_base._torch_view_refs.append(weakref.ref(view))
    return view
_torch_non_native_view_functions = ('flip', 'flipud', 'rot90', 'fliplr')

def _check_in_nested_sequence(sequence, value=None, _type=None):
    if False:
        while True:
            i = 10
    '\n    Check `sequence` for either a `value` or a value of type `_type`.\n\n    Helper to recursively check if a N-level nested `sequence` contains\n    either a `value` or contains a value of type `_type` and return a\n    boolean flag.\n    '
    if sequence is value or isinstance(sequence, _type):
        return True
    elif isinstance(sequence, (tuple, list)):
        if any((isinstance(_val, _type) or _val is value for _val in sequence)):
            return True
        else:
            return any((_check_in_nested_sequence(sub_sequence, value, _type) for sub_sequence in sequence if isinstance(sub_sequence, (tuple, list))))

def _get_preferred_device(args, kwargs):
    if False:
        i = 10
        return i + 15
    device = None
    if 'device' in kwargs and kwargs['device'] is not None:
        return device
    if not ivy.soft_device_mode:
        arr_arg = _get_first_array(*args, **kwargs)
        return ivy.default_device(item=arr_arg, as_native=True)
    return ivy.default_device(as_native=True)

def handle_array_function(fn):
    if False:
        i = 10
        return i + 15
    '\n    Wrap a function `fn` to be passed to array_function method.\n\n    Wrap a function to extract the relevant argument types to be passed\n    to array_function method.\n    '

    @functools.wraps(fn)
    def _handle_array_function(*args, **kwargs):
        if False:
            while True:
                i = 10
        overloaded_types = []
        overloaded_args = []
        for arg in args + tuple(kwargs.values()):
            if ivy.exists(arg):
                if not isinstance(arg, ivy.Container) and hasattr(arg, '__ivy_array_function__'):
                    if type(arg) not in overloaded_types:
                        overloaded_types.append(type(arg))
                        if arg.__ivy_array_function__ is not ivy.Array.__ivy_array_function__ and (not isinstance(arg, (ivy.Array, ivy.NativeArray))):
                            index = len(overloaded_args)
                            for (i, old_arg) in enumerate(overloaded_args):
                                if issubclass(type(arg), type(old_arg)):
                                    index = i
                                    break
                            overloaded_args.insert(index, arg)
                elif isinstance(arg, ivy.Container):
                    arg = ivy.Container.cont_flatten_key_chains(arg)
                    indices = ivy.nested_argwhere(arg, lambda x: hasattr(x, '__ivy_array_function__'))
                    for a in indices:
                        if type(getattr(arg, a[0])) not in overloaded_types:
                            overloaded_types.append(type(getattr(arg, a[0])))
                            if getattr(arg, a[0]).__ivy_array_function__ is not ivy.Array.__ivy_array_function__ and (not isinstance(getattr(arg, a[0]), (ivy.Array, ivy.NativeArray))):
                                index = len(overloaded_args)
                                for (i, old_arg) in enumerate(overloaded_args):
                                    if issubclass(type(getattr(arg, a[0])), type(old_arg)):
                                        index = i
                                        break
                                overloaded_args.insert(index, arg)
        (success, value) = try_array_function_override(ivy.__dict__[fn.__name__], overloaded_args, overloaded_types, args, kwargs)
        if success:
            return value
        return fn(*args, **kwargs)
    _handle_array_function.handle_array_function = True
    return _handle_array_function

def handle_array_like_without_promotion(fn: Callable) -> Callable:
    if False:
        i = 10
        return i + 15

    @functools.wraps(fn)
    def _handle_array_like_without_promotion(*args, **kwargs):
        if False:
            return 10
        args = list(args)
        num_args = len(args)
        try:
            type_hints = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return fn(*args, **kwargs)
        parameters = list(type_hints.keys())
        annotations = [param.annotation for param in type_hints.values()]
        device = _get_preferred_device(args, kwargs)
        for (i, (annotation, parameter, arg)) in enumerate(zip(annotations, parameters, args)):
            annotation_str = str(annotation)
            if ('rray' in annotation_str or 'Tensor' in annotation_str) and parameter != 'out' and all((sq not in annotation_str for sq in ['Sequence', 'List', 'Tuple', 'float', 'int', 'bool'])):
                if i < num_args:
                    if _check_in_nested_sequence(arg, value=Ellipsis, _type=slice):
                        continue
                    if not ivy.is_array(arg):
                        args[i] = ivy.array(arg, device=device)
                elif parameters in kwargs:
                    kwarg = kwargs[parameter]
                    if not ivy.is_array(kwarg):
                        kwargs[parameter] = ivy.array(kwarg, device=device)
        return fn(*args, **kwargs)
    _handle_array_like_without_promotion.handle_array_like_without_promotion = True
    return _handle_array_like_without_promotion

def inputs_to_native_arrays(fn: Callable) -> Callable:
    if False:
        while True:
            i = 10

    @functools.wraps(fn)
    def _inputs_to_native_arrays(*args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Convert all `ivy.Array` instances in both the positional and keyword arguments\n        into `ivy.NativeArray` instances, and then calls the function with the updated\n        arguments.\n\n        Parameters\n        ----------\n        args\n            The arguments to be passed to the function.\n\n        kwargs\n            The keyword arguments to be passed to the function.\n\n        Returns\n        -------\n            The return of the function, with native arrays passed in the arguments.\n        '
        if not ivy.array_mode:
            return fn(*args, **kwargs)
        has_out = False
        out = None
        if 'out' in kwargs:
            out = kwargs['out']
            del kwargs['out']
            has_out = True
        (new_args, new_kwargs) = ivy.args_to_native(*args, **kwargs)
        if has_out:
            new_kwargs['out'] = out
        return fn(*new_args, **new_kwargs)
    _inputs_to_native_arrays.inputs_to_native_arrays = True
    return _inputs_to_native_arrays

def inputs_to_ivy_arrays(fn: Callable) -> Callable:
    if False:
        return 10

    @functools.wraps(fn)
    def _inputs_to_ivy_arrays(*args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Convert all `ivy.NativeArray` instances in both the positional and keyword\n        arguments into `ivy.Array` instances, and then calls the function with the\n        updated arguments.\n\n        Parameters\n        ----------\n        args\n            The arguments to be passed to the function.\n\n        kwargs\n            The keyword arguments to be passed to the function.\n\n        Returns\n        -------\n            The return of the function, with ivy arrays passed in the arguments.\n        '
        if not ivy.array_mode:
            warnings.warn('In the case of Compositional function, operators might cause inconsistent behavior when array_mode is set to False')
            return fn(*args, **kwargs)
        has_out = False
        if 'out' in kwargs:
            out = kwargs['out']
            has_out = True
        (ivy_args, ivy_kwargs) = ivy.args_to_ivy(*args, **kwargs, include_derived={'tuple': True})
        if has_out:
            ivy_kwargs['out'] = out
        return fn(*ivy_args, **ivy_kwargs)
    _inputs_to_ivy_arrays.inputs_to_ivy_arrays = True
    return _inputs_to_ivy_arrays

def inputs_to_native_shapes(fn: Callable) -> Callable:
    if False:
        return 10

    @functools.wraps(fn)
    def _inputs_to_native_shapes(*args, **kwargs):
        if False:
            return 10
        (args, kwargs) = ivy.nested_map(lambda x: x.shape if isinstance(x, ivy.Shape) and ivy.array_mode else x, [args, kwargs])
        return fn(*args, **kwargs)
    _inputs_to_native_shapes.inputs_to_native_shapes = True
    return _inputs_to_native_shapes

def outputs_to_ivy_shapes(fn: Callable) -> Callable:
    if False:
        print('Hello World!')

    @functools.wraps(fn)
    def _outputs_to_ivy_shapes(*args, **kwargs):
        if False:
            print('Hello World!')
        (args, kwargs) = ivy.nested_map(lambda x: x.shape if isinstance(x, ivy.Shape) and ivy.array_mode else x, [args, kwargs])
        return fn(*args, **kwargs)
    _outputs_to_ivy_shapes.outputs_to_ivy_shapes = True
    return _outputs_to_ivy_shapes

def to_native_shapes_and_back(fn: Callable) -> Callable:
    if False:
        print('Hello World!')
    '\n    Make `fn` receive `ivy.NativeShape` and return `ivy.Shape`.\n\n    Wrap `fn` so that input shapes are all converted to\n    `ivy.NativeShape` instances and return shapes are all converted to\n    `ivy.Shape` instances.\n    '
    return outputs_to_ivy_shapes(inputs_to_native_shapes(fn))

def outputs_to_ivy_arrays(fn: Callable) -> Callable:
    if False:
        print('Hello World!')

    @functools.wraps(fn)
    def _outputs_to_ivy_arrays(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Call the function, and then converts all `ivy.NativeArray` instances in the\n        function return into `ivy.Array` instances.\n\n        Parameters\n        ----------\n        args\n            The arguments to be passed to the function.\n\n        kwargs\n            The keyword arguments to be passed to the function.\n\n        Returns\n        -------\n            The return of the function, with native arrays as ivy arrays.\n        '
        ret = fn(*args, **kwargs)
        return ivy.to_ivy(ret, nested=True, include_derived={'tuple': True}) if ivy.array_mode else ret
    _outputs_to_ivy_arrays.outputs_to_ivy_arrays = True
    return _outputs_to_ivy_arrays

def output_to_native_arrays(fn: Callable) -> Callable:
    if False:
        for i in range(10):
            print('nop')
    '\n    Call the function, and then converts all `ivy.Array` instances in the function\n    return into `ivy.NativeArray` instances.\n\n    Parameters\n    ----------\n    args\n        The arguments to be passed to the function.\n\n    kwargs\n        The keyword arguments to be passed to the function.\n\n    Returns\n    -------\n        The return of the function, with ivy arrays as native arrays.\n    '

    @functools.wraps(fn)
    def _output_to_native_arrays(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        ret = fn(*args, **kwargs)
        return ivy.to_native(ret, nested=True, include_derived={'tuple': True})
    _output_to_native_arrays.outputs_to_native_arrays = True
    return _output_to_native_arrays

def to_ivy_arrays_and_back(fn: Callable) -> Callable:
    if False:
        i = 10
        return i + 15
    '\n    Make `fn` receive `ivy.Array` and return `ivy.NativeArray`.\n\n    Wrap `fn` so that input arrays are all converted to `ivy.Array`\n    instances and return arrays are all converted to `ivy.NativeArray`\n    instances.\n    '
    return output_to_native_arrays(inputs_to_ivy_arrays(fn))

def to_native_arrays_and_back(fn: Callable) -> Callable:
    if False:
        return 10
    '\n    Make `fn` receive `ivy.NativeArray` and return `ivy.Array`.\n\n    Wrap `fn` so that input arrays are all converted to\n    `ivy.NativeArray` instances and return arrays are all converted to\n    `ivy.Array` instances.\n    '
    return outputs_to_ivy_arrays(inputs_to_native_arrays(fn))

def frontend_outputs_to_ivy_arrays(fn: Callable) -> Callable:
    if False:
        for i in range(10):
            print('nop')
    '\n    Wrap `fn` and convert all frontend arrays in its return to ivy arrays.\n\n    Used in cases when a frontend function receives a callable (frontend\n    function) argument. To be able to use that callable in a composition\n    of ivy functions, its outputs need to be converted to ivy arrays.\n    '

    @functools.wraps(fn)
    def _outputs_to_ivy_arrays(*args, **kwargs):
        if False:
            print('Hello World!')
        ret = fn(*args, **kwargs)
        return ivy.nested_map(lambda x: x.ivy_array if hasattr(x, 'ivy_array') else x, ret, shallow=False)
    return _outputs_to_ivy_arrays

def handle_view(fn: Callable) -> Callable:
    if False:
        for i in range(10):
            print('nop')
    "\n    Wrap `fn` and performs view handling if copy is False.\n\n    Used for functional backends (Jax and TensorFlow). Checks if the\n    first arg is a view or original array by checking if the ._base\n    attribute is populated. If it's original it adds the returned array\n    to its view references, then the returned array adds the operation\n    to its manipulation stack and stores the original as its base. If\n    the first arg is a view, then the returned array copies its base and\n    manipulation stack, appends the new operation to the manipulation\n    stack and appends its reference to the base array's view_refs\n    attribute.\n    "

    @functools.wraps(fn)
    def _handle_view(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        ret = fn(*args, **kwargs)
        if 'copy' in kwargs and kwargs['copy'] or not ivy.is_ivy_array(args[0]):
            return ret
        original = args[0]
        if isinstance(ret, (list, tuple)):
            for (i, view) in enumerate(ret):
                ret[i] = _build_view(original, view, fn.__name__, args, kwargs, i)
        else:
            ret = _build_view(original, ret, fn.__name__, args, kwargs, None)
        return ret
    _handle_view.handle_view = True
    return _handle_view

def handle_view_indexing(fn: Callable) -> Callable:
    if False:
        while True:
            i = 10
    "\n    Wrap `fn` and performs view handling specifically for indexing.\n\n    As with NumPy it returns a copy if advanced indexing is performed.\n    Used for functional backends (Jax and TensorFlow). Checks if the\n    first arg is a view or original array by checking if the ._base\n    attribute is populated. If it's original it adds the returned array\n    to its view references, then the returned array adds the operation\n    to its manipulation stack and stores the original as its base. If\n    the first arg is a view, then the returned array copies its base and\n    manipulation stack, appends the new operation to the manipulation\n    stack and appends its reference to the base array's view_refs\n    attribute.\n    "

    @functools.wraps(fn)
    def _handle_view_indexing(*args, **kwargs):
        if False:
            print('Hello World!')
        ret = fn(*args, **kwargs)
        if 'copy' in kwargs and kwargs['copy'] or not ivy.is_ivy_array(args[0]):
            return ret
        query = kwargs['query'] if 'query' in kwargs else args[1]
        query = query if isinstance(query, tuple) else (query,)
        if [i for i in query if not isinstance(i, (slice, int))]:
            return ret
        original = args[0]
        ret = _build_view(original, ret, 'get_item', args, kwargs)
        return ret
    _handle_view_indexing.handle_view_indexing = True
    return _handle_view_indexing

def _convert_numpy_arrays_to_backend_specific(*args):
    if False:
        return 10
    if isinstance(args, np.ndarray):
        np_arr_idxs = ivy.nested_argwhere(args, lambda x: isinstance(x, np.ndarray))
        np_arr_val = ivy.multi_index_nest(args, np_arr_idxs)
        backend_arr_vals = [ivy.array(x).to_native() for x in np_arr_val]
        ivy.set_nest_at_indices(args, np_arr_idxs, backend_arr_vals)
    return args

def handle_numpy_arrays_in_specific_backend(fn: Callable) -> Callable:
    if False:
        return 10
    '\n    Wrap `fn` and converts all `numpy.ndarray` inputs to `torch.Tensor` instances.\n\n    Used for functional backends (PyTorch). Converts all `numpy.ndarray`\n    inputs to `torch.Tensor` instances.\n    '

    @functools.wraps(fn)
    def _handle_numpy_array_in_torch(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        args = _convert_numpy_arrays_to_backend_specific(*args)
        ret = fn(*args, **kwargs)
        return ret
    _handle_numpy_array_in_torch.handle_numpy_arrays_in_specific_backend = True
    return _handle_numpy_array_in_torch

def infer_dtype(fn: Callable) -> Callable:
    if False:
        i = 10
        return i + 15

    @functools.wraps(fn)
    def _infer_dtype(*args, dtype=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Determine the correct `dtype`, and then calls the function with the `dtype`\n        passed explicitly.\n\n        Parameters\n        ----------\n        args\n            The arguments to be passed to the function.\n\n        dtype\n            The data type for the function.\n\n        kwargs\n            The keyword arguments to be passed to the function.\n\n        Returns\n        -------\n            The return of the function, with `dtype` passed explicitly.\n        '
        arr = None if ivy.exists(dtype) else _get_first_array(*args, **kwargs)
        dtype = ivy.default_dtype(dtype=dtype, item=arr, as_native=True)
        ivy.utils.assertions._check_jax_x64_flag(dtype)
        return fn(*args, dtype=dtype, **kwargs)
    _infer_dtype.infer_dtype = True
    return _infer_dtype

def handle_device(fn: Callable) -> Callable:
    if False:
        print('Hello World!')

    @functools.wraps(fn)
    def _handle_device(*args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Move all array inputs of the function to `ivy.default_device()`.\n\n        Parameters\n        ----------\n        args\n            The arguments to be passed to the function.\n        kwargs\n            The keyword arguments to be passed to the function.\n\n        Returns\n        -------\n            The return of the function.\n        '
        dev = None
        if 'device' in kwargs and kwargs['device'] is not None:
            dev = ivy.as_native_dev(kwargs['device'])
        if ivy.soft_device_mode:
            with ivy.DefaultDevice(ivy.default_device(dev)):
                return ivy.handle_soft_device_variable(*args, fn=fn, **kwargs)
        inputs = args + tuple(kwargs.values())
        devices = tuple((ivy.dev(x) for x in inputs if ivy.is_native_array(x)))
        unique_devices = set(devices)
        if len(unique_devices) <= 1:
            dst_dev = dev if dev is not None else None if len(unique_devices) == 0 else next(iter(unique_devices))
            with ivy.DefaultDevice(ivy.default_device(dst_dev)):
                return ivy.handle_soft_device_variable(*args, fn=fn, **kwargs)
        elif len(unique_devices) > 1:
            raise ivy.utils.exceptions.IvyException(f'Expected all input arrays to be on the same device, but found at least two devices - {devices}, set `ivy.set_soft_device_mode(True)` to handle this problem.')
        return fn(*args, **kwargs)
    _handle_device.handle_device = True
    return _handle_device

def handle_out_argument(fn: Callable) -> Callable:
    if False:
        i = 10
        return i + 15
    handle_out_in_backend = hasattr(fn, 'support_native_out')

    @functools.wraps(fn)
    def _handle_out_argument(*args, out=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Call `fn` with the `out` argument handled correctly for performing an inplace\n        update.\n\n        Parameters\n        ----------\n        args\n            The arguments to be passed to the function.\n\n        out\n            The array to write the result to.\n\n        kwargs\n            The keyword arguments to be passed to the function.\n\n        Returns\n        -------\n            The return of the function, with `out` handled correctly for\n            inplace updates.\n        '
        nonlocal handle_out_in_backend
        if out is None:
            return fn(*args, out=out, **kwargs)
        if ivy.gradients._is_variable(out):
            handle_out_in_backend = False
        if handle_out_in_backend:
            native_out = ivy.to_native(out)
            ret = fn(*args, out=native_out, **kwargs)
            if isinstance(ret, (tuple, list)):
                for i in range(len(ret)):
                    ivy.inplace_update(out[i], ret[i])
                    if ivy.backend == 'torch':
                        _update_torch_views(out[i])
            else:
                ivy.inplace_update(out, ret)
                if ivy.backend == 'torch':
                    _update_torch_views(out)
            return out
        ret = fn(*args, **kwargs)
        if not ivy.is_array(ret) and (not ivy.is_ivy_container(ret)):
            return ivy.nested_multi_map(lambda x, _: ivy.inplace_update(x[0], ivy.astype(x[1], ivy.dtype(x[0]))), [out, ret])
        return ivy.inplace_update(out, ivy.astype(ret, ivy.dtype(out)))
    _handle_out_argument.handle_out_argument = True
    return _handle_out_argument

def _update_torch_views(x, visited_view=None):
    if False:
        for i in range(10):
            print('nop')
    if x._torch_view_refs != []:
        _update_torch_references(x, visited_view)
    if ivy.exists(x._torch_manipulation):
        (parent_tensor, fn_args_kwargs) = x._torch_manipulation
        (fn, args, kwargs) = fn_args_kwargs
        kwargs['copy'] = True
        if fn == 'rot90':
            kwargs = kwargs.copy()
            kwargs['k'] = -kwargs['k']
        parent_tensor.data[()] = ivy.__dict__[fn](x, *args, **kwargs).data
    if ivy.exists(x._torch_base):
        _update_torch_views(x._torch_base, visited_view=x)

def _update_torch_references(x, visited_view=None):
    if False:
        return 10
    for ref in x._torch_view_refs:
        view = ref()
        if ivy.exists(view) and view is not visited_view:
            (parent_tensor, fn_args_kwargs) = view._torch_manipulation
            (fn, args, kwargs) = fn_args_kwargs
            kwargs['copy'] = True
            view.data[()] = ivy.__dict__[fn](parent_tensor, *args, **kwargs).data
            if view._torch_view_refs != []:
                _update_torch_references(view)

def handle_nestable(fn: Callable) -> Callable:
    if False:
        for i in range(10):
            print('nop')
    fn_name = fn.__name__

    @functools.wraps(fn)
    def _handle_nestable(*args, **kwargs):
        if False:
            return 10
        '\n        Call `fn` with the *nestable* property of the function correctly handled. This\n        means mapping the function to the container leaves if any containers are passed\n        in the input.\n\n        Parameters\n        ----------\n        args\n            The arguments to be passed to the function.\n\n        kwargs\n            The keyword arguments to be passed to the function.\n\n        Returns\n        -------\n            The return of the function, with the nestable property handled correctly.\n        '
        if hasattr(ivy.Container, f'_static_{fn_name}'):
            cont_fn = getattr(ivy.Container, f'_static_{fn_name}')
        else:

            def cont_fn(*args, **kwargs):
                if False:
                    while True:
                        i = 10
                return ivy.Container.cont_multi_map_in_function(fn, *args, **kwargs)
        if ivy.nestable_mode and (ivy.nested_any(args, ivy.is_ivy_container, check_nests=True) or ivy.nested_any(kwargs, ivy.is_ivy_container, check_nests=True)):
            return cont_fn(*args, **kwargs)
        return fn(*args, **kwargs)
    _handle_nestable.handle_nestable = True
    return _handle_nestable

def handle_ragged(fn: Callable) -> Callable:
    if False:
        for i in range(10):
            print('nop')

    @functools.wraps(fn)
    def _handle_ragged(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Call `fn` with the *ragged* property of the function correctly handled. This\n        means mapping the function to the RaggedArray arrays if any RaggedArrays are\n        passed in the input.\n\n        Parameters\n        ----------\n        args\n            The arguments to be passed to the function.\n\n        kwargs\n            The keyword arguments to be passed to the function.\n\n        Returns\n        -------\n            The return of the function, with the ragged property handled correctly.\n        '

        def nested_fn(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return ivy.NestedArray.ragged_multi_map_in_function(fn, *args, **kwargs)
        if ivy.nested_any(args, ivy.is_ivy_nested_array, check_nests=True) or ivy.nested_any(kwargs, ivy.is_ivy_nested_array, check_nests=True):
            return nested_fn(*args, **kwargs)
        return fn(*args, **kwargs)
    _handle_ragged.handle_ragged = True
    return _handle_ragged

def handle_partial_mixed_function(fn) -> Callable:
    if False:
        i = 10
        return i + 15

    @functools.wraps(fn)
    def _handle_partial_mixed_function(*args, **kwargs):
        if False:
            return 10
        handle_mixed_in_backend = False
        if not hasattr(fn, 'partial_mixed_handler'):
            handle_mixed_in_backend = True
        else:
            compos = getattr(fn, 'compos')
            condition = getattr(fn, 'partial_mixed_handler')
        if handle_mixed_in_backend or condition(*args, **kwargs):
            return fn(*args, **kwargs)
        return compos(*args, **kwargs)
    _handle_partial_mixed_function.handle_partial_mixed_function = True
    return _handle_partial_mixed_function

def temp_asarray_wrapper(fn: Callable) -> Callable:
    if False:
        print('Hello World!')

    @functools.wraps(fn)
    def _temp_asarray_wrapper(*args, **kwargs):
        if False:
            return 10
        '\n        Convert `Tensor` into `ivy.Array` instances.\n\n        Convert all `Tensor` instances in both the positional and keyword arguments\n        into `ivy.Array` instances, and then call the function with the updated\n        arguments.\n        '

        def _to_ivy_array(x):
            if False:
                for i in range(10):
                    print('nop')
            if hasattr(x, 'ivy_array'):
                return x.ivy_array
            return x
        new_args = ivy.nested_map(_to_ivy_array, args, include_derived={'tuple': True}, shallow=False)
        new_kwargs = ivy.nested_map(_to_ivy_array, kwargs, include_derived={'tuple': True}, shallow=False)
        return fn(*new_args, **new_kwargs)
    _temp_asarray_wrapper.temp_asarray_wrapper = True
    return _temp_asarray_wrapper

def _wrap_function(key: str, to_wrap: Callable, original: Callable, compositional: bool=False) -> Callable:
    if False:
        return 10
    '\n    Apply wrapping to backend implementation `to_wrap` if the original implementation\n    `original` is also wrapped, and if `to_wrap` is not already wrapped. Attributes\n    `handle_nestable` etc are set during wrapping, hence indicate to us whether a\n    certain function has been wrapped or not. Also handles wrapping of the `linalg`\n    namespace.\n\n    Parameters\n    ----------\n    to_wrap\n        the new implementation to potentially wrap\n    original\n        the original implementation of `to_wrap` which tells us which wrappers we need.\n    compositional\n        indicates whether the function being wrapped is compositional\n        (Default Value = ``False``).\n\n    Returns\n    -------\n    ret\n        `to_wrap` appropriately wrapped if `to_wrap` is a function, otherwise just the\n        input is returned.\n    '
    if key == 'linalg':
        for (linalg_k, linalg_v) in to_wrap.__dict__.items():
            if isinstance(linalg_v, FunctionType) and linalg_k.lower() != 'namedtuple' and (linalg_k != 'with_unsupported_dtypes') and (not linalg_k.startswith('_')):
                to_wrap.__dict__[linalg_k] = _wrap_function(linalg_k, linalg_v, ivy.__dict__[linalg_k], compositional=compositional)
        return to_wrap
    if isinstance(to_wrap, FunctionType):
        for attr in original.__dict__.keys():
            if attr.startswith('_') or hasattr(ivy, attr) or attr == 'mixed_backend_wrappers':
                continue
            setattr(to_wrap, attr, getattr(original, attr))
        docstring_attr = ['__annotations__', '__doc__']
        for attr in docstring_attr:
            setattr(to_wrap, attr, getattr(original, attr))
        mixed_fn = hasattr(original, 'mixed_backend_wrappers') and original != to_wrap
        partial_mixed = mixed_fn and hasattr(original, 'handle_partial_mixed_function') and hasattr(to_wrap, 'partial_mixed_handler')
        (add_wrappers, skip_wrappers) = ([], [])
        if mixed_fn:
            backend_wrappers = getattr(original, 'mixed_backend_wrappers')
            add_wrappers = backend_wrappers.get('to_add')
            skip_wrappers = backend_wrappers.get('to_skip')
        for attr in FN_DECORATORS:
            if hasattr(original, attr) and (not hasattr(to_wrap, attr)):
                if partial_mixed and attr == 'handle_partial_mixed_function':
                    to_wrap.compos = original
                    to_wrap = handle_partial_mixed_function(to_wrap)
                if attr not in skip_wrappers:
                    to_wrap = getattr(ivy, attr)(to_wrap)
            if attr in add_wrappers:
                to_wrap = getattr(ivy, attr)(to_wrap)
        if partial_mixed:
            array_spec = to_wrap.compos.__dict__['array_spec']
            for attr in FN_DECORATORS[-1:FN_DECORATORS.index('handle_partial_mixed_function'):-1]:
                if hasattr(to_wrap.compos, attr):
                    to_wrap.compos = to_wrap.compos.__wrapped__
            to_wrap.compos.__dict__['array_spec'] = array_spec
    return to_wrap

def casting_modes_ops(fn, ret_dtype_target=None):
    if False:
        print('Hello World!')

    @functools.wraps(fn)
    def method(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        signature = inspect.signature(fn)
        arg_names = [param.name for param in signature.parameters.values()]
        intersect = set(ivy.function_unsupported_dtypes(fn)).difference(set(ivy.invalid_dtypes))
        if not intersect:
            intersect = set(ivy.function_unsupported_devices_and_dtypes(fn).get(ivy.default_device().split(':')[0], {None})).difference(set(ivy.invalid_dtypes))
            if not intersect:
                return fn(*args, **kwargs)
        to_cast = None
        if 'dtype' in kwargs and kwargs['dtype'] is not None:
            to_cast = kwargs['dtype']
            dtype = caster(kwargs['dtype'], intersect)
            if dtype:
                kwargs['dtype'] = ivy.as_native_dtype(dtype)

        def mini_helper(x):
            if False:
                i = 10
                return i + 15
            if not hasattr(x, 'dtype'):
                return x
            dtype = caster(x, intersect)
            if dtype:
                x = ivy.to_native(ivy.astype(x, ivy.as_native_dtype(dtype)))
            return x
        args = ivy.nested_map(mini_helper, args, include_derived=True)
        kwargs = ivy.nested_map(mini_helper, kwargs)
        if not to_cast and ret_dtype_target:
            for arg in ret_dtype_target:
                if arg:
                    (to_cast, arg_mod) = ivy.promote_types_of_inputs(to_cast, args[arg_names.index(arg)] if arg not in kwargs else kwargs[arg])
                    if arg not in kwargs:
                        args[arg_names.index(arg)] = arg_mod if not ivy.is_array(args[arg_names.index(arg)]) else args[arg_names.index(arg)]
                    else:
                        kwargs[arg] = arg_mod if not ivy.is_array(args[arg_names.index(arg)]) else kwargs[arg]
        return ivy.astype(fn(*args, **kwargs), ivy.to_native(to_cast)) if to_cast else fn(*args, **kwargs)
    return method

def _dtype_from_version(dic, version):
    if False:
        return 10
    if isinstance(version, str):
        version = ivy.functional.frontends.__dict__['versions'][version]
    if isinstance(version, dict):
        version = version['version']
    if not dic:
        raise Exception('No version found in the dictionary')
    if version in dic:
        return dic[version]
    version_tuple = tuple(map(int, version.split('.')))
    for key in dic.keys():
        kl = key.split(' ')
        k1 = tuple(map(int, kl[0].split('.')))
        if 'above' in key and k1 <= version_tuple:
            return dic[key]
        if 'below' in key and k1 >= version_tuple:
            return dic[key]
        if 'to' in key and k1 <= version_tuple <= tuple(map(int, kl[2].split('.'))):
            return dic[key]
    return ()

def _versioned_attribute_factory(attribute_function, base):
    if False:
        return 10

    class VersionedAttributes(base):
        """
        Class which add versioned attributes to a class, inheriting from `base`.

        Create a class which inherits `base` this way if isinstance is
        called on an instance of the class, it will return True if
        testing for the baseclass, such as isinstance(instance, tuple)
        if `base` is tuple.
        """

        def __init__(self):
            if False:
                print('Hello World!')
            self.attribute_function = attribute_function

        def __get__(self, instance=None, owner=None):
            if False:
                while True:
                    i = 10
            return self.attribute_function()

        def __iter__(self):
            if False:
                while True:
                    i = 10
            return iter(self.__get__())

        def __repr__(self):
            if False:
                for i in range(10):
                    print('nop')
            return repr(self.__get__())

        def __bool__(self):
            if False:
                while True:
                    i = 10
            return bool(self.__get__())
    return VersionedAttributes()

def _dtype_device_wrapper_creator(attrib, t):
    if False:
        print('Hello World!')
    '\n    Create a wrapper for a dtype or device attribute.\n\n    The wrapper returns the correct dtype or device for the current version of the\n    backend.\n\n    Parameters\n    ----------\n    attrib\n        The attribute name to be wrapped. for example, "unsupported_dtypes"\n    t\n        The type of the attribute. for example, "tuple"\n\n    Returns\n    -------\n    A wrapper function for the attribute.\n    '

    def _wrapper_outer(version_dict, version, exclusive=True, ret_dtype_target=None):
        if False:
            while True:
                i = 10

        def _wrapped(func):
            if False:
                i = 10
                return i + 15
            val = _versioned_attribute_factory(lambda : _dtype_from_version(version_dict, version), t)
            if hasattr(func, 'override'):
                return func
            if not exclusive:
                setattr(func, 'exclusive', True)
            has_attrib = [attribute for attribute in attribute_dict if hasattr(func, attribute)] or False
            if has_attrib:
                for attribs in has_attrib:
                    if not (attrib == attribs or (attrib, attribs) in attribute_conflict):
                        setattr(func, attrib, val)
                        setattr(func, 'dictionary_info', (version_dict, version))
                    elif hasattr(func, 'exclusive'):
                        if attrib == attribs:
                            old_version_dict = getattr(func, 'dictionary_info')[0]
                            old_version_dict.update(version_dict)
                            val = _versioned_attribute_factory(lambda : _dtype_from_version(version_dict, old_version_dict), t)
                            setattr(func, attrib, val)
                        else:
                            pass
            else:
                if not val and attrib.startswith('supported'):
                    setattr(func, f'un{attrib}', val)
                else:
                    setattr(func, attrib, val)
                setattr(func, 'dictionary_info', (version_dict, version))
            if 'frontends' in func.__module__:
                return func
            return casting_modes_ops(func, ret_dtype_target=ret_dtype_target)
        return _wrapped
    return _wrapper_outer

def _leaf_has_nans(x):
    if False:
        while True:
            i = 10
    if isinstance(x, ivy.Container):
        return x.has_nans()
    elif ivy.is_array(x):
        return ivy.isnan(x).any()
    elif np.isnan(x):
        return True
    return False

def _nest_has_nans(x):
    if False:
        i = 10
        return i + 15
    return ivy.nested_any(x, _leaf_has_nans)

def handle_nans(fn: Callable) -> Callable:
    if False:
        i = 10
        return i + 15

    @functools.wraps(fn)
    def _handle_nans(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Check for the existence of nans in all arrays in the `args` and `kwargs`.\n\n        The presence of nans is then handled depending on the enabled `nan_policy`.\n\n        Following policies apply:\n        raise_exception: raises an exception in case nans are present\n        warns: warns a user in case nans are present\n        nothing: does nothing\n\n        Parameters\n        ----------\n        args\n            The arguments to be passed to the function.\n        kwargs\n            The keyword arguments to be passed to the function.\n\n        Returns\n        -------\n            The return of the function, with handling of inputs based\n            on the selected `nan_policy`.\n        '
        nan_policy = ivy.nan_policy
        if nan_policy == 'nothing':
            return fn(*args, **kwargs)
        result = _nest_has_nans(args) or _nest_has_nans(kwargs)
        if result:
            if nan_policy == 'raise_exception':
                raise ivy.utils.exceptions.IvyException('Nans are not allowed in `raise_exception` policy.')
            elif nan_policy == 'warns':
                logging.warning('Nans are present in the input.')
        return fn(*args, **kwargs)
    _handle_nans.handle_nans = True
    return _handle_nans

def handle_complex_input(fn: Callable) -> Callable:
    if False:
        return 10

    @functools.wraps(fn)
    def _handle_complex_input(inp, *args, complex_mode: Literal['split', 'magnitude', 'jax']='jax', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check whether the first positional argument is an array of complex type, and if\n        so handle it according to the provided `complex_mode`.\n\n        The options are:\n        `"jax"` (default): emulate the behaviour of the JAX framework. If the function\n            has a `jax_like` attribute then this will be used to decide on the\n            behaviour (see below) and if not, then the entire array will be passed to\n            the function.\n        `"split"`: execute the function separately on the real and imaginary parts of\n            the input.\n        `"magnitude"`: execute the function on the magnitude of the input, and keep the\n            angle constant.\n\n        The `jax_like` attribute (which should be added to the function itself, and not\n        passed as a parameter) has the following options:\n        `"entire"` (default): pass the entire input to the function. This is best used\n            for purely mathematical operators which are already well defined on complex\n            inputs, as many backends will throw exceptions otherwise.\n        `"split"`: as the `"split"` option for `complex_mode`\n        `"magnitude"`: as the `"magnitude"` option for `complex_mode`\n        A callable function: the function will be called instead of the originally\n            decorated function. It will be passed `inp` and `*args` as positional\n            arguments, and the original `**kwargs` plus `fn_original` as keyword\n            arguments. The latter is the original function, in case the `jax_like`\n            function wishes to call it.\n\n        Parameters\n        ----------\n        inp\n            The first positional argument to the function, which is expected to be an\n            :class:`ivy.Array`.\n        args\n            The remaining positional arguments to be passed to the function.\n        complex_mode\n            Optional argument which specifies the method that will be used to handle\n            the input, if it is complex.\n        kwargs\n            The keyword arguments to be passed to the function.\n\n        Returns\n        -------\n            The return of the function, with handling of inputs based\n            on the selected `complex_mode`.\n\n        Examples\n        --------\n        Using the default `jax_like` behaviour\n\n        >>> @handle_complex_input\n        >>> def my_func(inp):\n        >>>     return ivy.ones_like(inp)\n\n        >>> x = ivy.array([1+1j, 3+4j, 5+12j])\n        >>> my_func(x)  # equivalent to setting complex_mode="jax"\n        ivy.array([1.+0.j, 1.+0.j, 1.+0.j])\n\n        >>> my_func(x, complex_mode="split")\n        ivy.array([1.+1.j, 1.+1.j, 1.+1.j])\n\n        >>> my_func(x, complex_mode="magnitude")\n        ivy.array([0.70710681+0.70710675j, 0.60000001+0.79999999j,\n                   0.38461535+0.92307694j])\n\n        Using non-default `jax_like` behaviour\n\n        >>> @handle_complex_input\n        >>> def my_func(inp):\n        >>>     return ivy.ones_like(inp)\n        >>> my_func.jax_like = "split"\n        >>> my_func(x, complex_mode="jax")\n        ivy.array([1.+1.j, 1.+1.j, 1.+1.j])\n\n        Using callable `jax_like` behaviour\n\n        >>> def _my_func_jax_like(inp, fn_original=None):\n        >>>     return fn_original(inp) * 3j\n        >>> @handle_complex_input\n        >>> def my_func(inp):\n        >>>     return ivy.ones_like(inp)\n        >>> my_func.jax_like = _my_func_jax_like\n        >>> my_func(x, complex_mode="jax")\n        ivy.array([0.+3.j, 0.+3.j, 0.+3.j])\n        '
        if not ivy.is_complex_dtype(inp):
            return fn(inp, *args, **kwargs)
        jax_like = fn.jax_like if hasattr(fn, 'jax_like') else 'entire'
        if complex_mode == 'split' or (complex_mode == 'jax' and jax_like == 'split'):
            real_inp = ivy.real(inp).data
            imag_inp = ivy.imag(inp).data
            if 'out' in kwargs and kwargs['out'] is not None:
                out = kwargs.pop('out')
                real_ret = fn(real_inp, *args, out=ivy.real(out), **kwargs)
                imag_ret = fn(imag_inp, *args, out=ivy.imag(out), **kwargs)
            else:
                real_ret = fn(real_inp, *args, **kwargs)
                imag_ret = fn(imag_inp, *args, **kwargs)
            return ivy.add(real_ret, ivy.multiply(ivy.array(1j, dtype=inp.dtype), imag_ret))
        elif complex_mode == 'magnitude' or (complex_mode == 'jax' and jax_like == 'magnitude'):
            mag_inp = ivy.abs(inp).data
            angle_inp = ivy.angle(inp).data
            return ivy.multiply(fn(mag_inp, *args, **kwargs), ivy.exp(ivy.multiply(1j, angle_inp)))
        elif complex_mode == 'jax' and jax_like == 'entire':
            return fn(inp, *args, **kwargs)
        elif complex_mode == 'jax':
            return jax_like(inp, *args, **kwargs, fn_original=fn)
        else:
            raise IvyValueError(f"complex_mode '{complex_mode}' is not recognised.")
    _handle_complex_input.handle_complex_input = True
    return _handle_complex_input

def handle_backend_invalid(fn: Callable) -> Callable:
    if False:
        for i in range(10):
            print('nop')

    @functools.wraps(fn)
    def _handle_backend_invalid(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Check if any of the arguments (or nested arguments) passed to the function are\n        instances of ivy.Array or ivy.NativeArray. If so, it returns the function. If\n        not, it raises an InvalidBackendException.\n\n        Parameters\n        ----------\n        args\n            The arguments to be passed to the function.\n\n        kwargs\n            The keyword arguments to be passed to the function.\n\n        Returns\n        -------\n            The return of the function if the current\n            backend matches the argument backend.\n            If not, it raises an InvalidBackendException\n        '
        array_indices = ivy.nested_argwhere([args, kwargs], lambda x: isinstance(x, ivy.Array))
        array_vals = ivy.multi_index_nest([args, kwargs], array_indices)

        def func(x):
            if False:
                for i in range(10):
                    print('nop')
            target_backend = ivy.utils.backend.handler._determine_backend_from_args(x)
            if target_backend is not None and ivy.backend != '' and (ivy.current_backend_str() != target_backend.backend):
                raise ivy.utils.exceptions.IvyInvalidBackendException(f'Operation not allowed. Array was instantiated with backend {target_backend.backend}. But current backend is {ivy.backend}. Please set dynamic=True for the array if you want to convert it to the target backend')
            return x
        ivy.nested_map(func, array_vals, include_derived=True)
        return fn(*args, **kwargs)
    _handle_backend_invalid.handle_backend_invalid = True
    return _handle_backend_invalid
attribute_dict = {'unsupported_dtypes', 'supported_dtypes', 'unsupported_devices', 'supported_devices', 'unsupported_device_and_dtype', 'supported_device_and_dtype'}
attribute_conflict = {('unsupported_devices', 'supported_devices'), ('supported_devices', 'unsupported_devices'), ('unsupported_device_and_dtype', 'supported_device_and_dtype'), ('supported_device_and_dtype', 'unsupported_device_and_dtype')}

def globals_getter_func(x=None):
    if False:
        i = 10
        return i + 15
    if not x:
        return globals()
    else:
        globals()[x[0]] = x[1]

class with_unsupported_dtypes(contextlib.ContextDecorator):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self.args = args
        self.kwargs = kwargs
        self.globals = {}

    def __call__(self, func=None):
        if False:
            print('Hello World!')
        if func:
            return _dtype_device_wrapper_creator('unsupported_dtypes', tuple)(*self.args, **self.kwargs)(func)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self.globals = globals_getter_func().copy()

    def __exit__(self, *exec):
        if False:
            while True:
                i = 10
        new_globals = set(globals_getter_func().keys())
        diff = new_globals.difference(set(self.globals))
        for item in diff:
            if globals_getter_func().get(item, None):
                if isinstance(globals_getter_func()[item], FunctionType):
                    globals_getter_func([item, _dtype_device_wrapper_creator('unsupported_dtypes', tuple)(*self.args, **self.kwargs)(globals_getter_func()[item])])

class with_supported_dtypes(contextlib.ContextDecorator):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.args = args
        self.kwargs = kwargs
        self.globals = {}

    def __call__(self, func=None):
        if False:
            for i in range(10):
                print('nop')
        if func:
            return _dtype_device_wrapper_creator('supported_dtypes', tuple)(*self.args, **self.kwargs)(func)

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self.globals = globals_getter_func().copy()

    def __exit__(self, *exec):
        if False:
            while True:
                i = 10
        new_globals = set(globals_getter_func().keys())
        diff = new_globals.difference(set(self.globals))
        for item in diff:
            if globals_getter_func().get(item, None):
                if isinstance(globals_getter_func()[item], FunctionType):
                    globals_getter_func([item, _dtype_device_wrapper_creator('supported_dtypes', tuple)(*self.args, **self.kwargs)(globals_getter_func()[item])])

class with_unsupported_devices(contextlib.ContextDecorator):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        self.args = args
        self.kwargs = kwargs
        self.globals = {}

    def __call__(self, func=None):
        if False:
            return 10
        if func:
            return _dtype_device_wrapper_creator('unsupported_devices', tuple)(*self.args, **self.kwargs)(func)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self.globals = globals_getter_func().copy()

    def __exit__(self, *exec):
        if False:
            for i in range(10):
                print('nop')
        new_globals = set(globals_getter_func().keys())
        diff = new_globals.difference(set(self.globals))
        for item in diff:
            if globals_getter_func().get(item, None):
                if isinstance(globals_getter_func()[item], FunctionType):
                    globals_getter_func([item, _dtype_device_wrapper_creator('unsupported_devices', tuple)(*self.args, **self.kwargs)(globals_getter_func()[item])])

class with_supported_devices(contextlib.ContextDecorator):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.args = args
        self.kwargs = kwargs
        self.globals = {}

    def __call__(self, func=None):
        if False:
            while True:
                i = 10
        if func:
            return _dtype_device_wrapper_creator('supported_devices', tuple)(*self.args, **self.kwargs)(func)

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self.globals = globals_getter_func().copy()

    def __exit__(self, *exec):
        if False:
            print('Hello World!')
        new_globals = set(globals_getter_func().keys())
        diff = new_globals.difference(set(self.globals))
        for item in diff:
            if globals_getter_func().get(item, None):
                if isinstance(globals_getter_func()[item], FunctionType):
                    globals_getter_func([item, _dtype_device_wrapper_creator('supported_devices', tuple)(*self.args, **self.kwargs)(globals_getter_func()[item])])

class with_unsupported_device_and_dtypes(contextlib.ContextDecorator):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        dicti = args[0]
        self.kwargs = kwargs
        for key in dicti.keys():
            nested_dic = {}
            for nested_key in dicti[key].keys():
                if nested_key == 'all':
                    nested_dic['cpu'] = dicti[key].get('cpu', ()) + tuple(dicti[key]['all'])
                    nested_dic['tpu'] = dicti[key].get('tpu', ()) + tuple(dicti[key]['all'])
                    nested_dic['gpu'] = dicti[key].get('gpu', ()) + tuple(dicti[key]['all'])
                else:
                    nested_dic[nested_key] = tuple(dicti[key][nested_key])
            dicti[key] = nested_dic
        args = (dicti, args[1])
        self.args = args
        self.globals = {}

    def __call__(self, func=None):
        if False:
            print('Hello World!')
        if func:
            return _dtype_device_wrapper_creator('unsupported_device_and_dtype', tuple)(*self.args, **self.kwargs)(func)

    def __enter__(self):
        if False:
            return 10
        self.globals = globals_getter_func().copy()

    def __exit__(self, *exec):
        if False:
            i = 10
            return i + 15
        new_globals = set(globals_getter_func().keys())
        diff = new_globals.difference(set(self.globals.keys()))
        for item in diff:
            if globals_getter_func().get(item, None):
                if isinstance(globals_getter_func()[item], FunctionType):
                    globals_getter_func([item, _dtype_device_wrapper_creator('unsupported_device_and_dtype', tuple)(*self.args, **self.kwargs)(globals_getter_func()[item])])

class with_supported_device_and_dtypes(contextlib.ContextDecorator):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        dicti = args[0]
        self.kwargs = kwargs
        for key in dicti.keys():
            nested_dic = {}
            for nested_key in dicti[key].keys():
                if nested_key == 'all':
                    nested_dic['cpu'] = dicti[key].get('cpu', ()) + tuple(dicti[key]['all'])
                    nested_dic['tpu'] = dicti[key].get('tpu', ()) + tuple(dicti[key]['all'])
                    nested_dic['gpu'] = dicti[key].get('gpu', ()) + tuple(dicti[key]['all'])
                else:
                    nested_dic[nested_key] = tuple(dicti[key][nested_key])
            dicti[key] = nested_dic
        args = (dicti, args[1])
        self.args = args
        self.globals = {}

    def __call__(self, func=None):
        if False:
            print('Hello World!')
        if func:
            return _dtype_device_wrapper_creator('supported_device_and_dtype', tuple)(*self.args, **self.kwargs)(func)

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self.globals = globals_getter_func().copy()

    def __exit__(self, *exec):
        if False:
            return 10
        new_globals = set(globals_getter_func().keys())
        diff = new_globals.difference(set(self.globals))
        for item in diff:
            if globals_getter_func().get(item, None):
                if isinstance(globals_getter_func()[item], FunctionType):
                    globals_getter_func([item, _dtype_device_wrapper_creator('supported_device_and_dtype', tuple)(*self.args, **self.kwargs)(globals_getter_func()[item])])

class override(contextlib.ContextDecorator):

    def __call__(self, func=None):
        if False:
            print('Hello World!')
        if func:
            setattr(func, 'override', 'override')
            return func

    def __enter__(self):
        if False:
            return 10
        self.globals = globals_getter_func().copy()

    def __exit__(self, *exec):
        if False:
            print('Hello World!')
        new_globals = set(globals().keys())
        diff = new_globals.difference(set(self.globals))
        for item in diff:
            if globals_getter_func().get(item, None):
                if isinstance(globals_getter_func()[item], FunctionType):
                    globals_getter_func([item, 'override'])