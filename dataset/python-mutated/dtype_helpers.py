import numpy as np
from hypothesis import strategies as st
from typing import Optional
import ivy
from ..pipeline_helper import BackendHandler, get_frontend_config
from . import number_helpers as nh
from . import array_helpers as ah
from .. import globals as test_globals
from ..globals import mod_backend
_dtype_kind_keys = {'valid', 'numeric', 'float', 'unsigned', 'integer', 'signed_integer', 'complex', 'real_and_complex', 'float_and_integer', 'float_and_complex', 'bool'}

def _get_fn_dtypes(framework: str, kind='valid', mixed_fn_dtypes='compositional'):
    if False:
        return 10
    all_devices_dtypes = test_globals.CURRENT_RUNNING_TEST.supported_device_dtypes[framework]
    if mixed_fn_dtypes in all_devices_dtypes:
        all_devices_dtypes = all_devices_dtypes[mixed_fn_dtypes]
    return all_devices_dtypes[test_globals.CURRENT_DEVICE_STRIPPED][kind]

def _get_type_dict(framework: str, kind: str, is_frontend_test=False):
    if False:
        while True:
            i = 10
    if mod_backend[framework]:
        (proc, input_queue, output_queue) = mod_backend[framework]
        input_queue.put(('_get_type_dict_helper', framework, kind, is_frontend_test))
        return output_queue.get()
    else:
        return _get_type_dict_helper(framework, kind, is_frontend_test)

def _get_type_dict_helper(framework, kind, is_frontend_test):
    if False:
        print('Hello World!')
    if is_frontend_test:
        framework_module = get_frontend_config(framework).supported_dtypes
    elif ivy.current_backend_str() == framework:
        framework_module = ivy
    else:
        with BackendHandler.update_backend(framework) as ivy_backend:
            framework_module = ivy_backend
    if kind == 'valid':
        return framework_module.valid_dtypes
    if kind == 'numeric':
        return framework_module.valid_numeric_dtypes
    if kind == 'integer':
        return framework_module.valid_int_dtypes
    if kind == 'float':
        return framework_module.valid_float_dtypes
    if kind == 'unsigned':
        return framework_module.valid_uint_dtypes
    if kind == 'signed_integer':
        return tuple(set(framework_module.valid_int_dtypes).difference(framework_module.valid_uint_dtypes))
    if kind == 'complex':
        return framework_module.valid_complex_dtypes
    if kind == 'real_and_complex':
        return tuple(set(framework_module.valid_numeric_dtypes).union(framework_module.valid_complex_dtypes))
    if kind == 'float_and_complex':
        return tuple(set(framework_module.valid_float_dtypes).union(framework_module.valid_complex_dtypes))
    if kind == 'float_and_integer':
        return tuple(set(framework_module.valid_float_dtypes).union(framework_module.valid_int_dtypes))
    if kind == 'bool':
        return tuple(set(framework_module.valid_dtypes).difference(framework_module.valid_numeric_dtypes))
    raise RuntimeError(f'{kind} is an unknown kind!')

@st.composite
def get_dtypes(draw, kind='valid', index=0, mixed_fn_compos=True, full=True, none=False, key=None, prune_function=True):
    if False:
        i = 10
        return i + 15
    '\n    Draws a valid dtypes for the test function. For frontend tests, it draws the data\n    types from the intersection between backend framework data types and frontend\n    framework dtypes, otherwise, draws it from backend framework data types.\n\n    Parameters\n    ----------\n    draw\n        special function that draws data randomly (but is reproducible) from a given\n        data-set (ex. list).\n    kind\n        Supported types are integer, float, valid, numeric, signed_integer, complex,\n        real_and_complex, float_and_complex, bool, and unsigned\n    index\n        list indexing in case a test needs to be skipped for a particular dtype(s)\n    mixed_fn_compos\n        boolean if True, the function will return the dtypes of the compositional\n        implementation for mixed partial functions and if False, it will return\n        the dtypes of the primary implementation.\n    full\n        returns the complete list of valid types\n    none\n        allow none in the list of valid types\n    key\n        if provided, a shared value will be drawn from the strategy and passed to the\n        function as the keyword argument with the given name.\n    prune_function\n        if True, the function will prune the data types to only include the ones that\n        are supported by the current function. If False, the function will return all\n        the data types supported by the current backend.\n\n    Returns\n    -------\n    ret\n        A strategy that draws dtype strings\n\n    Examples\n    --------\n    >>> get_dtypes()\n    [\'float16\',\n        \'uint8\',\n        \'complex128\',\n        \'bool\',\n        \'uint32\',\n        \'float64\',\n        \'int8\',\n        \'int16\',\n        \'complex64\',\n        \'float32\',\n        \'int32\',\n        \'uint16\',\n        \'int64\',\n        \'uint64\']\n\n    >>> get_dtypes(kind=\'valid\', full=False)\n    [\'int16\']\n\n    >>> get_dtypes(kind=\'valid\', full=False)\n    [\'uint16\']\n\n    >>> get_dtypes(kind=\'numeric\', full=False)\n    [\'complex64\']\n\n    >>> get_dtypes(kind=\'float\', full=False, key="leaky_relu")\n    [\'float16\']\n\n    >>> get_dtypes(kind=\'float\', full=False, key="searchsorted")\n    [\'bfloat16\']\n\n    >>> get_dtypes(kind=\'float\', full=False, key="dtype")\n    [\'float32\']\n\n    >>> get_dtypes("numeric", prune_function=False)\n    [\'int16\']\n\n    >>> get_dtypes("valid", prune_function=False)\n    [\'uint32\']\n\n    >>> get_dtypes("valid", prune_function=False)\n    [\'complex128\']\n\n    >>> get_dtypes("valid", prune_function=False)\n    [\'bool\']\n\n    >>> get_dtypes("valid", prune_function=False)\n    [\'float16\']\n    '
    mixed_fn_dtypes = 'compositional' if mixed_fn_compos else 'primary'
    if prune_function:
        retrieval_fn = _get_fn_dtypes
        if test_globals.CURRENT_RUNNING_TEST is not test_globals._Notsetval:
            valid_dtypes = set(retrieval_fn(test_globals.CURRENT_BACKEND, mixed_fn_dtypes=mixed_fn_dtypes, kind=kind))
        else:
            raise RuntimeError('No function is set to prune, calling prune_function=True without a function is redundant.')
    else:
        retrieval_fn = _get_type_dict
        valid_dtypes = set(retrieval_fn(test_globals.CURRENT_BACKEND, kind))
    if test_globals.CURRENT_FRONTEND is not test_globals._Notsetval:
        frontend_dtypes = _get_type_dict_helper(test_globals.CURRENT_FRONTEND, kind, True)
        valid_dtypes = valid_dtypes.intersection(frontend_dtypes)
    ground_truth_is_set = test_globals.CURRENT_GROUND_TRUTH_BACKEND is not test_globals._Notsetval
    if ground_truth_is_set:
        valid_dtypes = valid_dtypes.intersection(retrieval_fn(test_globals.CURRENT_GROUND_TRUTH_BACKEND, kind=kind))
    valid_dtypes = list(valid_dtypes)
    if none:
        valid_dtypes.append(None)
    if full:
        return valid_dtypes[index:]
    if key is None:
        return [draw(st.sampled_from(valid_dtypes[index:]))]
    return [draw(st.shared(st.sampled_from(valid_dtypes[index:]), key=key))]

@st.composite
def array_dtypes(draw, *, num_arrays=st.shared(nh.ints(min_value=1, max_value=4), key='num_arrays'), available_dtypes=get_dtypes('valid'), shared_dtype=False, array_api_dtypes=False):
    if False:
        print('Hello World!')
    '\n    Draws a list of data types.\n\n    Parameters\n    ----------\n    draw\n        special function that draws data randomly (but is reproducible) from a given\n        data-set (ex. list).\n    num_arrays\n        number of data types to be drawn.\n    available_dtypes\n        universe of available data types.\n    shared_dtype\n        if True, all data types in the list are same.\n    array_api_dtypes\n        if True, use data types that can be promoted with the array_api_promotion\n        table.\n\n    Returns\n    -------\n        A strategy that draws a list of data types.\n\n    Examples\n    --------\n    >>> array_dtypes(\n    ...     available_dtypes=get_dtypes("numeric"),\n    ...     shared_dtype=True,\n    ... )\n    [\'float64\']\n\n    >>> array_dtypes(\n    ...     available_dtypes=get_dtypes("numeric"),\n    ...     shared_dtype=True,\n    ... )\n    [\'int8\', \'int8\']\n\n    >>> array_dtypes(\n    ...     available_dtypes=get_dtypes("numeric"),\n    ...     shared_dtype=True,\n    ... )\n    [\'int32\', \'int32\', \'int32\', \'int32\']\n\n    >>> array_dtypes(\n    ...     num_arrays=5,\n    ...     available_dtypes=get_dtypes("valid"),\n    ...     shared_dtype=False,\n    ... )\n    [\'int8\', \'float64\', \'complex64\', \'int8\', \'bool\']\n\n    >>> array_dtypes(\n    ...     num_arrays=5,\n    ...     available_dtypes=get_dtypes("valid"),\n    ...     shared_dtype=False,\n    ... )\n    [\'bool\', \'complex64\', \'bool\', \'complex64\', \'bool\']\n\n    >>> array_dtypes(\n    ...     num_arrays=5,\n    ...     available_dtypes=get_dtypes("valid"),\n    ...     shared_dtype=False,\n    ... )\n    [\'float64\', \'int8\', \'float64\', \'int8\', \'float64\']\n    '
    if isinstance(available_dtypes, st._internal.SearchStrategy):
        available_dtypes = draw(available_dtypes)
    if not isinstance(num_arrays, int):
        num_arrays = draw(num_arrays)
    if num_arrays == 1:
        dtypes = draw(ah.list_of_size(x=st.sampled_from(available_dtypes), size=1))
    elif shared_dtype:
        dtypes = draw(ah.list_of_size(x=st.sampled_from(available_dtypes), size=1))
        dtypes = [dtypes[0] for _ in range(num_arrays)]
    else:
        unwanted_types = set(ivy.all_dtypes).difference(set(available_dtypes))
        if array_api_dtypes:
            pairs = ivy.array_api_promotion_table.keys()
        else:
            pairs = ivy.promotion_table.keys()
        [pair for pair in pairs if all((d in available_dtypes for d in pair))]
        available_dtypes = [pair for pair in pairs if not any((d in pair for d in unwanted_types))]
        dtypes = list(draw(st.sampled_from(available_dtypes)))
        if num_arrays > 2:
            dtypes += [dtypes[i % 2] for i in range(num_arrays - 2)]
    return dtypes

@st.composite
def get_castable_dtype(draw, available_dtypes, dtype: str, x: Optional[list]=None):
    if False:
        return 10
    '\n    Draws castable dtypes for the given dtype based on the current backend.\n\n    Parameters\n    ----------\n    draw\n        Special function that draws data randomly (but is reproducible) from a given\n        data-set (ex. list).\n    available_dtypes\n        Castable data types are drawn from this list randomly.\n    dtype\n        Data type from which to cast.\n    x\n        Optional list of values to cast.\n\n    Returns\n    -------\n    ret\n        A tuple of inputs and castable dtype.\n    '
    cast_dtype = draw(st.sampled_from(available_dtypes).filter(lambda value: cast_filter(value, dtype=dtype, x=x)))
    if x is None:
        return (dtype, cast_dtype)
    return (dtype, x, cast_dtype)

def cast_filter(d, dtype, x):
    if False:
        print('Hello World!')
    if mod_backend[test_globals.CURRENT_BACKEND]:
        (proc, input_queue, output_queue) = mod_backend[test_globals.CURRENT_BACKEND]
        input_queue.put(('cast_filter_helper', d, dtype, x, test_globals.CURRENT_BACKEND))
        return output_queue.get()
    else:
        return cast_filter_helper(d, dtype, x, test_globals.CURRENT_BACKEND)

def cast_filter_helper(d, dtype, x, current_backend):
    if False:
        while True:
            i = 10
    with BackendHandler.update_backend(current_backend) as ivy_backend:

        def bound_dtype_bits(d):
            if False:
                for i in range(10):
                    print('nop')
            return ivy_backend.dtype_bits(d) / 2 if ivy_backend.is_complex_dtype(d) else ivy_backend.dtype_bits(d)
        if ivy_backend.is_int_dtype(d):
            max_val = ivy_backend.iinfo(d).max
            min_val = ivy_backend.iinfo(d).min
        elif ivy_backend.is_float_dtype(d) or ivy_backend.is_complex_dtype(d):
            max_val = ivy_backend.finfo(d).max
            min_val = ivy_backend.finfo(d).min
        else:
            max_val = 1
            min_val = -1
        if x is None:
            if ivy_backend.is_int_dtype(dtype):
                max_x = ivy_backend.iinfo(dtype).max
                min_x = ivy_backend.iinfo(dtype).min
            elif ivy_backend.is_float_dtype(dtype) or ivy_backend.is_complex_dtype(dtype):
                max_x = ivy_backend.finfo(dtype).max
                min_x = ivy_backend.finfo(dtype).min
            else:
                max_x = 1
                min_x = -1
        else:
            max_x = np.max(np.asarray(x))
            min_x = np.min(np.asarray(x))
        return max_x <= max_val and min_x >= min_val and (bound_dtype_bits(d) >= bound_dtype_bits(dtype)) and (ivy_backend.is_complex_dtype(d) or not ivy_backend.is_complex_dtype(dtype)) and (min_x > 0 or not ivy_backend.is_uint_dtype(dtype))