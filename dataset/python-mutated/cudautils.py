from pickle import dumps
import cachetools
from numba import cuda
from numba.np import numpy_support
from cudf.utils._numba import _CUDFNumbaConfig

@cuda.jit
def gpu_window_sizes_from_offset(arr, window_sizes, offset):
    if False:
        return 10
    i = cuda.grid(1)
    j = i
    if i < arr.size:
        while j > -1:
            if arr[i] - arr[j] >= offset:
                break
            j -= 1
        window_sizes[i] = i - j

def window_sizes_from_offset(arr, offset):
    if False:
        return 10
    window_sizes = cuda.device_array(shape=arr.shape, dtype='int32')
    if arr.size > 0:
        with _CUDFNumbaConfig():
            gpu_window_sizes_from_offset.forall(arr.size)(arr, window_sizes, offset)
    return window_sizes

@cuda.jit
def gpu_grouped_window_sizes_from_offset(arr, window_sizes, group_starts, offset):
    if False:
        i = 10
        return i + 15
    i = cuda.grid(1)
    j = i
    if i < arr.size:
        while j > group_starts[i] - 1:
            if arr[i] - arr[j] >= offset:
                break
            j -= 1
        window_sizes[i] = i - j

def grouped_window_sizes_from_offset(arr, group_starts, offset):
    if False:
        print('Hello World!')
    window_sizes = cuda.device_array(shape=arr.shape, dtype='int32')
    if arr.size > 0:
        with _CUDFNumbaConfig():
            gpu_grouped_window_sizes_from_offset.forall(arr.size)(arr, window_sizes, group_starts, offset)
    return window_sizes
_udf_code_cache: cachetools.LRUCache = cachetools.LRUCache(maxsize=32)

def make_cache_key(udf, sig):
    if False:
        return 10
    '\n    Build a cache key for a user defined function. Used to avoid\n    recompiling the same function for the same set of types\n    '
    codebytes = udf.__code__.co_code
    constants = udf.__code__.co_consts
    names = udf.__code__.co_names
    if udf.__closure__ is not None:
        cvars = tuple((x.cell_contents for x in udf.__closure__))
        cvarbytes = dumps(cvars)
    else:
        cvarbytes = b''
    return (names, constants, codebytes, cvarbytes, sig)

def compile_udf(udf, type_signature):
    if False:
        i = 10
        return i + 15
    'Compile ``udf`` with `numba`\n\n    Compile a python callable function ``udf`` with\n    `numba.cuda.compile_ptx_for_current_device(device=True)` using\n    ``type_signature`` into CUDA PTX together with the generated output type.\n\n    The output is expected to be passed to the PTX parser in `libcudf`\n    to generate a CUDA device function to be inlined into CUDA kernels,\n    compiled at runtime and launched.\n\n    Parameters\n    ----------\n    udf:\n      a python callable function\n\n    type_signature:\n      a tuple that specifies types of each of the input parameters of ``udf``.\n      The types should be one in `numba.types` and could be converted from\n      numpy types with `numba.numpy_support.from_dtype(...)`.\n\n    Returns\n    -------\n    ptx_code:\n      The compiled CUDA PTX\n\n    output_type:\n      An numpy type\n\n    '
    import cudf.core.udf
    key = make_cache_key(udf, type_signature)
    res = _udf_code_cache.get(key)
    if res:
        return res
    (ptx_code, return_type) = cuda.compile_ptx_for_current_device(udf, type_signature, device=True)
    if not isinstance(return_type, cudf.core.udf.masked_typing.MaskedType):
        output_type = numpy_support.as_dtype(return_type).type
    else:
        output_type = return_type
    res = (ptx_code, output_type)
    _udf_code_cache[key] = res
    return res