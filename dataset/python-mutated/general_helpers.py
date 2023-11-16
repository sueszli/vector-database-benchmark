from hypothesis import assume, strategies as st
from typing import Tuple
from functools import lru_cache
import math
import numpy as np
import ivy
from . import array_helpers, number_helpers, dtype_helpers
from ..pipeline_helper import WithBackendContext
from ivy.functional.ivy.layers import _deconv_length
from ..globals import mod_backend

def matrix_is_stable(x, cond_limit=30):
    if False:
        return 10
    '\n    Check if a matrix is numerically stable or not.\n\n    Used to avoid numerical instabilities in further computationally heavy calculations.\n\n    Parameters\n    ----------\n    x\n        The original matrix whose condition number is to be determined.\n    cond_limit\n        The greater the condition number, the more ill-conditioned the matrix\n        will be, the more it will be prone to numerical instabilities.\n\n        There is no rule of thumb for what the exact condition number\n        should be to consider a matrix ill-conditioned(prone to numerical errors).\n        But, if the condition number is "1", the matrix is perfectly said to be a\n        well-conditioned matrix which will not be prone to any type of numerical\n        instabilities in further calculations, but that would probably be a\n        very simple matrix.\n\n        The cond_limit should start with "30", gradually decreasing it according\n        to our use, lower cond_limit would result in more numerically stable\n        matrices but more simple matrices.\n\n        The limit should always be in the range "1-30", greater the number greater\n        the computational instability. Should not increase 30, it leads to strong\n        multi-collinearity which leads to singularity.\n\n    Returns\n    -------\n    ret\n        If True, the matrix is suitable for further numerical computations.\n    '
    return np.all(np.linalg.cond(x.astype('float64')) <= cond_limit)

@lru_cache(None)
def apply_safety_factor(dtype, *, backend: str, min_value=None, max_value=None, abs_smallest_val=None, small_abs_safety_factor=1.1, large_abs_safety_factor=1.1, safety_factor_scale='linear'):
    if False:
        while True:
            i = 10
    "\n    Apply safety factor scaling to numeric data type.\n\n    Parameters\n    ----------\n    dtype\n        the data type to apply safety factor scaling to.\n    min_value\n        the minimum value of the data type.\n    max_value\n        the maximum value of the data type.\n    abs_smallest_val\n        the absolute smallest representable value of the data type.\n    large_abs_safety_factor\n        the safety factor to apply to the maximum value.\n    small_abs_safety_factor\n        the safety factor to apply to the minimum value.\n    safety_factor_scale\n        the scale to apply the safety factor to, either 'linear' or 'log'.\n\n    Returns\n    -------\n        A tuple of the minimum value, maximum value and absolute smallest representable\n    "
    assert small_abs_safety_factor >= 1, 'small_abs_safety_factor must be >= 1'
    assert large_abs_safety_factor >= 1, 'large_value_safety_factor must be >= 1'
    if 'float' in dtype or 'complex' in dtype:
        kind_dtype = 'float'
        if mod_backend[backend]:
            (proc, input_queue, output_queue) = mod_backend[backend]
            input_queue.put(('dtype_info_helper', backend, kind_dtype, dtype))
            dtype_info = output_queue.get()
        else:
            dtype_info = general_helpers_dtype_info_helper(backend=backend, kind_dtype=kind_dtype, dtype=dtype)
    elif 'int' in dtype:
        kind_dtype = 'int'
        if mod_backend[backend]:
            (proc, input_queue, output_queue) = mod_backend[backend]
            input_queue.put(('dtype_info_helper', backend, kind_dtype, dtype))
            dtype_info = output_queue.get()
        else:
            dtype_info = general_helpers_dtype_info_helper(backend=backend, kind_dtype=kind_dtype, dtype=dtype)
    else:
        raise TypeError(f'{dtype} is not a valid numeric data type only integers and floats')
    if min_value is None:
        min_value = dtype_info[1]
    if max_value is None:
        max_value = dtype_info[0]
    if safety_factor_scale == 'linear':
        min_value = min_value / large_abs_safety_factor
        max_value = max_value / large_abs_safety_factor
        if kind_dtype == 'float' and (not abs_smallest_val):
            abs_smallest_val = dtype_info[2] * small_abs_safety_factor
    elif safety_factor_scale == 'log':
        min_sign = math.copysign(1, min_value)
        min_value = abs(min_value) ** (1 / large_abs_safety_factor) * min_sign
        max_sign = math.copysign(1, max_value)
        max_value = abs(max_value) ** (1 / large_abs_safety_factor) * max_sign
        if kind_dtype == 'float' and (not abs_smallest_val):
            (m, e) = math.frexp(dtype_info[2])
            abs_smallest_val = m * 2 ** (e / small_abs_safety_factor)
    else:
        raise ValueError(f"{safety_factor_scale} is not a valid safety factor scale. use 'log' or 'linear'.")
    if kind_dtype == 'int':
        return (int(min_value), int(max_value), None)
    return (min_value, max_value, abs_smallest_val)

def general_helpers_dtype_info_helper(backend, kind_dtype, dtype):
    if False:
        while True:
            i = 10
    with WithBackendContext(backend) as ivy_backend:
        if kind_dtype == 'float':
            return (ivy_backend.finfo(dtype).max, ivy_backend.finfo(dtype).min, getattr(ivy_backend.finfo(dtype), 'smallest_normal', None))
        elif kind_dtype == 'int':
            return (ivy_backend.iinfo(dtype).max, ivy_backend.iinfo(dtype).min, getattr(ivy_backend.iinfo(dtype), 'smallest_normal', None))

class BroadcastError(ValueError):
    """Shapes do not broadcast with eachother."""

def _broadcast_shapes(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Tuple[int, ...]:
    if False:
        for i in range(10):
            print('nop')
    N1 = len(shape1)
    N2 = len(shape2)
    N = max(N1, N2)
    shape = [None for _ in range(N)]
    i = N - 1
    while i >= 0:
        n1 = N1 - N + i
        if N1 - N + i >= 0:
            d1 = shape1[n1]
        else:
            d1 = 1
        n2 = N2 - N + i
        if N2 - N + i >= 0:
            d2 = shape2[n2]
        else:
            d2 = 1
        if d1 == 1:
            shape[i] = d2
        elif d2 == 1:
            shape[i] = d1
        elif d1 == d2:
            shape[i] = d1
        else:
            raise BroadcastError()
        i = i - 1
    return tuple(shape)

def broadcast_shapes(*shapes: Tuple[int, ...]):
    if False:
        for i in range(10):
            print('nop')
    if len(shapes) == 0:
        raise ValueError('shapes=[] must be non-empty')
    elif len(shapes) == 1:
        return shapes[0]
    result = _broadcast_shapes(shapes[0], shapes[1])
    for i in range(2, len(shapes)):
        result = _broadcast_shapes(result, shapes[i])
    return result

@st.composite
def two_broadcastable_shapes(draw):
    if False:
        print('Hello World!')
    (shape1, shape2) = draw(array_helpers.mutually_broadcastable_shapes(2))
    assume(broadcast_shapes(shape1, shape2) == shape1)
    return (shape1, shape2)

@st.composite
def reshape_shapes(draw, *, shape):
    if False:
        return 10
    '\n    Draws a random shape with the same number of elements as the given shape.\n\n    Parameters\n    ----------\n    draw\n        special function that draws data randomly (but is reproducible) from a given\n        data-set (ex. list).\n    shape\n        list/strategy/tuple of integers representing an array shape.\n\n    Returns\n    -------\n        A strategy that draws a tuple.\n    '
    if isinstance(shape, st._internal.SearchStrategy):
        shape = draw(shape)
    size = 1 if len(shape) == 0 else math.prod(shape)
    rshape = draw(st.lists(number_helpers.ints(min_value=0)).filter(lambda s: math.prod(s) == size))
    return tuple(rshape)

@st.composite
def subsets(draw, *, elements):
    if False:
        while True:
            i = 10
    '\n    Draws a subset of elements from the given elements.\n\n    Parameters\n    ----------\n    draw\n        special function that draws data randomly (but is reproducible) from a given\n        data-set (ex. list).\n    elements\n        set of elements to be drawn from.\n\n    Returns\n    -------\n        A strategy that draws a subset of elements.\n    '
    return tuple((e for e in elements if draw(st.booleans())))

@st.composite
def get_shape(draw, *, allow_none=False, min_num_dims=0, max_num_dims=5, min_dim_size=1, max_dim_size=10):
    if False:
        for i in range(10):
            print('nop')
    '\n    Draws a tuple of integers drawn randomly from [min_dim_size, max_dim_size] of size\n    drawn from min_num_dims to max_num_dims. Useful for randomly drawing the shape of an\n    array.\n\n    Parameters\n    ----------\n    draw\n        special function that draws data randomly (but is reproducible) from a given\n        data-set (ex. list).\n    allow_none\n        if True, allow for the result to be None.\n    min_num_dims\n        minimum size of the tuple.\n    max_num_dims\n        maximum size of the tuple.\n    min_dim_size\n        minimum value of each integer in the tuple.\n    max_dim_size\n        maximum value of each integer in the tuple.\n\n    Returns\n    -------\n        A strategy that draws a tuple.\n    '
    if allow_none:
        shape = draw(st.none() | st.lists(number_helpers.ints(min_value=min_dim_size, max_value=max_dim_size), min_size=min_num_dims, max_size=max_num_dims))
    else:
        shape = draw(st.lists(number_helpers.ints(min_value=min_dim_size, max_value=max_dim_size), min_size=min_num_dims, max_size=max_num_dims))
    if shape is None:
        return shape
    return tuple(shape)

@st.composite
def get_mean_std(draw, *, dtype):
    if False:
        for i in range(10):
            print('nop')
    '\n    Draws two integers representing the mean and standard deviation for a given data\n    type.\n\n    Parameters\n    ----------\n    draw\n        special function that draws data randomly (but is reproducible) from a given\n        data-set (ex. list).\n    dtype\n        data type.\n\n    Returns\n    -------\n    A strategy that can be used in the @given hypothesis decorator.\n    '
    none_or_float = none_or_float = number_helpers.floats(dtype=dtype) | st.none()
    values = draw(array_helpers.list_of_size(x=none_or_float, size=2))
    values[1] = abs(values[1]) if values[1] else None
    return (values[0], values[1])

@st.composite
def get_bounds(draw, *, dtype):
    if False:
        i = 10
        return i + 15
    '\n    Draws two numbers; low and high, for a given data type such that low < high.\n\n    Parameters\n    ----------\n    draw\n        special function that draws data randomly (but is reproducible) from a given\n        data-set (ex. list).\n    dtype\n        data type.\n\n    Returns\n    -------\n        A strategy that draws a list of two numbers.\n    '
    if 'int' in dtype:
        values = draw(array_helpers.array_values(dtype=dtype, shape=2))
        (values[0], values[1]) = (abs(values[0]), abs(values[1]))
        (low, high) = (min(values), max(values))
        if low == high:
            return draw(get_bounds(dtype=dtype))
    else:
        none_or_float = number_helpers.floats(dtype=dtype) | st.none()
        values = draw(array_helpers.list_of_size(x=none_or_float, size=2))
        if values[0] is not None and values[1] is not None:
            (low, high) = (min(values), max(values))
        else:
            (low, high) = (values[0], values[1])
        if ivy.default(low, 0.0) >= ivy.default(high, 1.0):
            return draw(get_bounds(dtype=dtype))
    return [low, high]

@st.composite
def get_axis(draw, *, shape, allow_neg=True, allow_none=False, sort_values=True, unique=True, min_size=1, max_size=None, force_tuple=False, force_int=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Draws one or more axis for the given shape.\n\n    Parameters\n    ----------\n    draw\n        special function that draws data randomly (but is reproducible) from a given\n        data-set (ex. list).\n    shape\n        shape of the array as a tuple, or a hypothesis strategy from which the shape\n        will be drawn\n    allow_neg\n        boolean; if True, allow negative axes to be drawn\n    allow_none\n        boolean; if True, allow None to be drawn\n    sort_values\n        boolean; if True, and a tuple of axes is drawn, tuple is sorted in increasing\n        fashion\n    unique\n        boolean; if True, and a tuple of axes is drawn, all axes drawn will be unique\n    min_size\n        int or hypothesis strategy; if a tuple of axes is drawn, the minimum number of\n        axes drawn\n    max_size\n        int or hypothesis strategy; if a tuple of axes is drawn, the maximum number of\n        axes drawn.\n        If None and unique is True, then it is set to the number of axes in the shape\n    force_tuple\n        boolean, if true, all axis will be returned as a tuple. If force_tuple and\n        force_int are true, then an AssertionError is raised\n    force_int\n        boolean, if true, all axis will be returned as an int. If force_tuple and\n        force_int are true, then an AssertionError is raised\n\n    Returns\n    -------\n        A strategy that draws an axis or axes.\n    '
    assert not (force_int and force_tuple), 'Cannot return an int and a tuple. If both are valid then set both to False.'
    if isinstance(shape, st._internal.SearchStrategy):
        shape = draw(shape)
    if isinstance(min_size, st._internal.SearchStrategy):
        min_size = draw(min_size)
    if isinstance(max_size, st._internal.SearchStrategy):
        max_size = draw(max_size)
    axes = len(shape)
    lower_axes_bound = axes if allow_neg else 0
    if max_size is None and unique:
        max_size = max(axes, min_size)
    valid_strategies = []
    if allow_none:
        valid_strategies.append(st.none())
    if min_size > 1:
        force_tuple = True
    if not force_tuple:
        if axes == 0:
            valid_strategies.append(st.just(0))
        else:
            valid_strategies.append(st.integers(-lower_axes_bound, axes - 1))
    if not force_int:
        if axes == 0:
            valid_strategies.append(st.lists(st.just(0), min_size=min_size, max_size=max_size))
        else:
            valid_strategies.append(st.lists(st.integers(-lower_axes_bound, axes - 1), min_size=min_size, max_size=max_size, unique=unique))
    axis = draw(st.one_of(*valid_strategies).filter(lambda x: all((i != axes + j for i in x for j in x)) if isinstance(x, list) and unique and allow_neg else True))
    if isinstance(axis, list):
        if sort_values:

            def sort_key(ele, max_len):
                if False:
                    for i in range(10):
                        print('nop')
                if ele < 0:
                    return ele + max_len
                return ele
            axis.sort(key=lambda ele: sort_key(ele, axes))
        axis = tuple(axis)
    return axis

@st.composite
def x_and_filters(draw, dim: int=2, transpose: bool=False, depthwise=False, mixed_fn_compos=True):
    if False:
        return 10
    '\n    Draws a random x and filters for a convolution.\n\n    Parameters\n    ----------\n    draw\n        special function that draws data randomly (but is reproducible) from a given\n        data-set (ex. list).\n    dim\n        the dimension of the convolution\n    transpose\n        if True, draw a transpose convolution\n    depthwise\n        if True, draw a depthwise convolution\n\n    Returns\n    -------\n        A strategy that draws a random x and filters for a convolution.\n    '
    strides = draw(st.integers(min_value=1, max_value=2))
    padding = draw(st.sampled_from(['SAME', 'VALID']))
    batch_size = draw(st.integers(1, 5))
    filter_shape = draw(get_shape(min_num_dims=dim, max_num_dims=dim, min_dim_size=1, max_dim_size=5))
    input_channels = draw(st.integers(1, 5))
    output_channels = draw(st.integers(1, 5))
    dilations = draw(st.integers(1, 2))
    dtype = draw(dtype_helpers.get_dtypes('float', mixed_fn_compos=mixed_fn_compos, full=False))
    if dim == 2:
        data_format = draw(st.sampled_from(['NCHW']))
    elif dim == 1:
        data_format = draw(st.sampled_from(['NWC', 'NCW']))
    else:
        data_format = draw(st.sampled_from(['NDHWC', 'NCDHW']))
    x_dim = []
    if transpose:
        output_shape = []
        x_dim = draw(get_shape(min_num_dims=dim, max_num_dims=dim, min_dim_size=1, max_dim_size=20))
        for i in range(dim):
            output_shape.append(_deconv_length(x_dim[i], strides, filter_shape[i], padding, dilations))
    else:
        for i in range(dim):
            min_x = filter_shape[i] + (filter_shape[i] - 1) * (dilations - 1)
            x_dim.append(draw(st.integers(min_x, 100)))
        x_dim = tuple(x_dim)
    if not depthwise:
        filter_shape = filter_shape + (input_channels, output_channels)
    else:
        filter_shape = filter_shape + (input_channels,)
    if data_format in ['NHWC', 'NWC', 'NDHWC']:
        x_shape = (batch_size,) + x_dim + (input_channels,)
    else:
        x_shape = (batch_size, input_channels) + x_dim
    vals = draw(array_helpers.array_values(shape=x_shape, dtype=dtype[0], large_abs_safety_factor=3, small_abs_safety_factor=4, safety_factor_scale='log'))
    filters = draw(array_helpers.array_values(shape=filter_shape, dtype=dtype[0], large_abs_safety_factor=3, small_abs_safety_factor=4, safety_factor_scale='log'))
    if transpose:
        return (dtype, vals, filters, dilations, data_format, strides, padding, output_shape)
    return (dtype, vals, filters, dilations, data_format, strides, padding)

@st.composite
def embedding_helper(draw, mixed_fn_compos=True):
    if False:
        i = 10
        return i + 15
    '\n    Obtain weights for embeddings, the corresponding indices, the padding indices.\n\n    Parameters\n    ----------\n    draw\n        special function that draws data randomly (but is reproducible) from a given\n        data-set (ex. list).\n\n    Returns\n    -------\n        A strategy for generating a tuple\n    '
    (dtype_weight, weight) = draw(array_helpers.dtype_and_values(available_dtypes=[x for x in draw(dtype_helpers.get_dtypes('numeric', mixed_fn_compos=mixed_fn_compos)) if 'float' in x or 'complex' in x], min_num_dims=2, max_num_dims=2, min_dim_size=1, min_value=-10000.0, max_value=10000.0))
    (num_embeddings, embedding_dim) = weight[0].shape
    (dtype_indices, indices) = draw(array_helpers.dtype_and_values(available_dtypes=['int32', 'int64'], min_num_dims=2, min_dim_size=1, min_value=0, max_value=num_embeddings - 1).filter(lambda x: x[1][0].shape[-1] == embedding_dim))
    padding_idx = draw(st.integers(min_value=0, max_value=num_embeddings - 1))
    return (dtype_indices + dtype_weight, indices[0], weight[0], padding_idx)