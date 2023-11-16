"""
Peak finding functions.

Some of the functions defined here were ported directly from CuSignal under
terms of the MIT license, under the following notice:

Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import math
import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx import jit

def _get_typename(dtype):
    if False:
        while True:
            i = 10
    typename = get_typename(dtype)
    if cupy.dtype(dtype).kind == 'c':
        typename = 'thrust::' + typename
    elif typename == 'float16':
        if runtime.is_hip:
            typename = '__half'
        else:
            typename = 'half'
    return typename
FLOAT_TYPES = [cupy.float16, cupy.float32, cupy.float64]
INT_TYPES = [cupy.int8, cupy.int16, cupy.int32, cupy.int64]
UNSIGNED_TYPES = [cupy.uint8, cupy.uint16, cupy.uint32, cupy.uint64]
FLOAT_INT_TYPES = FLOAT_TYPES + INT_TYPES
TYPES = FLOAT_INT_TYPES + UNSIGNED_TYPES
TYPE_NAMES = [_get_typename(t) for t in TYPES]
FLOAT_INT_NAMES = [_get_typename(t) for t in FLOAT_INT_TYPES]
_modedict = {cupy.less: 0, cupy.greater: 1, cupy.less_equal: 2, cupy.greater_equal: 3, cupy.equal: 4, cupy.not_equal: 5}
if runtime.is_hip:
    PEAKS_KERNEL_BASE = '\n    #include <hip/hip_runtime.h>\n'
else:
    PEAKS_KERNEL_BASE = '\n#include <cuda_runtime.h>\n#include <device_launch_parameters.h>\n'
PEAKS_KERNEL = PEAKS_KERNEL_BASE + '\n#include <cupy/math_constants.h>\n#include <cupy/carray.cuh>\n#include <cupy/complex.cuh>\n\ntemplate<typename T>\n__global__ void local_maxima_1d(\n        const int n, const T* __restrict__ x, long long* midpoints,\n        long long* left_edges, long long* right_edges) {\n\n    const int orig_idx = blockDim.x * blockIdx.x + threadIdx.x;\n    const int idx = orig_idx + 1;\n\n    if(idx >= n - 1) {\n        return;\n    }\n\n    long long midpoint = -1;\n    long long left = -1;\n    long long right = -1;\n\n    if(x[idx - 1] < x[idx]) {\n        int i_ahead = idx + 1;\n\n        while(i_ahead < n - 1 && x[i_ahead] == x[idx]) {\n            i_ahead++;\n        }\n\n        if(x[i_ahead] < x[idx]) {\n            left = idx;\n            right = i_ahead - 1;\n            midpoint = (left + right) / 2;\n        }\n    }\n\n    midpoints[orig_idx] = midpoint;\n    left_edges[orig_idx] = left;\n    right_edges[orig_idx] = right;\n}\n\ntemplate<typename T>\n__global__ void peak_prominences(\n        const int n, const int n_peaks, const T* __restrict__ x,\n        const long long* __restrict__ peaks, const long long wlen,\n        T* prominences, long long* left_bases, long long* right_bases) {\n\n    const int idx = blockDim.x * blockIdx.x + threadIdx.x;\n    if(idx >= n_peaks) {\n        return;\n    }\n\n    const long long peak = peaks[idx];\n    long long i_min = 0;\n    long long i_max = n - 1;\n\n    if(wlen >= 2) {\n        i_min = max(peak - wlen / 2, i_min);\n        i_max = min(peak + wlen / 2, i_max);\n    }\n\n    left_bases[idx] = peak;\n    long long i = peak;\n    T left_min = x[peak];\n\n    while(i_min <= i && x[i] <= x[peak]) {\n        if(x[i] < left_min) {\n            left_min = x[i];\n            left_bases[idx] = i;\n        }\n        i--;\n    }\n\n    right_bases[idx] = peak;\n    i = peak;\n    T right_min = x[peak];\n\n    while(i <= i_max && x[i] <= x[peak]) {\n        if(x[i] < right_min) {\n            right_min = x[i];\n            right_bases[idx] = i;\n        }\n        i++;\n    }\n\n    prominences[idx] = x[peak] - max(left_min, right_min);\n}\n\ntemplate<>\n__global__ void peak_prominences<half>(\n        const int n, const int n_peaks, const half* __restrict__ x,\n        const long long* __restrict__ peaks, const long long wlen,\n        half* prominences, long long* left_bases, long long* right_bases) {\n\n    const int idx = blockDim.x * blockIdx.x + threadIdx.x;\n    if(idx >= n_peaks) {\n        return;\n    }\n\n    const long long peak = peaks[idx];\n    long long i_min = 0;\n    long long i_max = n - 1;\n\n    if(wlen >= 2) {\n        i_min = max(peak - wlen / 2, i_min);\n        i_max = min(peak + wlen / 2, i_max);\n    }\n\n    left_bases[idx] = peak;\n    long long i = peak;\n    half left_min = x[peak];\n\n    while(i_min <= i && x[i] <= x[peak]) {\n        if(x[i] < left_min) {\n            left_min = x[i];\n            left_bases[idx] = i;\n        }\n        i--;\n    }\n\n    right_bases[idx] = peak;\n    i = peak;\n    half right_min = x[peak];\n\n    while(i <= i_max && x[i] <= x[peak]) {\n        if(x[i] < right_min) {\n            right_min = x[i];\n            right_bases[idx] = i;\n        }\n        i++;\n    }\n\n    prominences[idx] = x[peak] - __hmax(left_min, right_min);\n}\n\ntemplate<typename T>\n__global__ void peak_widths(\n        const int n, const T* __restrict__ x,\n        const long long* __restrict__ peaks,\n        const double rel_height,\n        const T* __restrict__ prominences,\n        const long long* __restrict__ left_bases,\n        const long long* __restrict__ right_bases,\n        double* widths, double* width_heights,\n        double* left_ips, double* right_ips) {\n\n    const int idx = blockDim.x * blockIdx.x + threadIdx.x;\n    if(idx >= n) {\n        return;\n    }\n\n    long long i_min = left_bases[idx];\n    long long i_max = right_bases[idx];\n    long long peak = peaks[idx];\n\n    double height = x[peak] - prominences[idx] * rel_height;\n    width_heights[idx] = height;\n\n    // Find intersection point on left side\n    long long i = peak;\n    while (i_min < i && height < x[i]) {\n        i--;\n    }\n\n    double left_ip = (double) i;\n    if(x[i] < height) {\n        // Interpolate if true intersection height is between samples\n        left_ip += (height - x[i]) / (x[i + 1] - x[i]);\n    }\n\n    // Find intersection point on right side\n    i = peak;\n    while(i < i_max && height < x[i]) {\n        i++;\n    }\n\n    double right_ip = (double) i;\n    if(x[i] < height) {\n        // Interpolate if true intersection height is between samples\n        right_ip -= (height - x[i]) / (x[i - 1] - x[i]);\n    }\n\n    widths[idx] = right_ip - left_ip;\n    left_ips[idx] = left_ip;\n    right_ips[idx] = right_ip;\n}\n\ntemplate<>\n__global__ void peak_widths<half>(\n        const int n, const half* __restrict__ x,\n        const long long* __restrict__ peaks,\n        const double rel_height,\n        const half* __restrict__ prominences,\n        const long long* __restrict__ left_bases,\n        const long long* __restrict__ right_bases,\n        double* widths, double* width_heights,\n        double* left_ips, double* right_ips) {\n\n    const int idx = blockDim.x * blockIdx.x + threadIdx.x;\n    if(idx >= n) {\n        return;\n    }\n\n    long long i_min = left_bases[idx];\n    long long i_max = right_bases[idx];\n    long long peak = peaks[idx];\n\n    double height = ((double) x[peak]) - ((double) prominences[idx]) * rel_height;\n    width_heights[idx] = height;\n\n    // Find intersection point on left side\n    long long i = peak;\n    while (i_min < i && height < ((double) x[i])) {\n        i--;\n    }\n\n    double left_ip = (double) i;\n    if(((double) x[i]) < height) {\n        // Interpolate if true intersection height is between samples\n        left_ip += (height - ((double) x[i])) / ((double) (x[i + 1] - x[i]));\n    }\n\n    // Find intersection point on right side\n    i = peak;\n    while(i < i_max && height < ((double) x[i])) {\n        i++;\n    }\n\n    double right_ip = (double) i;\n    if(((double) x[i]) < height) {\n        // Interpolate if true intersection height is between samples\n        right_ip -= (height - ((double) x[i])) / ((double) (x[i - 1] - x[i]));\n    }\n\n    widths[idx] = right_ip - left_ip;\n    left_ips[idx] = left_ip;\n    right_ips[idx] = right_ip;\n}\n'
PEAKS_MODULE = cupy.RawModule(code=PEAKS_KERNEL, options=('-std=c++11',), name_expressions=[f'local_maxima_1d<{x}>' for x in TYPE_NAMES] + [f'peak_prominences<{x}>' for x in TYPE_NAMES] + [f'peak_widths<{x}>' for x in TYPE_NAMES])
ARGREL_KERNEL = '\n#include <cupy/math_constants.h>\n#include <cupy/carray.cuh>\n#include <cupy/complex.cuh>\n\ntemplate<typename T>\n__device__ __forceinline__ bool less( const T &a, const T &b ) {\n    return ( a < b );\n}\n\ntemplate<typename T>\n__device__ __forceinline__ bool greater( const T &a, const T &b ) {\n    return ( a > b );\n}\n\ntemplate<typename T>\n__device__ __forceinline__ bool less_equal( const T &a, const T &b ) {\n    return ( a <= b );\n}\n\ntemplate<typename T>\n__device__ __forceinline__ bool greater_equal( const T &a, const T &b ) {\n    return ( a >= b );\n}\n\ntemplate<typename T>\n__device__ __forceinline__ bool equal( const T &a, const T &b ) {\n    return ( a == b );\n}\n\ntemplate<typename T>\n__device__ __forceinline__ bool not_equal( const T &a, const T &b ) {\n    return ( a != b );\n}\n\n__device__ __forceinline__ void clip_plus(\n        const bool &clip, const int &n, int &plus ) {\n    if ( clip ) {\n        if ( plus >= n ) {\n            plus = n - 1;\n        }\n    } else {\n        if ( plus >= n ) {\n            plus -= n;\n        }\n    }\n}\n\n__device__ __forceinline__ void clip_minus(\n        const bool &clip, const int &n, int &minus ) {\n    if ( clip ) {\n        if ( minus < 0 ) {\n            minus = 0;\n        }\n    } else {\n        if ( minus < 0 ) {\n            minus += n;\n        }\n    }\n}\n\ntemplate<typename T>\n__device__ bool compare(const int comp, const T &a, const T &b) {\n    if(comp == 0) {\n        return less(a, b);\n    } else if(comp == 1) {\n        return greater(a, b);\n    } else if(comp == 2) {\n        return less_equal(a, b);\n    } else if(comp == 3) {\n        return greater_equal(a, b);\n    } else if(comp == 4) {\n        return equal(a, b);\n    } else {\n        return not_equal(a, b);\n    }\n}\n\ntemplate<typename T>\n__global__ void boolrelextrema_1D( const int  n,\n                                   const int  order,\n                                   const bool clip,\n                                   const int  comp,\n                                   const T *__restrict__ inp,\n                                   bool *__restrict__ results) {\n\n    const int tx { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };\n    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };\n\n    for ( int tid = tx; tid < n; tid += stride ) {\n\n        const T data { inp[tid] };\n        bool    temp { true };\n\n        for ( int o = 1; o < ( order + 1 ); o++ ) {\n            int plus { tid + o };\n            int minus { tid - o };\n\n            clip_plus( clip, n, plus );\n            clip_minus( clip, n, minus );\n\n            temp &= compare<T>( comp,  data, inp[plus] );\n            temp &= compare<T>( comp, data, inp[minus] );\n        }\n        results[tid] = temp;\n    }\n}\n\ntemplate<typename T>\n__global__ void boolrelextrema_2D( const int  in_x,\n                                   const int  in_y,\n                                   const int  order,\n                                   const bool clip,\n                                   const int  comp,\n                                   const int  axis,\n                                   const T *__restrict__ inp,\n                                   bool *__restrict__ results) {\n\n    const int ty { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };\n    const int tx { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };\n\n    if ( ( tx < in_y ) && ( ty < in_x ) ) {\n        int tid { tx * in_x + ty };\n\n        const T data { inp[tid] };\n        bool    temp { true };\n\n        for ( int o = 1; o < ( order + 1 ); o++ ) {\n\n            int plus {};\n            int minus {};\n\n            if ( axis == 0 ) {\n                plus  = tx + o;\n                minus = tx - o;\n\n                clip_plus( clip, in_y, plus );\n                clip_minus( clip, in_y, minus );\n\n                plus  = plus * in_x + ty;\n                minus = minus * in_x + ty;\n            } else {\n                plus  = ty + o;\n                minus = ty - o;\n\n                clip_plus( clip, in_x, plus );\n                clip_minus( clip, in_x, minus );\n\n                plus  = tx * in_x + plus;\n                minus = tx * in_x + minus;\n            }\n\n            temp &= compare<T>( comp, data, inp[plus] );\n            temp &= compare<T>( comp, data, inp[minus] );\n        }\n        results[tid] = temp;\n    }\n}\n'
ARGREL_MODULE = cupy.RawModule(code=ARGREL_KERNEL, options=('-std=c++11',), name_expressions=[f'boolrelextrema_1D<{x}>' for x in FLOAT_INT_NAMES] + [f'boolrelextrema_2D<{x}>' for x in FLOAT_INT_NAMES])

def _get_module_func(module, func_name, *template_args):
    if False:
        while True:
            i = 10
    args_dtypes = [_get_typename(arg.dtype) for arg in template_args]
    template = ', '.join(args_dtypes)
    kernel_name = f'{func_name}<{template}>' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel

def _local_maxima_1d(x):
    if False:
        i = 10
        return i + 15
    samples = x.shape[0] - 2
    block_sz = 128
    n_blocks = (samples + block_sz - 1) // block_sz
    midpoints = cupy.empty(samples, dtype=cupy.int64)
    left_edges = cupy.empty(samples, dtype=cupy.int64)
    right_edges = cupy.empty(samples, dtype=cupy.int64)
    local_max_kernel = _get_module_func(PEAKS_MODULE, 'local_maxima_1d', x)
    local_max_kernel((n_blocks,), (block_sz,), (x.shape[0], x, midpoints, left_edges, right_edges))
    pos_idx = midpoints > 0
    midpoints = midpoints[pos_idx]
    left_edges = left_edges[pos_idx]
    right_edges = right_edges[pos_idx]
    return (midpoints, left_edges, right_edges)

def _unpack_condition_args(interval, x, peaks):
    if False:
        return 10
    '\n    Parse condition arguments for `find_peaks`.\n\n    Parameters\n    ----------\n    interval : number or ndarray or sequence\n        Either a number or ndarray or a 2-element sequence of the former. The\n        first value is always interpreted as `imin` and the second,\n        if supplied, as `imax`.\n    x : ndarray\n        The signal with `peaks`.\n    peaks : ndarray\n        An array with indices used to reduce `imin` and / or `imax` if those\n        are arrays.\n\n    Returns\n    -------\n    imin, imax : number or ndarray or None\n        Minimal and maximal value in `argument`.\n\n    Raises\n    ------\n    ValueError :\n        If interval border is given as array and its size does not match the\n        size of `x`.\n    '
    try:
        (imin, imax) = interval
    except (TypeError, ValueError):
        (imin, imax) = (interval, None)
    if isinstance(imin, cupy.ndarray):
        if imin.size != x.size:
            raise ValueError('array size of lower interval border must match x')
        imin = imin[peaks]
    if isinstance(imax, cupy.ndarray):
        if imax.size != x.size:
            raise ValueError('array size of upper interval border must match x')
        imax = imax[peaks]
    return (imin, imax)

def _select_by_property(peak_properties, pmin, pmax):
    if False:
        i = 10
        return i + 15
    '\n    Evaluate where the generic property of peaks confirms to an interval.\n\n    Parameters\n    ----------\n    peak_properties : ndarray\n        An array with properties for each peak.\n    pmin : None or number or ndarray\n        Lower interval boundary for `peak_properties`. ``None``\n        is interpreted as an open border.\n    pmax : None or number or ndarray\n        Upper interval boundary for `peak_properties`. ``None``\n        is interpreted as an open border.\n\n    Returns\n    -------\n    keep : bool\n        A boolean mask evaluating to true where `peak_properties` confirms\n        to the interval.\n\n    See Also\n    --------\n    find_peaks\n\n    '
    keep = cupy.ones(peak_properties.size, dtype=bool)
    if pmin is not None:
        keep &= pmin <= peak_properties
    if pmax is not None:
        keep &= peak_properties <= pmax
    return keep

def _select_by_peak_threshold(x, peaks, tmin, tmax):
    if False:
        return 10
    '\n    Evaluate which peaks fulfill the threshold condition.\n\n    Parameters\n    ----------\n    x : ndarray\n        A 1-D array which is indexable by `peaks`.\n    peaks : ndarray\n        Indices of peaks in `x`.\n    tmin, tmax : scalar or ndarray or None\n         Minimal and / or maximal required thresholds. If supplied as ndarrays\n         their size must match `peaks`. ``None`` is interpreted as an open\n         border.\n\n    Returns\n    -------\n    keep : bool\n        A boolean mask evaluating to true where `peaks` fulfill the threshold\n        condition.\n    left_thresholds, right_thresholds : ndarray\n        Array matching `peak` containing the thresholds of each peak on\n        both sides.\n\n    '
    stacked_thresholds = cupy.vstack([x[peaks] - x[peaks - 1], x[peaks] - x[peaks + 1]])
    keep = cupy.ones(peaks.size, dtype=bool)
    if tmin is not None:
        min_thresholds = cupy.min(stacked_thresholds, axis=0)
        keep &= tmin <= min_thresholds
    if tmax is not None:
        max_thresholds = cupy.max(stacked_thresholds, axis=0)
        keep &= max_thresholds <= tmax
    return (keep, stacked_thresholds[0], stacked_thresholds[1])

def _select_by_peak_distance(peaks, priority, distance):
    if False:
        for i in range(10):
            print('nop')
    "\n    Evaluate which peaks fulfill the distance condition.\n\n    Parameters\n    ----------\n    peaks : ndarray\n        Indices of peaks in `vector`.\n    priority : ndarray\n        An array matching `peaks` used to determine priority of each peak. A\n        peak with a higher priority value is kept over one with a lower one.\n    distance : np.float64\n        Minimal distance that peaks must be spaced.\n\n    Returns\n    -------\n    keep : ndarray[bool]\n        A boolean mask evaluating to true where `peaks` fulfill the distance\n        condition.\n\n    Notes\n    -----\n    Declaring the input arrays as C-contiguous doesn't seem to have performance\n    advantages.\n    "
    peaks_size = peaks.shape[0]
    distance_ = cupy.ceil(distance)
    keep = cupy.ones(peaks_size, dtype=cupy.bool_)
    priority_to_position = cupy.argsort(priority)
    for i in range(peaks_size - 1, -1, -1):
        j = priority_to_position[i]
        if keep[j] == 0:
            continue
        k = j - 1
        while 0 <= k and peaks[j] - peaks[k] < distance_:
            keep[k] = 0
            k -= 1
        k = j + 1
        while k < peaks_size and peaks[k] - peaks[j] < distance_:
            keep[k] = 0
            k += 1
    return keep

def _arg_x_as_expected(value):
    if False:
        print('Hello World!')
    'Ensure argument `x` is a 1-D C-contiguous array.\n\n    Returns\n    -------\n    value : ndarray\n        A 1-D C-contiguous array.\n    '
    value = cupy.asarray(value, order='C')
    if value.ndim != 1:
        raise ValueError('`x` must be a 1-D array')
    return value

def _arg_wlen_as_expected(value):
    if False:
        i = 10
        return i + 15
    'Ensure argument `wlen` is of type `np.intp` and larger than 1.\n\n    Used in `peak_prominences` and `peak_widths`.\n\n    Returns\n    -------\n    value : np.intp\n        The original `value` rounded up to an integer or -1 if `value` was\n        None.\n    '
    if value is None:
        value = -1
    elif 1 < value:
        if not cupy.can_cast(value, cupy.int64, 'safe'):
            value = math.ceil(value)
        value = int(value)
    else:
        raise ValueError('`wlen` must be larger than 1, was {}'.format(value))
    return value

def _arg_peaks_as_expected(value):
    if False:
        for i in range(10):
            print('nop')
    "Ensure argument `peaks` is a 1-D C-contiguous array of dtype('int64').\n\n    Used in `peak_prominences` and `peak_widths` to make `peaks` compatible\n    with the signature of the wrapped Cython functions.\n\n    Returns\n    -------\n    value : ndarray\n        A 1-D C-contiguous array with dtype('int64').\n    "
    value = cupy.asarray(value)
    if value.size == 0:
        value = cupy.array([], dtype=cupy.int64)
    try:
        value = value.astype(cupy.int64, order='C', copy=False)
    except TypeError as e:
        raise TypeError("cannot safely cast `peaks` to dtype('intp')") from e
    if value.ndim != 1:
        raise ValueError('`peaks` must be a 1-D array')
    return value

@jit.rawkernel()
def _check_prominence_invalid(n, peaks, left_bases, right_bases, out):
    if False:
        for i in range(10):
            print('nop')
    tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    i_min = left_bases[tid]
    i_max = right_bases[tid]
    peak = peaks[tid]
    valid = 0 <= i_min and i_min <= peak and (peak <= i_max) and (i_max < n)
    out[tid] = not valid

def _peak_prominences(x, peaks, wlen=None, check=False):
    if False:
        for i in range(10):
            print('nop')
    if check and cupy.any(cupy.logical_or(peaks < 0, peaks > x.shape[0] - 1)):
        raise ValueError('peaks are not a valid index')
    prominences = cupy.empty(peaks.shape[0], dtype=x.dtype)
    left_bases = cupy.empty(peaks.shape[0], dtype=cupy.int64)
    right_bases = cupy.empty(peaks.shape[0], dtype=cupy.int64)
    n = peaks.shape[0]
    block_sz = 128
    n_blocks = (n + block_sz - 1) // block_sz
    peak_prom_kernel = _get_module_func(PEAKS_MODULE, 'peak_prominences', x)
    peak_prom_kernel((n_blocks,), (block_sz,), (x.shape[0], n, x, peaks, wlen, prominences, left_bases, right_bases))
    return (prominences, left_bases, right_bases)

def _peak_widths(x, peaks, rel_height, prominences, left_bases, right_bases, check=False):
    if False:
        while True:
            i = 10
    if rel_height < 0:
        raise ValueError('`rel_height` must be greater or equal to 0.0')
    if prominences is None:
        raise TypeError('prominences must not be None')
    if left_bases is None:
        raise TypeError('left_bases must not be None')
    if right_bases is None:
        raise TypeError('right_bases must not be None')
    if not peaks.shape[0] == prominences.shape[0] == left_bases.shape[0] == right_bases.shape[0]:
        raise ValueError('arrays in `prominence_data` must have the same shape as `peaks`')
    n = peaks.shape[0]
    block_sz = 128
    n_blocks = (n + block_sz - 1) // block_sz
    if check and n > 0:
        invalid = cupy.zeros(n, dtype=cupy.bool_)
        _check_prominence_invalid((n_blocks,), (block_sz,), (x.shape[0], peaks, left_bases, right_bases, invalid))
        if cupy.any(invalid):
            raise ValueError('prominence data is invalid')
    widths = cupy.empty(peaks.shape[0], dtype=cupy.float64)
    width_heights = cupy.empty(peaks.shape[0], dtype=cupy.float64)
    left_ips = cupy.empty(peaks.shape[0], dtype=cupy.float64)
    right_ips = cupy.empty(peaks.shape[0], dtype=cupy.float64)
    peak_widths_kernel = _get_module_func(PEAKS_MODULE, 'peak_widths', x)
    peak_widths_kernel((n_blocks,), (block_sz,), (n, x, peaks, rel_height, prominences, left_bases, right_bases, widths, width_heights, left_ips, right_ips))
    return (widths, width_heights, left_ips, right_ips)

def peak_prominences(x, peaks, wlen=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate the prominence of each peak in a signal.\n\n    The prominence of a peak measures how much a peak stands out from the\n    surrounding baseline of the signal and is defined as the vertical distance\n    between the peak and its lowest contour line.\n\n    Parameters\n    ----------\n    x : sequence\n        A signal with peaks.\n    peaks : sequence\n        Indices of peaks in `x`.\n    wlen : int, optional\n        A window length in samples that optionally limits the evaluated area\n        for each peak to a subset of `x`. The peak is always placed in the\n        middle of the window therefore the given length is rounded up to the\n        next odd integer. This parameter can speed up the calculation\n        (see Notes).\n\n    Returns\n    -------\n    prominences : ndarray\n        The calculated prominences for each peak in `peaks`.\n    left_bases, right_bases : ndarray\n        The peaks\' bases as indices in `x` to the left and right of each peak.\n        The higher base of each pair is a peak\'s lowest contour line.\n\n    Raises\n    ------\n    ValueError\n        If a value in `peaks` is an invalid index for `x`.\n\n    Warns\n    -----\n    PeakPropertyWarning\n        For indices in `peaks` that don\'t point to valid local maxima in `x`,\n        the returned prominence will be 0 and this warning is raised. This\n        also happens if `wlen` is smaller than the plateau size of a peak.\n\n    Warnings\n    --------\n    This function may return unexpected results for data containing NaNs. To\n    avoid this, NaNs should either be removed or replaced.\n\n    See Also\n    --------\n    find_peaks\n        Find peaks inside a signal based on peak properties.\n    peak_widths\n        Calculate the width of peaks.\n\n    Notes\n    -----\n    Strategy to compute a peak\'s prominence:\n\n    1. Extend a horizontal line from the current peak to the left and right\n       until the line either reaches the window border (see `wlen`) or\n       intersects the signal again at the slope of a higher peak. An\n       intersection with a peak of the same height is ignored.\n    2. On each side find the minimal signal value within the interval defined\n       above. These points are the peak\'s bases.\n    3. The higher one of the two bases marks the peak\'s lowest contour line.\n       The prominence can then be calculated as the vertical difference between\n       the peaks height itself and its lowest contour line.\n\n    Searching for the peak\'s bases can be slow for large `x` with periodic\n    behavior because large chunks or even the full signal need to be evaluated\n    for the first algorithmic step. This evaluation area can be limited with\n    the parameter `wlen` which restricts the algorithm to a window around the\n    current peak and can shorten the calculation time if the window length is\n    short in relation to `x`.\n    However, this may stop the algorithm from finding the true global contour\n    line if the peak\'s true bases are outside this window. Instead, a higher\n    contour line is found within the restricted window leading to a smaller\n    calculated prominence. In practice, this is only relevant for the highest\n    set of peaks in `x`. This behavior may even be used intentionally to\n    calculate "local" prominences.\n\n    '
    x = _arg_x_as_expected(x)
    peaks = _arg_peaks_as_expected(peaks)
    wlen = _arg_wlen_as_expected(wlen)
    return _peak_prominences(x, peaks, wlen, check=True)

def peak_widths(x, peaks, rel_height=0.5, prominence_data=None, wlen=None):
    if False:
        print('Hello World!')
    "\n    Calculate the width of each peak in a signal.\n\n    This function calculates the width of a peak in samples at a relative\n    distance to the peak's height and prominence.\n\n    Parameters\n    ----------\n    x : sequence\n        A signal with peaks.\n    peaks : sequence\n        Indices of peaks in `x`.\n    rel_height : float, optional\n        Chooses the relative height at which the peak width is measured as a\n        percentage of its prominence. 1.0 calculates the width of the peak at\n        its lowest contour line while 0.5 evaluates at half the prominence\n        height. Must be at least 0. See notes for further explanation.\n    prominence_data : tuple, optional\n        A tuple of three arrays matching the output of `peak_prominences` when\n        called with the same arguments `x` and `peaks`. This data are\n        calculated internally if not provided.\n    wlen : int, optional\n        A window length in samples passed to `peak_prominences` as an optional\n        argument for internal calculation of `prominence_data`. This argument\n        is ignored if `prominence_data` is given.\n\n    Returns\n    -------\n    widths : ndarray\n        The widths for each peak in samples.\n    width_heights : ndarray\n        The height of the contour lines at which the `widths` where evaluated.\n    left_ips, right_ips : ndarray\n        Interpolated positions of left and right intersection points of a\n        horizontal line at the respective evaluation height.\n\n    Raises\n    ------\n    ValueError\n        If `prominence_data` is supplied but doesn't satisfy the condition\n        ``0 <= left_base <= peak <= right_base < x.shape[0]`` for each peak,\n        has the wrong dtype, is not C-contiguous or does not have the same\n        shape.\n\n    Warns\n    -----\n    PeakPropertyWarning\n        Raised if any calculated width is 0. This may stem from the supplied\n        `prominence_data` or if `rel_height` is set to 0.\n\n    Warnings\n    --------\n    This function may return unexpected results for data containing NaNs. To\n    avoid this, NaNs should either be removed or replaced.\n\n    See Also\n    --------\n    find_peaks\n        Find peaks inside a signal based on peak properties.\n    peak_prominences\n        Calculate the prominence of peaks.\n\n    Notes\n    -----\n    The basic algorithm to calculate a peak's width is as follows:\n\n    * Calculate the evaluation height :math:`h_{eval}` with the formula\n      :math:`h_{eval} = h_{Peak} - P \\cdot R`, where :math:`h_{Peak}` is the\n      height of the peak itself, :math:`P` is the peak's prominence and\n      :math:`R` a positive ratio specified with the argument `rel_height`.\n    * Draw a horizontal line at the evaluation height to both sides, starting\n      at the peak's current vertical position until the lines either intersect\n      a slope, the signal border or cross the vertical position of the peak's\n      base (see `peak_prominences` for an definition). For the first case,\n      intersection with the signal, the true intersection point is estimated\n      with linear interpolation.\n    * Calculate the width as the horizontal distance between the chosen\n      endpoints on both sides. As a consequence of this the maximal possible\n      width for each peak is the horizontal distance between its bases.\n\n    As shown above to calculate a peak's width its prominence and bases must be\n    known. You can supply these yourself with the argument `prominence_data`.\n    Otherwise, they are internally calculated (see `peak_prominences`).\n    "
    x = _arg_x_as_expected(x)
    peaks = _arg_peaks_as_expected(peaks)
    if prominence_data is None:
        wlen = _arg_wlen_as_expected(wlen)
        prominence_data = _peak_prominences(x, peaks, wlen, check=True)
    return _peak_widths(x, peaks, rel_height, *prominence_data, check=True)

def find_peaks(x, height=None, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Find peaks inside a signal based on peak properties.\n\n    This function takes a 1-D array and finds all local maxima by\n    simple comparison of neighboring values. Optionally, a subset of these\n    peaks can be selected by specifying conditions for a peak's properties.\n\n    Parameters\n    ----------\n    x : sequence\n        A signal with peaks.\n    height : number or ndarray or sequence, optional\n        Required height of peaks. Either a number, ``None``, an array matching\n        `x` or a 2-element sequence of the former. The first element is\n        always interpreted as the  minimal and the second, if supplied, as the\n        maximal required height.\n    threshold : number or ndarray or sequence, optional\n        Required threshold of peaks, the vertical distance to its neighboring\n        samples. Either a number, ``None``, an array matching `x` or a\n        2-element sequence of the former. The first element is always\n        interpreted as the  minimal and the second, if supplied, as the maximal\n        required threshold.\n    distance : number, optional\n        Required minimal horizontal distance (>= 1) in samples between\n        neighbouring peaks. Smaller peaks are removed first until the condition\n        is fulfilled for all remaining peaks.\n    prominence : number or ndarray or sequence, optional\n        Required prominence of peaks. Either a number, ``None``, an array\n        matching `x` or a 2-element sequence of the former. The first\n        element is always interpreted as the  minimal and the second, if\n        supplied, as the maximal required prominence.\n    width : number or ndarray or sequence, optional\n        Required width of peaks in samples. Either a number, ``None``, an array\n        matching `x` or a 2-element sequence of the former. The first\n        element is always interpreted as the  minimal and the second, if\n        supplied, as the maximal required width.\n    wlen : int, optional\n        Used for calculation of the peaks prominences, thus it is only used if\n        one of the arguments `prominence` or `width` is given. See argument\n        `wlen` in `peak_prominences` for a full description of its effects.\n    rel_height : float, optional\n        Used for calculation of the peaks width, thus it is only used if\n        `width` is given. See argument  `rel_height` in `peak_widths` for\n        a full description of its effects.\n    plateau_size : number or ndarray or sequence, optional\n        Required size of the flat top of peaks in samples. Either a number,\n        ``None``, an array matching `x` or a 2-element sequence of the former.\n        The first element is always interpreted as the minimal and the second,\n        if supplied as the maximal required plateau size.\n\n        .. versionadded:: 1.2.0\n\n    Returns\n    -------\n    peaks : ndarray\n        Indices of peaks in `x` that satisfy all given conditions.\n    properties : dict\n        A dictionary containing properties of the returned peaks which were\n        calculated as intermediate results during evaluation of the specified\n        conditions:\n\n        * 'peak_heights'\n              If `height` is given, the height of each peak in `x`.\n        * 'left_thresholds', 'right_thresholds'\n              If `threshold` is given, these keys contain a peaks vertical\n              distance to its neighbouring samples.\n        * 'prominences', 'right_bases', 'left_bases'\n              If `prominence` is given, these keys are accessible. See\n              `peak_prominences` for a description of their content.\n        * 'width_heights', 'left_ips', 'right_ips'\n              If `width` is given, these keys are accessible. See `peak_widths`\n              for a description of their content.\n        * 'plateau_sizes', left_edges', 'right_edges'\n              If `plateau_size` is given, these keys are accessible and contain\n              the indices of a peak's edges (edges are still part of the\n              plateau) and the calculated plateau sizes.\n\n        To calculate and return properties without excluding peaks, provide the\n        open interval ``(None, None)`` as a value to the appropriate argument\n        (excluding `distance`).\n\n    Warns\n    -----\n    PeakPropertyWarning\n        Raised if a peak's properties have unexpected values (see\n        `peak_prominences` and `peak_widths`).\n\n    Warnings\n    --------\n    This function may return unexpected results for data containing NaNs. To\n    avoid this, NaNs should either be removed or replaced.\n\n    See Also\n    --------\n    find_peaks_cwt\n        Find peaks using the wavelet transformation.\n    peak_prominences\n        Directly calculate the prominence of peaks.\n    peak_widths\n        Directly calculate the width of peaks.\n\n    Notes\n    -----\n    In the context of this function, a peak or local maximum is defined as any\n    sample whose two direct neighbours have a smaller amplitude. For flat peaks\n    (more than one sample of equal amplitude wide) the index of the middle\n    sample is returned (rounded down in case the number of samples is even).\n    For noisy signals the peak locations can be off because the noise might\n    change the position of local maxima. In those cases consider smoothing the\n    signal before searching for peaks or use other peak finding and fitting\n    methods (like `find_peaks_cwt`).\n\n    Some additional comments on specifying conditions:\n\n    * Almost all conditions (excluding `distance`) can be given as half-open or\n      closed intervals, e.g., ``1`` or ``(1, None)`` defines the half-open\n      interval :math:`[1, \\infty]` while ``(None, 1)`` defines the interval\n      :math:`[-\\infty, 1]`. The open interval ``(None, None)`` can be specified\n      as well, which returns the matching properties without exclusion of peaks.\n    * The border is always included in the interval used to select valid peaks.\n    * For several conditions the interval borders can be specified with\n      arrays matching `x` in shape which enables dynamic constrains based on\n      the sample position.\n    * The conditions are evaluated in the following order: `plateau_size`,\n      `height`, `threshold`, `distance`, `prominence`, `width`. In most cases\n      this order is the fastest one because faster operations are applied first\n      to reduce the number of peaks that need to be evaluated later.\n    * While indices in `peaks` are guaranteed to be at least `distance` samples\n      apart, edges of flat peaks may be closer than the allowed `distance`.\n    * Use `wlen` to reduce the time it takes to evaluate the conditions for\n      `prominence` or `width` if `x` is large or has many local maxima\n      (see `peak_prominences`).\n    "
    x = _arg_x_as_expected(x)
    if distance is not None and distance < 1:
        raise ValueError('`distance` must be greater or equal to 1')
    (peaks, left_edges, right_edges) = _local_maxima_1d(x)
    properties = {}
    if plateau_size is not None:
        plateau_sizes = right_edges - left_edges + 1
        (pmin, pmax) = _unpack_condition_args(plateau_size, x, peaks)
        keep = _select_by_property(plateau_sizes, pmin, pmax)
        peaks = peaks[keep]
        properties['plateau_sizes'] = plateau_sizes
        properties['left_edges'] = left_edges
        properties['right_edges'] = right_edges
        properties = {key: array[keep] for (key, array) in properties.items()}
    if height is not None:
        peak_heights = x[peaks]
        (hmin, hmax) = _unpack_condition_args(height, x, peaks)
        keep = _select_by_property(peak_heights, hmin, hmax)
        peaks = peaks[keep]
        properties['peak_heights'] = peak_heights
        properties = {key: array[keep] for (key, array) in properties.items()}
    if threshold is not None:
        (tmin, tmax) = _unpack_condition_args(threshold, x, peaks)
        (keep, left_thresholds, right_thresholds) = _select_by_peak_threshold(x, peaks, tmin, tmax)
        peaks = peaks[keep]
        properties['left_thresholds'] = left_thresholds
        properties['right_thresholds'] = right_thresholds
        properties = {key: array[keep] for (key, array) in properties.items()}
    if distance is not None:
        keep = _select_by_peak_distance(peaks, x[peaks], distance)
        peaks = peaks[keep]
        properties = {key: array[keep] for (key, array) in properties.items()}
    if prominence is not None or width is not None:
        wlen = _arg_wlen_as_expected(wlen)
        properties.update(zip(['prominences', 'left_bases', 'right_bases'], _peak_prominences(x, peaks, wlen=wlen)))
    if prominence is not None:
        (pmin, pmax) = _unpack_condition_args(prominence, x, peaks)
        keep = _select_by_property(properties['prominences'], pmin, pmax)
        peaks = peaks[keep]
        properties = {key: array[keep] for (key, array) in properties.items()}
    if width is not None:
        properties.update(zip(['widths', 'width_heights', 'left_ips', 'right_ips'], _peak_widths(x, peaks, rel_height, properties['prominences'], properties['left_bases'], properties['right_bases'])))
        (wmin, wmax) = _unpack_condition_args(width, x, peaks)
        keep = _select_by_property(properties['widths'], wmin, wmax)
        peaks = peaks[keep]
        properties = {key: array[keep] for (key, array) in properties.items()}
    return (peaks, properties)

def _peak_finding(data, comparator, axis, order, mode, results):
    if False:
        return 10
    comp = _modedict[comparator]
    clip = mode == 'clip'
    device_id = cupy.cuda.Device()
    num_blocks = (device_id.attributes['MultiProcessorCount'] * 20,)
    block_sz = (512,)
    call_args = (data.shape[axis], order, clip, comp, data, results)
    kernel_name = 'boolrelextrema_1D'
    if data.ndim > 1:
        kernel_name = 'boolrelextrema_2D'
        (block_sz_x, block_sz_y) = (16, 16)
        n_blocks_x = (data.shape[1] + block_sz_x - 1) // block_sz_x
        n_blocks_y = (data.shape[0] + block_sz_y - 1) // block_sz_y
        block_sz = (block_sz_x, block_sz_y)
        num_blocks = (n_blocks_x, n_blocks_y)
        call_args = (data.shape[1], data.shape[0], order, clip, comp, axis, data, results)
    boolrelextrema = _get_module_func(ARGREL_MODULE, kernel_name, data)
    boolrelextrema(num_blocks, block_sz, call_args)

def _boolrelextrema(data, comparator, axis=0, order=1, mode='clip'):
    if False:
        i = 10
        return i + 15
    "\n    Calculate the relative extrema of `data`.\n\n    Relative extrema are calculated by finding locations where\n    ``comparator(data[n], data[n+1:n+order+1])`` is True.\n\n    Parameters\n    ----------\n    data : ndarray\n        Array in which to find the relative extrema.\n    comparator : callable\n        Function to use to compare two data points.\n        Should take two arrays as arguments.\n    axis : int, optional\n        Axis over which to select from `data`.  Default is 0.\n    order : int, optional\n        How many points on each side to use for the comparison\n        to consider ``comparator(n,n+x)`` to be True.\n    mode : str, optional\n        How the edges of the vector are treated. 'wrap' (wrap around) or\n        'clip' (treat overflow as the same as the last (or first) element).\n        Default 'clip'. See cupy.take.\n\n    Returns\n    -------\n    extrema : ndarray\n        Boolean array of the same shape as `data` that is True at an extrema,\n        False otherwise.\n\n    See also\n    --------\n    argrelmax, argrelmin\n    "
    if int(order) != order or order < 1:
        raise ValueError('Order must be an int >= 1')
    if data.ndim < 3:
        results = cupy.empty(data.shape, dtype=bool)
        _peak_finding(data, comparator, axis, order, mode, results)
    else:
        datalen = data.shape[axis]
        locs = cupy.arange(0, datalen)
        results = cupy.ones(data.shape, dtype=bool)
        main = cupy.take(data, locs, axis=axis)
        for shift in cupy.arange(1, order + 1):
            if mode == 'clip':
                p_locs = cupy.clip(locs + shift, a_min=None, a_max=datalen - 1)
                m_locs = cupy.clip(locs - shift, a_min=0, a_max=None)
            else:
                p_locs = locs + shift
                m_locs = locs - shift
            plus = cupy.take(data, p_locs, axis=axis)
            minus = cupy.take(data, m_locs, axis=axis)
            results &= comparator(main, plus)
            results &= comparator(main, minus)
            if ~results.any():
                return results
    return results

def argrelmin(data, axis=0, order=1, mode='clip'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Calculate the relative minima of `data`.\n\n    Parameters\n    ----------\n    data : ndarray\n        Array in which to find the relative minima.\n    axis : int, optional\n        Axis over which to select from `data`.  Default is 0.\n    order : int, optional\n        How many points on each side to use for the comparison\n        to consider ``comparator(n, n+x)`` to be True.\n    mode : str, optional\n        How the edges of the vector are treated.\n        Available options are 'wrap' (wrap around) or 'clip' (treat overflow\n        as the same as the last (or first) element).\n        Default 'clip'. See cupy.take.\n\n\n    Returns\n    -------\n    extrema : tuple of ndarrays\n        Indices of the minima in arrays of integers.  ``extrema[k]`` is\n        the array of indices of axis `k` of `data`.  Note that the\n        return value is a tuple even when `data` is one-dimensional.\n\n    See Also\n    --------\n    argrelextrema, argrelmax, find_peaks\n\n    Notes\n    -----\n    This function uses `argrelextrema` with cupy.less as comparator. Therefore\n    it requires a strict inequality on both sides of a value to consider it a\n    minimum. This means flat minima (more than one sample wide) are not\n    detected. In case of one-dimensional `data` `find_peaks` can be used to\n    detect all local minima, including flat ones, by calling it with negated\n    `data`.\n\n    Examples\n    --------\n    >>> from cupyx.scipy.signal import argrelmin\n    >>> import cupy\n    >>> x = cupy.array([2, 1, 2, 3, 2, 0, 1, 0])\n    >>> argrelmin(x)\n    (array([1, 5]),)\n    >>> y = cupy.array([[1, 2, 1, 2],\n    ...               [2, 2, 0, 0],\n    ...               [5, 3, 4, 4]])\n    ...\n    >>> argrelmin(y, axis=1)\n    (array([0, 2]), array([2, 1]))\n\n    "
    data = cupy.asarray(data)
    return argrelextrema(data, cupy.less, axis, order, mode)

def argrelmax(data, axis=0, order=1, mode='clip'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Calculate the relative maxima of `data`.\n\n    Parameters\n    ----------\n    data : ndarray\n        Array in which to find the relative maxima.\n    axis : int, optional\n        Axis over which to select from `data`.  Default is 0.\n    order : int, optional\n        How many points on each side to use for the comparison\n        to consider ``comparator(n, n+x)`` to be True.\n    mode : str, optional\n        How the edges of the vector are treated.\n        Available options are 'wrap' (wrap around) or 'clip' (treat overflow\n        as the same as the last (or first) element).\n        Default 'clip'. See cupy.take.\n\n    Returns\n    -------\n    extrema : tuple of ndarrays\n        Indices of the maxima in arrays of integers.  ``extrema[k]`` is\n        the array of indices of axis `k` of `data`.  Note that the\n        return value is a tuple even when `data` is one-dimensional.\n\n    See Also\n    --------\n    argrelextrema, argrelmin, find_peaks\n\n    Notes\n    -----\n    This function uses `argrelextrema` with cupy.greater as comparator.\n    Therefore it requires a strict inequality on both sides of a value to\n    consider it a maximum. This means flat maxima (more than one sample wide)\n    are not detected. In case of one-dimensional `data` `find_peaks` can be\n    used to detect all local maxima, including flat ones.\n\n    Examples\n    --------\n    >>> from cupyx.scipy.signal import argrelmax\n    >>> import cupy\n    >>> x = cupy.array([2, 1, 2, 3, 2, 0, 1, 0])\n    >>> argrelmax(x)\n    (array([3, 6]),)\n    >>> y = cupy.array([[1, 2, 1, 2],\n    ...               [2, 2, 0, 0],\n    ...               [5, 3, 4, 4]])\n    ...\n    >>> argrelmax(y, axis=1)\n    (array([0]), array([1]))\n    "
    data = cupy.asarray(data)
    return argrelextrema(data, cupy.greater, axis, order, mode)

def argrelextrema(data, comparator, axis=0, order=1, mode='clip'):
    if False:
        for i in range(10):
            print('nop')
    "\n    Calculate the relative extrema of `data`.\n\n    Parameters\n    ----------\n    data : ndarray\n        Array in which to find the relative extrema.\n    comparator : callable\n        Function to use to compare two data points.\n        Should take two arrays as arguments.\n    axis : int, optional\n        Axis over which to select from `data`.  Default is 0.\n    order : int, optional\n        How many points on each side to use for the comparison\n        to consider ``comparator(n, n+x)`` to be True.\n    mode : str, optional\n        How the edges of the vector are treated.\n        Available options are 'wrap' (wrap around) or 'clip' (treat overflow\n        as the same as the last (or first) element).\n        Default 'clip'. See cupy.take.\n\n    Returns\n    -------\n    extrema : tuple of ndarrays\n        Indices of the maxima in arrays of integers.  ``extrema[k]`` is\n        the array of indices of axis `k` of `data`.  Note that the\n        return value is a tuple even when `data` is one-dimensional.\n\n    See Also\n    --------\n    argrelmin, argrelmax\n\n    Examples\n    --------\n    >>> from cupyx.scipy.signal import argrelextrema\n    >>> import cupy\n    >>> x = cupy.array([2, 1, 2, 3, 2, 0, 1, 0])\n    >>> argrelextrema(x, cupy.greater)\n    (array([3, 6]),)\n    >>> y = cupy.array([[1, 2, 1, 2],\n    ...               [2, 2, 0, 0],\n    ...               [5, 3, 4, 4]])\n    ...\n    >>> argrelextrema(y, cupy.less, axis=1)\n    (array([0, 2]), array([2, 1]))\n\n    "
    data = cupy.asarray(data)
    results = _boolrelextrema(data, comparator, axis, order, mode)
    if mode == 'raise':
        raise NotImplementedError("CuPy `take` doesn't support `mode='raise'`.")
    return cupy.nonzero(results)