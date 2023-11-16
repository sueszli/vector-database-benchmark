import math
import cupy
from cupy._core import internal
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx.scipy import special as spec
from cupyx.scipy.interpolate._bspline import BSpline, _get_dtype
import numpy as np
try:
    from math import comb
    _comb = comb
except ImportError:

    def _comb(n, k):
        if False:
            while True:
                i = 10
        return math.factorial(n) // (math.factorial(n - k) * math.factorial(k))
MAX_DIMS = 64
TYPES = ['double', 'thrust::complex<double>']
INT_TYPES = ['int', 'long long']
INTERVAL_KERNEL = '\n#include <cupy/complex.cuh>\n\n#define le_or_ge(x, y, r) ((r) ? ((x) < (y)) : ((x) > (y)))\n#define ge_or_le(x, y, r) ((r) ? ((x) > (y)) : ((x) < (y)))\n#define geq_or_leq(x, y, r) ((r) ? ((x) >= (y)) : ((x) <= (y)))\n\n__device__ long long find_breakpoint_position(\n        const double* breakpoints, const double xp, bool extrapolate,\n        const int total_breakpoints, const bool* pasc) {\n\n    double a = *&breakpoints[0];\n    double b = *&breakpoints[total_breakpoints - 1];\n    bool asc = pasc[0];\n\n    if(isnan(xp)) {\n        return -1;\n    }\n\n    if(le_or_ge(xp, a, asc) || ge_or_le(xp, b, asc)) {\n        if(!extrapolate) {\n            return -1;\n        } else if(le_or_ge(xp, a, asc)) {\n            return 0;\n        } else {  // ge_or_le(xp, b, asc)\n            return total_breakpoints - 2;\n        }\n    } else if (xp == b) {\n        return total_breakpoints - 2;\n    }\n\n    int left = 0;\n    int right = total_breakpoints - 2;\n    int mid;\n\n    if(le_or_ge(xp, *&breakpoints[left + 1], asc)) {\n        right = left;\n    }\n\n    bool found = false;\n\n    while(left < right && !found) {\n        mid = ((right + left) / 2);\n        if(le_or_ge(xp, *&breakpoints[mid], asc)) {\n            right = mid;\n        } else if (geq_or_leq(xp, *&breakpoints[mid + 1], asc)) {\n            left = mid + 1;\n        } else {\n            found = true;\n            left = mid;\n        }\n    }\n\n    return left;\n\n}\n\n__global__ void find_breakpoint_position_1d(\n        const double* breakpoints, const double* x, long long* out,\n        bool extrapolate, int total_x, int total_breakpoints,\n        const bool* pasc) {\n\n    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n    if(idx >= total_x) {\n        return;\n    }\n\n    const double xp = *&x[idx];\n    out[idx] = find_breakpoint_position(\n        breakpoints, xp, extrapolate, total_breakpoints, pasc);\n}\n\n__global__ void find_breakpoint_position_nd(\n        const double* breakpoints, const double* x, long long* out,\n        bool extrapolate, int total_x, const long long* x_dims,\n        const long long* breakpoints_sizes,\n        const long long* breakpoints_strides) {\n\n    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n    if(idx >= total_x) {\n        return;\n    }\n\n    const long long x_dim = *&x_dims[idx];\n    const long long stride = breakpoints_strides[x_dim];\n    const double* dim_breakpoints = breakpoints + stride;\n    const int num_breakpoints = *&breakpoints_sizes[x_dim];\n\n    const bool asc = true;\n    const double xp = *&x[idx];\n    out[idx] = find_breakpoint_position(\n        dim_breakpoints, xp, extrapolate, num_breakpoints, &asc);\n}\n'
INTERVAL_MODULE = cupy.RawModule(code=INTERVAL_KERNEL, options=('-std=c++11',), name_expressions=['find_breakpoint_position_1d', 'find_breakpoint_position_nd'])
if runtime.is_hip:
    BASE_HEADERS = '#include <hip/hip_runtime.h>\n'
else:
    BASE_HEADERS = '#include <cuda_runtime.h>\n#include <device_launch_parameters.h>\n'
PPOLY_KERNEL = BASE_HEADERS + '\n#include <cupy/complex.cuh>\n#include <cupy/math_constants.h>\n\ntemplate<typename T>\n__device__ T eval_poly_1(\n        const double s, const T* coef, long long ci, int cj, int dx,\n        const long long* c_dims, const long long stride_0,\n        const long long stride_1) {\n    int kp, k;\n    T res, z;\n    double prefactor;\n\n    res = 0.0;\n    z = 1.0;\n\n    if(dx < 0) {\n        for(int i = 0; i < -dx; i++) {\n            z *= s;\n        }\n    }\n\n    int c_dim_0 = (int) *&c_dims[0];\n\n    for(kp = 0; kp < c_dim_0; kp++) {\n        if(dx == 0) {\n            prefactor = 1.0;\n        } else if(dx > 0) {\n            if(kp < dx) {\n                continue;\n            } else {\n                prefactor = 1.0;\n                for(k = kp; k > kp - dx; k--) {\n                    prefactor *= k;\n                }\n            }\n        } else {\n            prefactor = 1.0;\n            for(k = kp; k < kp - dx; k++) {\n                prefactor /= k + 1;\n            }\n        }\n\n        int off = stride_0 * (c_dim_0 - kp - 1) + stride_1 * ci + cj;\n        T cur_coef = *&coef[off];\n        res += cur_coef * z * ((T) prefactor);\n\n        if((kp < c_dim_0 - 1) && kp >= dx) {\n            z *= s;\n        }\n\n    }\n\n    return res;\n\n}\n\ntemplate<typename T>\n__global__ void eval_ppoly(\n        const T* coef, const double* breakpoints, const double* x,\n        const long long* intervals, int dx, const long long* c_dims,\n        const long long* c_strides, int num_x, T* out) {\n\n    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n\n    if(idx >= num_x) {\n        return;\n    }\n\n    double xp = *&x[idx];\n    long long interval = *&intervals[idx];\n    double breakpoint = *&breakpoints[interval];\n\n    const int num_c = *&c_dims[2];\n    const long long stride_0 = *&c_strides[0];\n    const long long stride_1 = *&c_strides[1];\n\n    if(interval < 0) {\n        for(int j = 0; j < num_c; j++) {\n            out[num_c * idx + j] = CUDART_NAN;\n        }\n        return;\n    }\n\n    for(int j = 0; j < num_c; j++) {\n        T res = eval_poly_1<T>(\n            xp - breakpoint, coef, interval, ((long long) (j)), dx,\n            c_dims, stride_0, stride_1);\n        out[num_c * idx + j] = res;\n    }\n}\n\ntemplate<typename T>\n__global__ void eval_ppoly_nd(\n        const T* coef, const double* xs, const double* xp,\n        const long long* intervals, const long long* dx,\n        const long long* ks, T* c2_all, const long long* c_dims,\n        const long long* c_strides, const long long* xs_strides,\n        const long long* xs_offsets, const long long* ks_strides,\n        const int num_x, const int ndims, const int num_ks, T* out) {\n\n    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n    if(idx >= num_x) {\n        return;\n    }\n\n    const long long c_dim0 = c_dims[0];\n    const int num_c = *&c_dims[2];\n    const long long c_stride0 = c_strides[0];\n    const long long c_stride1 = c_strides[1];\n\n    const double* xp_dims = xp + ndims * idx;\n    const long long* xp_intervals = intervals + ndims * idx;\n    T* c2 = c2_all + c_dim0 * idx;\n\n    bool invalid = false;\n    for(int i = 0; i < ndims && !invalid; i++) {\n        invalid = xp_intervals[i] < 0;\n    }\n\n    if(invalid) {\n        for(int j = 0; j < num_c; j++) {\n            out[num_c * idx + j] = CUDART_NAN;\n        }\n        return;\n    }\n\n    long long pos = 0;\n    for(int k = 0; k < ndims; k++) {\n        pos += xp_intervals[k] * xs_strides[k];\n    }\n\n    for(int jp = 0; jp < num_c; jp++) {\n        for(int i = 0; i < c_dim0; i++) {\n            c2[i] = coef[c_stride0 * i + c_stride1 * pos + jp];\n        }\n\n        for(int k = ndims - 1; k >= 0; k--) {\n            const long long interval = xp_intervals[k];\n            const long long xs_offset = xs_offsets[k];\n            const double* dim_breakpoints = xs + xs_offset;\n            const double xval = xp_dims[k] - dim_breakpoints[interval];\n\n            const long long k_off = ks_strides[k];\n            const long long dim_ks = ks[k];\n            int kpos = 0;\n\n            for(int ko = 0; ko < k_off; ko++) {\n                const T* c2_off = c2 + kpos;\n                const int k_dx = dx[k];\n                T res = eval_poly_1<T>(\n                    xval, c2_off, ((long long) 0), 0, k_dx,\n                    &dim_ks, ((long long) 1), ((long long) 1));\n                c2[ko] = res;\n                kpos += dim_ks;\n            }\n        }\n\n        out[num_c * idx + jp] =  c2[0];\n    }\n}\n\ntemplate<typename T>\n__global__ void fix_continuity(\n        T* coef, const double* breakpoints, const int order,\n        const long long* c_dims, const long long* c_strides,\n        int num_breakpoints) {\n\n    const long long c_size0 = *&c_dims[0];\n    const long long c_size2 = *&c_dims[2];\n    const long long stride_0 = *&c_strides[0];\n    const long long stride_1 = *&c_strides[1];\n    const long long stride_2 = *&c_strides[2];\n\n    for(int idx = 1; idx < num_breakpoints - 1; idx++) {\n        const double breakpoint = *&breakpoints[idx];\n        const long long interval = idx - 1;\n        const double breakpoint_interval = *&breakpoints[interval];\n\n        for(int jp = 0; jp < c_size2; jp++) {\n            for(int dx = order; dx > -1; dx--) {\n                T res = eval_poly_1<T>(\n                    breakpoint - breakpoint_interval, coef,\n                    interval, jp, dx, c_dims, stride_0, stride_1);\n\n                for(int kp = 0; kp < dx; kp++) {\n                    res /= kp + 1;\n                }\n\n                const long long c_idx = (\n                    stride_0 * (c_size0 - dx - 1) + stride_1 * idx +\n                    stride_2 * jp);\n\n                coef[c_idx] = res;\n            }\n        }\n    }\n}\n\ntemplate<typename T>\n__global__ void integrate(\n        const T* coef, const double* breakpoints,\n        const double* a_val, const double* b_val,\n        const long long* start, const long long* end,\n        const long long* c_dims, const long long* c_strides,\n        const bool* pasc, T* out) {\n\n    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n    const long long c_dim2 = *&c_dims[2];\n\n    if(idx >= c_dim2) {\n        return;\n    }\n\n    const bool asc = pasc[0];\n    const long long start_interval = asc ? *&start[0] : *&end[0];\n    const long long end_interval = asc ? *&end[0] : *&start[0];\n    const double a = asc ? *&a_val[0] : *&b_val[0];\n    const double b = asc ? *&b_val[0] : *&a_val[0];\n\n    const long long stride_0 = *&c_strides[0];\n    const long long stride_1 = *&c_strides[1];\n\n    if(start_interval < 0 || end_interval < 0) {\n        out[idx] = CUDART_NAN;\n        return;\n    }\n\n    T vtot = 0;\n    T vb;\n    T va;\n    for(int interval = start_interval; interval <= end_interval; interval++) {\n        const double breakpoint = *&breakpoints[interval];\n        if(interval == end_interval) {\n            vb = eval_poly_1<T>(\n                b - breakpoint, coef, interval, idx, -1, c_dims,\n                stride_0, stride_1);\n        } else {\n            const double next_breakpoint = *&breakpoints[interval + 1];\n            vb = eval_poly_1<T>(\n                next_breakpoint - breakpoint, coef, interval,\n                idx, -1, c_dims, stride_0, stride_1);\n        }\n\n        if(interval == start_interval) {\n            va = eval_poly_1<T>(\n                a - breakpoint, coef, interval, idx, -1, c_dims,\n                stride_0, stride_1);\n        } else {\n            va = eval_poly_1<T>(\n                0, coef, interval, idx, -1, c_dims,\n                stride_0, stride_1);\n        }\n\n        vtot += (vb - va);\n    }\n\n    if(!asc) {\n        vtot = -vtot;\n    }\n\n    out[idx] = vtot;\n\n}\n'
PPOLY_MODULE = cupy.RawModule(code=PPOLY_KERNEL, options=('-std=c++11',), name_expressions=[f'eval_ppoly<{type_name}>' for type_name in TYPES] + [f'eval_ppoly_nd<{type_name}>' for type_name in TYPES] + [f'fix_continuity<{type_name}>' for type_name in TYPES] + [f'integrate<{type_name}>' for type_name in TYPES])
BPOLY_KERNEL = BASE_HEADERS + '\n#include <cupy/complex.cuh>\n#include <cupy/math_constants.h>\n\ntemplate<typename T>\n__device__ T eval_bpoly1(\n        const double s, const T* coef, const long long ci, const long long cj,\n        const long long c_dims_0, const long long c_strides_0,\n        const long long c_strides_1) {\n\n    const long long k = c_dims_0 - 1;\n    const double s1 = 1 - s;\n    T res;\n\n    const long long i0 = 0 * c_strides_0 + ci * c_strides_1 + cj;\n    const long long i1 = 1 * c_strides_0 + ci * c_strides_1 + cj;\n    const long long i2 = 2 * c_strides_0 + ci * c_strides_1 + cj;\n    const long long i3 = 3 * c_strides_0 + ci * c_strides_1 + cj;\n\n    if(k == 0) {\n        res = coef[i0];\n    } else if(k == 1) {\n        res = coef[i0] * s1 + coef[i1] * s;\n    } else if(k == 2) {\n        res = coef[i0] * s1 * s1 + coef[i1] * 2.0 * s1 * s + coef[i2] * s * s;\n    } else if(k == 3) {\n        res = (coef[i0] * s1 * s1 * s1 + coef[i1] * 3.0 * s1 * s1 * s +\n               coef[i2] * 3.0 * s1 * s * s + coef[i3] * s * s * s);\n    } else {\n        T comb = 1;\n        res = 0;\n        for(int j = 0; j < k + 1; j++) {\n            const long long idx = j * c_strides_0 + ci * c_strides_1 + cj;\n            res += (comb * pow(s, ((double) j)) * pow(s1, ((double) k) - j) *\n                    coef[idx]);\n            comb *= 1.0 * (k - j) / (j + 1.0);\n        }\n    }\n\n    return res;\n}\n\ntemplate<typename T>\n__device__ T eval_bpoly1_deriv(\n        const double s, const T* coef, const long long ci, const long long cj,\n        int dx, T* wrk, const long long c_dims_0, const long long c_strides_0,\n        const long long c_strides_1, const long long wrk_dims_0,\n        const long long wrk_strides_0, const long long wrk_strides_1) {\n\n    T res, term;\n    double comb, poch;\n\n    const long long k = c_dims_0 - 1;\n\n    if(dx == 0) {\n        res = eval_bpoly1<T>(s, coef, ci, cj, c_dims_0, c_strides_0,\n                             c_strides_1);\n    } else {\n        poch = 1.0;\n        for(int a = 0; a < dx; a++) {\n            poch *= k - a;\n        }\n\n        term = 0;\n        for(int a = 0; a < k - dx + 1; a++) {\n            term = 0;\n            comb = 1;\n            for(int j = 0; j < dx + 1; j++) {\n                const long long idx = (c_strides_0 * (j + a) +\n                                       c_strides_1 * ci + cj);\n                term += coef[idx] * pow(-1.0, ((double) (j + dx))) * comb;\n                comb *= 1.0 * (dx - j) / (j + 1);\n            }\n            wrk[a] = term * poch;\n        }\n\n        res = eval_bpoly1<T>(s, wrk, 0, 0, wrk_dims_0, wrk_strides_0,\n                             wrk_strides_1);\n    }\n    return res;\n}\n\ntemplate<typename T>\n__global__ void eval_bpoly(\n        const T* coef, const double* breakpoints, const double* x,\n        const long long* intervals, int dx, T* wrk, const long long* c_dims,\n        const long long* c_strides, const long long* wrk_dims,\n        const long long* wrk_strides, int num_x, T* out) {\n\n    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n\n    if(idx >= num_x) {\n        return;\n    }\n\n    double xp = *&x[idx];\n    long long interval = *&intervals[idx];\n    const int num_c = *&c_dims[2];\n\n    const long long c_dims_0 = *&c_dims[0];\n    const long long c_strides_0 = *&c_strides[0];\n    const long long c_strides_1 = *&c_strides[1];\n\n    const long long wrk_dims_0 = *&wrk_dims[0];\n    const long long wrk_strides_0 = *&wrk_strides[0];\n    const long long wrk_strides_1 = *&wrk_strides[1];\n\n    if(interval < 0) {\n        for(int j = 0; j < num_c; j++) {\n            out[num_c * idx + j] = CUDART_NAN;\n        }\n        return;\n    }\n\n    const double ds = breakpoints[interval + 1] - breakpoints[interval];\n    const double ds_dx = pow(ds, ((double) dx));\n    T* off_wrk = wrk + idx * (c_dims_0 - dx);\n\n    for(int j = 0; j < num_c; j++) {\n        T res;\n        const double s = (xp - breakpoints[interval]) / ds;\n        if(dx == 0) {\n            res = eval_bpoly1<T>(\n                s, coef, interval, ((long long) (j)), c_dims_0, c_strides_0,\n                c_strides_1);\n        } else {\n            res = eval_bpoly1_deriv<T>(\n                s, coef, interval, ((long long) (j)), dx,\n                off_wrk, c_dims_0, c_strides_0, c_strides_1,\n                wrk_dims_0, wrk_strides_0, wrk_strides_1) / ds_dx;\n        }\n        out[num_c * idx + j] = res;\n    }\n\n}\n'
BPOLY_MODULE = cupy.RawModule(code=BPOLY_KERNEL, options=('-std=c++11',), name_expressions=[f'eval_bpoly<{type_name}>' for type_name in TYPES])

def _get_module_func(module, func_name, *template_args):
    if False:
        i = 10
        return i + 15

    def _get_typename(dtype):
        if False:
            i = 10
            return i + 15
        typename = get_typename(dtype)
        if dtype.kind == 'c':
            typename = 'thrust::' + typename
        return typename
    args_dtypes = [_get_typename(arg.dtype) for arg in template_args]
    template = ', '.join(args_dtypes)
    kernel_name = f'{func_name}<{template}>' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel

def _ppoly_evaluate(c, x, xp, dx, extrapolate, out):
    if False:
        while True:
            i = 10
    '\n    Evaluate a piecewise polynomial.\n\n    Parameters\n    ----------\n    c : ndarray, shape (k, m, n)\n        Coefficients local polynomials of order `k-1` in `m` intervals.\n        There are `n` polynomials in each interval.\n        Coefficient of highest order-term comes first.\n    x : ndarray, shape (m+1,)\n        Breakpoints of polynomials.\n    xp : ndarray, shape (r,)\n        Points to evaluate the piecewise polynomial at.\n    dx : int\n        Order of derivative to evaluate.  The derivative is evaluated\n        piecewise and may have discontinuities.\n    extrapolate : bool\n        Whether to extrapolate to out-of-bounds points based on first\n        and last intervals, or to return NaNs.\n    out : ndarray, shape (r, n)\n        Value of each polynomial at each of the input points.\n        This argument is modified in-place.\n    '
    ascending = x[-1] >= x[0]
    intervals = cupy.empty(xp.shape, dtype=cupy.int64)
    interval_kernel = INTERVAL_MODULE.get_function('find_breakpoint_position_1d')
    interval_kernel(((xp.shape[0] + 128 - 1) // 128,), (128,), (x, xp, intervals, extrapolate, xp.shape[0], x.shape[0], ascending))
    c_shape = cupy.asarray(c.shape, dtype=cupy.int64)
    c_strides = cupy.asarray(c.strides, dtype=cupy.int64) // c.itemsize
    ppoly_kernel = _get_module_func(PPOLY_MODULE, 'eval_ppoly', c)
    ppoly_kernel(((xp.shape[0] + 128 - 1) // 128,), (128,), (c, x, xp, intervals, dx, c_shape, c_strides, xp.shape[0], out))

def _ndppoly_evaluate(c, xs, ks, xp, dx, extrapolate, out):
    if False:
        print('Hello World!')
    '\n    Evaluate a piecewise tensor-product polynomial.\n\n    Parameters\n    ----------\n    c : ndarray, shape (k_1*...*k_d, m_1*...*m_d, n)\n        Coefficients local polynomials of order `k-1` in\n        `m_1`, ..., `m_d` intervals. There are `n` polynomials\n        in each interval.\n    xs : d-tuple of ndarray of shape (m_d+1,) each\n        Breakpoints of polynomials\n    ks : ndarray of int, shape (d,)\n        Orders of polynomials in each dimension\n    xp : ndarray, shape (r, d)\n        Points to evaluate the piecewise polynomial at.\n    dx : ndarray of int, shape (d,)\n        Orders of derivative to evaluate.  The derivative is evaluated\n        piecewise and may have discontinuities.\n    extrapolate : int, optional\n        Whether to extrapolate to out-of-bounds points based on first\n        and last intervals, or to return NaNs.\n    out : ndarray, shape (r, n)\n        Value of each polynomial at each of the input points.\n        For points outside the span ``x[0] ... x[-1]``,\n        ``nan`` is returned.\n        This argument is modified in-place.\n    '
    num_samples = xp.shape[0]
    total_xp = xp.size
    ndims = len(xs)
    num_ks = ks.size
    xs_sizes = cupy.asarray([x.size for x in xs], dtype=cupy.int64)
    xs_offsets = cupy.cumsum(xs_sizes)
    xs_offsets = cupy.r_[0, xs_offsets[:-1]]
    xs_complete = cupy.r_[xs]
    xs_sizes_m1 = xs_sizes - 1
    xs_strides = cupy.cumprod(xs_sizes_m1[:0:-1])
    xs_strides = cupy.r_[xs_strides[::-1], 1]
    intervals = cupy.empty(xp.shape, dtype=cupy.int64)
    dim_seq = cupy.arange(ndims, dtype=cupy.int64)
    xp_dims = cupy.broadcast_to(cupy.expand_dims(dim_seq, 0), (num_samples, ndims))
    xp_dims = xp_dims.copy()
    interval_kernel = INTERVAL_MODULE.get_function('find_breakpoint_position_nd')
    interval_kernel(((total_xp + 128 - 1) // 128,), (128,), (xs_complete, xp, intervals, extrapolate, total_xp, xp_dims, xs_sizes, xs_offsets))
    c_shape = cupy.asarray(c.shape, dtype=cupy.int64)
    c_strides = cupy.asarray(c.strides, dtype=cupy.int64) // c.itemsize
    c2 = cupy.zeros((num_samples * c.shape[0], 1, 1), dtype=_get_dtype(c))
    ks_strides = cupy.cumprod(cupy.r_[1, ks])
    ks_strides = ks_strides[:-1]
    ppoly_kernel = _get_module_func(PPOLY_MODULE, 'eval_ppoly_nd', c)
    ppoly_kernel(((num_samples + 128 - 1) // 128,), (128,), (c, xs_complete, xp, intervals, dx, ks, c2, c_shape, c_strides, xs_strides, xs_offsets, ks_strides, num_samples, ndims, num_ks, out))

def _fix_continuity(c, x, order):
    if False:
        for i in range(10):
            print('nop')
    '\n    Make a piecewise polynomial continuously differentiable to given order.\n\n    Parameters\n    ----------\n    c : ndarray, shape (k, m, n)\n        Coefficients local polynomials of order `k-1` in `m` intervals.\n        There are `n` polynomials in each interval.\n        Coefficient of highest order-term comes first.\n\n        Coefficients c[-order-1:] are modified in-place.\n    x : ndarray, shape (m+1,)\n        Breakpoints of polynomials\n    order : int\n        Order up to which enforce piecewise differentiability.\n    '
    c_shape = cupy.asarray(c.shape, dtype=cupy.int64)
    c_strides = cupy.asarray(c.strides, dtype=cupy.int64) // c.itemsize
    continuity_kernel = _get_module_func(PPOLY_MODULE, 'fix_continuity', c)
    continuity_kernel((1,), (1,), (c, x, order, c_shape, c_strides, x.shape[0]))

def _integrate(c, x, a, b, extrapolate, out):
    if False:
        i = 10
        return i + 15
    '\n    Compute integral over a piecewise polynomial.\n\n    Parameters\n    ----------\n    c : ndarray, shape (k, m, n)\n        Coefficients local polynomials of order `k-1` in `m` intervals.\n    x : ndarray, shape (m+1,)\n        Breakpoints of polynomials\n    a : double\n        Start point of integration.\n    b : double\n        End point of integration.\n    extrapolate : bool, optional\n        Whether to extrapolate to out-of-bounds points based on first\n        and last intervals, or to return NaNs.\n    out : ndarray, shape (n,)\n        Integral of the piecewise polynomial, assuming the polynomial\n        is zero outside the range (x[0], x[-1]).\n        This argument is modified in-place.\n    '
    ascending = x[-1] >= x[0]
    a = cupy.asarray([a], dtype=cupy.float64)
    b = cupy.asarray([b], dtype=cupy.float64)
    start_interval = cupy.empty(a.shape, dtype=cupy.int64)
    end_interval = cupy.empty(b.shape, dtype=cupy.int64)
    interval_kernel = INTERVAL_MODULE.get_function('find_breakpoint_position_1d')
    interval_kernel(((a.shape[0] + 128 - 1) // 128,), (128,), (x, a, start_interval, extrapolate, a.shape[0], x.shape[0], ascending))
    interval_kernel(((b.shape[0] + 128 - 1) // 128,), (128,), (x, b, end_interval, extrapolate, b.shape[0], x.shape[0], ascending))
    c_shape = cupy.asarray(c.shape, dtype=cupy.int64)
    c_strides = cupy.asarray(c.strides, dtype=cupy.int64) // c.itemsize
    int_kernel = _get_module_func(PPOLY_MODULE, 'integrate', c)
    int_kernel(((c.shape[2] + 128 - 1) // 128,), (128,), (c, x, a, b, start_interval, end_interval, c_shape, c_strides, ascending, out))

def _bpoly_evaluate(c, x, xp, dx, extrapolate, out):
    if False:
        for i in range(10):
            print('nop')
    '\n    Evaluate a Bernstein polynomial.\n\n    Parameters\n    ----------\n    c : ndarray, shape (k, m, n)\n        Coefficients local polynomials of order `k-1` in `m` intervals.\n        There are `n` polynomials in each interval.\n        Coefficient of highest order-term comes first.\n    x : ndarray, shape (m+1,)\n        Breakpoints of polynomials.\n    xp : ndarray, shape (r,)\n        Points to evaluate the piecewise polynomial at.\n    dx : int\n        Order of derivative to evaluate.  The derivative is evaluated\n        piecewise and may have discontinuities.\n    extrapolate : bool\n        Whether to extrapolate to out-of-bounds points based on first\n        and last intervals, or to return NaNs.\n    out : ndarray, shape (r, n)\n        Value of each polynomial at each of the input points.\n        This argument is modified in-place.\n    '
    ascending = x[-1] >= x[0]
    intervals = cupy.empty(xp.shape, dtype=cupy.int64)
    interval_kernel = INTERVAL_MODULE.get_function('find_breakpoint_position_1d')
    interval_kernel(((xp.shape[0] + 128 - 1) // 128,), (128,), (x, xp, intervals, extrapolate, xp.shape[0], x.shape[0], ascending))
    c_shape = cupy.asarray(c.shape, dtype=cupy.int64)
    c_strides = cupy.asarray(c.strides, dtype=cupy.int64) // c.itemsize
    wrk = cupy.empty((xp.shape[0] * (c.shape[0] - dx), 1, 1), dtype=_get_dtype(c))
    wrk_shape = cupy.asarray([c.shape[0] - dx, 1, 1], dtype=cupy.int64)
    wrk_strides = cupy.asarray(wrk.strides, dtype=cupy.int64) // wrk.itemsize
    bpoly_kernel = _get_module_func(BPOLY_MODULE, 'eval_bpoly', c)
    bpoly_kernel(((xp.shape[0] + 128 - 1) // 128,), (128,), (c, x, xp, intervals, dx, wrk, c_shape, c_strides, wrk_shape, wrk_strides, xp.shape[0], out))

def _ndim_coords_from_arrays(points, ndim=None):
    if False:
        while True:
            i = 10
    '\n    Convert a tuple of coordinate arrays to a (..., ndim)-shaped array.\n    '
    if isinstance(points, tuple) and len(points) == 1:
        points = cupy.asarray(points[0])
    if isinstance(points, tuple):
        p = cupy.broadcast_arrays(*[cupy.asarray(x) for x in points])
        p = [cupy.expand_dims(x, -1) for x in p]
        points = cupy.concatenate(p, axis=-1)
    else:
        points = cupy.asarray(points)
        if points.ndim == 1:
            if ndim is None:
                points = points.reshape(-1, 1)
            else:
                points = points.reshape(-1, ndim)
    return points

class _PPolyBase:
    """Base class for piecewise polynomials."""
    __slots__ = ('c', 'x', 'extrapolate', 'axis')

    def __init__(self, c, x, extrapolate=None, axis=0):
        if False:
            while True:
                i = 10
        self.c = cupy.asarray(c)
        self.x = cupy.ascontiguousarray(x, dtype=cupy.float64)
        if extrapolate is None:
            extrapolate = True
        elif extrapolate != 'periodic':
            extrapolate = bool(extrapolate)
        self.extrapolate = extrapolate
        if self.c.ndim < 2:
            raise ValueError('Coefficients array must be at least 2-dimensional.')
        if not 0 <= axis < self.c.ndim - 1:
            raise ValueError('axis=%s must be between 0 and %s' % (axis, self.c.ndim - 1))
        self.axis = axis
        if axis != 0:
            self.c = cupy.moveaxis(self.c, axis + 1, 0)
            self.c = cupy.moveaxis(self.c, axis + 1, 0)
        if self.x.ndim != 1:
            raise ValueError('x must be 1-dimensional')
        if self.x.size < 2:
            raise ValueError('at least 2 breakpoints are needed')
        if self.c.ndim < 2:
            raise ValueError('c must have at least 2 dimensions')
        if self.c.shape[0] == 0:
            raise ValueError('polynomial must be at least of order 0')
        if self.c.shape[1] != self.x.size - 1:
            raise ValueError('number of coefficients != len(x)-1')
        dx = cupy.diff(self.x)
        if not (cupy.all(dx >= 0) or cupy.all(dx <= 0)):
            raise ValueError('`x` must be strictly increasing or decreasing.')
        dtype = self._get_dtype(self.c.dtype)
        self.c = cupy.ascontiguousarray(self.c, dtype=dtype)

    def _get_dtype(self, dtype):
        if False:
            while True:
                i = 10
        if cupy.issubdtype(dtype, cupy.complexfloating) or cupy.issubdtype(self.c.dtype, cupy.complexfloating):
            return cupy.complex_
        else:
            return cupy.float_

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None, axis=0):
        if False:
            print('Hello World!')
        '\n        Construct the piecewise polynomial without making checks.\n        Takes the same parameters as the constructor. Input arguments\n        ``c`` and ``x`` must be arrays of the correct shape and type. The\n        ``c`` array can only be of dtypes float and complex, and ``x``\n        array must have dtype float.\n        '
        self = object.__new__(cls)
        self.c = c
        self.x = x
        self.axis = axis
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = extrapolate
        return self

    def _ensure_c_contiguous(self):
        if False:
            i = 10
            return i + 15
        '\n        c and x may be modified by the user. The Cython code expects\n        that they are C contiguous.\n        '
        if not self.x.flags.c_contiguous:
            self.x = self.x.copy()
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()

    def extend(self, c, x):
        if False:
            print('Hello World!')
        '\n        Add additional breakpoints and coefficients to the polynomial.\n\n        Parameters\n        ----------\n        c : ndarray, size (k, m, ...)\n            Additional coefficients for polynomials in intervals. Note that\n            the first additional interval will be formed using one of the\n            ``self.x`` end points.\n        x : ndarray, size (m,)\n            Additional breakpoints. Must be sorted in the same order as\n            ``self.x`` and either to the right or to the left of the current\n            breakpoints.\n        '
        c = cupy.asarray(c)
        x = cupy.asarray(x)
        if c.ndim < 2:
            raise ValueError('invalid dimensions for c')
        if x.ndim != 1:
            raise ValueError('invalid dimensions for x')
        if x.shape[0] != c.shape[1]:
            raise ValueError('Shapes of x {} and c {} are incompatible'.format(x.shape, c.shape))
        if c.shape[2:] != self.c.shape[2:] or c.ndim != self.c.ndim:
            raise ValueError('Shapes of c {} and self.c {} are incompatible'.format(c.shape, self.c.shape))
        if c.size == 0:
            return
        dx = cupy.diff(x)
        if not (cupy.all(dx >= 0) or cupy.all(dx <= 0)):
            raise ValueError('`x` is not sorted.')
        if self.x[-1] >= self.x[0]:
            if not x[-1] >= x[0]:
                raise ValueError('`x` is in the different order than `self.x`.')
            if x[0] >= self.x[-1]:
                action = 'append'
            elif x[-1] <= self.x[0]:
                action = 'prepend'
            else:
                raise ValueError('`x` is neither on the left or on the right from `self.x`.')
        else:
            if not x[-1] <= x[0]:
                raise ValueError('`x` is in the different order than `self.x`.')
            if x[0] <= self.x[-1]:
                action = 'append'
            elif x[-1] >= self.x[0]:
                action = 'prepend'
            else:
                raise ValueError('`x` is neither on the left or on the right from `self.x`.')
        dtype = self._get_dtype(c.dtype)
        k2 = max(c.shape[0], self.c.shape[0])
        c2 = cupy.zeros((k2, self.c.shape[1] + c.shape[1]) + self.c.shape[2:], dtype=dtype)
        if action == 'append':
            c2[k2 - self.c.shape[0]:, :self.c.shape[1]] = self.c
            c2[k2 - c.shape[0]:, self.c.shape[1]:] = c
            self.x = cupy.r_[self.x, x]
        elif action == 'prepend':
            c2[k2 - self.c.shape[0]:, :c.shape[1]] = c
            c2[k2 - c.shape[0]:, c.shape[1]:] = self.c
            self.x = cupy.r_[x, self.x]
        self.c = c2

    def __call__(self, x, nu=0, extrapolate=None):
        if False:
            while True:
                i = 10
        "\n        Evaluate the piecewise polynomial or its derivative.\n\n        Parameters\n        ----------\n        x : array_like\n            Points to evaluate the interpolant at.\n        nu : int, optional\n            Order of derivative to evaluate. Must be non-negative.\n        extrapolate : {bool, 'periodic', None}, optional\n            If bool, determines whether to extrapolate to out-of-bounds points\n            based on first and last intervals, or to return NaNs.\n            If 'periodic', periodic extrapolation is used.\n            If None (default), use `self.extrapolate`.\n\n        Returns\n        -------\n        y : array_like\n            Interpolated values. Shape is determined by replacing\n            the interpolation axis in the original array with the shape of x.\n\n        Notes\n        -----\n        Derivatives are evaluated piecewise for each polynomial\n        segment, even if the polynomial is not differentiable at the\n        breakpoints. The polynomial intervals are considered half-open,\n        ``[a, b)``, except for the last interval which is closed\n        ``[a, b]``.\n        "
        if extrapolate is None:
            extrapolate = self.extrapolate
        x = cupy.asarray(x)
        (x_shape, x_ndim) = (x.shape, x.ndim)
        x = cupy.ascontiguousarray(x.ravel(), dtype=cupy.float_)
        if extrapolate == 'periodic':
            x = self.x[0] + (x - self.x[0]) % (self.x[-1] - self.x[0])
            extrapolate = False
        out = cupy.empty((len(x), int(np.prod(self.c.shape[2:]))), dtype=self.c.dtype)
        self._ensure_c_contiguous()
        self._evaluate(x, nu, extrapolate, out)
        out = out.reshape(x_shape + self.c.shape[2:])
        if self.axis != 0:
            dims = list(range(out.ndim))
            dims = dims[x_ndim:x_ndim + self.axis] + dims[:x_ndim] + dims[x_ndim + self.axis:]
            out = out.transpose(dims)
        return out

class PPoly(_PPolyBase):
    """
    Piecewise polynomial in terms of coefficients and breakpoints
    The polynomial between ``x[i]`` and ``x[i + 1]`` is written in the
    local power basis::

        S = sum(c[m, i] * (xp - x[i]) ** (k - m) for m in range(k + 1))

    where ``k`` is the degree of the polynomial.

    Parameters
    ----------
    c : ndarray, shape (k, m, ...)
        Polynomial coefficients, order `k` and `m` intervals.
    x : ndarray, shape (m+1,)
        Polynomial breakpoints. Must be sorted in either increasing or
        decreasing order.
    extrapolate : bool or 'periodic', optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.

    Attributes
    ----------
    x : ndarray
        Breakpoints.
    c : ndarray
        Coefficients of the polynomials. They are reshaped
        to a 3-D array with the last dimension representing
        the trailing dimensions of the original coefficient array.
    axis : int
        Interpolation axis.

    See also
    --------
    BPoly : piecewise polynomials in the Bernstein basis

    Notes
    -----
    High-order polynomials in the power basis can be numerically
    unstable. Precision problems can start to appear for orders
    larger than 20-30.

    .. seealso:: :class:`scipy.interpolate.BSpline`
    """

    def _evaluate(self, x, nu, extrapolate, out):
        if False:
            return 10
        _ppoly_evaluate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1), self.x, x, nu, bool(extrapolate), out)

    def derivative(self, nu=1):
        if False:
            while True:
                i = 10
        '\n        Construct a new piecewise polynomial representing the derivative.\n\n        Parameters\n        ----------\n        nu : int, optional\n            Order of derivative to evaluate. Default is 1, i.e., compute the\n            first derivative. If negative, the antiderivative is returned.\n\n        Returns\n        -------\n        pp : PPoly\n            Piecewise polynomial of order k2 = k - n representing the\n            derivative of this polynomial.\n\n        Notes\n        -----\n        Derivatives are evaluated piecewise for each polynomial\n        segment, even if the polynomial is not differentiable at the\n        breakpoints. The polynomial intervals are considered half-open,\n        ``[a, b)``, except for the last interval which is closed\n        ``[a, b]``.\n        '
        if nu < 0:
            return self.antiderivative(-nu)
        if nu == 0:
            c2 = self.c.copy()
        else:
            c2 = self.c[:-nu, :].copy()
        if c2.shape[0] == 0:
            c2 = cupy.zeros((1,) + c2.shape[1:], dtype=c2.dtype)
        factor = spec.poch(cupy.arange(c2.shape[0], 0, -1), nu)
        c2 *= factor[(slice(None),) + (None,) * (c2.ndim - 1)]
        return self.construct_fast(c2, self.x, self.extrapolate, self.axis)

    def antiderivative(self, nu=1):
        if False:
            i = 10
            return i + 15
        "\n        Construct a new piecewise polynomial representing the antiderivative.\n        Antiderivative is also the indefinite integral of the function,\n        and derivative is its inverse operation.\n\n        Parameters\n        ----------\n        nu : int, optional\n            Order of antiderivative to evaluate. Default is 1, i.e., compute\n            the first integral. If negative, the derivative is returned.\n\n        Returns\n        -------\n        pp : PPoly\n            Piecewise polynomial of order k2 = k + n representing\n            the antiderivative of this polynomial.\n\n        Notes\n        -----\n        The antiderivative returned by this function is continuous and\n        continuously differentiable to order n-1, up to floating point\n        rounding error.\n\n        If antiderivative is computed and ``self.extrapolate='periodic'``,\n        it will be set to False for the returned instance. This is done because\n        the antiderivative is no longer periodic and its correct evaluation\n        outside of the initially given x interval is difficult.\n        "
        if nu <= 0:
            return self.derivative(-nu)
        c = cupy.zeros((self.c.shape[0] + nu, self.c.shape[1]) + self.c.shape[2:], dtype=self.c.dtype)
        c[:-nu] = self.c
        factor = spec.poch(cupy.arange(self.c.shape[0], 0, -1), nu)
        c[:-nu] /= factor[(slice(None),) + (None,) * (c.ndim - 1)]
        self._ensure_c_contiguous()
        _fix_continuity(c.reshape(c.shape[0], c.shape[1], -1), self.x, nu - 1)
        if self.extrapolate == 'periodic':
            extrapolate = False
        else:
            extrapolate = self.extrapolate
        return self.construct_fast(c, self.x, extrapolate, self.axis)

    def integrate(self, a, b, extrapolate=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Compute a definite integral over a piecewise polynomial.\n\n        Parameters\n        ----------\n        a : float\n            Lower integration bound\n        b : float\n            Upper integration bound\n        extrapolate : {bool, 'periodic', None}, optional\n            If bool, determines whether to extrapolate to out-of-bounds points\n            based on first and last intervals, or to return NaNs.\n            If 'periodic', periodic extrapolation is used.\n            If None (default), use `self.extrapolate`.\n\n        Returns\n        -------\n        ig : array_like\n            Definite integral of the piecewise polynomial over [a, b]\n        "
        if extrapolate is None:
            extrapolate = self.extrapolate
        sign = 1
        if b < a:
            (a, b) = (b, a)
            sign = -1
        range_int = cupy.empty((int(np.prod(self.c.shape[2:])),), dtype=self.c.dtype)
        self._ensure_c_contiguous()
        if extrapolate == 'periodic':
            (xs, xe) = (self.x[0], self.x[-1])
            period = xe - xs
            interval = b - a
            (n_periods, left) = divmod(interval, period)
            if n_periods > 0:
                _integrate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1), self.x, xs, xe, False, out=range_int)
                range_int *= n_periods
            else:
                range_int.fill(0)
            a = xs + (a - xs) % period
            b = a + left
            remainder_int = cupy.empty_like(range_int)
            if b <= xe:
                _integrate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1), self.x, a, b, False, out=remainder_int)
                range_int += remainder_int
            else:
                _integrate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1), self.x, a, xe, False, out=remainder_int)
                range_int += remainder_int
                _integrate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1), self.x, xs, xs + left + a - xe, False, out=remainder_int)
                range_int += remainder_int
        else:
            _integrate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1), self.x, a, b, bool(extrapolate), out=range_int)
        range_int *= sign
        return range_int.reshape(self.c.shape[2:])

    def solve(self, y=0.0, discontinuity=True, extrapolate=None):
        if False:
            return 10
        "\n        Find real solutions of the equation ``pp(x) == y``.\n\n        Parameters\n        ----------\n        y : float, optional\n            Right-hand side. Default is zero.\n        discontinuity : bool, optional\n            Whether to report sign changes across discontinuities at\n            breakpoints as roots.\n        extrapolate : {bool, 'periodic', None}, optional\n            If bool, determines whether to return roots from the polynomial\n            extrapolated based on first and last intervals, 'periodic' works\n            the same as False. If None (default), use `self.extrapolate`.\n\n        Returns\n        -------\n        roots : ndarray\n            Roots of the polynomial(s).\n            If the PPoly object describes multiple polynomials, the\n            return value is an object array whose each element is an\n            ndarray containing the roots.\n\n        Notes\n        -----\n        This routine works only on real-valued polynomials.\n        If the piecewise polynomial contains sections that are\n        identically zero, the root list will contain the start point\n        of the corresponding interval, followed by a ``nan`` value.\n        If the polynomial is discontinuous across a breakpoint, and\n        there is a sign change across the breakpoint, this is reported\n        if the `discont` parameter is True.\n\n        At the moment, there is not an actual implementation.\n        "
        raise NotImplementedError('At the moment there is not a GPU implementation for solve')

    def roots(self, discontinuity=True, extrapolate=None):
        if False:
            print('Hello World!')
        "\n        Find real roots of the piecewise polynomial.\n\n        Parameters\n        ----------\n        discontinuity : bool, optional\n            Whether to report sign changes across discontinuities at\n            breakpoints as roots.\n        extrapolate : {bool, 'periodic', None}, optional\n            If bool, determines whether to return roots from the polynomial\n            extrapolated based on first and last intervals, 'periodic' works\n            the same as False. If None (default), use `self.extrapolate`.\n\n        Returns\n        -------\n        roots : ndarray\n            Roots of the polynomial(s).\n            If the PPoly object describes multiple polynomials, the\n            return value is an object array whose each element is an\n            ndarray containing the roots.\n\n        See Also\n        --------\n        PPoly.solve\n        "
        return self.solve(0, discontinuity, extrapolate)

    @classmethod
    def from_spline(cls, tck, extrapolate=None):
        if False:
            print('Hello World!')
        "\n        Construct a piecewise polynomial from a spline\n\n        Parameters\n        ----------\n        tck\n            A spline, as a (knots, coefficients, degree) tuple or\n            a BSpline object.\n        extrapolate : bool or 'periodic', optional\n            If bool, determines whether to extrapolate to out-of-bounds points\n            based on first and last intervals, or to return NaNs.\n            If 'periodic', periodic extrapolation is used. Default is True.\n        "
        if isinstance(tck, BSpline):
            (t, c, k) = tck.tck
            if extrapolate is None:
                extrapolate = tck.extrapolate
        else:
            (t, c, k) = tck
        spl = BSpline(t, c, k, extrapolate=extrapolate)
        cvals = cupy.empty((k + 1, len(t) - 1), dtype=c.dtype)
        for m in range(k, -1, -1):
            y = spl(t[:-1], nu=m)
            cvals[k - m, :] = y / spec.gamma(m + 1)
        return cls.construct_fast(cvals, t, extrapolate)

    @classmethod
    def from_bernstein_basis(cls, bp, extrapolate=None):
        if False:
            while True:
                i = 10
        "\n        Construct a piecewise polynomial in the power basis\n        from a polynomial in Bernstein basis.\n\n        Parameters\n        ----------\n        bp : BPoly\n            A Bernstein basis polynomial, as created by BPoly\n        extrapolate : bool or 'periodic', optional\n            If bool, determines whether to extrapolate to out-of-bounds points\n            based on first and last intervals, or to return NaNs.\n            If 'periodic', periodic extrapolation is used. Default is True.\n        "
        if not isinstance(bp, BPoly):
            raise TypeError('.from_bernstein_basis only accepts BPoly instances. Got %s instead.' % type(bp))
        dx = cupy.diff(bp.x)
        k = bp.c.shape[0] - 1
        rest = (None,) * (bp.c.ndim - 2)
        c = cupy.zeros_like(bp.c)
        for a in range(k + 1):
            factor = (-1) ** a * _comb(k, a) * bp.c[a]
            for s in range(a, k + 1):
                val = _comb(k - a, s - a) * (-1) ** s
                c[k - s] += factor * val / dx[(slice(None),) + rest] ** s
        if extrapolate is None:
            extrapolate = bp.extrapolate
        return cls.construct_fast(c, bp.x, extrapolate, bp.axis)

class BPoly(_PPolyBase):
    """
    Piecewise polynomial in terms of coefficients and breakpoints.

    The polynomial between ``x[i]`` and ``x[i + 1]`` is written in the

    Bernstein polynomial basis::

        S = sum(c[a, i] * b(a, k; x) for a in range(k+1)),

    where ``k`` is the degree of the polynomial, and::

        b(a, k; x) = binom(k, a) * t**a * (1 - t)**(k - a),

    with ``t = (x - x[i]) / (x[i+1] - x[i])`` and ``binom`` is the binomial
    coefficient.

    Parameters
    ----------
    c : ndarray, shape (k, m, ...)
        Polynomial coefficients, order `k` and `m` intervals
    x : ndarray, shape (m+1,)
        Polynomial breakpoints. Must be sorted in either increasing or
        decreasing order.
    extrapolate : bool, optional
        If bool, determines whether to extrapolate to out-of-bounds points
        based on first and last intervals, or to return NaNs. If 'periodic',
        periodic extrapolation is used. Default is True.
    axis : int, optional
        Interpolation axis. Default is zero.

    Attributes
    ----------
    x : ndarray
        Breakpoints.
    c : ndarray
        Coefficients of the polynomials. They are reshaped
        to a 3-D array with the last dimension representing
        the trailing dimensions of the original coefficient array.
    axis : int
        Interpolation axis.

    See also
    --------
    PPoly : piecewise polynomials in the power basis

    Notes
    -----
    Properties of Bernstein polynomials are well documented in the literature,
    see for example [1]_ [2]_ [3]_.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Bernstein_polynomial
    .. [2] Kenneth I. Joy, Bernstein polynomials,
       http://www.idav.ucdavis.edu/education/CAGDNotes/Bernstein-Polynomials.pdf
    .. [3] E. H. Doha, A. H. Bhrawy, and M. A. Saker, Boundary Value Problems,
           vol 2011, article ID 829546,
           `10.1155/2011/829543 <https://doi.org/10.1155/2011/829543>`_.

    Examples
    --------
    >>> from cupyx.scipy.interpolate import BPoly
    >>> x = [0, 1]
    >>> c = [[1], [2], [3]]
    >>> bp = BPoly(c, x)

    This creates a 2nd order polynomial

    .. math::

        B(x) = 1 \\times b_{0, 2}(x) + 2 \\times b_{1, 2}(x) +
               3 \\times b_{2, 2}(x) \\\\
             = 1 \\times (1-x)^2 + 2 \\times 2 x (1 - x) + 3 \\times x^2
    """

    def _evaluate(self, x, nu, extrapolate, out):
        if False:
            while True:
                i = 10
        if nu < 0:
            raise NotImplementedError('Cannot do antiderivatives in the B-basis yet.')
        _bpoly_evaluate(self.c.reshape(self.c.shape[0], self.c.shape[1], -1), self.x, x, nu, bool(extrapolate), out)

    def derivative(self, nu=1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new piecewise polynomial representing the derivative.\n\n        Parameters\n        ----------\n        nu : int, optional\n            Order of derivative to evaluate. Default is 1, i.e., compute the\n            first derivative. If negative, the antiderivative is returned.\n\n        Returns\n        -------\n        bp : BPoly\n            Piecewise polynomial of order k - nu representing the derivative of\n            this polynomial.\n        '
        if nu < 0:
            return self.antiderivative(-nu)
        if nu > 1:
            bp = self
            for k in range(nu):
                bp = bp.derivative()
            return bp
        if nu == 0:
            c2 = self.c.copy()
        else:
            rest = (None,) * (self.c.ndim - 2)
            k = self.c.shape[0] - 1
            dx = cupy.diff(self.x)[(None, slice(None)) + rest]
            c2 = k * cupy.diff(self.c, axis=0) / dx
        if c2.shape[0] == 0:
            c2 = cupy.zeros((1,) + c2.shape[1:], dtype=c2.dtype)
        return self.construct_fast(c2, self.x, self.extrapolate, self.axis)

    def antiderivative(self, nu=1):
        if False:
            return 10
        "\n        Construct a new piecewise polynomial representing the antiderivative.\n\n        Parameters\n        ----------\n        nu : int, optional\n            Order of antiderivative to evaluate. Default is 1, i.e., compute\n            the first integral. If negative, the derivative is returned.\n\n        Returns\n        -------\n        bp : BPoly\n            Piecewise polynomial of order k + nu representing the\n            antiderivative of this polynomial.\n\n        Notes\n        -----\n        If antiderivative is computed and ``self.extrapolate='periodic'``,\n        it will be set to False for the returned instance. This is done because\n        the antiderivative is no longer periodic and its correct evaluation\n        outside of the initially given x interval is difficult.\n        "
        if nu <= 0:
            return self.derivative(-nu)
        if nu > 1:
            bp = self
            for k in range(nu):
                bp = bp.antiderivative()
            return bp
        (c, x) = (self.c, self.x)
        k = c.shape[0]
        c2 = cupy.zeros((k + 1,) + c.shape[1:], dtype=c.dtype)
        c2[1:, ...] = cupy.cumsum(c, axis=0) / k
        delta = x[1:] - x[:-1]
        c2 *= delta[(None, slice(None)) + (None,) * (c.ndim - 2)]
        c2[:, 1:] += cupy.cumsum(c2[k, :], axis=0)[:-1]
        if self.extrapolate == 'periodic':
            extrapolate = False
        else:
            extrapolate = self.extrapolate
        return self.construct_fast(c2, x, extrapolate, axis=self.axis)

    def integrate(self, a, b, extrapolate=None):
        if False:
            while True:
                i = 10
        "\n        Compute a definite integral over a piecewise polynomial.\n\n        Parameters\n        ----------\n        a : float\n            Lower integration bound\n        b : float\n            Upper integration bound\n        extrapolate : {bool, 'periodic', None}, optional\n            Whether to extrapolate to out-of-bounds points based on first\n            and last intervals, or to return NaNs. If 'periodic', periodic\n            extrapolation is used. If None (default), use `self.extrapolate`.\n\n        Returns\n        -------\n        array_like\n            Definite integral of the piecewise polynomial over [a, b]\n        "
        ib = self.antiderivative()
        if extrapolate is None:
            extrapolate = self.extrapolate
        if extrapolate != 'periodic':
            ib.extrapolate = extrapolate
        if extrapolate == 'periodic':
            if a <= b:
                sign = 1
            else:
                (a, b) = (b, a)
                sign = -1
            (xs, xe) = (self.x[0], self.x[-1])
            period = xe - xs
            interval = b - a
            (n_periods, left) = divmod(interval, period)
            res = n_periods * (ib(xe) - ib(xs))
            a = xs + (a - xs) % period
            b = a + left
            if b <= xe:
                res += ib(b) - ib(a)
            else:
                res += ib(xe) - ib(a) + ib(xs + left + a - xe) - ib(xs)
            return sign * res
        else:
            return ib(b) - ib(a)

    def extend(self, c, x):
        if False:
            for i in range(10):
                print('nop')
        k = max(self.c.shape[0], c.shape[0])
        self.c = self._raise_degree(self.c, k - self.c.shape[0])
        c = self._raise_degree(c, k - c.shape[0])
        return _PPolyBase.extend(self, c, x)
    extend.__doc__ = _PPolyBase.extend.__doc__

    @staticmethod
    def _raise_degree(c, d):
        if False:
            while True:
                i = 10
        '\n        Raise a degree of a polynomial in the Bernstein basis.\n\n        Given the coefficients of a polynomial degree `k`, return (the\n        coefficients of) the equivalent polynomial of degree `k+d`.\n\n        Parameters\n        ----------\n        c : array_like\n            coefficient array, 1-D\n        d : integer\n\n        Returns\n        -------\n        array\n            coefficient array, 1-D array of length `c.shape[0] + d`\n\n        Notes\n        -----\n        This uses the fact that a Bernstein polynomial `b_{a, k}` can be\n        identically represented as a linear combination of polynomials of\n        a higher degree `k+d`:\n\n            .. math:: b_{a, k} = comb(k, a) \\sum_{j=0}^{d} b_{a+j, k+d} \\\n                                 comb(d, j) / comb(k+d, a+j)\n        '
        if d == 0:
            return c
        k = c.shape[0] - 1
        out = cupy.zeros((c.shape[0] + d,) + c.shape[1:], dtype=c.dtype)
        for a in range(c.shape[0]):
            f = c[a] * _comb(k, a)
            for j in range(d + 1):
                out[a + j] += f * _comb(d, j) / _comb(k + d, a + j)
        return out

    @classmethod
    def from_power_basis(cls, pp, extrapolate=None):
        if False:
            print('Hello World!')
        "\n        Construct a piecewise polynomial in Bernstein basis\n        from a power basis polynomial.\n\n        Parameters\n        ----------\n        pp : PPoly\n            A piecewise polynomial in the power basis\n        extrapolate : bool or 'periodic', optional\n            If bool, determines whether to extrapolate to out-of-bounds points\n            based on first and last intervals, or to return NaNs.\n            If 'periodic', periodic extrapolation is used. Default is True.\n        "
        if not isinstance(pp, PPoly):
            raise TypeError('.from_power_basis only accepts PPoly instances. Got %s instead.' % type(pp))
        dx = cupy.diff(pp.x)
        k = pp.c.shape[0] - 1
        rest = (None,) * (pp.c.ndim - 2)
        c = cupy.zeros_like(pp.c)
        for a in range(k + 1):
            factor = pp.c[a] / _comb(k, k - a) * dx[(slice(None),) + rest] ** (k - a)
            for j in range(k - a, k + 1):
                c[j] += factor * _comb(j, k - a)
        if extrapolate is None:
            extrapolate = pp.extrapolate
        return cls.construct_fast(c, pp.x, extrapolate, pp.axis)

    @classmethod
    def from_derivatives(cls, xi, yi, orders=None, extrapolate=None):
        if False:
            i = 10
            return i + 15
        "\n        Construct a piecewise polynomial in the Bernstein basis,\n        compatible with the specified values and derivatives at breakpoints.\n\n        Parameters\n        ----------\n        xi : array_like\n            sorted 1-D array of x-coordinates\n        yi : array_like or list of array_likes\n            ``yi[i][j]`` is the ``j`` th derivative known at ``xi[i]``\n        orders : None or int or array_like of ints. Default: None.\n            Specifies the degree of local polynomials. If not None, some\n            derivatives are ignored.\n        extrapolate : bool or 'periodic', optional\n            If bool, determines whether to extrapolate to out-of-bounds points\n            based on first and last intervals, or to return NaNs.\n            If 'periodic', periodic extrapolation is used. Default is True.\n\n        Notes\n        -----\n        If ``k`` derivatives are specified at a breakpoint ``x``, the\n        constructed polynomial is exactly ``k`` times continuously\n        differentiable at ``x``, unless the ``order`` is provided explicitly.\n        In the latter case, the smoothness of the polynomial at\n        the breakpoint is controlled by the ``order``.\n\n        Deduces the number of derivatives to match at each end\n        from ``order`` and the number of derivatives available. If\n        possible it uses the same number of derivatives from\n        each end; if the number is odd it tries to take the\n        extra one from y2. In any case if not enough derivatives\n        are available at one end or another it draws enough to\n        make up the total from the other end.\n\n        If the order is too high and not enough derivatives are available,\n        an exception is raised.\n\n        Examples\n        --------\n        >>> from cupyx.scipy.interpolate import BPoly\n        >>> BPoly.from_derivatives([0, 1], [[1, 2], [3, 4]])\n\n        Creates a polynomial `f(x)` of degree 3, defined on `[0, 1]`\n        such that `f(0) = 1, df/dx(0) = 2, f(1) = 3, df/dx(1) = 4`\n\n        >>> BPoly.from_derivatives([0, 1, 2], [[0, 1], [0], [2]])\n\n        Creates a piecewise polynomial `f(x)`, such that\n        `f(0) = f(1) = 0`, `f(2) = 2`, and `df/dx(0) = 1`.\n        Based on the number of derivatives provided, the order of the\n        local polynomials is 2 on `[0, 1]` and 1 on `[1, 2]`.\n        Notice that no restriction is imposed on the derivatives at\n        ``x = 1`` and ``x = 2``.\n\n        Indeed, the explicit form of the polynomial is::\n\n            f(x) = | x * (1 - x),  0 <= x < 1\n                   | 2 * (x - 1),  1 <= x <= 2\n\n        So that f'(1-0) = -1 and f'(1+0) = 2\n        "
        xi = cupy.asarray(xi)
        if len(xi) != len(yi):
            raise ValueError('xi and yi need to have the same length')
        if cupy.any(xi[1:] - xi[:1] <= 0):
            raise ValueError('x coordinates are not in increasing order')
        m = len(xi) - 1
        try:
            k = max((len(yi[i]) + len(yi[i + 1]) for i in range(m)))
        except TypeError as e:
            raise ValueError('Using a 1-D array for y? Please .reshape(-1, 1).') from e
        if orders is None:
            orders = [None] * m
        else:
            if isinstance(orders, (int, cupy.integer)):
                orders = [orders] * m
            k = max(k, max(orders))
            if any((o <= 0 for o in orders)):
                raise ValueError('Orders must be positive.')
        c = []
        for i in range(m):
            (y1, y2) = (yi[i], yi[i + 1])
            if orders[i] is None:
                (n1, n2) = (len(y1), len(y2))
            else:
                n = orders[i] + 1
                n1 = min(n // 2, len(y1))
                n2 = min(n - n1, len(y2))
                n1 = min(n - n2, len(y2))
                if n1 + n2 != n:
                    mesg = 'Point %g has %d derivatives, point %g has %d derivatives, but order %d requested' % (xi[i], len(y1), xi[i + 1], len(y2), orders[i])
                    raise ValueError(mesg)
                if not (n1 <= len(y1) and n2 <= len(y2)):
                    raise ValueError('`order` input incompatible with length y1 or y2.')
            b = BPoly._construct_from_derivatives(xi[i], xi[i + 1], y1[:n1], y2[:n2])
            if len(b) < k:
                b = BPoly._raise_degree(b, k - len(b))
            c.append(b)
        c = cupy.asarray(c)
        return cls(c.swapaxes(0, 1), xi, extrapolate)

    @staticmethod
    def _construct_from_derivatives(xa, xb, ya, yb):
        if False:
            while True:
                i = 10
        "\n        Compute the coefficients of a polynomial in the Bernstein basis\n        given the values and derivatives at the edges.\n\n        Return the coefficients of a polynomial in the Bernstein basis\n        defined on ``[xa, xb]`` and having the values and derivatives at the\n        endpoints `xa` and `xb` as specified by `ya`` and `yb`.\n\n        The polynomial constructed is of the minimal possible degree, i.e.,\n        if the lengths of `ya` and `yb` are `na` and `nb`, the degree\n        of the polynomial is ``na + nb - 1``.\n\n        Parameters\n        ----------\n        xa : float\n            Left-hand end point of the interval\n        xb : float\n            Right-hand end point of the interval\n        ya : array_like\n            Derivatives at `xa`. `ya[0]` is the value of the function, and\n            `ya[i]` for ``i > 0`` is the value of the ``i``th derivative.\n        yb : array_like\n            Derivatives at `xb`.\n\n        Returns\n        -------\n        array\n            coefficient array of a polynomial having specified derivatives\n\n        Notes\n        -----\n        This uses several facts from life of Bernstein basis functions.\n        First of all,\n\n            .. math:: b'_{a, n} = n (b_{a-1, n-1} - b_{a, n-1})\n\n        If B(x) is a linear combination of the form\n\n            .. math:: B(x) = \\sum_{a=0}^{n} c_a b_{a, n},\n\n        then :math: B'(x) = n \\sum_{a=0}^{n-1} (c_{a+1} - c_{a}) b_{a, n-1}.\n        Iterating the latter one, one finds for the q-th derivative\n\n            .. math:: B^{q}(x) = n!/(n-q)! \\sum_{a=0}^{n-q} Q_a b_{a, n-q},\n\n        with\n\n            .. math:: Q_a = \\sum_{j=0}^{q} (-)^{j+q} comb(q, j) c_{j+a}\n\n        This way, only `a=0` contributes to :math: `B^{q}(x = xa)`, and\n        `c_q` are found one by one by iterating `q = 0, ..., na`.\n\n        At ``x = xb`` it's the same with ``a = n - q``.\n        "
        (ya, yb) = (cupy.asarray(ya), cupy.asarray(yb))
        if ya.shape[1:] != yb.shape[1:]:
            raise ValueError('Shapes of ya {} and yb {} are incompatible'.format(ya.shape, yb.shape))
        (dta, dtb) = (ya.dtype, yb.dtype)
        if cupy.issubdtype(dta, cupy.complexfloating) or cupy.issubdtype(dtb, cupy.complexfloating):
            dt = cupy.complex_
        else:
            dt = cupy.float_
        (na, nb) = (len(ya), len(yb))
        n = na + nb
        c = cupy.empty((na + nb,) + ya.shape[1:], dtype=dt)
        for q in range(0, na):
            c[q] = ya[q] / spec.poch(n - q, q) * (xb - xa) ** q
            for j in range(0, q):
                c[q] -= (-1) ** (j + q) * _comb(q, j) * c[j]
        for q in range(0, nb):
            c[-q - 1] = yb[q] / spec.poch(n - q, q) * (-1) ** q * (xb - xa) ** q
            for j in range(0, q):
                c[-q - 1] -= (-1) ** (j + 1) * _comb(q, j + 1) * c[-q + j]
        return c

class NdPPoly:
    """
    Piecewise tensor product polynomial

    The value at point ``xp = (x', y', z', ...)`` is evaluated by first
    computing the interval indices `i` such that::

        x[0][i[0]] <= x' < x[0][i[0]+1]
        x[1][i[1]] <= y' < x[1][i[1]+1]
        ...

    and then computing::

        S = sum(c[k0-m0-1,...,kn-mn-1,i[0],...,i[n]]
                * (xp[0] - x[0][i[0]])**m0
                * ...
                * (xp[n] - x[n][i[n]])**mn
                for m0 in range(k[0]+1)
                ...
                for mn in range(k[n]+1))

    where ``k[j]`` is the degree of the polynomial in dimension j. This
    representation is the piecewise multivariate power basis.

    Parameters
    ----------
    c : ndarray, shape (k0, ..., kn, m0, ..., mn, ...)
        Polynomial coefficients, with polynomial order `kj` and
        `mj+1` intervals for each dimension `j`.
    x : ndim-tuple of ndarrays, shapes (mj+1,)
        Polynomial breakpoints for each dimension. These must be
        sorted in increasing order.
    extrapolate : bool, optional
        Whether to extrapolate to out-of-bounds points based on first
        and last intervals, or to return NaNs. Default: True.

    Attributes
    ----------
    x : tuple of ndarrays
        Breakpoints.
    c : ndarray
        Coefficients of the polynomials.

    See also
    --------
    PPoly : piecewise polynomials in 1D

    Notes
    -----
    High-order polynomials in the power basis can be numerically
    unstable.
    """

    def __init__(self, c, x, extrapolate=None):
        if False:
            for i in range(10):
                print('nop')
        self.x = tuple((cupy.ascontiguousarray(v, dtype=cupy.float64) for v in x))
        self.c = cupy.asarray(c)
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = bool(extrapolate)
        ndim = len(self.x)
        if any((v.ndim != 1 for v in self.x)):
            raise ValueError('x arrays must all be 1-dimensional')
        if any((v.size < 2 for v in self.x)):
            raise ValueError('x arrays must all contain at least 2 points')
        if c.ndim < 2 * ndim:
            raise ValueError('c must have at least 2*len(x) dimensions')
        if any((cupy.any(v[1:] - v[:-1] < 0) for v in self.x)):
            raise ValueError('x-coordinates are not in increasing order')
        if any((a != b.size - 1 for (a, b) in zip(c.shape[ndim:2 * ndim], self.x))):
            raise ValueError('x and c do not agree on the number of intervals')
        dtype = self._get_dtype(self.c.dtype)
        self.c = cupy.ascontiguousarray(self.c, dtype=dtype)

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None):
        if False:
            print('Hello World!')
        '\n        Construct the piecewise polynomial without making checks.\n\n        Takes the same parameters as the constructor. Input arguments\n        ``c`` and ``x`` must be arrays of the correct shape and type.  The\n        ``c`` array can only be of dtypes float and complex, and ``x``\n        array must have dtype float.\n        '
        self = object.__new__(cls)
        self.c = c
        self.x = x
        if extrapolate is None:
            extrapolate = True
        self.extrapolate = extrapolate
        return self

    def _get_dtype(self, dtype):
        if False:
            return 10
        if cupy.issubdtype(dtype, cupy.complexfloating) or cupy.issubdtype(self.c.dtype, cupy.complexfloating):
            return cupy.complex_
        else:
            return cupy.float_

    def _ensure_c_contiguous(self):
        if False:
            print('Hello World!')
        if not self.c.flags.c_contiguous:
            self.c = self.c.copy()
        if not isinstance(self.x, tuple):
            self.x = tuple(self.x)

    def __call__(self, x, nu=None, extrapolate=None):
        if False:
            while True:
                i = 10
        '\n        Evaluate the piecewise polynomial or its derivative\n\n        Parameters\n        ----------\n        x : array-like\n            Points to evaluate the interpolant at.\n        nu : tuple, optional\n            Orders of derivatives to evaluate. Each must be non-negative.\n        extrapolate : bool, optional\n            Whether to extrapolate to out-of-bounds points based on first\n            and last intervals, or to return NaNs.\n\n        Returns\n        -------\n        y : array-like\n            Interpolated values. Shape is determined by replacing\n            the interpolation axis in the original array with the shape of x.\n\n        Notes\n        -----\n        Derivatives are evaluated piecewise for each polynomial\n        segment, even if the polynomial is not differentiable at the\n        breakpoints. The polynomial intervals are considered half-open,\n        ``[a, b)``, except for the last interval which is closed\n        ``[a, b]``.\n        '
        if extrapolate is None:
            extrapolate = self.extrapolate
        else:
            extrapolate = bool(extrapolate)
        ndim = len(self.x)
        x = _ndim_coords_from_arrays(x)
        x_shape = x.shape
        x = cupy.ascontiguousarray(x.reshape(-1, x.shape[-1]), dtype=cupy.float64)
        if nu is None:
            nu = cupy.zeros((ndim,), dtype=cupy.int64)
        else:
            nu = cupy.asarray(nu, dtype=cupy.int64)
            if nu.ndim != 1 or nu.shape[0] != ndim:
                raise ValueError('invalid number of derivative orders nu')
        dim1 = int(np.prod(self.c.shape[:ndim]))
        dim2 = int(np.prod(self.c.shape[ndim:2 * ndim]))
        dim3 = int(np.prod(self.c.shape[2 * ndim:]))
        ks = cupy.asarray(self.c.shape[:ndim], dtype=cupy.int64)
        out = cupy.empty((x.shape[0], dim3), dtype=self.c.dtype)
        self._ensure_c_contiguous()
        _ndppoly_evaluate(self.c.reshape(dim1, dim2, dim3), self.x, ks, x, nu, bool(extrapolate), out)
        return out.reshape(x_shape[:-1] + self.c.shape[2 * ndim:])

    def _derivative_inplace(self, nu, axis):
        if False:
            while True:
                i = 10
        '\n        Compute 1-D derivative along a selected dimension in-place\n        May result to non-contiguous c array.\n        '
        if nu < 0:
            return self._antiderivative_inplace(-nu, axis)
        ndim = len(self.x)
        axis = axis % ndim
        if nu == 0:
            return
        else:
            sl = [slice(None)] * ndim
            sl[axis] = slice(None, -nu, None)
            c2 = self.c[tuple(sl)]
        if c2.shape[axis] == 0:
            shp = list(c2.shape)
            shp[axis] = 1
            c2 = cupy.zeros(shp, dtype=c2.dtype)
        factor = spec.poch(cupy.arange(c2.shape[axis], 0, -1), nu)
        sl = [None] * c2.ndim
        sl[axis] = slice(None)
        c2 *= factor[tuple(sl)]
        self.c = c2

    def _antiderivative_inplace(self, nu, axis):
        if False:
            i = 10
            return i + 15
        '\n        Compute 1-D antiderivative along a selected dimension\n        May result to non-contiguous c array.\n        '
        if nu <= 0:
            return self._derivative_inplace(-nu, axis)
        ndim = len(self.x)
        axis = axis % ndim
        perm = list(range(ndim))
        (perm[0], perm[axis]) = (perm[axis], perm[0])
        perm = perm + list(range(ndim, self.c.ndim))
        c = self.c.transpose(perm)
        c2 = cupy.zeros((c.shape[0] + nu,) + c.shape[1:], dtype=c.dtype)
        c2[:-nu] = c
        factor = spec.poch(cupy.arange(c.shape[0], 0, -1), nu)
        c2[:-nu] /= factor[(slice(None),) + (None,) * (c.ndim - 1)]
        perm2 = list(range(c2.ndim))
        (perm2[1], perm2[ndim + axis]) = (perm2[ndim + axis], perm2[1])
        c2 = c2.transpose(perm2)
        c2 = c2.copy()
        _fix_continuity(c2.reshape(c2.shape[0], c2.shape[1], -1), self.x[axis], nu - 1)
        c2 = c2.transpose(perm2)
        c2 = c2.transpose(perm)
        self.c = c2

    def derivative(self, nu):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new piecewise polynomial representing the derivative.\n\n        Parameters\n        ----------\n        nu : ndim-tuple of int\n            Order of derivatives to evaluate for each dimension.\n            If negative, the antiderivative is returned.\n\n        Returns\n        -------\n        pp : NdPPoly\n            Piecewise polynomial of orders (k[0] - nu[0], ..., k[n] - nu[n])\n            representing the derivative of this polynomial.\n\n        Notes\n        -----\n        Derivatives are evaluated piecewise for each polynomial\n        segment, even if the polynomial is not differentiable at the\n        breakpoints. The polynomial intervals in each dimension are\n        considered half-open, ``[a, b)``, except for the last interval\n        which is closed ``[a, b]``.\n        '
        p = self.construct_fast(self.c.copy(), self.x, self.extrapolate)
        for (axis, n) in enumerate(nu):
            p._derivative_inplace(n, axis)
        p._ensure_c_contiguous()
        return p

    def antiderivative(self, nu):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a new piecewise polynomial representing the antiderivative.\n        Antiderivative is also the indefinite integral of the function,\n        and derivative is its inverse operation.\n\n        Parameters\n        ----------\n        nu : ndim-tuple of int\n            Order of derivatives to evaluate for each dimension.\n            If negative, the derivative is returned.\n\n        Returns\n        -------\n        pp : PPoly\n            Piecewise polynomial of order k2 = k + n representing\n            the antiderivative of this polynomial.\n\n        Notes\n        -----\n        The antiderivative returned by this function is continuous and\n        continuously differentiable to order n-1, up to floating point\n        rounding error.\n        '
        p = self.construct_fast(self.c.copy(), self.x, self.extrapolate)
        for (axis, n) in enumerate(nu):
            p._antiderivative_inplace(n, axis)
        p._ensure_c_contiguous()
        return p

    def integrate_1d(self, a, b, axis, extrapolate=None):
        if False:
            print('Hello World!')
        '\n        Compute NdPPoly representation for one dimensional definite integral\n        The result is a piecewise polynomial representing the integral:\n\n        .. math::\n           p(y, z, ...) = \\int_a^b dx\\, p(x, y, z, ...)\n\n        where the dimension integrated over is specified with the\n        `axis` parameter.\n\n        Parameters\n        ----------\n        a, b : float\n            Lower and upper bound for integration.\n        axis : int\n            Dimension over which to compute the 1-D integrals\n        extrapolate : bool, optional\n            Whether to extrapolate to out-of-bounds points based on first\n            and last intervals, or to return NaNs.\n\n        Returns\n        -------\n        ig : NdPPoly or array-like\n            Definite integral of the piecewise polynomial over [a, b].\n            If the polynomial was 1D, an array is returned,\n            otherwise, an NdPPoly object.\n        '
        if extrapolate is None:
            extrapolate = self.extrapolate
        else:
            extrapolate = bool(extrapolate)
        ndim = len(self.x)
        axis = int(axis) % ndim
        c = self.c
        swap = list(range(c.ndim))
        swap.insert(0, swap[axis])
        del swap[axis + 1]
        swap.insert(1, swap[ndim + axis])
        del swap[ndim + axis + 1]
        c = c.transpose(swap)
        p = PPoly.construct_fast(c.reshape(c.shape[0], c.shape[1], -1), self.x[axis], extrapolate=extrapolate)
        out = p.integrate(a, b, extrapolate=extrapolate)
        if ndim == 1:
            return out.reshape(c.shape[2:])
        else:
            c = out.reshape(c.shape[2:])
            x = self.x[:axis] + self.x[axis + 1:]
            return self.construct_fast(c, x, extrapolate=extrapolate)

    def integrate(self, ranges, extrapolate=None):
        if False:
            i = 10
            return i + 15
        '\n        Compute a definite integral over a piecewise polynomial.\n\n        Parameters\n        ----------\n        ranges : ndim-tuple of 2-tuples float\n            Sequence of lower and upper bounds for each dimension,\n            ``[(a[0], b[0]), ..., (a[ndim-1], b[ndim-1])]``\n        extrapolate : bool, optional\n            Whether to extrapolate to out-of-bounds points based on first\n            and last intervals, or to return NaNs.\n\n        Returns\n        -------\n        ig : array_like\n            Definite integral of the piecewise polynomial over\n            [a[0], b[0]] x ... x [a[ndim-1], b[ndim-1]]\n        '
        ndim = len(self.x)
        if extrapolate is None:
            extrapolate = self.extrapolate
        else:
            extrapolate = bool(extrapolate)
        if not hasattr(ranges, '__len__') or len(ranges) != ndim:
            raise ValueError('Range not a sequence of correct length')
        self._ensure_c_contiguous()
        c = self.c
        for (n, (a, b)) in enumerate(ranges):
            swap = list(range(c.ndim))
            swap.insert(1, swap[ndim - n])
            del swap[ndim - n + 1]
            c = c.transpose(swap)
            p = PPoly.construct_fast(c, self.x[n], extrapolate=extrapolate)
            out = p.integrate(a, b, extrapolate=extrapolate)
            c = out.reshape(c.shape[2:])
        return c