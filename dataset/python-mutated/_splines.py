import cupy
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupy._core.internal import _normalize_axis_index
from cupyx.scipy.signal._signaltools import lfilter
from cupyx.scipy.signal._arraytools import axis_slice, axis_assign, axis_reverse
from cupyx.scipy.signal._iir_utils import collapse_2d, apply_iir_sos
if runtime.is_hip:
    SYMIIR2_KERNEL = '#include <hip/hip_runtime.h>\n'
else:
    SYMIIR2_KERNEL = '\n#include <cuda_runtime.h>\n#include <device_launch_parameters.h>\n'
SYMIIR2_KERNEL = SYMIIR2_KERNEL + '\n#include <cupy/math_constants.h>\n#include <cupy/carray.cuh>\n\ntemplate<typename T>\n__device__ T _compute_symiirorder2_fwd_hc(\n        const int k, const T cs, const T r, const T omega) {\n    T base;\n\n    if(k < 0) {\n        return 0;\n    }\n\n    if(omega == 0.0) {\n        base = cs * pow(r, ((T) k)) * (k + 1);\n    } else if(omega == M_PI) {\n        base = cs * pow(r, ((T) k)) * (k + 1) * (1 - 2 * (k % 2));\n    } else {\n        base = (cs * pow(r, ((T) k)) * sin(omega * (k + 1)) /\n                sin(omega));\n    }\n    return base;\n}\n\ntemplate<typename T>\n__global__ void compute_symiirorder2_fwd_sc(\n        const int n, const int off, const T* cs_ptr, const T* r_ptr,\n        const T* omega_ptr, const double precision, bool* valid, T* out) {\n\n    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n    if(idx + off >= n) {\n        return;\n    }\n\n    const T cs = cs_ptr[0];\n    const T r = r_ptr[0];\n    const T omega = omega_ptr[0];\n\n    T val = _compute_symiirorder2_fwd_hc<T>(idx + off + 1, cs, r, omega);\n    T err = val * val;\n\n    out[idx] = val;\n    valid[idx] = err <= precision;\n}\n\ntemplate<typename T>\n__device__ T _compute_symiirorder2_bwd_hs(\n        const int ki, const T cs, const T rsq, const T omega) {\n    T c0;\n    T gamma;\n\n    T cssq = cs * cs;\n    int k = abs(ki);\n    T rsupk = pow(rsq, ((T) k) / ((T) 2.0));\n\n\n    if(omega == 0.0) {\n        c0 = (1 + rsq) / ((1 - rsq) * (1 - rsq) * (1 - rsq)) * cssq;\n        gamma = (1 - rsq) / (1 + rsq);\n        return c0 * rsupk * (1 + gamma * k);\n    }\n\n    if(omega == M_PI) {\n        c0 = (1 + rsq) / ((1 - rsq) * (1 - rsq) * (1 - rsq)) * cssq;\n        gamma = (1 - rsq) / (1 + rsq) * (1 - 2 * (k % 2));\n        return c0 * rsupk * (1 + gamma * k);\n    }\n\n    c0 = (cssq * (1.0 + rsq) / (1.0 - rsq) /\n                (1 - 2 * rsq * cos(2 * omega) + rsq * rsq));\n    gamma = (1.0 - rsq) / (1.0 + rsq) / tan(omega);\n    return c0 * rsupk * (cos(omega * k) + gamma * sin(omega * k));\n}\n\ntemplate<typename T>\n__global__ void compute_symiirorder2_bwd_sc(\n        const int n, const int off, const int l_off, const int r_off,\n        const T* cs_ptr, const T* rsq_ptr, const T* omega_ptr,\n        const double precision, bool* valid, T* out) {\n\n    int idx = blockDim.x * blockIdx.x + threadIdx.x;\n    if(idx + off >= n) {\n        return;\n    }\n\n    const T cs = cs_ptr[0];\n    const T rsq = rsq_ptr[0];\n    const T omega = omega_ptr[0];\n\n    T v1 = _compute_symiirorder2_bwd_hs<T>(idx + l_off + off, cs, rsq, omega);\n    T v2 = _compute_symiirorder2_bwd_hs<T>(idx + r_off + off, cs, rsq, omega);\n\n    T diff = v1 + v2;\n    T err = diff * diff;\n    out[idx] = diff;\n    valid[idx] = err <= precision;\n}\n'
SYMIIR2_MODULE = cupy.RawModule(code=SYMIIR2_KERNEL, options=('-std=c++11',), name_expressions=[f'compute_symiirorder2_bwd_sc<{t}>' for t in ['float', 'double']] + [f'compute_symiirorder2_fwd_sc<{t}>' for t in ['float', 'double']])

def _get_module_func(module, func_name, *template_args):
    if False:
        print('Hello World!')
    args_dtypes = [get_typename(arg.dtype) for arg in template_args]
    template = ', '.join(args_dtypes)
    kernel_name = f'{func_name}<{template}>' if template_args else func_name
    kernel = module.get_function(kernel_name)
    return kernel

def _find_initial_cond(all_valid, cum_poly, n, off=0, axis=-1):
    if False:
        while True:
            i = 10
    indices = cupy.where(all_valid)[0] + 1 + off
    zi = cupy.nan
    if indices.size > 0:
        zi = cupy.where(indices[0] >= n, cupy.nan, axis_slice(cum_poly, indices[0] - 1 - off, indices[0] - off, axis=axis))
    return zi

def _symiirorder1_nd(input, c0, z1, precision=-1.0, axis=-1):
    if False:
        i = 10
        return i + 15
    axis = _normalize_axis_index(axis, input.ndim)
    input_shape = input.shape
    input_ndim = input.ndim
    if input.ndim > 1:
        (input, input_shape) = collapse_2d(input, axis)
    if cupy.abs(z1) >= 1:
        raise ValueError('|z1| must be less than 1.0')
    if precision <= 0.0 or precision > 1.0:
        if input.dtype is cupy.dtype(cupy.float64):
            precision = 1e-06
        elif input.dtype is cupy.dtype(cupy.float32):
            precision = 0.001
        else:
            precision = 10 ** (-cupy.finfo(input.dtype).iexp)
    precision *= precision
    pos = cupy.arange(1, input_shape[-1] + 1, dtype=input.dtype)
    pow_z1 = z1 ** pos
    diff = pow_z1 * cupy.conjugate(pow_z1)
    cum_poly = cupy.cumsum(pow_z1 * input, axis=-1) + axis_slice(input, 0, 1, axis=-1)
    all_valid = diff <= precision
    zi = _find_initial_cond(all_valid, cum_poly, input_shape[-1])
    if cupy.any(cupy.isnan(zi)):
        raise ValueError('Sum to find symmetric boundary conditions did not converge.')
    zi_shape = (1, 4)
    if input_ndim > 1:
        zi_shape = (1, input.shape[0], 4)
    all_zi = cupy.zeros(zi_shape, dtype=input.dtype)
    all_zi = axis_assign(all_zi, zi, 3, 4)
    coef = cupy.r_[1, 0, 0, 1, -z1, 0]
    coef = cupy.atleast_2d(coef)
    (y1, _) = apply_iir_sos(axis_slice(input, 1), coef, zi=all_zi, dtype=input.dtype, apply_fir=False)
    y1 = cupy.c_[zi, y1]
    zi = -c0 / (z1 - 1.0) * axis_slice(y1, -1)
    all_zi = axis_assign(all_zi, zi, 3, 4)
    coef = cupy.r_[c0, 0, 0, 1, -z1, 0]
    coef = cupy.atleast_2d(coef)
    (out, _) = apply_iir_sos(axis_slice(y1, -2, step=-1), coef, zi=all_zi, dtype=input.dtype)
    if input_ndim > 1:
        out = cupy.c_[axis_reverse(out), zi]
    else:
        out = cupy.r_[axis_reverse(out), zi]
    if input_ndim > 1:
        out = out.reshape(input_shape)
        out = cupy.moveaxis(out, -1, axis)
        if not out.flags.c_contiguous:
            out = out.copy()
    return out

def symiirorder1(input, c0, z1, precision=-1.0):
    if False:
        while True:
            i = 10
    '\n    Implement a smoothing IIR filter with mirror-symmetric boundary conditions\n    using a cascade of first-order sections.  The second section uses a\n    reversed sequence.  This implements a system with the following\n    transfer function and mirror-symmetric boundary conditions::\n\n                           c0\n           H(z) = ---------------------\n                   (1-z1/z) (1 - z1 z)\n\n    The resulting signal will have mirror symmetric boundary conditions\n    as well.\n\n    Parameters\n    ----------\n    input : ndarray\n        The input signal.\n    c0, z1 : scalar\n        Parameters in the transfer function.\n    precision :\n        Specifies the precision for calculating initial conditions\n        of the recursive filter based on mirror-symmetric input.\n\n    Returns\n    -------\n    output : ndarray\n        The filtered signal.\n    '
    c0 = cupy.asarray([c0], input.dtype)
    z1 = cupy.asarray([z1], input.dtype)
    if cupy.abs(z1) >= 1:
        raise ValueError('|z1| must be less than 1.0')
    if precision <= 0.0 or precision > 1.0:
        precision = cupy.finfo(input.dtype).resolution
    precision *= precision
    pos = cupy.arange(1, input.size + 1, dtype=input.dtype)
    pow_z1 = z1 ** pos
    diff = pow_z1 * cupy.conjugate(pow_z1)
    cum_poly = cupy.cumsum(pow_z1 * input) + input[0]
    all_valid = diff <= precision
    zi = _find_initial_cond(all_valid, cum_poly, input.size)
    if cupy.isnan(zi):
        raise ValueError('Sum to find symmetric boundary conditions did not converge.')
    a = cupy.r_[1, -z1]
    a = a.astype(input.dtype)
    (y1, _) = lfilter(cupy.ones(1, dtype=input.dtype), a, input[1:], zi=zi)
    y1 = cupy.r_[zi, y1]
    zi = -c0 / (z1 - 1.0) * y1[-1]
    a = cupy.r_[1, -z1]
    a = a.astype(input.dtype)
    (out, _) = lfilter(c0, a, y1[:-1][::-1], zi=zi)
    return cupy.r_[out[::-1], zi]

def _compute_symiirorder2_fwd_hc(k, cs, r, omega):
    if False:
        for i in range(10):
            print('nop')
    base = None
    if omega == 0.0:
        base = cs * cupy.power(r, k) * (k + 1)
    elif omega == cupy.pi:
        base = cs * cupy.power(r, k) * (k + 1) * (1 - 2 * (k % 2))
    else:
        base = cs * cupy.power(r, k) * cupy.sin(omega * (k + 1)) / cupy.sin(omega)
    return cupy.where(k < 0, 0.0, base)

def _compute_symiirorder2_bwd_hs(k, cs, rsq, omega):
    if False:
        i = 10
        return i + 15
    cssq = cs * cs
    k = cupy.abs(k)
    rsupk = cupy.power(rsq, k / 2.0)
    if omega == 0.0:
        c0 = (1 + rsq) / ((1 - rsq) * (1 - rsq) * (1 - rsq)) * cssq
        gamma = (1 - rsq) / (1 + rsq)
        return c0 * rsupk * (1 + gamma * k)
    if omega == cupy.pi:
        c0 = (1 + rsq) / ((1 - rsq) * (1 - rsq) * (1 - rsq)) * cssq
        gamma = (1 - rsq) / (1 + rsq) * (1 - 2 * (k % 2))
        return c0 * rsupk * (1 + gamma * k)
    c0 = cssq * (1.0 + rsq) / (1.0 - rsq) / (1 - 2 * rsq * cupy.cos(2 * omega) + rsq * rsq)
    gamma = (1.0 - rsq) / (1.0 + rsq) / cupy.tan(omega)
    return c0 * rsupk * (cupy.cos(omega * k) + gamma * cupy.sin(omega * k))

def _symiirorder2_nd(input, r, omega, precision=-1.0, axis=-1):
    if False:
        for i in range(10):
            print('nop')
    if r >= 1.0:
        raise ValueError('r must be less than 1.0')
    if precision <= 0.0 or precision > 1.0:
        if input.dtype is cupy.dtype(cupy.float64):
            precision = 1e-11
        elif input.dtype is cupy.dtype(cupy.float32):
            precision = 1e-06
        else:
            precision = 10 ** (-cupy.finfo(input.dtype).iexp)
    axis = _normalize_axis_index(axis, input.ndim)
    input_shape = input.shape
    input_ndim = input.ndim
    if input.ndim > 1:
        (input, input_shape) = collapse_2d(input, axis)
    block_sz = 128
    rsq = r * r
    a2 = 2 * r * cupy.cos(omega)
    a3 = -rsq
    cs = cupy.atleast_1d(1 - 2 * r * cupy.cos(omega) + rsq)
    omega = cupy.asarray(omega, cs.dtype)
    r = cupy.asarray(r, cs.dtype)
    rsq = cupy.asarray(rsq, cs.dtype)
    precision *= precision
    compute_symiirorder2_fwd_sc = _get_module_func(SYMIIR2_MODULE, 'compute_symiirorder2_fwd_sc', cs)
    diff = cupy.empty((block_sz + 1,), dtype=cs.dtype)
    all_valid = cupy.empty((block_sz + 1,), dtype=cupy.bool_)
    starting_diff = cupy.arange(2, dtype=input.dtype)
    starting_diff = _compute_symiirorder2_fwd_hc(starting_diff, cs, r, omega)
    y0 = cupy.nan
    y1 = cupy.nan
    for i in range(0, input.shape[-1] + 2, block_sz):
        compute_symiirorder2_fwd_sc((1,), (block_sz + 1,), (input.shape[-1] + 2, i, cs, r, omega, precision, all_valid, diff))
        input_slice = axis_slice(input, i, i + block_sz)
        diff_y0 = diff[:-1][:input_slice.shape[-1]]
        diff_y1 = diff[1:][:input_slice.shape[-1]]
        if cupy.isnan(y0):
            cum_poly_y0 = cupy.cumsum(diff_y0 * input_slice, axis=-1) + starting_diff[0] * axis_slice(input, 0, 1)
            y0 = _find_initial_cond(all_valid[:-1][:input_slice.shape[-1]], cum_poly_y0, input.shape[-1], i)
        if cupy.isnan(y1):
            cum_poly_y1 = cupy.cumsum(diff_y1 * input_slice, axis=-1) + starting_diff[0] * axis_slice(input, 1, 2) + starting_diff[1] * axis_slice(input, 0, 1)
            y1 = _find_initial_cond(all_valid[1:][:input_slice.shape[-1]], cum_poly_y1, input.shape[-1], i)
        if not cupy.any(cupy.isnan(cupy.r_[y0, y1])):
            break
    if cupy.any(cupy.isnan(cupy.r_[y0, y1])):
        raise ValueError('Sum to find symmetric boundary conditions did not converge.')
    zi_shape = (1, 4)
    if input_ndim > 1:
        zi_shape = (1, input.shape[0], 4)
    sos = cupy.atleast_2d(cupy.r_[cs, 0, 0, 1, -a2, -a3])
    sos = sos.astype(input.dtype)
    all_zi = cupy.zeros(zi_shape, dtype=input.dtype)
    all_zi = axis_assign(all_zi, y0, 2, 3)
    all_zi = axis_assign(all_zi, y1, 3, 4)
    (y_fwd, _) = apply_iir_sos(axis_slice(input, 2), sos, zi=all_zi, dtype=input.dtype)
    if input_ndim > 1:
        y_fwd = cupy.c_[y0, y1, y_fwd]
    else:
        y_fwd = cupy.r_[y0, y1, y_fwd]
    compute_symiirorder2_bwd_sc = _get_module_func(SYMIIR2_MODULE, 'compute_symiirorder2_bwd_sc', cs)
    diff = cupy.empty((block_sz,), dtype=cs.dtype)
    all_valid = cupy.empty((block_sz,), dtype=cupy.bool_)
    rev_input = axis_reverse(input)
    y0 = cupy.nan
    for i in range(0, input.shape[-1] + 1, block_sz):
        compute_symiirorder2_bwd_sc((1,), (block_sz,), (input.shape[-1] + 1, i, 0, 1, cs, cupy.asarray(rsq, cs.dtype), cupy.asarray(omega, cs.dtype), precision, all_valid, diff))
        input_slice = axis_slice(rev_input, i, i + block_sz)
        cum_poly_y0 = cupy.cumsum(diff[:input_slice.shape[-1]] * input_slice, axis=-1)
        y0 = _find_initial_cond(all_valid[:input_slice.shape[-1]], cum_poly_y0, input.shape[-1], i)
        if not cupy.any(cupy.isnan(y0)):
            break
    if cupy.any(cupy.isnan(y0)):
        raise ValueError('Sum to find symmetric boundary conditions did not converge.')
    y1 = cupy.nan
    for i in range(0, input.shape[-1] + 1, block_sz):
        compute_symiirorder2_bwd_sc((1,), (block_sz,), (input.size + 1, i, -1, 2, cs, cupy.asarray(rsq, cs.dtype), cupy.asarray(omega, cs.dtype), precision, all_valid, diff))
        input_slice = axis_slice(rev_input, i, i + block_sz)
        cum_poly_y1 = cupy.cumsum(diff[:input_slice.shape[-1]] * input_slice, axis=-1)
        y1 = _find_initial_cond(all_valid[:input_slice.size], cum_poly_y1, input.size, i)
        if not cupy.any(cupy.isnan(y1)):
            break
    if cupy.any(cupy.isnan(y1)):
        raise ValueError('Sum to find symmetric boundary conditions did not converge.')
    all_zi = axis_assign(all_zi, y0, 2, 3)
    all_zi = axis_assign(all_zi, y1, 3, 4)
    (out, _) = apply_iir_sos(axis_slice(y_fwd, -3, step=-1), sos, zi=all_zi)
    if input_ndim > 1:
        out = cupy.c_[axis_reverse(out), y1, y0]
    else:
        out = cupy.r_[axis_reverse(out), y1, y0]
    if input_ndim > 1:
        out = out.reshape(input_shape)
        out = cupy.moveaxis(out, -1, axis)
        if not out.flags.c_contiguous:
            out = out.copy()
    return out

def symiirorder2(input, r, omega, precision=-1.0):
    if False:
        i = 10
        return i + 15
    '\n    Implement a smoothing IIR filter with mirror-symmetric boundary conditions\n    using a cascade of second-order sections.  The second section uses a\n    reversed sequence.  This implements the following transfer function::\n\n                                  cs^2\n         H(z) = ---------------------------------------\n                (1 - a2/z - a3/z^2) (1 - a2 z - a3 z^2 )\n\n    where::\n\n          a2 = 2 * r * cos(omega)\n          a3 = - r ** 2\n          cs = 1 - 2 * r * cos(omega) + r ** 2\n\n    Parameters\n    ----------\n    input : ndarray\n        The input signal.\n    r, omega : float\n        Parameters in the transfer function.\n    precision : float\n        Specifies the precision for calculating initial conditions\n        of the recursive filter based on mirror-symmetric input.\n\n    Returns\n    -------\n    output : ndarray\n        The filtered signal.\n    '
    return _symiirorder2_nd(input, r, omega, precision)