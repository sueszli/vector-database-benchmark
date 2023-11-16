"""
Spline poles and boundary handling implemented as in SciPy

https://github.com/scipy/scipy/blob/master/scipy/ndimage/src/ni_splines.c
"""
import functools
import math
import operator
import textwrap
import cupy

def get_poles(order):
    if False:
        for i in range(10):
            print('nop')
    if order == 2:
        return (-0.1715728752538099,)
    elif order == 3:
        return (-0.2679491924311227,)
    elif order == 4:
        return (-0.36134122590022016, -0.013725429297339121)
    elif order == 5:
        return (-0.4305753470999738, -0.04309628820326465)
    else:
        raise ValueError('only order 2-5 supported')

def get_gain(poles):
    if False:
        for i in range(10):
            print('nop')
    return functools.reduce(operator.mul, [(1.0 - z) * (1.0 - 1.0 / z) for z in poles])

def _causal_init_code(mode):
    if False:
        print('Hello World!')
    'Code for causal initialization step of IIR filtering.\n\n    c is a 1d array of length n and z is a filter pole\n    '
    code = f'\n        // causal init for mode={mode}'
    if mode == 'mirror':
        code += '\n        z_i = z;\n        z_n_1 = pow(z, (P)(n - 1));\n\n        c[0] = c[0] + z_n_1 * c[(n - 1) * element_stride];\n        for (i = 1; i < min(n - 1, static_cast<idx_t>({n_boundary})); ++i) {{\n            c[0] += z_i * (c[i * element_stride] +\n                           z_n_1 * c[(n - 1 - i) * element_stride]);\n            z_i *= z;\n        }}\n        c[0] /= 1 - z_n_1 * z_n_1;'
    elif mode == 'grid-wrap':
        code += '\n        z_i = z;\n\n        for (i = 1; i < min(n, static_cast<idx_t>({n_boundary})); ++i) {{\n            c[0] += z_i * c[(n - i) * element_stride];\n            z_i *= z;\n        }}\n        c[0] /= 1 - z_i; /* z_i = pow(z, n) */'
    elif mode == 'reflect':
        code += '\n        z_i = z;\n        z_n = pow(z, (P)n);\n        c0 = c[0];\n\n        c[0] = c[0] + z_n * c[(n - 1) * element_stride];\n        for (i = 1; i < min(n, static_cast<idx_t>({n_boundary})); ++i) {{\n            c[0] += z_i * (c[i * element_stride] +\n                           z_n * c[(n - 1 - i) * element_stride]);\n            z_i *= z;\n        }}\n        c[0] *= z / (1 - z_n * z_n);\n        c[0] += c0;'
    else:
        raise ValueError('invalid mode: {}'.format(mode))
    return code

def _anticausal_init_code(mode):
    if False:
        while True:
            i = 10
    'Code for the anti-causal initialization step of IIR filtering.\n\n    c is a 1d array of length n and z is a filter pole\n    '
    code = f'\n        // anti-causal init for mode={mode}'
    if mode == 'mirror':
        code += '\n        c[(n - 1) * element_stride] = (\n            z * c[(n - 2) * element_stride] +\n            c[(n - 1) * element_stride]) * z / (z * z - 1);'
    elif mode == 'grid-wrap':
        code += '\n        z_i = z;\n\n        for (i = 0; i < min(n - 1, static_cast<idx_t>({n_boundary})); ++i) {{\n            c[(n - 1) * element_stride] += z_i * c[i * element_stride];\n            z_i *= z;\n        }}\n        c[(n - 1) * element_stride] *= z / (z_i - 1); /* z_i = pow(z, n) */'
    elif mode == 'reflect':
        code += '\n        c[(n - 1) * element_stride] *= z / (z - 1);'
    else:
        raise ValueError('invalid mode: {}'.format(mode))
    return code

def _get_spline_mode(mode):
    if False:
        while True:
            i = 10
    'spline boundary mode for interpolation with order >= 2.'
    if mode in ['mirror', 'reflect', 'grid-wrap']:
        return mode
    elif mode == 'grid-mirror':
        return 'reflect'
    return 'reflect' if mode == 'nearest' else 'mirror'

def _get_spline1d_code(mode, poles, n_boundary):
    if False:
        while True:
            i = 10
    'Generates the code required for IIR filtering of a single 1d signal.\n\n    Prefiltering is done by causal filtering followed by anti-causal filtering.\n    Multiple boundary conditions have been implemented.\n    '
    code = ['\n    __device__ void spline_prefilter1d(\n        T* __restrict__ c, idx_t signal_length, idx_t element_stride)\n    {{']
    code.append('\n        idx_t i, n = signal_length;\n        P z, z_i;')
    mode = _get_spline_mode(mode)
    if mode == 'mirror':
        code.append('\n        P z_n_1;')
    elif mode == 'reflect':
        code.append('\n        P z_n;\n        T c0;')
    for pole in poles:
        code.append(f'\n        // select the current pole\n        z = {pole};')
        code.append(_causal_init_code(mode))
        code.append('\n        // apply the causal filter for the current pole\n        for (i = 1; i < n; ++i) {{\n            c[i * element_stride] += z * c[(i - 1) * element_stride];\n        }}')
        code.append('\n        #ifdef __HIP_DEVICE_COMPILE__\n        __syncthreads();\n        #endif\n        ')
        code.append(_anticausal_init_code(mode))
        code.append('\n        // apply the anti-causal filter for the current pole\n        for (i = n - 2; i >= 0; --i) {{\n            c[i * element_stride] = z * (c[(i + 1) * element_stride] -\n                                         c[i * element_stride]);\n        }}')
    code += ['\n    }}']
    return textwrap.dedent('\n'.join(code)).format(n_boundary=n_boundary)
_FILTER_GENERAL = '\n#include "cupy/carray.cuh"\n#include "cupy/complex.cuh"\ntypedef {data_type} T;\ntypedef {pole_type} P;\ntypedef {index_type} idx_t;\ntemplate <typename T>\n__device__ T* row(\n        T* ptr, idx_t i, idx_t axis, idx_t ndim, const idx_t* shape) {{\n    idx_t index = 0, stride = 1;\n    for (idx_t a = ndim - 1; a > 0; --a) {{\n        if (a != axis) {{\n            index += (i % shape[a]) * stride;\n            i /= shape[a];\n        }}\n        stride *= shape[a];\n    }}\n    return ptr + index + stride * i;\n}}\n'
_batch_spline1d_strided_template = '\nextern "C" __global__\n__launch_bounds__({block_size})\nvoid {kernel_name}(T* __restrict__ y, const idx_t* __restrict__ info) {{\n    const idx_t n_signals = info[0], n_samples = info[1],\n        * __restrict__ shape = info+2;\n    idx_t y_elem_stride = 1;\n    for (int a = {ndim} - 1; a > {axis}; --a) {{ y_elem_stride *= shape[a]; }}\n    idx_t unraveled_idx = blockDim.x * blockIdx.x + threadIdx.x;\n    idx_t batch_idx = unraveled_idx;\n    if (batch_idx < n_signals)\n    {{\n        T* __restrict__ y_i = row(y, batch_idx, {axis}, {ndim}, shape);\n        spline_prefilter1d(y_i, n_samples, y_elem_stride);\n    }}\n}}\n'

@cupy.memoize(for_each_device=True)
def get_raw_spline1d_kernel(axis, ndim, mode, order, index_type='int', data_type='double', pole_type='double', block_size=128):
    if False:
        while True:
            i = 10
    'Generate a kernel for applying a spline prefilter along a given axis.'
    poles = get_poles(order)
    largest_pole = max([abs(p) for p in poles])
    tol = 1e-10 if pole_type == 'float' else 1e-18
    n_boundary = math.ceil(math.log(tol, largest_pole))
    code = _FILTER_GENERAL.format(index_type=index_type, data_type=data_type, pole_type=pole_type)
    code += _get_spline1d_code(mode, poles, n_boundary)
    mode_str = mode.replace('-', '_')
    kernel_name = f'cupyx_scipy_ndimage_spline_filter_{ndim}d_ord{order}_axis{axis}_{mode_str}'
    code += _batch_spline1d_strided_template.format(ndim=ndim, axis=axis, block_size=block_size, kernel_name=kernel_name)
    return cupy.RawKernel(code, kernel_name)