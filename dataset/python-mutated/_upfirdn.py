"""
upfirdn implementation.

Functions defined here were ported directly from cuSignal under
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
from math import ceil
import cupy
_upfirdn_modes = ['constant', 'wrap', 'edge', 'smooth', 'symmetric', 'reflect', 'antisymmetric', 'antireflect', 'line']
UPFIRDN_KERNEL = '\n#include <cupy/complex.cuh>\n\n///////////////////////////////////////////////////////////////////////////////\n//                              UPFIRDN1D                                    //\n///////////////////////////////////////////////////////////////////////////////\n\ntemplate<typename T>\n__device__ void _cupy_upfirdn1D( const T *__restrict__ inp,\n                                 const T *__restrict__ h_trans_flip,\n                                 const int up,\n                                 const int down,\n                                 const int axis,\n                                 const int x_shape_a,\n                                 const int h_per_phase,\n                                 const int padded_len,\n                                 T *__restrict__ out,\n                                 const int outW ) {\n\n    const int t { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };\n    const int stride { static_cast<int>( blockDim.x * gridDim.x ) };\n\n    for ( size_t tid = t; tid < outW; tid += stride ) {\n\n#if ( __CUDACC_VER_MAJOR__ >= 11 ) && ( __CUDACC_VER_MINOR__ >= 2 )\n        __builtin_assume( padded_len > 0 );\n        __builtin_assume( up > 0 );\n        __builtin_assume( down > 0 );\n        __builtin_assume( tid > 0 );\n#endif\n\n        const int x_idx { static_cast<int>( ( tid * down ) / up ) % padded_len };\n        int       h_idx { static_cast<int>( ( tid * down ) % up * h_per_phase ) };\n        int       x_conv_idx { x_idx - h_per_phase + 1 };\n\n        if ( x_conv_idx < 0 ) {\n            h_idx -= x_conv_idx;\n            x_conv_idx = 0;\n        }\n\n        T temp {};\n\n        int stop = ( x_shape_a < ( x_idx + 1 ) ) ? x_shape_a : ( x_idx + 1 );\n\n        for ( int x_c = x_conv_idx; x_c < stop; x_c++ ) {\n            temp += inp[x_c] * h_trans_flip[h_idx];\n            h_idx += 1;\n        }\n        out[tid] = temp;\n    }\n}\n\nextern "C" __global__ void __launch_bounds__( 512 ) _cupy_upfirdn1D_float32( const float *__restrict__ inp,\n                                                                             const float *__restrict__ h_trans_flip,\n                                                                             const int up,\n                                                                             const int down,\n                                                                             const int axis,\n                                                                             const int x_shape_a,\n                                                                             const int h_per_phase,\n                                                                             const int padded_len,\n                                                                             float *__restrict__ out,\n                                                                             const int outW ) {\n    _cupy_upfirdn1D<float>( inp, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW );\n}\n\nextern "C" __global__ void __launch_bounds__( 512 ) _cupy_upfirdn1D_float64( const double *__restrict__ inp,\n                                                                             const double *__restrict__ h_trans_flip,\n                                                                             const int up,\n                                                                             const int down,\n                                                                             const int axis,\n                                                                             const int x_shape_a,\n                                                                             const int h_per_phase,\n                                                                             const int padded_len,\n                                                                             double *__restrict__ out,\n                                                                             const int outW ) {\n    _cupy_upfirdn1D<double>( inp, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW );\n}\n\nextern "C" __global__ void __launch_bounds__( 512 )\n    _cupy_upfirdn1D_complex64( const thrust::complex<float> *__restrict__ inp,\n                               const thrust::complex<float> *__restrict__ h_trans_flip,\n                               const int up,\n                               const int down,\n                               const int axis,\n                               const int x_shape_a,\n                               const int h_per_phase,\n                               const int padded_len,\n                               thrust::complex<float> *__restrict__ out,\n                               const int outW ) {\n    _cupy_upfirdn1D<thrust::complex<float>>(\n        inp, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW );\n}\n\nextern "C" __global__ void __launch_bounds__( 512 )\n    _cupy_upfirdn1D_complex128( const thrust::complex<double> *__restrict__ inp,\n                                const thrust::complex<double> *__restrict__ h_trans_flip,\n                                const int up,\n                                const int down,\n                                const int axis,\n                                const int x_shape_a,\n                                const int h_per_phase,\n                                const int padded_len,\n                                thrust::complex<double> *__restrict__ out,\n                                const int outW ) {\n    _cupy_upfirdn1D<thrust::complex<double>>(\n        inp, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW );\n}\n\n///////////////////////////////////////////////////////////////////////////////\n//                              UPFIRDN2D                                    //\n///////////////////////////////////////////////////////////////////////////////\n\ntemplate<typename T>\n__device__ void _cupy_upfirdn2D( const T *__restrict__ inp,\n                                 const int inpH,\n                                 const T *__restrict__ h_trans_flip,\n                                 const int up,\n                                 const int down,\n                                 const int axis,\n                                 const int x_shape_a,\n                                 const int h_per_phase,\n                                 const int padded_len,\n                                 T *__restrict__ out,\n                                 const int outW,\n                                 const int outH ) {\n\n    const int ty { static_cast<int>( blockIdx.x * blockDim.x + threadIdx.x ) };\n    const int tx { static_cast<int>( blockIdx.y * blockDim.y + threadIdx.y ) };\n\n    const int stride_y { static_cast<int>( blockDim.x * gridDim.x ) };\n    const int stride_x { static_cast<int>( blockDim.y * gridDim.y ) };\n\n    for ( int x = tx; x < outH; x += stride_x ) {\n        for ( int y = ty; y < outW; y += stride_y ) {\n            int x_idx {};\n            int h_idx {};\n\n#if ( __CUDACC_VER_MAJOR__ >= 11 ) && ( __CUDACC_VER_MINOR__ >= 2 )\n            __builtin_assume( padded_len > 0 );\n            __builtin_assume( up > 0 );\n            __builtin_assume( down > 0 );\n#endif\n\n            if ( axis == 1 ) {\n#if ( __CUDACC_VER_MAJOR__ >= 11 ) && ( __CUDACC_VER_MINOR__ >= 2 )\n                __builtin_assume( x > 0 );\n#endif\n                x_idx = ( static_cast<int>( x * down ) / up ) % padded_len;\n                h_idx = ( x * down ) % up * h_per_phase;\n            } else {\n#if ( __CUDACC_VER_MAJOR__ >= 11 ) && ( __CUDACC_VER_MINOR__ >= 2 )\n                __builtin_assume( y > 0 );\n#endif\n                x_idx = ( static_cast<int>( y * down ) / up ) % padded_len;\n                h_idx = ( y * down ) % up * h_per_phase;\n            }\n\n            int x_conv_idx { x_idx - h_per_phase + 1 };\n            if ( x_conv_idx < 0 ) {\n                h_idx -= x_conv_idx;\n                x_conv_idx = 0;\n            }\n\n            T temp {};\n\n            int stop = ( x_shape_a < ( x_idx + 1 ) ) ? x_shape_a : ( x_idx + 1 );\n\n            for ( int x_c = x_conv_idx; x_c < stop; x_c++ ) {\n                if ( axis == 1 ) {\n                    temp += inp[y * inpH + x_c] * h_trans_flip[h_idx];\n                } else {\n                    temp += inp[x_c * inpH + x] * h_trans_flip[h_idx];\n                }\n                h_idx += 1;\n            }\n            out[y * outH + x] = temp;\n        }\n    }\n}\n\nextern "C" __global__ void __launch_bounds__( 64 ) _cupy_upfirdn2D_float32( const float *__restrict__ inp,\n                                                                            const int inpH,\n                                                                            const float *__restrict__ h_trans_flip,\n                                                                            const int up,\n                                                                            const int down,\n                                                                            const int axis,\n                                                                            const int x_shape_a,\n                                                                            const int h_per_phase,\n                                                                            const int padded_len,\n                                                                            float *__restrict__ out,\n                                                                            const int outW,\n                                                                            const int outH ) {\n    _cupy_upfirdn2D<float>(\n        inp, inpH, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW, outH );\n}\n\nextern "C" __global__ void _cupy_upfirdn2D_float64( const double *__restrict__ inp,\n                                                    const int inpH,\n                                                    const double *__restrict__ h_trans_flip,\n                                                    const int up,\n                                                    const int down,\n                                                    const int axis,\n                                                    const int x_shape_a,\n                                                    const int h_per_phase,\n                                                    const int padded_len,\n                                                    double *__restrict__ out,\n                                                    const int outW,\n                                                    const int outH ) {\n    _cupy_upfirdn2D<double>(\n        inp, inpH, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW, outH );\n}\n\nextern "C" __global__ void __launch_bounds__( 64 )\n    _cupy_upfirdn2D_complex64( const thrust::complex<float> *__restrict__ inp,\n                               const int inpH,\n                               const thrust::complex<float> *__restrict__ h_trans_flip,\n                               const int up,\n                               const int down,\n                               const int axis,\n                               const int x_shape_a,\n                               const int h_per_phase,\n                               const int padded_len,\n                               thrust::complex<float> *__restrict__ out,\n                               const int outW,\n                               const int outH ) {\n    _cupy_upfirdn2D<thrust::complex<float>>(\n        inp, inpH, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW, outH );\n}\n\nextern "C" __global__ void __launch_bounds__( 64 )\n    _cupy_upfirdn2D_complex128( const thrust::complex<double> *__restrict__ inp,\n                                const int inpH,\n                                const thrust::complex<double> *__restrict__ h_trans_flip,\n                                const int up,\n                                const int down,\n                                const int axis,\n                                const int x_shape_a,\n                                const int h_per_phase,\n                                const int padded_len,\n                                thrust::complex<double> *__restrict__ out,\n                                const int outW,\n                                const int outH ) {\n    _cupy_upfirdn2D<thrust::complex<double>>(\n        inp, inpH, h_trans_flip, up, down, axis, x_shape_a, h_per_phase, padded_len, out, outW, outH );\n}\n'
UPFIRDN_MODULE = cupy.RawModule(code=UPFIRDN_KERNEL, options=('-std=c++11',), name_expressions=['_cupy_upfirdn1D_float32', '_cupy_upfirdn1D_float64', '_cupy_upfirdn1D_complex64', '_cupy_upfirdn1D_complex128', '_cupy_upfirdn2D_float32', '_cupy_upfirdn2D_float64', '_cupy_upfirdn2D_complex64', '_cupy_upfirdn2D_complex128'])

def _pad_h(h, up):
    if False:
        return 10
    'Store coefficients in a transposed, flipped arrangement.\n    For example, suppose upRate is 3, and the\n    input number of coefficients is 10, represented as h[0], ..., h[9].\n    Then the internal buffer will look like this::\n       h[9], h[6], h[3], h[0],   // flipped phase 0 coefs\n       0,    h[7], h[4], h[1],   // flipped phase 1 coefs (zero-padded)\n       0,    h[8], h[5], h[2],   // flipped phase 2 coefs (zero-padded)\n    '
    h_padlen = len(h) + -len(h) % up
    h_full = cupy.zeros(h_padlen, h.dtype)
    h_full[:len(h)] = h
    h_full = h_full.reshape(-1, up).T[:, ::-1].ravel()
    return h_full

def _output_len(len_h, in_len, up, down):
    if False:
        for i in range(10):
            print('nop')
    return ((in_len - 1) * up + len_h - 1) // down + 1

def _get_max_gdx():
    if False:
        while True:
            i = 10
    device_id = cupy.cuda.Device()
    return device_id.attributes['MaxGridDimX']

def _get_max_gdy():
    if False:
        i = 10
        return i + 15
    device_id = cupy.cuda.Device()
    return device_id.attributes['MaxGridDimY']

def _get_tpb_bpg():
    if False:
        while True:
            i = 10
    device_id = cupy.cuda.Device()
    numSM = device_id.attributes['MultiProcessorCount']
    threadsperblock = 512
    blockspergrid = numSM * 20
    return (threadsperblock, blockspergrid)

class _UpFIRDn(object):

    def __init__(self, h, x_dtype, up, down):
        if False:
            return 10
        'Helper for resampling'
        h = cupy.asarray(h)
        if h.ndim != 1 or h.size == 0:
            raise ValueError('h must be 1D with non-zero length')
        self._output_type = cupy.result_type(h.dtype, x_dtype, cupy.float32)
        h = cupy.asarray(h, self._output_type)
        self._up = int(up)
        self._down = int(down)
        if self._up < 1 or self._down < 1:
            raise ValueError('Both up and down must be >= 1')
        self._h_trans_flip = _pad_h(h, self._up)
        self._h_trans_flip = cupy.asarray(self._h_trans_flip)
        self._h_trans_flip = cupy.ascontiguousarray(self._h_trans_flip)
        self._h_len_orig = len(h)

    def apply_filter(self, x, axis):
        if False:
            return 10
        'Apply the prepared filter to the specified axis of a nD signal x'
        x = cupy.asarray(x, self._output_type)
        output_len = _output_len(self._h_len_orig, x.shape[axis], self._up, self._down)
        output_shape = list(x.shape)
        output_shape[axis] = output_len
        out = cupy.empty(output_shape, dtype=self._output_type, order='C')
        axis = axis % x.ndim
        x_shape_a = x.shape[axis]
        h_per_phase = len(self._h_trans_flip) // self._up
        padded_len = x.shape[axis] + len(self._h_trans_flip) // self._up - 1
        if out.ndim == 1:
            (threadsperblock, blockspergrid) = _get_tpb_bpg()
            kernel = UPFIRDN_MODULE.get_function(f'_cupy_upfirdn1D_{out.dtype.name}')
            kernel(((x.shape[0] + 128 - 1) // 128,), (128,), (x, self._h_trans_flip, self._up, self._down, axis, x_shape_a, h_per_phase, padded_len, out, out.shape[0]))
        elif out.ndim == 2:
            threadsperblock = (8, 8)
            blocks = ceil(out.shape[0] / threadsperblock[0])
            blockspergrid_x = blocks if blocks < _get_max_gdx() else _get_max_gdx()
            blocks = ceil(out.shape[1] / threadsperblock[1])
            blockspergrid_y = blocks if blocks < _get_max_gdy() else _get_max_gdy()
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            kernel = UPFIRDN_MODULE.get_function(f'_cupy_upfirdn2D_{out.dtype.name}')
            kernel(threadsperblock, blockspergrid, (x, x.shape[1], self._h_trans_flip, self._up, self._down, axis, x_shape_a, h_per_phase, padded_len, out, out.shape[0], out.shape[1]))
        else:
            raise NotImplementedError('upfirdn() requires ndim <= 2')
        return out

def upfirdn(h, x, up=1, down=1, axis=-1, mode=None, cval=0):
    if False:
        print('Hello World!')
    '\n    Upsample, FIR filter, and downsample.\n\n    Parameters\n    ----------\n    h : array_like\n        1-dimensional FIR (finite-impulse response) filter coefficients.\n    x : array_like\n        Input signal array.\n    up : int, optional\n        Upsampling rate. Default is 1.\n    down : int, optional\n        Downsampling rate. Default is 1.\n    axis : int, optional\n        The axis of the input data array along which to apply the\n        linear filter. The filter is applied to each subarray along\n        this axis. Default is -1.\n    mode : str, optional\n        This parameter is not implemented.\n    cval : float, optional\n        This parameter is not implemented.\n\n    Returns\n    -------\n    y : ndarray\n        The output signal array. Dimensions will be the same as `x` except\n        for along `axis`, which will change size according to the `h`,\n        `up`,  and `down` parameters.\n\n    Notes\n    -----\n    The algorithm is an implementation of the block diagram shown on page 129\n    of the Vaidyanathan text [1]_ (Figure 4.3-8d).\n\n    The direct approach of upsampling by factor of P with zero insertion,\n    FIR filtering of length ``N``, and downsampling by factor of Q is\n    O(N*Q) per output sample. The polyphase implementation used here is\n    O(N/P).\n\n    See Also\n    --------\n    scipy.signal.upfirdn\n\n    References\n    ----------\n    .. [1] P. P. Vaidyanathan, Multirate Systems and Filter Banks,\n       Prentice Hall, 1993.\n    '
    if mode is not None or cval != 0:
        raise NotImplementedError(f'mode = {mode!r} and cval ={cval!r} not implemented.')
    ufd = _UpFIRDn(h, x.dtype, int(up), int(down))
    return ufd.apply_filter(x, axis)