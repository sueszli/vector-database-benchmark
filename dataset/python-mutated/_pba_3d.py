import math
import os
import cupy
import numpy as np
from ._util import _get_inttype
from ._pba_2d import _check_distances, _check_indices, _distance_tranform_arg_check, _generate_indices_ops, _generate_shape, _get_block_size, lcm
pba3d_defines_template = '\n\n#define MARKER     {marker}\n#define MAX_INT    {max_int}\n#define BLOCKSIZE  {block_size_3d}\n\n'
pba3d_defines_encode_32bit = '\n// Sites     : ENCODE(x, y, z, 0, 0)\n// Not sites : ENCODE(0, 0, 0, 1, 0) or MARKER\n#define ENCODED_INT_TYPE int\n#define ZERO 0\n#define ONE 1\n#define ENCODE(x, y, z, a, b)  (((x) << 20) | ((y) << 10) | (z) | ((a) << 31) | ((b) << 30))\n#define DECODE(value, x, y, z)     x = ((value) >> 20) & 0x3ff;     y = ((value) >> 10) & 0x3ff;     z = (value) & 0x3ff\n\n#define NOTSITE(value)  (((value) >> 31) & 1)\n#define HASNEXT(value)  (((value) >> 30) & 1)\n\n#define GET_X(value)    (((value) >> 20) & 0x3ff)\n#define GET_Y(value)    (((value) >> 10) & 0x3ff)\n#define GET_Z(value)    ((NOTSITE((value))) ? MAX_INT : ((value) & 0x3ff))\n\n'
pba3d_defines_encode_64bit = '\n// Sites     : ENCODE(x, y, z, 0, 0)\n// Not sites : ENCODE(0, 0, 0, 1, 0) or MARKER\n#define ENCODED_INT_TYPE long long\n#define ZERO 0L\n#define ONE 1L\n#define ENCODE(x, y, z, a, b)  (((x) << 40) | ((y) << 20) | (z) | ((a) << 61) | ((b) << 60))\n#define DECODE(value, x, y, z)     x = ((value) >> 40) & 0xfffff;     y = ((value) >> 20) & 0xfffff;     z = (value) & 0xfffff\n\n#define NOTSITE(value)  (((value) >> 61) & 1)\n#define HASNEXT(value)  (((value) >> 60) & 1)\n\n#define GET_X(value)    (((value) >> 40) & 0xfffff)\n#define GET_Y(value)    (((value) >> 20) & 0xfffff)\n#define GET_Z(value)    ((NOTSITE((value))) ? MAX_INT : ((value) & 0xfffff))\n\n'

@cupy.memoize(True)
def get_pba3d_src(block_size_3d=32, marker=-2147483648, max_int=2147483647, size_max=1024):
    if False:
        return 10
    pba3d_code = pba3d_defines_template.format(block_size_3d=block_size_3d, marker=marker, max_int=max_int)
    if size_max > 1024:
        pba3d_code += pba3d_defines_encode_64bit
    else:
        pba3d_code += pba3d_defines_encode_32bit
    kernel_directory = os.path.join(os.path.dirname(__file__), 'cuda')
    with open(os.path.join(kernel_directory, 'pba_kernels_3d.h'), 'rt') as f:
        pba3d_kernels = '\n'.join(f.readlines())
    pba3d_code += pba3d_kernels
    return pba3d_code

@cupy.memoize(for_each_device=True)
def _get_encode3d_kernel(size_max, marker=-2147483648):
    if False:
        return 10
    'Pack array coordinates into a single integer.'
    if size_max > 1024:
        int_type = 'ptrdiff_t'
    else:
        int_type = 'int'
    if size_max > 1024:
        value = '(((x) << 40) | ((y) << 20) | (z))'
    else:
        value = '(((x) << 20) | ((y) << 10) | (z))'
    code = f'\n    if (arr[i]) {{\n        out[i] = {marker};\n    }} else {{\n        {int_type} shape_2 = arr.shape()[2];\n        {int_type} shape_1 = arr.shape()[1];\n        {int_type} _i = i;\n        {int_type} x = _i % shape_2;\n        _i /= shape_2;\n        {int_type} y = _i % shape_1;\n        _i /= shape_1;\n        {int_type} z = _i;\n        out[i] = {value};\n    }}\n    '
    return cupy.ElementwiseKernel(in_params='raw B arr', out_params='raw I out', operation=code, options=('--std=c++11',))

def encode3d(arr, marker=-2147483648, bit_depth=32, size_max=1024):
    if False:
        for i in range(10):
            print('nop')
    if arr.ndim != 3:
        raise ValueError('only 3d arr suppported')
    if bit_depth not in [32, 64]:
        raise ValueError('only bit_depth of 32 or 64 is supported')
    if size_max > 1024:
        dtype = np.int64
    else:
        dtype = np.int32
    image = cupy.zeros(arr.shape, dtype=dtype, order='C')
    kern = _get_encode3d_kernel(size_max, marker=marker)
    kern(arr, image, size=image.size)
    return image

def _get_decode3d_code(size_max, int_type=''):
    if False:
        while True:
            i = 10
    if size_max > 1024:
        code = f'\n        {int_type} x = (encoded >> 40) & 0xfffff;\n        {int_type} y = (encoded >> 20) & 0xfffff;\n        {int_type} z = encoded & 0xfffff;\n        '
    else:
        code = f'\n        {int_type} x = (encoded >> 20) & 0x3ff;\n        {int_type} y = (encoded >> 10) & 0x3ff;\n        {int_type} z = encoded & 0x3ff;\n        '
    return code

@cupy.memoize(for_each_device=True)
def _get_decode3d_kernel(size_max):
    if False:
        while True:
            i = 10
    'Unpack 3 coordinates encoded as a single integer.'
    code = _get_decode3d_code(size_max, int_type='')
    return cupy.ElementwiseKernel(in_params='E encoded', out_params='I x, I y, I z', operation=code, options=('--std=c++11',))

def decode3d(encoded, size_max=1024):
    if False:
        i = 10
        return i + 15
    coord_dtype = cupy.int32 if size_max < 2 ** 31 else cupy.int64
    x = cupy.empty_like(encoded, dtype=coord_dtype)
    y = cupy.empty_like(x)
    z = cupy.empty_like(x)
    kern = _get_decode3d_kernel(size_max)
    kern(encoded, x, y, z)
    return (x, y, z)

def _determine_padding(shape, block_size, m1, m2, m3, blockx, blocky):
    if False:
        for i in range(10):
            print('nop')
    LCM = lcm(block_size, m1, m2, m3, blockx, blocky)
    (orig_sz, orig_sy, orig_sx) = shape
    round_up = False
    if orig_sx % LCM != 0:
        round_up = True
        sx = LCM * math.ceil(orig_sx / LCM)
    else:
        sx = orig_sx
    if orig_sy % LCM != 0:
        round_up = True
        sy = LCM * math.ceil(orig_sy / LCM)
    else:
        sy = orig_sy
    if orig_sz % LCM != 0:
        round_up = True
        sz = LCM * math.ceil(orig_sz / LCM)
    else:
        sz = orig_sz
    aniso = not sx == sy == sz
    if aniso or round_up:
        smax = max(sz, sy, sx)
        padding_width = ((0, smax - orig_sz), (0, smax - orig_sy), (0, smax - orig_sx))
    else:
        padding_width = None
    return padding_width

def _generate_distance_computation(int_type, dist_int_type):
    if False:
        print('Hello World!')
    '\n    Compute euclidean distance from current coordinate (ind_0, ind_1, ind_2) to\n    the coordinates of the nearest point (z, y, x).'
    return f'\n    {int_type} tmp = z - ind_0;\n    {dist_int_type} sq_dist = tmp * tmp;\n    tmp = y - ind_1;\n    sq_dist += tmp * tmp;\n    tmp = x - ind_2;\n    sq_dist += tmp * tmp;\n    dist[i] = sqrt(static_cast<F>(sq_dist));\n    '

def _get_distance_kernel_code(int_type, dist_int_type, raw_out_var=True):
    if False:
        while True:
            i = 10
    code = _generate_shape(ndim=3, int_type=int_type, var_name='dist', raw_var=raw_out_var)
    code += _generate_indices_ops(ndim=3, int_type=int_type)
    code += _generate_distance_computation(int_type, dist_int_type)
    return code

@cupy.memoize(for_each_device=True)
def _get_distance_kernel(int_type, large_dist=False):
    if False:
        while True:
            i = 10
    'Returns kernel computing the Euclidean distance from coordinates.'
    dist_int_type = 'ptrdiff_t' if large_dist else 'int'
    operation = _get_distance_kernel_code(int_type, dist_int_type, raw_out_var=True)
    return cupy.ElementwiseKernel(in_params='I z, I y, I x', out_params='raw F dist', operation=operation, options=('--std=c++11',))

def _generate_aniso_distance_computation():
    if False:
        return 10
    '\n    Compute euclidean distance from current coordinate (ind_0, ind_1, ind_2) to\n    the coordinates of the nearest point (z, y, x).'
    return '\n    F tmp = static_cast<F>(z - ind_0) * sampling[0];\n    F sq_dist = tmp * tmp;\n    tmp = static_cast<F>(y - ind_1) * sampling[1];\n    sq_dist += tmp * tmp;\n    tmp = static_cast<F>(x - ind_2) * sampling[2];\n    sq_dist += tmp * tmp;\n    dist[i] = sqrt(static_cast<F>(sq_dist));\n    '

def _get_aniso_distance_kernel_code(int_type, raw_out_var=True):
    if False:
        for i in range(10):
            print('nop')
    code = _generate_shape(ndim=3, int_type=int_type, var_name='dist', raw_var=raw_out_var)
    code += _generate_indices_ops(ndim=3, int_type=int_type)
    code += _generate_aniso_distance_computation()
    return code

@cupy.memoize(for_each_device=True)
def _get_aniso_distance_kernel(int_type):
    if False:
        i = 10
        return i + 15
    'Returns kernel computing the Euclidean distance from coordinates with\n    axis spacing != 1.'
    operation = _get_aniso_distance_kernel_code(int_type, raw_out_var=True)
    return cupy.ElementwiseKernel(in_params='I z, I y, I x, raw F sampling', out_params='raw F dist', operation=operation, options=('--std=c++11',))

@cupy.memoize(for_each_device=True)
def _get_decode_as_distance_kernel(size_max, large_dist=False, sampling=None):
    if False:
        i = 10
        return i + 15
    'Fused decode3d and distance computation.\n\n    This kernel is for use when `return_distances=True`, but\n    `return_indices=False`. It replaces the separate calls to\n    `_get_decode3d_kernel` and `_get_distance_kernel`, avoiding the overhead of\n    generating full arrays containing the coordinates since the coordinate\n    arrays are not going to be returned.\n    '
    if sampling is None:
        dist_int_type = 'ptrdiff_t' if large_dist else 'int'
    int_type = 'int'
    code = _get_decode3d_code(size_max, int_type=int_type)
    code += _generate_shape(ndim=3, int_type=int_type, var_name='dist', raw_var=True)
    code += _generate_indices_ops(ndim=3, int_type=int_type)
    if sampling is None:
        code += _generate_distance_computation(int_type, dist_int_type)
        in_params = 'E encoded'
    else:
        code += _generate_aniso_distance_computation()
        in_params = 'E encoded, raw F sampling'
    return cupy.ElementwiseKernel(in_params=in_params, out_params='raw F dist', operation=code, options=('--std=c++11',))

def _pba_3d(arr, sampling=None, return_distances=True, return_indices=False, block_params=None, check_warp_size=False, *, float64_distances=False, distances=None, indices=None):
    if False:
        for i in range(10):
            print('nop')
    indices_inplace = isinstance(indices, cupy.ndarray)
    dt_inplace = isinstance(distances, cupy.ndarray)
    _distance_tranform_arg_check(dt_inplace, indices_inplace, return_distances, return_indices)
    if arr.ndim != 3:
        raise ValueError(f'expected a 3D array, got {arr.ndim}D')
    if block_params is None:
        m1 = 1
        m2 = 1
        m3 = 2
    else:
        (m1, m2, m3) = block_params
    s_min = min(arr.shape)
    if s_min <= 4:
        blockx = 4
    elif s_min <= 8:
        blockx = 8
    elif s_min <= 16:
        blockx = 16
    else:
        blockx = 32
    blocky = 4
    block_size = _get_block_size(check_warp_size)
    (orig_sz, orig_sy, orig_sx) = arr.shape
    padding_width = _determine_padding(arr.shape, block_size, m1, m2, m3, blockx, blocky)
    if padding_width is not None:
        arr = cupy.pad(arr, padding_width, mode='constant', constant_values=1)
    size = arr.shape[0]
    size_max = max(arr.shape)
    input_arr = encode3d(arr, size_max=size_max)
    buffer_idx = 0
    output = cupy.zeros_like(input_arr)
    pba_images = [input_arr, output]
    block = (blockx, blocky, 1)
    grid = (size // block[0], size // block[1], 1)
    pba3d = cupy.RawModule(code=get_pba3d_src(block_size_3d=block_size, size_max=size_max))
    kernelFloodZ = pba3d.get_function('kernelFloodZ')
    if sampling is None:
        kernelMaurerAxis = pba3d.get_function('kernelMaurerAxis')
        kernelColorAxis = pba3d.get_function('kernelColorAxis')
        sampling_args = ()
    else:
        kernelMaurerAxis = pba3d.get_function('kernelMaurerAxisWithSpacing')
        kernelColorAxis = pba3d.get_function('kernelColorAxisWithSpacing')
        sampling = tuple(map(float, sampling))
        sampling_args = (sampling[2], sampling[1], sampling[0])
    kernelFloodZ(grid, block, (pba_images[buffer_idx], pba_images[1 - buffer_idx], size))
    buffer_idx = 1 - buffer_idx
    block = (blockx, blocky, 1)
    grid = (size // block[0], size // block[1], 1)
    kernelMaurerAxis(grid, block, (pba_images[buffer_idx], pba_images[1 - buffer_idx], size) + sampling_args)
    block = (block_size, m3, 1)
    grid = (size // block[0], size, 1)
    kernelColorAxis(grid, block, (pba_images[1 - buffer_idx], pba_images[buffer_idx], size) + sampling_args)
    if sampling is not None:
        sampling_args = (sampling[1], sampling[2], sampling[0])
    block = (blockx, blocky, 1)
    grid = (size // block[0], size // block[1], 1)
    kernelMaurerAxis(grid, block, (pba_images[buffer_idx], pba_images[1 - buffer_idx], size) + sampling_args)
    block = (block_size, m3, 1)
    grid = (size // block[0], size, 1)
    kernelColorAxis(grid, block, (pba_images[1 - buffer_idx], pba_images[buffer_idx], size) + sampling_args)
    output = pba_images[buffer_idx]
    if return_distances:
        out_shape = (orig_sz, orig_sy, orig_sx)
        dtype_out = cupy.float64 if float64_distances else cupy.float32
        if dt_inplace:
            _check_distances(distances, out_shape, dtype_out)
        else:
            distances = cupy.zeros(out_shape, dtype=dtype_out)
        max_possible_dist = sum(((s - 1) ** 2 for s in out_shape))
        large_dist = max_possible_dist >= 2 ** 31
        if not return_indices:
            kern = _get_decode_as_distance_kernel(size_max=size_max, large_dist=large_dist, sampling=sampling)
            if sampling is None:
                kern(output[:orig_sz, :orig_sy, :orig_sx], distances)
            else:
                sampling = cupy.asarray(sampling, dtype=distances.dtype)
                kern(output[:orig_sz, :orig_sy, :orig_sx], sampling, distances)
            return (distances,)
    if return_indices:
        (x, y, z) = decode3d(output[:orig_sz, :orig_sy, :orig_sx], size_max=size_max)
    vals = ()
    if return_distances:
        if sampling is None:
            kern = _get_distance_kernel(int_type=_get_inttype(distances), large_dist=large_dist)
            kern(z, y, x, distances)
        else:
            kern = _get_aniso_distance_kernel(int_type=_get_inttype(distances))
            sampling = cupy.asarray(sampling, dtype=distances.dtype)
            kern(z, y, x, sampling, distances)
        vals = vals + (distances,)
    if return_indices:
        if indices_inplace:
            _check_indices(indices, (arr.ndim,) + arr.shape, x.dtype.itemsize)
            indices[0, ...] = z
            indices[1, ...] = y
            indices[2, ...] = x
        else:
            indices = cupy.stack((z, y, x), axis=0)
        vals = vals + (indices,)
    return vals