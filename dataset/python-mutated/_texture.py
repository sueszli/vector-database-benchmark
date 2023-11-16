import cupy
from cupy import _core
from cupy.cuda import texture
from cupy.cuda import runtime
_affine_transform_2d_array_kernel = _core.ElementwiseKernel('U texObj, raw float32 m, uint64 width', 'T transformed_image', '\n    float3 pixel = make_float3(\n        (float)(i / width),\n        (float)(i % width),\n        1.0f\n    );\n    float x = dot(pixel, make_float3(m[0],  m[1],  m[2])) + .5f;\n    float y = dot(pixel, make_float3(m[3],  m[4],  m[5])) + .5f;\n    transformed_image = tex2D<T>(texObj, y, x);\n    ', 'cupyx_texture_affine_transformation_2d_array', preamble='\n    inline __host__ __device__ float dot(float3 a, float3 b)\n    {\n        return a.x * b.x + a.y * b.y + a.z * b.z;\n    }\n    ')
_affine_transform_3d_array_kernel = _core.ElementwiseKernel('U texObj, raw float32 m, uint64 height, uint64 width', 'T transformed_volume', '\n    float4 voxel = make_float4(\n        (float)(i / (width * height)),\n        (float)((i % (width * height)) / width),\n        (float)((i % (width * height)) % width),\n        1.0f\n    );\n    float x = dot(voxel, make_float4(m[0],  m[1],  m[2],  m[3])) + .5f;\n    float y = dot(voxel, make_float4(m[4],  m[5],  m[6],  m[7])) + .5f;\n    float z = dot(voxel, make_float4(m[8],  m[9],  m[10], m[11])) + .5f;\n    transformed_volume = tex3D<T>(texObj, z, y, x);\n    ', 'cupyx_texture_affine_transformation_3d_array', preamble='\n    inline __host__ __device__ float dot(float4 a, float4 b)\n    {\n        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;\n    }\n    ')

def _create_texture_object(data, address_mode: str, filter_mode: str, read_mode: str, border_color=0):
    if False:
        return 10
    if cupy.issubdtype(data.dtype, cupy.unsignedinteger):
        fmt_kind = runtime.cudaChannelFormatKindUnsigned
    elif cupy.issubdtype(data.dtype, cupy.integer):
        fmt_kind = runtime.cudaChannelFormatKindSigned
    elif cupy.issubdtype(data.dtype, cupy.floating):
        fmt_kind = runtime.cudaChannelFormatKindFloat
    else:
        raise ValueError(f'Unsupported data type {data.dtype}')
    if address_mode == 'nearest':
        address_mode = runtime.cudaAddressModeClamp
    elif address_mode == 'constant':
        address_mode = runtime.cudaAddressModeBorder
    else:
        raise ValueError(f'Unsupported address mode {address_mode} (supported: constant, nearest)')
    if filter_mode == 'nearest':
        filter_mode = runtime.cudaFilterModePoint
    elif filter_mode == 'linear':
        filter_mode = runtime.cudaFilterModeLinear
    else:
        raise ValueError(f'Unsupported filter mode {filter_mode} (supported: nearest, linear)')
    if read_mode == 'element_type':
        read_mode = runtime.cudaReadModeElementType
    elif read_mode == 'normalized_float':
        read_mode = runtime.cudaReadModeNormalizedFloat
    else:
        raise ValueError(f'Unsupported read mode {read_mode} (supported: element_type, normalized_float)')
    texture_fmt = texture.ChannelFormatDescriptor(data.itemsize * 8, 0, 0, 0, fmt_kind)
    array = texture.CUDAarray(texture_fmt, *data.shape[::-1])
    res_desc = texture.ResourceDescriptor(runtime.cudaResourceTypeArray, cuArr=array)
    tex_desc = texture.TextureDescriptor((address_mode,) * data.ndim, filter_mode, read_mode, borderColors=(border_color,))
    tex_obj = texture.TextureObject(res_desc, tex_desc)
    array.copy_from(data)
    return tex_obj

def affine_transformation(data, transformation_matrix, output_shape=None, output=None, interpolation: str='linear', mode: str='constant', border_value=0):
    if False:
        print('Hello World!')
    "\n    Apply an affine transformation.\n\n    The method uses texture memory and supports only 2D and 3D float32 arrays\n    without channel dimension.\n\n    Args:\n        data (cupy.ndarray): The input array or texture object.\n        transformation_matrix (cupy.ndarray): Affine transformation matrix.\n            Must be a homogeneous and have shape ``(ndim + 1, ndim + 1)``.\n        output_shape (tuple of ints): Shape of output. If not specified,\n            the input array shape is used. Default is None.\n        output (cupy.ndarray or ~cupy.dtype): The array in which to place the\n            output, or the dtype of the returned array. If not specified,\n            creates the output array with shape of ``output_shape``. Default is\n            None.\n        interpolation (str): Specifies interpolation mode: ``'linear'`` or\n            ``'nearest'``. Default is ``'linear'``.\n        mode (str): Specifies addressing mode for points outside of the array:\n            (`'constant'``, ``'nearest'``). Default is ``'constant'``.\n        border_value: Specifies value to be used for coordinates outside\n            of the array for ``'constant'`` mode. Default is 0.\n\n    Returns:\n        cupy.ndarray:\n            The transformed input.\n\n    .. seealso:: :func:`cupyx.scipy.ndimage.affine_transform`\n    "
    ndim = data.ndim
    if ndim < 2 or ndim > 3:
        raise ValueError('Texture memory affine transformation is defined only for 2D and 3D arrays without channel dimension.')
    dtype = data.dtype
    if dtype != cupy.float32:
        raise ValueError(f'Texture memory affine transformation is available only for float32 data type (not {dtype})')
    if interpolation not in ['linear', 'nearest']:
        raise ValueError(f'Unsupported interpolation {interpolation} (supported: linear, nearest)')
    if transformation_matrix.shape != (ndim + 1, ndim + 1):
        raise ValueError('Matrix must be have shape (ndim + 1, ndim + 1)')
    texture_object = _create_texture_object(data, address_mode=mode, filter_mode=interpolation, read_mode='element_type', border_color=border_value)
    if ndim == 2:
        kernel = _affine_transform_2d_array_kernel
    else:
        kernel = _affine_transform_3d_array_kernel
    if output_shape is None:
        output_shape = data.shape
    if output is None:
        output = cupy.zeros(output_shape, dtype=dtype)
    elif isinstance(output, (type, cupy.dtype)):
        if output != cupy.float32:
            raise ValueError(f'Texture memory affine transformation is available only for float32 data type (not {output})')
        output = cupy.zeros(output_shape, dtype=output)
    elif isinstance(output, cupy.ndarray):
        if output.shape != output_shape:
            raise ValueError('Output shapes do not match')
    else:
        raise ValueError('Output must be None, cupy.ndarray or cupy.dtype')
    kernel(texture_object, transformation_matrix, *output_shape[1:], output)
    return output