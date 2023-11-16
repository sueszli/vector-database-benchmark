import numpy
import pytest
import cupy
from cupy import testing
from cupy.cuda import runtime
from cupy.cuda.texture import ChannelFormatDescriptor, CUDAarray, ResourceDescriptor, TextureDescriptor, TextureObject, SurfaceObject
if cupy.cuda.runtime.is_hip:
    pytest.skip('HIP texture support is not yet ready', allow_module_level=True)

@testing.parameterize(*testing.product({'xp': ('numpy', 'cupy'), 'stream': (True, False), 'dimensions': ((68, 0, 0), (68, 19, 0), (68, 19, 31)), 'n_channels': (1, 2, 4), 'dtype': (numpy.float16, numpy.float32, numpy.int8, numpy.int16, numpy.int32, numpy.uint8, numpy.uint16, numpy.uint32), 'c_contiguous': (True, False)}))
class TestCUDAarray:

    def test_array_gen_cpy(self):
        if False:
            return 10
        xp = numpy if self.xp == 'numpy' else cupy
        stream = None if not self.stream else cupy.cuda.Stream()
        (width, height, depth) = self.dimensions
        n_channel = self.n_channels
        dim = 3 if depth != 0 else 2 if height != 0 else 1
        shape = (depth, height, n_channel * width) if dim == 3 else (height, n_channel * width) if dim == 2 else (n_channel * width,)
        if self.dtype in (numpy.float16, numpy.float32):
            arr = xp.random.random(shape).astype(self.dtype)
            kind = runtime.cudaChannelFormatKindFloat
        else:
            arr = xp.random.randint(100, size=shape, dtype=self.dtype)
            if self.dtype in (numpy.int8, numpy.int16, numpy.int32):
                kind = runtime.cudaChannelFormatKindSigned
            else:
                kind = runtime.cudaChannelFormatKindUnsigned
        if self.c_contiguous:
            arr2 = xp.zeros_like(arr)
            assert arr.flags.c_contiguous
            assert arr2.flags.c_contiguous
        else:
            arr = arr[..., ::2]
            arr2 = xp.zeros_like(arr)
            width = arr.shape[-1] // n_channel
            assert not arr.flags.c_contiguous
            assert arr2.flags.c_contiguous
            assert arr.shape[-1] == n_channel * width
        ch_bits = [0, 0, 0, 0]
        for i in range(n_channel):
            ch_bits[i] = arr.dtype.itemsize * 8
        ch = ChannelFormatDescriptor(*ch_bits, kind)
        cu_arr = CUDAarray(ch, width, height, depth)
        if stream is not None:
            s = cupy.cuda.get_current_stream()
            e = s.record()
            stream.wait_event(e)
        cu_arr.copy_from(arr, stream)
        cu_arr.copy_to(arr2, stream)
        if stream is not None:
            stream.synchronize()
        assert (arr == arr2).all()
source_texobj = '\nextern "C"{\n__global__ void copyKernel1Dfetch(float* output,\n                                  cudaTextureObject_t texObj,\n                                  int width)\n{\n    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;\n\n    // Read from texture and write to global memory\n    if (x < width)\n        output[x] = tex1Dfetch<float>(texObj, x);\n}\n\n__global__ void copyKernel1D(float* output,\n                             cudaTextureObject_t texObj,\n                             int width)\n{\n    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;\n\n    // Read from texture and write to global memory\n    float u = x;\n    if (x < width)\n        output[x] = tex1D<float>(texObj, u);\n}\n\n__global__ void copyKernel2D(float* output,\n                             cudaTextureObject_t texObj,\n                             int width, int height)\n{\n    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;\n    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;\n\n    // Read from texture and write to global memory\n    float u = x;\n    float v = y;\n    if (x < width && y < height)\n        output[y * width + x] = tex2D<float>(texObj, u, v);\n}\n\n__global__ void copyKernel3D(float* output,\n                             cudaTextureObject_t texObj,\n                             int width, int height, int depth)\n{\n    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;\n    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;\n    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;\n\n    // Read from texture and write to global memory\n    float u = x;\n    float v = y;\n    float w = z;\n    if (x < width && y < height && z < depth)\n        output[z*width*height+y*width+x] = tex3D<float>(texObj, u, v, w);\n}\n\n__global__ void copyKernel3D_4ch(float* output_x,\n                                 float* output_y,\n                                 float* output_z,\n                                 float* output_w,\n                                 cudaTextureObject_t texObj,\n                                 int width, int height, int depth)\n{\n    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;\n    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;\n    unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;\n    float4 data;\n\n    // Read from texture, separate channels, and write to global memory\n    float u = x;\n    float v = y;\n    float w = z;\n    if (x < width && y < height && z < depth) {\n        data = tex3D<float4>(texObj, u, v, w);\n        output_x[z*width*height+y*width+x] = data.x;\n        output_y[z*width*height+y*width+x] = data.y;\n        output_z[z*width*height+y*width+x] = data.z;\n        output_w[z*width*height+y*width+x] = data.w;\n    }\n}\n}\n'

@testing.parameterize(*testing.product({'dimensions': ((64, 0, 0), (64, 32, 0), (64, 32, 19)), 'mem_type': ('CUDAarray', 'linear', 'pitch2D'), 'target': ('object',)}))
class TestTexture:

    def test_fetch_float_texture(self):
        if False:
            i = 10
            return i + 15
        (width, height, depth) = self.dimensions
        dim = 3 if depth != 0 else 2 if height != 0 else 1
        if self.mem_type == 'linear' and dim != 1 or (self.mem_type == 'pitch2D' and dim != 2):
            pytest.skip('The test case {0} is inapplicable for {1} and thus skipped.'.format(self.dimensions, self.mem_type))
        shape = (depth, height, width) if dim == 3 else (height, width) if dim == 2 else (width,)
        tex_data = cupy.random.random(shape, dtype=cupy.float32)
        real_output = cupy.zeros_like(tex_data)
        ch = ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
        assert tex_data.flags['C_CONTIGUOUS']
        assert real_output.flags['C_CONTIGUOUS']
        if self.mem_type == 'CUDAarray':
            arr = CUDAarray(ch, width, height, depth)
            expected_output = cupy.zeros_like(tex_data)
            assert expected_output.flags['C_CONTIGUOUS']
            arr.copy_from(tex_data)
            arr.copy_to(expected_output)
        else:
            arr = tex_data
            expected_output = tex_data
        if self.mem_type == 'CUDAarray':
            res = ResourceDescriptor(runtime.cudaResourceTypeArray, cuArr=arr)
        elif self.mem_type == 'linear':
            res = ResourceDescriptor(runtime.cudaResourceTypeLinear, arr=arr, chDesc=ch, sizeInBytes=arr.size * arr.dtype.itemsize)
        else:
            res = ResourceDescriptor(runtime.cudaResourceTypePitch2D, arr=arr, chDesc=ch, width=width, height=height, pitchInBytes=width * arr.dtype.itemsize)
        address_mode = (runtime.cudaAddressModeClamp, runtime.cudaAddressModeClamp)
        tex = TextureDescriptor(address_mode, runtime.cudaFilterModePoint, runtime.cudaReadModeElementType)
        if self.target == 'object':
            texobj = TextureObject(res, tex)
            mod = cupy.RawModule(code=source_texobj)
        else:
            assert False
        ker_name = 'copyKernel'
        ker_name += '3D' if dim == 3 else '2D' if dim == 2 else '1D'
        ker_name += 'fetch' if self.mem_type == 'linear' else ''
        ker = mod.get_function(ker_name)
        block = (4, 4, 2) if dim == 3 else (4, 4) if dim == 2 else (4,)
        grid = ()
        args = (real_output,)
        if self.target == 'object':
            args = args + (texobj,)
        if dim >= 1:
            grid_x = (width + block[0] - 1) // block[0]
            grid = grid + (grid_x,)
            args = args + (width,)
        if dim >= 2:
            grid_y = (height + block[1] - 1) // block[1]
            grid = grid + (grid_y,)
            args = args + (height,)
        if dim == 3:
            grid_z = (depth + block[2] - 1) // block[2]
            grid = grid + (grid_z,)
            args = args + (depth,)
        ker(grid, block, args)
        assert (real_output == expected_output).all()

@testing.parameterize(*testing.product({'target': ('object',)}))
class TestTextureVectorType:

    def test_fetch_float4_texture(self):
        if False:
            i = 10
            return i + 15
        width = 47
        height = 39
        depth = 11
        n_channel = 4
        in_shape = (depth, height, n_channel * width)
        out_shape = (depth, height, width)
        tex_data = cupy.random.random(in_shape, dtype=cupy.float32)
        real_output_x = cupy.zeros(out_shape, dtype=cupy.float32)
        real_output_y = cupy.zeros(out_shape, dtype=cupy.float32)
        real_output_z = cupy.zeros(out_shape, dtype=cupy.float32)
        real_output_w = cupy.zeros(out_shape, dtype=cupy.float32)
        ch = ChannelFormatDescriptor(32, 32, 32, 32, runtime.cudaChannelFormatKindFloat)
        arr = CUDAarray(ch, width, height, depth)
        arr.copy_from(tex_data)
        res = ResourceDescriptor(runtime.cudaResourceTypeArray, cuArr=arr)
        address_mode = (runtime.cudaAddressModeClamp, runtime.cudaAddressModeClamp)
        tex = TextureDescriptor(address_mode, runtime.cudaFilterModePoint, runtime.cudaReadModeElementType)
        if self.target == 'object':
            texobj = TextureObject(res, tex)
            mod = cupy.RawModule(code=source_texobj)
        else:
            assert False
        ker_name = 'copyKernel3D_4ch'
        ker = mod.get_function(ker_name)
        block = (4, 4, 2)
        grid = ((width + block[0] - 1) // block[0], (height + block[1] - 1) // block[1], (depth + block[2] - 1) // block[2])
        args = (real_output_x, real_output_y, real_output_z, real_output_w)
        if self.target == 'object':
            args = args + (texobj,)
        args = args + (width, height, depth)
        ker(grid, block, args)
        assert (real_output_x == tex_data[..., 0::4]).all()
        assert (real_output_y == tex_data[..., 1::4]).all()
        assert (real_output_z == tex_data[..., 2::4]).all()
        assert (real_output_w == tex_data[..., 3::4]).all()
source_surfobj = '\nextern "C" {\n__global__ void writeKernel1D(cudaSurfaceObject_t surf,\n                              int width)\n{\n    unsigned int w = blockIdx.x * blockDim.x + threadIdx.x;\n\n    if (w < width)\n    {\n        float value = w;\n        value *= 3.0;\n        surf1Dwrite(value, surf, w * 4);\n    }\n}\n\n__global__ void writeKernel2D(cudaSurfaceObject_t surf,\n                              int width, int height)\n{\n    unsigned int w = blockIdx.x * blockDim.x + threadIdx.x;\n    unsigned int h = blockIdx.y * blockDim.y + threadIdx.y;\n\n    if (w < width && h < height)\n    {\n        float value = h * width + w;\n        value *= 3.0;\n        surf2Dwrite(value, surf, w * 4, h);\n    }\n}\n\n__global__ void writeKernel3D(cudaSurfaceObject_t surf,\n                              int width, int height, int depth)\n{\n    unsigned int w = blockIdx.x * blockDim.x + threadIdx.x;\n    unsigned int h = blockIdx.y * blockDim.y + threadIdx.y;\n    unsigned int d = blockIdx.z * blockDim.z + threadIdx.z;\n\n    if (w < width && h < height && d < depth)\n    {\n        float value = d * width * height + h * width + w;\n        value *= 3.0;\n        surf3Dwrite(value, surf, w * 4, h, d);\n    }\n}\n}\n'

@testing.parameterize(*testing.product({'dimensions': ((64, 0, 0), (64, 32, 0), (64, 32, 32))}))
class TestSurface:

    def test_write_float_surface(self):
        if False:
            i = 10
            return i + 15
        (width, height, depth) = self.dimensions
        dim = 3 if depth != 0 else 2 if height != 0 else 1
        shape = (depth, height, width) if dim == 3 else (height, width) if dim == 2 else (width,)
        real_output = cupy.zeros(shape, dtype=cupy.float32)
        assert real_output.flags['C_CONTIGUOUS']
        ch = ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
        expected_output = cupy.arange(numpy.prod(shape), dtype=cupy.float32)
        expected_output = expected_output.reshape(shape) * 3.0
        assert expected_output.flags['C_CONTIGUOUS']
        arr = CUDAarray(ch, width, height, depth, runtime.cudaArraySurfaceLoadStore)
        arr.copy_from(real_output)
        res = ResourceDescriptor(runtime.cudaResourceTypeArray, cuArr=arr)
        surfobj = SurfaceObject(res)
        mod = cupy.RawModule(code=source_surfobj)
        ker_name = 'writeKernel'
        ker_name += '3D' if dim == 3 else '2D' if dim == 2 else '1D'
        ker = mod.get_function(ker_name)
        block = (4, 4, 2) if dim == 3 else (4, 4) if dim == 2 else (4,)
        grid = ()
        args = (surfobj,)
        if dim >= 1:
            grid_x = (width + block[0] - 1) // block[0]
            grid = grid + (grid_x,)
            args = args + (width,)
        if dim >= 2:
            grid_y = (height + block[1] - 1) // block[1]
            grid = grid + (grid_y,)
            args = args + (height,)
        if dim == 3:
            grid_z = (depth + block[2] - 1) // block[2]
            grid = grid + (grid_z,)
            args = args + (depth,)
        ker(grid, block, args)
        arr.copy_to(real_output)
        assert (real_output == expected_output).all()