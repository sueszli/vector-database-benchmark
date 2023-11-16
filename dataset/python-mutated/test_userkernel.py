import unittest
import numpy
import pytest
import cupy
from cupy import testing
from cupy.cuda import runtime
from cupy.cuda.texture import ChannelFormatDescriptor, CUDAarray, ResourceDescriptor, TextureDescriptor, TextureObject

class TestUserkernel(unittest.TestCase):

    def test_manual_indexing(self, n=100):
        if False:
            while True:
                i = 10
        in1 = cupy.random.uniform(-1, 1, n).astype(cupy.float32)
        in2 = cupy.random.uniform(-1, 1, n).astype(cupy.float32)
        uesr_kernel_1 = cupy.ElementwiseKernel('T x, T y', 'T z', '\n                z = x + y;\n            ', 'uesr_kernel_1')
        out1 = uesr_kernel_1(in1, in2)
        uesr_kernel_2 = cupy.ElementwiseKernel('raw T x, raw T y', 'raw T z', '\n                z[i] = x[i] + y[i];\n            ', 'uesr_kernel_2')
        out2 = uesr_kernel_2(in1, in2, size=n)
        testing.assert_array_equal(out1, out2)

    def test_python_scalar(self):
        if False:
            return 10
        for typ in (int, float, bool):
            dtype = numpy.dtype(typ).type
            in1_cpu = numpy.random.randint(0, 1, (4, 5)).astype(dtype)
            in1 = cupy.array(in1_cpu)
            scalar_value = typ(2)
            uesr_kernel_1 = cupy.ElementwiseKernel('T x, T y', 'T z', '\n                    z = x + y;\n                ', 'uesr_kernel_1')
            out1 = uesr_kernel_1(in1, scalar_value)
            expected = in1_cpu + dtype(2)
            testing.assert_array_equal(out1, expected)

    @testing.for_all_dtypes()
    def test_numpy_scalar(self, dtype):
        if False:
            while True:
                i = 10
        in1_cpu = numpy.random.randint(0, 1, (4, 5)).astype(dtype)
        in1 = cupy.array(in1_cpu)
        scalar_value = dtype(2)
        uesr_kernel_1 = cupy.ElementwiseKernel('T x, T y', 'T z', '\n                z = x + y;\n            ', 'uesr_kernel_1')
        out1 = uesr_kernel_1(in1, scalar_value)
        expected = in1_cpu + dtype(2)
        testing.assert_array_equal(out1, expected)

    def test_cached_code(self):
        if False:
            i = 10
            return i + 15
        in1 = cupy.random.uniform(-1, 1, 100).astype(cupy.float32)
        in2 = cupy.random.uniform(-1, 1, 100).astype(cupy.float32)
        user_kernel_1 = cupy.ElementwiseKernel('T x, T y', 'T z', '\n                z = x + y;\n            ', 'uesr_kernel_1')
        assert len(user_kernel_1._cached_codes) == 0
        user_kernel_1(in1, in2)
        assert len(user_kernel_1._cached_codes) == 1
        user_kernel_1(in1, in2)
        assert len(user_kernel_1._cached_codes) == 1
        user_kernel_1(in1.astype(cupy.float64), in2.astype(cupy.float64))
        assert len(user_kernel_1._cached_codes) == 2

class TestElementwiseKernelSize(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.arr1 = cupy.array([1, 2], dtype='float32')
        self.arr2 = cupy.array([3, 4], dtype='float32')

    def raises_size_not_allowed(self):
        if False:
            while True:
                i = 10
        return pytest.raises(ValueError, match="^Specified \\'size\\' can")

    def raises_size_required(self):
        if False:
            return 10
        return pytest.raises(ValueError, match='^Loop size is undecided\\.')

    def create_kernel(self, input_raw, output_raw):
        if False:
            while True:
                i = 10
        input_types = ', '.join(['{}float32 x{}'.format('raw ' if raw else '', i) for (i, raw) in enumerate(input_raw)])
        output_types = ', '.join(['{}float32 y{}'.format('raw ' if raw else '', i) for (i, raw) in enumerate(output_raw)])
        return cupy.ElementwiseKernel(input_types, output_types, '', 'kernel')

    def test_all_raws(self):
        if False:
            return 10
        kernel1 = self.create_kernel((True, True), (False,))
        kernel1(self.arr1, self.arr2, size=2)
        with self.raises_size_required():
            kernel1(self.arr1, self.arr2)
        kernel2 = self.create_kernel((True, True), (True,))
        kernel2(self.arr1, self.arr2, size=2)
        with self.raises_size_required():
            kernel2(self.arr1, self.arr2)

    def test_all_nonraws(self):
        if False:
            i = 10
            return i + 15
        kernel1 = self.create_kernel((False, False), (False,))
        with self.raises_size_not_allowed():
            kernel1(self.arr1, self.arr2, size=2)
        kernel2 = self.create_kernel((False, False), (True,))
        with self.raises_size_not_allowed():
            kernel2(self.arr1, self.arr2, size=2)

    def test_some_nonraws(self):
        if False:
            print('Hello World!')
        kernel1 = self.create_kernel((True, False), (False,))
        with self.raises_size_not_allowed():
            kernel1(self.arr1, self.arr2, size=2)
        kernel2 = self.create_kernel((False, True), (False,))
        with self.raises_size_not_allowed():
            kernel2(self.arr1, self.arr2, size=2)
        kernel3 = self.create_kernel((True, False), (True,))
        with self.raises_size_not_allowed():
            kernel3(self.arr1, self.arr2, size=2)
        kernel4 = self.create_kernel((False, True), (True,))
        with self.raises_size_not_allowed():
            kernel4(self.arr1, self.arr2, size=2)

    def test_scalars_and_nonraws(self):
        if False:
            for i in range(10):
                print('nop')
        kernel1 = self.create_kernel((False, False), (False,))
        with self.raises_size_not_allowed():
            kernel1(self.arr1, 7, size=2)
        kernel2 = self.create_kernel((False, False), (False,))
        with self.raises_size_not_allowed():
            kernel2(7, self.arr1, size=2)
        kernel3 = self.create_kernel((False, False), (True,))
        with self.raises_size_not_allowed():
            kernel3(self.arr1, 7, size=2)
        kernel4 = self.create_kernel((False, False), (True,))
        with self.raises_size_not_allowed():
            kernel4(7, self.arr1, size=2)

    def test_scalars_and_raws_and_nonraws(self):
        if False:
            i = 10
            return i + 15
        kernel1 = self.create_kernel((False, False, True), (False,))
        with self.raises_size_not_allowed():
            kernel1(self.arr1, 7, self.arr2, size=2)
        kernel2 = self.create_kernel((False, False, True), (True,))
        with self.raises_size_not_allowed():
            kernel2(self.arr1, 7, self.arr2, size=2)

    def test_scalars_and_raws(self):
        if False:
            while True:
                i = 10
        kernel1 = self.create_kernel((True, False), (False,))
        kernel1(self.arr1, 7, size=2)
        with self.raises_size_required():
            kernel1(self.arr1, 7)
        kernel2 = self.create_kernel((False, True), (False,))
        kernel2(7, self.arr1, size=2)
        with self.raises_size_required():
            kernel2(7, self.arr1)
        kernel3 = self.create_kernel((True, False), (True,))
        kernel3(self.arr1, 7, size=2)
        with self.raises_size_required():
            kernel3(self.arr1, 7)
        kernel4 = self.create_kernel((False, True), (True,))
        kernel4(7, self.arr1, size=2)
        with self.raises_size_required():
            kernel4(7, self.arr1)

    def test_size_determined_by_output(self):
        if False:
            while True:
                i = 10
        kernel1 = self.create_kernel((True,), (False,))
        kernel1(self.arr1, self.arr2)
        with self.raises_size_not_allowed():
            kernel1(self.arr1, self.arr2, size=2)
        kernel2 = self.create_kernel((False,), (False,))
        kernel2(self.arr1, self.arr2)
        with self.raises_size_not_allowed():
            kernel2(7, self.arr2, size=2)
        kernel3 = self.create_kernel((), (False,))
        kernel3(self.arr1)
        with self.raises_size_not_allowed():
            kernel3(self.arr1, size=2)

    def test_no_input_and_raw_output(self):
        if False:
            i = 10
            return i + 15
        kernel1 = self.create_kernel((), (True,))
        kernel1(self.arr1, size=2)
        with self.raises_size_required():
            kernel1(self.arr1)

@testing.parameterize(*testing.product({'value': [-1, 2 ** 32, 2 ** 63 - 1, -2 ** 63]}))
class TestUserkernelScalar(unittest.TestCase):

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_scalar(self, xp, dtype):
        if False:
            while True:
                i = 10
        x = testing.shaped_arange((2, 3, 4), xp, dtype)
        if xp is numpy:
            y = numpy.array(self.value).astype(dtype)
            return x + y
        else:
            kernel = cupy.ElementwiseKernel('T x, T y', 'T z', 'z = x + y')
            return kernel(x, self.value)

class TestUserkernelManualBlockSize(unittest.TestCase):

    def test_invalid_block_size(self):
        if False:
            print('Hello World!')
        x = testing.shaped_arange((2, 3, 4), cupy, cupy.float32)
        kernel = cupy.ElementwiseKernel('T x, T y', 'T z', 'z = x + y')
        with pytest.raises(ValueError):
            kernel(x, 1, block_size=0)

    def test_block_size(self):
        if False:
            return 10
        x = testing.shaped_arange((2, 3, 4), cupy, cupy.float32)
        kernel = cupy.ElementwiseKernel('T x, T y', 'T z', 'z = x + y')
        y = kernel(x, 1, block_size=1)
        testing.assert_array_equal(y, x + 1)

@testing.parameterize(*testing.product({'dimensions': ((64, 0, 0), (64, 32, 0), (64, 32, 19))}))
@pytest.mark.skipif(runtime.is_hip, reason='texture support on HIP is not yet implemented')
class TestElementwiseKernelTexture(unittest.TestCase):

    def _prep_texture(self):
        if False:
            print('Hello World!')
        (width, height, depth) = self.dimensions
        dim = 3 if depth != 0 else 2 if height != 0 else 1
        shape = (depth, height, width) if dim == 3 else (height, width) if dim == 2 else (width,)
        self.shape = shape
        tex_data = cupy.random.random(shape, dtype=cupy.float32)
        ch = ChannelFormatDescriptor(32, 0, 0, 0, runtime.cudaChannelFormatKindFloat)
        arr = CUDAarray(ch, width, height, depth)
        arr.copy_from(tex_data)
        self.data = tex_data
        res = ResourceDescriptor(runtime.cudaResourceTypeArray, cuArr=arr)
        address_mode = (runtime.cudaAddressModeClamp, runtime.cudaAddressModeClamp)
        tex = TextureDescriptor(address_mode, runtime.cudaFilterModePoint, runtime.cudaReadModeElementType)
        return TextureObject(res, tex)

    def _prep_kernel1D(self):
        if False:
            print('Hello World!')
        return cupy.ElementwiseKernel('T x, U texObj', 'T y', '\n            T temp = tex1D<T>(texObj,\n                              float(i)\n                              );\n            y = temp + x;\n            ', name='test_tex1D')

    def _prep_kernel2D(self):
        if False:
            while True:
                i = 10
        return cupy.ElementwiseKernel('T x, U texObj, uint64 width', 'T y', '\n            T temp = tex2D<T>(texObj,\n                              (float)(i % width),\n                              (float)(i / width)\n                              );\n            y = temp + x;\n            ', name='test_tex2D')

    def _prep_kernel3D(self):
        if False:
            print('Hello World!')
        return cupy.ElementwiseKernel('T x, U texObj, uint64 width, uint64 height', 'T y', '\n            T temp = tex3D<T>(texObj,\n                              (float)((i % (width * height)) % width),\n                              (float)((i % (width * height)) / width),\n                              (float)((i / (width * height)))\n                              );\n            y = temp + x;\n            ', name='test_tex3D')

    def test_texture_input(self):
        if False:
            i = 10
            return i + 15
        (width, height, depth) = self.dimensions
        dim = 3 if depth != 0 else 2 if height != 0 else 1
        texobj = self._prep_texture()
        ker = getattr(self, f'_prep_kernel{dim}D')()
        args = [None, texobj]
        size = width
        if height > 0:
            size *= height
            args.append(width)
        if depth > 0:
            size *= depth
            args.append(height)
        in_arr = cupy.arange(size, dtype=cupy.float32)
        in_arr = in_arr.reshape(self.shape)
        args[0] = in_arr
        out_arr = ker(*args)
        expected = in_arr + self.data
        testing.assert_allclose(out_arr, expected)