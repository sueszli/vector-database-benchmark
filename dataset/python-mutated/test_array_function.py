import unittest
import numpy
import cupy
from cupy import testing

class TestArrayFunction(unittest.TestCase):

    @testing.with_requires('numpy>=1.17.0')
    def test_array_function(self):
        if False:
            print('Hello World!')
        a = numpy.random.randn(100, 100)
        a_cpu = numpy.asarray(a)
        a_gpu = cupy.asarray(a)
        qr_cpu = numpy.linalg.qr(a_cpu)
        qr_gpu = numpy.linalg.qr(a_gpu)
        if isinstance(qr_cpu, tuple):
            for (b_cpu, b_gpu) in zip(qr_cpu, qr_gpu):
                assert b_cpu.dtype == b_gpu.dtype
                cupy.testing.assert_allclose(b_cpu, b_gpu, atol=0.0001)
        else:
            assert qr_cpu.dtype == qr_gpu.dtype
            cupy.testing.assert_allclose(qr_cpu, qr_gpu, atol=0.0001)

    @testing.with_requires('numpy>=1.17.0')
    def test_array_function2(self):
        if False:
            while True:
                i = 10
        a = numpy.random.randn(100, 100)
        a_cpu = numpy.asarray(a)
        a_gpu = cupy.asarray(a)
        out_cpu = numpy.sum(a_cpu, axis=1)
        out_gpu = numpy.sum(a_gpu, axis=1)
        assert out_cpu.dtype == out_gpu.dtype
        cupy.testing.assert_allclose(out_cpu, out_gpu, atol=0.0001)

    @testing.with_requires('numpy>=1.17.0')
    @testing.numpy_cupy_equal()
    def test_array_function_can_cast(self, xp):
        if False:
            while True:
                i = 10
        return numpy.can_cast(xp.arange(2), 'f4')

    @testing.with_requires('numpy>=1.17.0')
    @testing.numpy_cupy_equal()
    def test_array_function_common_type(self, xp):
        if False:
            while True:
                i = 10
        return numpy.common_type(xp.arange(2, dtype='f8'), xp.arange(2, dtype='f4'))

    @testing.with_requires('numpy>=1.17.0')
    @testing.numpy_cupy_equal()
    def test_array_function_result_type(self, xp):
        if False:
            for i in range(10):
                print('nop')
        return numpy.result_type(3, xp.arange(2, dtype='f8'))