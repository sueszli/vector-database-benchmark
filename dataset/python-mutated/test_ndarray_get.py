import unittest
import cupy
from cupy import cuda
from cupy import testing
import numpy
from numpy import testing as np_testing

class TestArrayGet(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.stream = cuda.Stream.null

    def check_get(self, f, stream, order='C', blocking=True):
        if False:
            return 10
        a_gpu = f(cupy)
        a_cpu = a_gpu.get(stream, order=order, blocking=blocking)
        if stream:
            stream.synchronize()
        b_cpu = f(numpy)
        np_testing.assert_array_equal(a_cpu, b_cpu)
        if order == 'F' or (order == 'A' and a_gpu.flags.f_contiguous):
            assert a_cpu.flags.f_contiguous
        else:
            assert a_cpu.flags.c_contiguous

    @testing.for_orders('CFA')
    @testing.for_all_dtypes()
    def test_contiguous_array(self, dtype, order):
        if False:
            return 10

        def contiguous_array(xp):
            if False:
                return 10
            return testing.shaped_arange((3,), xp, dtype, order)
        self.check_get(contiguous_array, None, order)

    @testing.for_orders('CFA')
    @testing.for_all_dtypes()
    def test_non_contiguous_array(self, dtype, order):
        if False:
            i = 10
            return i + 15

        def non_contiguous_array(xp):
            if False:
                for i in range(10):
                    print('nop')
            return testing.shaped_arange((3, 3), xp, dtype, order)[0::2, 0::2]
        self.check_get(non_contiguous_array, None, order)

    @testing.for_orders('CFA')
    @testing.for_all_dtypes()
    def test_contiguous_array_stream(self, dtype, order):
        if False:
            return 10

        def contiguous_array(xp):
            if False:
                while True:
                    i = 10
            return testing.shaped_arange((3,), xp, dtype, order)
        self.check_get(contiguous_array, self.stream, order)

    @testing.for_orders('CFA')
    @testing.for_all_dtypes()
    def test_contiguous_array_stream_nonblocking(self, dtype, order):
        if False:
            return 10

        def contiguous_array(xp):
            if False:
                while True:
                    i = 10
            return testing.shaped_arange((3,), xp, dtype, order)
        self.check_get(contiguous_array, self.stream, order, False)

    @testing.for_orders('CFA')
    @testing.for_all_dtypes()
    def test_non_contiguous_array_stream(self, dtype, order):
        if False:
            print('Hello World!')

        def non_contiguous_array(xp):
            if False:
                while True:
                    i = 10
            return testing.shaped_arange((3, 3), xp, dtype, order)[0::2, 0::2]
        self.check_get(non_contiguous_array, self.stream)

    @testing.multi_gpu(2)
    @testing.for_orders('CFA')
    @testing.for_all_dtypes()
    def test_get_multigpu(self, dtype, order):
        if False:
            return 10
        with cuda.Device(1):
            src = testing.shaped_arange((2, 3), cupy, dtype, order)
            src = cupy.asfortranarray(src)
        with cuda.Device(0):
            dst = src.get()
        expected = testing.shaped_arange((2, 3), numpy, dtype, order)
        np_testing.assert_array_equal(dst, expected)

class TestArrayGetWithOut(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.stream = cuda.Stream.null

    def check_get(self, f, out, stream):
        if False:
            return 10
        a_gpu = f(cupy)
        a_cpu = a_gpu.get(stream, out=out)
        if stream:
            stream.synchronize()
        b_cpu = f(numpy)
        assert a_cpu is out
        np_testing.assert_array_equal(a_cpu, b_cpu)

    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_contiguous_array(self, dtype, order):
        if False:
            return 10

        def contiguous_array(xp):
            if False:
                i = 10
                return i + 15
            return testing.shaped_arange((3,), xp, dtype, order)
        out = numpy.empty((3,), dtype, order)
        self.check_get(contiguous_array, out, None)

    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_contiguous_array_cross(self, dtype, order):
        if False:
            for i in range(10):
                print('nop')

        def contiguous_array(xp):
            if False:
                while True:
                    i = 10
            return testing.shaped_arange((3,), xp, dtype, order)
        out_order = 'C' if order == 'F' else 'F'
        out = numpy.empty((3,), dtype, out_order)
        self.check_get(contiguous_array, out, None)

    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_contiguous_array_with_error(self, dtype, order):
        if False:
            i = 10
            return i + 15
        out = numpy.empty((3, 3), dtype)[0:2, 0:2]
        with self.assertRaises(RuntimeError):
            a_gpu = testing.shaped_arange((3, 3), cupy, dtype, order)[0:2, 0:2]
            a_gpu.get(out=out)

    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_non_contiguous_array(self, dtype, order):
        if False:
            i = 10
            return i + 15

        def non_contiguous_array(xp):
            if False:
                return 10
            return testing.shaped_arange((3, 3), xp, dtype, order)[0::2, 0::2]
        out = numpy.empty((2, 2), dtype, order)
        self.check_get(non_contiguous_array, out, None)

    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_contiguous_array_stream(self, dtype, order):
        if False:
            while True:
                i = 10

        def contiguous_array(xp):
            if False:
                i = 10
                return i + 15
            return testing.shaped_arange((3,), xp, dtype, order)
        out = numpy.empty((3,), dtype, order)
        self.check_get(contiguous_array, out, self.stream)

    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_non_contiguous_array_stream(self, dtype, order):
        if False:
            i = 10
            return i + 15

        def non_contiguous_array(xp):
            if False:
                for i in range(10):
                    print('nop')
            return testing.shaped_arange((3, 3), xp, dtype, order)[0::2, 0::2]
        out = numpy.empty((2, 2), dtype, order)
        self.check_get(non_contiguous_array, out, self.stream)

    @testing.multi_gpu(2)
    @testing.for_orders('CF')
    @testing.for_all_dtypes()
    def test_get_multigpu(self, dtype, order):
        if False:
            for i in range(10):
                print('nop')
        with cuda.Device(1):
            src = testing.shaped_arange((2, 3), cupy, dtype, order)
            src = cupy.asfortranarray(src)
        with cuda.Device(0):
            dst = numpy.empty((2, 3), dtype, order)
            src.get(out=dst)
        expected = testing.shaped_arange((2, 3), numpy, dtype, order)
        np_testing.assert_array_equal(dst, expected)