import unittest
import numpy
import pytest
import cupy
from cupyx import jit
from cupy import testing

class TestDeviceFunction(unittest.TestCase):

    def test_device_function(self):
        if False:
            i = 10
            return i + 15

        @jit.rawkernel()
        def f(x, y, z):
            if False:
                while True:
                    i = 10
            tid = jit.threadIdx.x
            z[tid] = g(x[tid], y[tid]) + x[tid] + y[tid]

        @jit.rawkernel(device=True)
        def g(x, y):
            if False:
                return 10
            x += 1
            y += 1
            return x + y
        x = testing.shaped_random((30,), dtype=numpy.int32, seed=0)
        y = testing.shaped_random((30,), dtype=numpy.int32, seed=1)
        z = testing.shaped_random((30,), dtype=numpy.int32, seed=2)
        f((1,), (30,), (x, y, z))
        assert (z == (x + y + 1) * 2).all()

    def test_device_function_duplicated_names(self):
        if False:
            while True:
                i = 10

        @jit.rawkernel()
        def f(x, y, z):
            if False:
                print('Hello World!')
            tid = jit.threadIdx.x
            z[tid] = g(10)(x[tid], y[tid])
            z[tid] += g(20)(x[tid], y[tid])
            z[tid] += g(30)(x[tid], y[tid])

        def g(const):
            if False:
                while True:
                    i = 10

            @jit.rawkernel(device=True)
            def f(x, y):
                if False:
                    i = 10
                    return i + 15
                return x + y + const
            return f
        x = testing.shaped_random((30,), dtype=numpy.int32, seed=0)
        y = testing.shaped_random((30,), dtype=numpy.int32, seed=1)
        z = testing.shaped_random((30,), dtype=numpy.int32, seed=2)
        f((1,), (30,), (x, y, z))
        assert (z == (x + y) * 3 + 60).all()

    def test_device_function_recursive(self):
        if False:
            for i in range(10):
                print('nop')

        @jit.rawkernel()
        def f(x, y, z):
            if False:
                while True:
                    i = 10
            tid = jit.threadIdx.x
            z[tid] = g(x[tid], y[tid])

        @jit.rawkernel(device=True)
        def g(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x + y + g(x, y)
        x = testing.shaped_random((30,), dtype=numpy.int32, seed=0)
        y = testing.shaped_random((30,), dtype=numpy.int32, seed=1)
        z = testing.shaped_random((30,), dtype=numpy.int32, seed=2)
        with pytest.raises(ValueError):
            f((1,), (30,), (x, y, z))

    def test_device_function_template_recursion(self):
        if False:
            while True:
                i = 10

        @jit.rawkernel()
        def f(x, y):
            if False:
                i = 10
                return i + 15
            tid = jit.threadIdx.x
            y[tid] = x[tid]
            jit.syncthreads()
            g(1)(y)

        def g(step):
            if False:
                print('Hello World!')

            @jit.rawkernel(device=True)
            def f(x):
                if False:
                    i = 10
                    return i + 15
                if step < 256:
                    tid = jit.threadIdx.x
                    if tid % (step * 2) == 0:
                        x[tid] += x[tid + step]
                        jit.syncthreads()
                        g(step * 2)(x)
            return f
        x = testing.shaped_random((256,), dtype=numpy.int32, seed=0)
        y = testing.shaped_random((256,), dtype=numpy.int32, seed=1)
        f((1,), (256,), (x, y))
        assert y[0] == x.sum()

    def test_device_function_called_once(self):
        if False:
            for i in range(10):
                print('nop')

        @jit.rawkernel(device=True)
        def g(x):
            if False:
                return 10
            x[0] += 1
            return 1

        @jit.rawkernel()
        def f(x):
            if False:
                print('Hello World!')
            x[g(x)] += 1
        x = cupy.array([0, 0])
        f((1,), (1,), (x,))
        testing.assert_array_equal(x, [1, 1])