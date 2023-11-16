import re
import types
import numpy as np
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from numba import cuda, jit, float32, int32
from numba.core.errors import TypingError

class TestDeviceFunc(CUDATestCase):

    def test_use_add2f(self):
        if False:
            for i in range(10):
                print('nop')

        @cuda.jit('float32(float32, float32)', device=True)
        def add2f(a, b):
            if False:
                while True:
                    i = 10
            return a + b

        def use_add2f(ary):
            if False:
                while True:
                    i = 10
            i = cuda.grid(1)
            ary[i] = add2f(ary[i], ary[i])
        compiled = cuda.jit('void(float32[:])')(use_add2f)
        nelem = 10
        ary = np.arange(nelem, dtype=np.float32)
        exp = ary + ary
        compiled[1, nelem](ary)
        self.assertTrue(np.all(ary == exp), (ary, exp))

    def test_indirect_add2f(self):
        if False:
            i = 10
            return i + 15

        @cuda.jit('float32(float32, float32)', device=True)
        def add2f(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return a + b

        @cuda.jit('float32(float32, float32)', device=True)
        def indirect(a, b):
            if False:
                i = 10
                return i + 15
            return add2f(a, b)

        def indirect_add2f(ary):
            if False:
                while True:
                    i = 10
            i = cuda.grid(1)
            ary[i] = indirect(ary[i], ary[i])
        compiled = cuda.jit('void(float32[:])')(indirect_add2f)
        nelem = 10
        ary = np.arange(nelem, dtype=np.float32)
        exp = ary + ary
        compiled[1, nelem](ary)
        self.assertTrue(np.all(ary == exp), (ary, exp))

    def _check_cpu_dispatcher(self, add):
        if False:
            print('Hello World!')

        @cuda.jit
        def add_kernel(ary):
            if False:
                while True:
                    i = 10
            i = cuda.grid(1)
            ary[i] = add(ary[i], 1)
        ary = np.arange(10)
        expect = ary + 1
        add_kernel[1, ary.size](ary)
        np.testing.assert_equal(expect, ary)

    def test_cpu_dispatcher(self):
        if False:
            print('Hello World!')

        @jit
        def add(a, b):
            if False:
                i = 10
                return i + 15
            return a + b
        self._check_cpu_dispatcher(add)

    @skip_on_cudasim('not supported in cudasim')
    def test_cpu_dispatcher_invalid(self):
        if False:
            return 10

        @jit('(i4, i4)')
        def add(a, b):
            if False:
                i = 10
                return i + 15
            return a + b
        with self.assertRaises(TypingError) as raises:
            self._check_cpu_dispatcher(add)
        msg = "Untyped global name 'add':.*using cpu function on device"
        expected = re.compile(msg)
        self.assertTrue(expected.search(str(raises.exception)) is not None)

    def test_cpu_dispatcher_other_module(self):
        if False:
            i = 10
            return i + 15

        @jit
        def add(a, b):
            if False:
                i = 10
                return i + 15
            return a + b
        mymod = types.ModuleType(name='mymod')
        mymod.add = add
        del add

        @cuda.jit
        def add_kernel(ary):
            if False:
                return 10
            i = cuda.grid(1)
            ary[i] = mymod.add(ary[i], 1)
        ary = np.arange(10)
        expect = ary + 1
        add_kernel[1, ary.size](ary)
        np.testing.assert_equal(expect, ary)

    @skip_on_cudasim('not supported in cudasim')
    def test_inspect_llvm(self):
        if False:
            print('Hello World!')

        @cuda.jit(device=True)
        def foo(x, y):
            if False:
                while True:
                    i = 10
            return x + y
        args = (int32, int32)
        cres = foo.compile_device(args)
        fname = cres.fndesc.mangled_name
        self.assertIn('foo', fname)
        llvm = foo.inspect_llvm(args)
        self.assertIn(fname, llvm)

    @skip_on_cudasim('not supported in cudasim')
    def test_inspect_asm(self):
        if False:
            i = 10
            return i + 15

        @cuda.jit(device=True)
        def foo(x, y):
            if False:
                return 10
            return x + y
        args = (int32, int32)
        cres = foo.compile_device(args)
        fname = cres.fndesc.mangled_name
        self.assertIn('foo', fname)
        ptx = foo.inspect_asm(args)
        self.assertIn(fname, ptx)

    @skip_on_cudasim('not supported in cudasim')
    def test_inspect_sass_disallowed(self):
        if False:
            while True:
                i = 10

        @cuda.jit(device=True)
        def foo(x, y):
            if False:
                return 10
            return x + y
        with self.assertRaises(RuntimeError) as raises:
            foo.inspect_sass((int32, int32))
        self.assertIn('Cannot inspect SASS of a device function', str(raises.exception))

    @skip_on_cudasim('cudasim will allow calling any function')
    def test_device_func_as_kernel_disallowed(self):
        if False:
            for i in range(10):
                print('nop')

        @cuda.jit(device=True)
        def f():
            if False:
                for i in range(10):
                    print('nop')
            pass
        with self.assertRaises(RuntimeError) as raises:
            f[1, 1]()
        self.assertIn('Cannot compile a device function as a kernel', str(raises.exception))

    @skip_on_cudasim('cudasim ignores casting by jit decorator signature')
    def test_device_casting(self):
        if False:
            return 10

        @cuda.jit('int32(int32, int32, int32, int32)', device=True)
        def rgba(r, g, b, a):
            if False:
                return 10
            return (r & 255) << 16 | (g & 255) << 8 | (b & 255) << 0 | (a & 255) << 24

        @cuda.jit
        def rgba_caller(x, channels):
            if False:
                return 10
            x[0] = rgba(channels[0], channels[1], channels[2], channels[3])
        x = cuda.device_array(1, dtype=np.int32)
        channels = cuda.to_device(np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        rgba_caller[1, 1](x, channels)
        self.assertEqual(67174915, x[0])

    def _test_declare_device(self, decl):
        if False:
            return 10
        self.assertEqual(decl.name, 'f1')
        self.assertEqual(decl.sig.args, (float32[:],))
        self.assertEqual(decl.sig.return_type, int32)

    @skip_on_cudasim('cudasim does not check signatures')
    def test_declare_device_signature(self):
        if False:
            return 10
        f1 = cuda.declare_device('f1', int32(float32[:]))
        self._test_declare_device(f1)

    @skip_on_cudasim('cudasim does not check signatures')
    def test_declare_device_string(self):
        if False:
            while True:
                i = 10
        f1 = cuda.declare_device('f1', 'int32(float32[:])')
        self._test_declare_device(f1)

    @skip_on_cudasim('cudasim does not check signatures')
    def test_bad_declare_device_tuple(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(TypeError, 'Return type'):
            cuda.declare_device('f1', (float32[:],))

    @skip_on_cudasim('cudasim does not check signatures')
    def test_bad_declare_device_string(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesRegex(TypeError, 'Return type'):
            cuda.declare_device('f1', '(float32[:],)')
if __name__ == '__main__':
    unittest.main()