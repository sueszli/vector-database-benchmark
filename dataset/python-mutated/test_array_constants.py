import numpy as np
import unittest
from numba.core.compiler import compile_isolated
from numba.core.errors import TypingError
from numba import jit, typeof
from numba.core import types
from numba.tests.support import skip_m1_llvm_rtdyld_failure
a0 = np.array(42)
s1 = np.int32(64)
a1 = np.arange(12)
a2 = a1[::2]
a3 = a1.reshape((3, 4)).T
dt = np.dtype([('x', np.int8), ('y', 'S3')])
a4 = np.arange(32, dtype=np.int8).view(dt)
a5 = a4[::-2]
a6 = np.frombuffer(b'XXXX_array_contents_XXXX', dtype=np.float32)
myarray = np.array([1])

def getitem0(i):
    if False:
        for i in range(10):
            print('nop')
    return a0[()]

def getitem1(i):
    if False:
        return 10
    return a1[i]

def getitem2(i):
    if False:
        for i in range(10):
            print('nop')
    return a2[i]

def getitem3(i):
    if False:
        print('Hello World!')
    return a3[i]

def getitem4(i):
    if False:
        for i in range(10):
            print('nop')
    return a4[i]

def getitem5(i):
    if False:
        print('Hello World!')
    return a5[i]

def getitem6(i):
    if False:
        return 10
    return a6[i]

def use_arrayscalar_const():
    if False:
        return 10
    return s1

def write_to_global_array():
    if False:
        while True:
            i = 10
    myarray[0] = 1

def bytes_as_const_array():
    if False:
        for i in range(10):
            print('nop')
    return np.frombuffer(b'foo', dtype=np.uint8)

class TestConstantArray(unittest.TestCase):
    """
    Test array constants.
    """

    def check_array_const(self, pyfunc):
        if False:
            for i in range(10):
                print('nop')
        cres = compile_isolated(pyfunc, (types.int32,))
        cfunc = cres.entry_point
        for i in [0, 1, 2]:
            np.testing.assert_array_equal(pyfunc(i), cfunc(i))

    def test_array_const_0d(self):
        if False:
            while True:
                i = 10
        self.check_array_const(getitem0)

    def test_array_const_1d_contig(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_array_const(getitem1)

    def test_array_const_1d_noncontig(self):
        if False:
            print('Hello World!')
        self.check_array_const(getitem2)

    def test_array_const_2d(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_array_const(getitem3)

    def test_record_array_const_contig(self):
        if False:
            print('Hello World!')
        self.check_array_const(getitem4)

    def test_record_array_const_noncontig(self):
        if False:
            i = 10
            return i + 15
        self.check_array_const(getitem5)

    def test_array_const_alignment(self):
        if False:
            return 10
        '\n        Issue #1933: the array declaration in the LLVM IR must have\n        the right alignment specified.\n        '
        sig = (types.intp,)
        cfunc = jit(sig, nopython=True)(getitem6)
        ir = cfunc.inspect_llvm(sig)
        for line in ir.splitlines():
            if 'XXXX_array_contents_XXXX' in line:
                self.assertIn('constant [24 x i8]', line)
                self.assertIn(', align 4', line)
                break
        else:
            self.fail('could not find array declaration in LLVM IR')

    def test_arrayscalar_const(self):
        if False:
            return 10
        pyfunc = use_arrayscalar_const
        cres = compile_isolated(pyfunc, ())
        cfunc = cres.entry_point
        self.assertEqual(pyfunc(), cfunc())

    def test_write_to_global_array(self):
        if False:
            i = 10
            return i + 15
        pyfunc = write_to_global_array
        with self.assertRaises(TypingError):
            compile_isolated(pyfunc, ())

    def test_issue_1850(self):
        if False:
            i = 10
            return i + 15
        '\n        This issue is caused by an unresolved bug in numpy since version 1.6.\n        See numpy GH issue #3147.\n        '
        constarr = np.array([86])

        def pyfunc():
            if False:
                return 10
            return constarr[0]
        cres = compile_isolated(pyfunc, ())
        out = cres.entry_point()
        self.assertEqual(out, 86)

    @skip_m1_llvm_rtdyld_failure
    def test_too_big_to_freeze(self):
        if False:
            print('Hello World!')
        "\n        Test issue https://github.com/numba/numba/issues/2188 where freezing\n        a constant array into the code that's prohibitively long and consumes\n        too much RAM.\n        "

        def test(biggie):
            if False:
                while True:
                    i = 10
            expect = np.copy(biggie)
            self.assertEqual(typeof(biggie), typeof(expect))

            def pyfunc():
                if False:
                    i = 10
                    return i + 15
                return biggie
            cres = compile_isolated(pyfunc, ())
            self.assertLess(len(cres.library.get_llvm_str()), biggie.nbytes)
            out = cres.entry_point()
            self.assertIs(biggie, out)
            del out
            biggie = None
            out = cres.entry_point()
            np.testing.assert_equal(expect, out)
            self.assertEqual(typeof(expect), typeof(out))
        nelem = 10 ** 7
        c_array = np.arange(nelem).reshape(nelem)
        f_array = np.asfortranarray(np.random.random((2, nelem // 2)))
        self.assertEqual(typeof(c_array).layout, 'C')
        self.assertEqual(typeof(f_array).layout, 'F')
        test(c_array)
        test(f_array)

class TestConstantBytes(unittest.TestCase):

    def test_constant_bytes(self):
        if False:
            return 10
        pyfunc = bytes_as_const_array
        cres = compile_isolated(pyfunc, ())
        cfunc = cres.entry_point
        np.testing.assert_array_equal(pyfunc(), cfunc())
if __name__ == '__main__':
    unittest.main()