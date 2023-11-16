"""
Test return values
"""
import math
import unittest
from numba.core.compiler import compile_isolated, Flags
from numba.core import types
from numba.core.errors import TypingError, NumbaTypeError
enable_pyobj_flags = Flags()
enable_pyobj_flags.enable_pyobject = True
no_pyobj_flags = Flags()

def get_nopython_func():
    if False:
        i = 10
        return i + 15
    return abs

def get_pyobj_func():
    if False:
        i = 10
        return i + 15
    return open

def get_module_func():
    if False:
        while True:
            i = 10
    return math.floor

class TestReturnValues(unittest.TestCase):

    def test_nopython_func(self, flags=enable_pyobj_flags):
        if False:
            i = 10
            return i + 15
        pyfunc = get_nopython_func
        cr = compile_isolated(pyfunc, (), flags=flags)
        cfunc = cr.entry_point
        if flags == enable_pyobj_flags:
            result = cfunc()
            self.assertEqual(result, abs)
        else:
            result = cfunc()

    def test_nopython_func_npm(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(NumbaTypeError):
            self.test_nopython_func(flags=no_pyobj_flags)

    def test_pyobj_func(self, flags=enable_pyobj_flags):
        if False:
            print('Hello World!')
        pyfunc = get_pyobj_func
        cr = compile_isolated(pyfunc, (), flags=flags)
        cfunc = cr.entry_point
        if flags == enable_pyobj_flags:
            result = cfunc()
            self.assertEqual(result, open)
        else:
            result = cfunc()

    def test_pyobj_func_npm(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(TypingError):
            self.test_pyobj_func(flags=no_pyobj_flags)

    def test_module_func(self, flags=enable_pyobj_flags):
        if False:
            for i in range(10):
                print('nop')
        pyfunc = get_module_func
        cr = compile_isolated(pyfunc, (), flags=flags)
        cfunc = cr.entry_point
        if flags == enable_pyobj_flags:
            result = cfunc()
            self.assertEqual(result, math.floor)
        else:
            result = cfunc()

    def test_module_func_npm(self):
        if False:
            return 10
        with self.assertRaises(NumbaTypeError):
            self.test_module_func(flags=no_pyobj_flags)
if __name__ == '__main__':
    unittest.main()