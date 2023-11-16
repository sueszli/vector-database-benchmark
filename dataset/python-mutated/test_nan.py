import unittest
from numba.core.compiler import compile_isolated, Flags
from numba.core import types
enable_pyobj_flags = Flags()
enable_pyobj_flags.enable_pyobject = True
no_pyobj_flags = Flags()

def isnan(x):
    if False:
        print('Hello World!')
    return x != x

def isequal(x):
    if False:
        for i in range(10):
            print('nop')
    return x == x

class TestNaN(unittest.TestCase):

    def test_nans(self, flags=enable_pyobj_flags):
        if False:
            print('Hello World!')
        pyfunc = isnan
        cr = compile_isolated(pyfunc, (types.float64,), flags=flags)
        cfunc = cr.entry_point
        self.assertTrue(cfunc(float('nan')))
        self.assertFalse(cfunc(1.0))
        pyfunc = isequal
        cr = compile_isolated(pyfunc, (types.float64,), flags=flags)
        cfunc = cr.entry_point
        self.assertFalse(cfunc(float('nan')))
        self.assertTrue(cfunc(1.0))

    def test_nans_npm(self):
        if False:
            print('Hello World!')
        self.test_nans(flags=no_pyobj_flags)
if __name__ == '__main__':
    unittest.main()