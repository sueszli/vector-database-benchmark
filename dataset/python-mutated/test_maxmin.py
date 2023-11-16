from numba.core.compiler import compile_isolated
from numba.core import types
import unittest

def domax3(a, b, c):
    if False:
        return 10
    return max(a, b, c)

def domin3(a, b, c):
    if False:
        for i in range(10):
            print('nop')
    return min(a, b, c)

class TestMaxMin(unittest.TestCase):

    def test_max3(self):
        if False:
            return 10
        pyfunc = domax3
        argtys = (types.int32, types.float32, types.double)
        cres = compile_isolated(pyfunc, argtys)
        cfunc = cres.entry_point
        a = 1
        b = 2
        c = 3
        self.assertEqual(pyfunc(a, b, c), cfunc(a, b, c))

    def test_min3(self):
        if False:
            return 10
        pyfunc = domin3
        argtys = (types.int32, types.float32, types.double)
        cres = compile_isolated(pyfunc, argtys)
        cfunc = cres.entry_point
        a = 1
        b = 2
        c = 3
        self.assertEqual(pyfunc(a, b, c), cfunc(a, b, c))
if __name__ == '__main__':
    unittest.main()