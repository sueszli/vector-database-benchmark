import random
import numpy as np
from numba import njit
from numba.core import types
import unittest

class TestMulti3(unittest.TestCase):
    """
    This test is only relevant for 32-bit architectures.

    Test __multi3 implementation in _helperlib.c.
    The symbol defines a i128 multiplication.
    It is necessary for working around an issue in LLVM (see issue #969).
    The symbol does not exist in 32-bit platform, and should not be used by
    LLVM.  However, optimization passes will create i65 multiplication that
    is then lowered to __multi3.
    """

    def test_multi3(self):
        if False:
            return 10

        @njit('(int64,)')
        def func(x):
            if False:
                return 10
            res = 0
            for i in range(x):
                res += i
            return res
        x_cases = [-1, 0, 1, 3, 4, 8, 4294967295 - 1, 4294967295, 4294967295 + 1, 81985529216486895, -81985529216486895]
        for _ in range(500):
            x_cases.append(random.randint(0, 4294967295))

        def expected(x):
            if False:
                for i in range(10):
                    print('nop')
            if x <= 0:
                return 0
            return x * (x - 1) // 2 & 2 ** 64 - 1
        for x in x_cases:
            self.assertEqual(expected(x), func(x))
if __name__ == '__main__':
    unittest.main()