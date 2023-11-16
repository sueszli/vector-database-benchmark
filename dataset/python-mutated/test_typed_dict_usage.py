import unittest
from numba.tests.support import captured_stdout

class DocsTypedDictUsageTest(unittest.TestCase):

    def test_ex_typed_dict_from_cpython(self):
        if False:
            for i in range(10):
                print('nop')
        with captured_stdout():
            import numpy as np
            from numba import njit
            from numba.core import types
            from numba.typed import Dict
            d = Dict.empty(key_type=types.unicode_type, value_type=types.float64[:])
            d['posx'] = np.asarray([1, 0.5, 2], dtype='f8')
            d['posy'] = np.asarray([1.5, 3.5, 2], dtype='f8')
            d['velx'] = np.asarray([0.5, 0, 0.7], dtype='f8')
            d['vely'] = np.asarray([0.2, -0.2, 0.1], dtype='f8')

            @njit
            def move(d):
                if False:
                    print('Hello World!')
                d['posx'] += d['velx']
                d['posy'] += d['vely']
            print('posx: ', d['posx'])
            print('posy: ', d['posy'])
            move(d)
            print('posx: ', d['posx'])
            print('posy: ', d['posy'])
        np.testing.assert_array_equal(d['posx'], [1.5, 0.5, 2.7])
        np.testing.assert_array_equal(d['posy'], [1.7, 3.3, 2.1])

    def test_ex_typed_dict_njit(self):
        if False:
            return 10
        with captured_stdout():
            import numpy as np
            from numba import njit
            from numba.core import types
            from numba.typed import Dict
            float_array = types.float64[:]

            @njit
            def foo():
                if False:
                    while True:
                        i = 10
                d = Dict.empty(key_type=types.unicode_type, value_type=float_array)
                d['posx'] = np.arange(3).astype(np.float64)
                d['posy'] = np.arange(3, 6).astype(np.float64)
                return d
            d = foo()
            print(d)
        np.testing.assert_array_equal(d['posx'], [0, 1, 2])
        np.testing.assert_array_equal(d['posy'], [3, 4, 5])

    def test_ex_inferred_dict_njit(self):
        if False:
            return 10
        with captured_stdout():
            from numba import njit
            import numpy as np

            @njit
            def foo():
                if False:
                    i = 10
                    return i + 15
                d = dict()
                k = {1: np.arange(1), 2: np.arange(2)}
                d[3] = np.arange(3)
                d[5] = np.arange(5)
                return (d, k)
            (d, k) = foo()
            print(d)
            print(k)
        np.testing.assert_array_equal(d[3], [0, 1, 2])
        np.testing.assert_array_equal(d[5], [0, 1, 2, 3, 4])
        np.testing.assert_array_equal(k[1], [0])
        np.testing.assert_array_equal(k[2], [0, 1])
if __name__ == '__main__':
    unittest.main()