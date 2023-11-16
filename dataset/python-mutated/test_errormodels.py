"""
Test setting/overriding error models
"""
from numba import jit
import unittest

class TestErrorModel(unittest.TestCase):

    def test_div_by_zero_python(self):
        if False:
            for i in range(10):
                print('nop')

        @jit
        def model_python(val):
            if False:
                print('Hello World!')
            return 1 / val
        with self.assertRaises(ZeroDivisionError):
            model_python(0)

    def test_div_by_zero_numpy(self):
        if False:
            print('Hello World!')

        @jit(error_model='numpy')
        def model_numpy(val):
            if False:
                while True:
                    i = 10
            return 1 / val
        self.assertEqual(model_numpy(0), float('inf'))
if __name__ == '__main__':
    unittest.main()