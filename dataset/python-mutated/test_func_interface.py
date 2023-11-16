import unittest
from numba import jit

class TestFuncInterface(unittest.TestCase):

    def test_jit_function_docstring(self):
        if False:
            i = 10
            return i + 15

        def add(x, y):
            if False:
                while True:
                    i = 10
            'Return sum of two numbers'
            return x + y
        c_add = jit(add)
        self.assertEqual(c_add.__doc__, 'Return sum of two numbers')

    def test_jit_function_name(self):
        if False:
            for i in range(10):
                print('nop')

        def add(x, y):
            if False:
                print('Hello World!')
            return x + y
        c_add = jit(add)
        self.assertEqual(c_add.__name__, 'add')

    def test_jit_function_module(self):
        if False:
            print('Hello World!')

        def add(x, y):
            if False:
                return 10
            return x + y
        c_add = jit(add)
        self.assertEqual(c_add.__module__, add.__module__)

    def test_jit_function_code_object(self):
        if False:
            i = 10
            return i + 15

        def add(x, y):
            if False:
                print('Hello World!')
            return x + y
        c_add = jit(add)
        self.assertEqual(c_add.__code__, add.__code__)
        self.assertEqual(c_add.func_code, add.__code__)
if __name__ == '__main__':
    unittest.main()