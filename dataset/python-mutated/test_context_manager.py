import unittest
from paddle import base

class TestContextManagerRaiseException(unittest.TestCase):

    def test_func1(self):
        if False:
            i = 10
            return i + 15

        def foo():
            if False:
                for i in range(10):
                    print('nop')
            with base.dygraph.guard():
                print('raise error in context manager')
                raise TypeError('error')
        self.assertRaises(TypeError, foo)

    def test_func2(self):
        if False:
            while True:
                i = 10
        self.assertEqual(base.in_dygraph_mode(), False)
if __name__ == '__main__':
    unittest.main()