import unittest
from paddle.utils.lazy_import import try_import

class TestUtilsLazyImport(unittest.TestCase):

    def setup(self):
        if False:
            while True:
                i = 10
        pass

    def func_test_lazy_import(self):
        if False:
            i = 10
            return i + 15
        paddle = try_import('paddle')
        self.assertIsNotNone(paddle.__version__)
        with self.assertRaises(ImportError) as context:
            paddle2 = try_import('paddle2')
        self.assertTrue('require additional dependencies that have to be' in str(context.exception))
        with self.assertRaises(ImportError) as context:
            paddle2 = try_import('paddle2', 'paddle2 is not installed')
        self.assertTrue('paddle2 is not installed' in str(context.exception))

    def test_lazy_import(self):
        if False:
            for i in range(10):
                print('nop')
        self.func_test_lazy_import()
if __name__ == '__main__':
    unittest.main()