import builtins
import unittest
from paddle.check_import_scipy import check_import_scipy

def my_import(name, globals=None, locals=None, fromlist=(), level=0):
    if False:
        print('Hello World!')
    raise ImportError('DLL load failed, unittest: import scipy failed')

class importTest(unittest.TestCase):

    def test_import(self):
        if False:
            for i in range(10):
                print('nop')
        testOsName = 'nt'
        old_import = builtins.__import__
        builtins.__import__ = my_import
        self.assertRaises(ImportError, check_import_scipy, testOsName)
        builtins.__import__ = old_import
if __name__ == '__main__':
    unittest.main()