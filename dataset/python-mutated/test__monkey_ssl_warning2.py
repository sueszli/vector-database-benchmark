import unittest
import warnings
import sys
from ssl import SSLContext

class MySubclass(SSLContext):
    pass

class Test(unittest.TestCase):

    @unittest.skipIf(sys.version_info[:2] < (3, 6), 'Only on Python 3.6+')
    def test_ssl_subclass_and_module_reference(self):
        if False:
            i = 10
            return i + 15
        from gevent import monkey
        self.assertFalse(monkey.saved)
        with warnings.catch_warnings(record=True) as issued_warnings:
            warnings.simplefilter('always')
            monkey.patch_all()
            monkey.patch_all()
        issued_warnings = [x for x in issued_warnings if isinstance(x.message, monkey.MonkeyPatchWarning)]
        self.assertEqual(1, len(issued_warnings))
        message = issued_warnings[0].message
        self.assertIn('Modules that had direct imports', str(message))
        self.assertIn('Subclasses (NOT patched)', str(message))
if __name__ == '__main__':
    unittest.main()