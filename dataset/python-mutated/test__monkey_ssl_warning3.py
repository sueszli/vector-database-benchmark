import unittest
import warnings
import sys
import ssl

class MySubclass(ssl.SSLContext):
    pass

class Test(unittest.TestCase):

    @unittest.skipIf(sys.version_info[:2] < (3, 6), 'Only on Python 3.6+')
    def test_ssl_subclass_and_module_reference(self):
        if False:
            for i in range(10):
                print('nop')
        from gevent import monkey
        self.assertFalse(monkey.saved)
        with warnings.catch_warnings(record=True) as issued_warnings:
            warnings.simplefilter('always')
            monkey.patch_all()
            monkey.patch_all()
        issued_warnings = [x for x in issued_warnings if isinstance(x.message, monkey.MonkeyPatchWarning)]
        self.assertEqual(1, len(issued_warnings))
        message = str(issued_warnings[0].message)
        self.assertNotIn('Modules that had direct imports', message)
        self.assertIn('Subclasses (NOT patched)', message)
        self.assertNotIn('gevent.', message)
if __name__ == '__main__':
    unittest.main()