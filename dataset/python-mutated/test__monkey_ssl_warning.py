import unittest
import warnings

class Test(unittest.TestCase):

    def test_with_pkg_resources(self):
        if False:
            i = 10
            return i + 15
        __import__('pkg_resources')
        from gevent import monkey
        self.assertFalse(monkey.saved)
        with warnings.catch_warnings(record=True) as issued_warnings:
            warnings.simplefilter('always')
            monkey.patch_all()
            monkey.patch_all()
        issued_warnings = [x for x in issued_warnings if isinstance(x.message, monkey.MonkeyPatchWarning)]
        self.assertFalse(issued_warnings, [str(i) for i in issued_warnings])
        self.assertEqual(0, len(issued_warnings))
if __name__ == '__main__':
    unittest.main()