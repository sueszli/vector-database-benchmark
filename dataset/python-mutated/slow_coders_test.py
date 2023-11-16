"""Unit tests for uncompiled implementation of coder impls."""
import logging
import unittest
from apache_beam.coders.coders_test_common import *

@unittest.skip('Remove non-cython tests.https://github.com/apache/beam/issues/28307')
class SlowCoders(unittest.TestCase):

    def test_using_slow_impl(self):
        if False:
            return 10
        try:
            from Cython.Build import cythonize
            self.skipTest('Found cython, cannot test non-compiled implementation.')
        except ImportError:
            with self.assertRaises(ImportError):
                import apache_beam.coders.stream
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()