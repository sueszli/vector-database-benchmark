import warnings
from pelican.tests.support import unittest

class TestSuiteTest(unittest.TestCase):

    def test_error_on_warning(self):
        if False:
            return 10
        with self.assertRaises(UserWarning):
            warnings.warn('test warning')