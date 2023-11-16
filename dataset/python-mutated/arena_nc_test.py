"""Negative compilation unit tests for arena API."""
import unittest
from google3.testing.pybase import fake_target_util
from google3.testing.pybase import unittest

class ArenaNcTest(unittest.TestCase):

    def testCompilerErrors(self):
        if False:
            i = 10
            return i + 15
        'Runs a list of tests to verify compiler error messages.'
        test_specs = [('ARENA_PRIVATE_CONSTRUCTOR', ['calling a protected constructor']), ('SANITY', None)]
        fake_target_util.AssertCcCompilerErrors(self, 'google3/google/protobuf/arena_nc', 'arena_nc.o', test_specs)
if __name__ == '__main__':
    unittest.main()