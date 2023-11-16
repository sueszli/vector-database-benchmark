"""Unit tests for compiled implementation of coder impls."""
import logging
import unittest
from apache_beam.coders.coders_test_common import *
from apache_beam.tools import utils

class FastCoders(unittest.TestCase):

    def test_using_fast_impl(self):
        if False:
            return 10
        try:
            utils.check_compiled('apache_beam.coders.coder_impl')
        except RuntimeError:
            self.skipTest('Cython is not installed')
        import apache_beam.coders.stream
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()