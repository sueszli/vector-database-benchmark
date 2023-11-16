import os
import unittest

class DriverTest(unittest.TestCase):

    def test_success(self):
        if False:
            while True:
                i = 10
        pass

    def test_skip(self):
        if False:
            return 10
        raise unittest.SkipTest('expected skip')

    def test_driver_passthrough(self):
        if False:
            return 10
        self.assertTrue(os.environ.get('TEST_ENVIRON_SET') is None)