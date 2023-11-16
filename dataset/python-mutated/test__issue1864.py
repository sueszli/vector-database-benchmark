import sys
import unittest
from gevent import testing as greentest

class TestSubnormalFloatsAreNotDisabled(unittest.TestCase):

    @greentest.skipOnCI('Some of our tests we compile with -Ofast, which breaks this.')
    def test_subnormal_is_not_zero(self):
        if False:
            return 10
        __import__('gevent')
        self.assertGreater(sys.float_info.min / 2, 0.0)
if __name__ == '__main__':
    unittest.main()