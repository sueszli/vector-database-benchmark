"""
Test Timeout helper class.
"""
import sys
import unittest
import time
from serial import serialutil

class TestTimeoutClass(unittest.TestCase):
    """Test the Timeout class"""

    def test_simple_timeout(self):
        if False:
            for i in range(10):
                print('nop')
        'Test simple timeout'
        t = serialutil.Timeout(2)
        self.assertFalse(t.expired())
        self.assertTrue(t.time_left() > 0)
        time.sleep(2.1)
        self.assertTrue(t.expired())
        self.assertEqual(t.time_left(), 0)

    def test_non_blocking(self):
        if False:
            print('Hello World!')
        'Test nonblocking case (0)'
        t = serialutil.Timeout(0)
        self.assertTrue(t.is_non_blocking)
        self.assertFalse(t.is_infinite)
        self.assertTrue(t.expired())

    def test_blocking(self):
        if False:
            while True:
                i = 10
        'Test no timeout (None)'
        t = serialutil.Timeout(None)
        self.assertFalse(t.is_non_blocking)
        self.assertTrue(t.is_infinite)

    def test_changing_clock(self):
        if False:
            return 10
        'Test recovery from changing clock'

        class T(serialutil.Timeout):

            def TIME(self):
                if False:
                    return 10
                return test_time
        test_time = 1000
        t = T(10)
        self.assertEqual(t.target_time, 1010)
        self.assertFalse(t.expired())
        self.assertTrue(t.time_left() > 0)
        test_time = 100
        self.assertTrue(t.time_left() > 0)
        self.assertTrue(t.time_left() <= 10)
        self.assertEqual(t.target_time, 110)
        test_time = 10000
        self.assertEqual(t.time_left(), 0)
if __name__ == '__main__':
    sys.stdout.write(__doc__)
    if len(sys.argv) > 1:
        PORT = sys.argv[1]
    sys.argv[1:] = ['-v']
    unittest.main()