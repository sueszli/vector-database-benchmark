"""Tests for exclusive access feature.
"""
import os
import unittest
import sys
import serial
PORT = 'loop://'

class Test_exclusive(unittest.TestCase):
    """Test serial port locking"""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        with serial.serial_for_url(PORT, do_not_open=True) as x:
            if not isinstance(x, serial.Serial):
                raise unittest.SkipTest('exclusive test only compatible with real serial port')

    def test_exclusive_none(self):
        if False:
            i = 10
            return i + 15
        'test for exclusive=None'
        with serial.Serial(PORT, exclusive=None):
            pass

    @unittest.skipUnless(os.name == 'posix', 'exclusive=False not supported on platform')
    def test_exclusive_false(self):
        if False:
            while True:
                i = 10
        'test for exclusive=False'
        with serial.Serial(PORT, exclusive=False):
            pass

    @unittest.skipUnless(os.name in ('posix', 'nt'), 'exclusive=True setting not supported on platform')
    def test_exclusive_true(self):
        if False:
            print('Hello World!')
        'test for exclusive=True'
        with serial.Serial(PORT, exclusive=True):
            with self.assertRaises(serial.SerialException):
                serial.Serial(PORT, exclusive=True)

    @unittest.skipUnless(os.name == 'nt', 'platform is not restricted to exclusive=True (and None)')
    def test_exclusive_only_true(self):
        if False:
            i = 10
            return i + 15
        'test if exclusive=False is not supported'
        with self.assertRaises(ValueError):
            serial.Serial(PORT, exclusive=False)
if __name__ == '__main__':
    sys.stdout.write(__doc__)
    if len(sys.argv) > 1:
        PORT = sys.argv[1]
    sys.stdout.write('Testing port: {!r}\n'.format(PORT))
    sys.argv[1:] = ['-v']
    unittest.main()