"""Test RFC 2217 related functionality.
"""
import unittest
import serial
import serial.rfc2217

class Test_RFC2217(unittest.TestCase):
    """Test RFC 2217 related functionality"""

    def test_failed_connection(self):
        if False:
            print('Hello World!')
        s = serial.serial_for_url('rfc2217://127.99.99.99:2217', do_not_open=True)
        self.assertRaises(serial.SerialException, s.open)
        self.assertFalse(s.is_open)
        s.close()
        s = serial.serial_for_url('rfc2217://127goingtofail', do_not_open=True)
        self.assertRaises(serial.SerialException, s.open)
        self.assertFalse(s.is_open)
        s.close()
        s = serial.serial_for_url('rfc2217://irrelevant', do_not_open=True)
        self.assertFalse(s.is_open)
        s.close()
if __name__ == '__main__':
    import sys
    sys.stdout.write(__doc__)
    sys.stdout.write('Testing connection on localhost\n')
    sys.argv[1:] = ['-v']
    unittest.main()