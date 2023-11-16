"""Some tests for the serial module.
Part of pyserial (http://pyserial.sf.net)  (C)2001-2009 cliechti@gmx.net

Intended to be run on different platforms, to ensure portability of
the code.

This modules contains test for the interaction between Serial and the io
library. This only works on Python 2.6+ that introduced the io library.

For all these tests a simple hardware is required.
Loopback HW adapter:
Shortcut these pin pairs:
 TX  <-> RX
 RTS <-> CTS
 DTR <-> DSR

On a 9 pole DSUB these are the pins (2-3) (4-6) (7-8)
"""
import io
import sys
import unittest
import serial
PORT = 'loop://'

class Test_SerialAndIO(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.s = serial.serial_for_url(PORT, timeout=1)
        self.io = io.TextIOWrapper(io.BufferedRWPair(self.s, self.s))

    def tearDown(self):
        if False:
            return 10
        self.s.close()

    def test_hello_raw(self):
        if False:
            i = 10
            return i + 15
        self.io.write(b'hello\n'.decode('utf-8'))
        self.io.flush()
        hello = self.io.readline()
        self.assertEqual(hello, b'hello\n'.decode('utf-8'))
if __name__ == '__main__':
    import sys
    sys.stdout.write(__doc__)
    if len(sys.argv) > 1:
        PORT = sys.argv[1]
    sys.stdout.write('Testing port: {!r}\n'.format(PORT))
    sys.argv[1:] = ['-v']
    unittest.main()