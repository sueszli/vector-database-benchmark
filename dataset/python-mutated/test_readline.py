"""Some tests for the serial module.
Part of pyserial (http://pyserial.sf.net)  (C)2010 cliechti@gmx.net

Intended to be run on different platforms, to ensure portability of
the code.

For all these tests a simple hardware is required.
Loopback HW adapter:
Shortcut these pin pairs:
 TX  <-> RX
 RTS <-> CTS
 DTR <-> DSR

On a 9 pole DSUB these are the pins (2-3) (4-6) (7-8)
"""
import unittest
import sys
import serial
PORT = 'loop://'
if sys.version_info >= (3, 0):

    def data(string):
        if False:
            i = 10
            return i + 15
        return bytes(string, 'latin1')
else:

    def data(string):
        if False:
            while True:
                i = 10
        return string

class Test_Readline(unittest.TestCase):
    """Test readline function"""

    def setUp(self):
        if False:
            return 10
        self.s = serial.serial_for_url(PORT, timeout=1)

    def tearDown(self):
        if False:
            return 10
        self.s.close()

    def test_readline(self):
        if False:
            i = 10
            return i + 15
        'Test readline method'
        self.s.write(serial.to_bytes([49, 10, 50, 10, 51, 10]))
        self.assertEqual(self.s.readline(), serial.to_bytes([49, 10]))
        self.assertEqual(self.s.readline(), serial.to_bytes([50, 10]))
        self.assertEqual(self.s.readline(), serial.to_bytes([51, 10]))
        self.assertEqual(self.s.readline(), serial.to_bytes([]))

    def test_readlines(self):
        if False:
            i = 10
            return i + 15
        'Test readlines method'
        self.s.write(serial.to_bytes([49, 10, 50, 10, 51, 10]))
        self.assertEqual(self.s.readlines(), [serial.to_bytes([49, 10]), serial.to_bytes([50, 10]), serial.to_bytes([51, 10])])

    def test_xreadlines(self):
        if False:
            return 10
        'Test xreadlines method (skipped for io based systems)'
        if hasattr(self.s, 'xreadlines'):
            self.s.write(serial.to_bytes([49, 10, 50, 10, 51, 10]))
            self.assertEqual(list(self.s.xreadlines()), [serial.to_bytes([49, 10]), serial.to_bytes([50, 10]), serial.to_bytes([51, 10])])

    def test_for_in(self):
        if False:
            print('Hello World!')
        'Test for line in s'
        self.s.write(serial.to_bytes([49, 10, 50, 10, 51, 10]))
        lines = []
        for line in self.s:
            lines.append(line)
        self.assertEqual(lines, [serial.to_bytes([49, 10]), serial.to_bytes([50, 10]), serial.to_bytes([51, 10])])

    def test_alternate_eol(self):
        if False:
            i = 10
            return i + 15
        'Test readline with alternative eol settings (skipped for io based systems)'
        if hasattr(self.s, 'xreadlines'):
            self.s.write(serial.to_bytes('no\rno\nyes\r\n'))
            self.assertEqual(self.s.readline(eol=serial.to_bytes('\r\n')), serial.to_bytes('no\rno\nyes\r\n'))
if __name__ == '__main__':
    import sys
    sys.stdout.write(__doc__)
    if len(sys.argv) > 1:
        PORT = sys.argv[1]
    sys.stdout.write('Testing port: {!r}\n'.format(PORT))
    sys.argv[1:] = ['-v']
    unittest.main()