"""Some tests for the serial module.
Part of pyserial (http://pyserial.sf.net)  (C)2002-2003 cliechti@gmx.net

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
BAUDRATE = 115200
if sys.version_info >= (3, 0):
    bytes_0to255 = bytes(range(256))
else:
    bytes_0to255 = ''.join([chr(x) for x in range(256)])

class TestHighLoad(unittest.TestCase):
    """Test sending and receiving large amount of data"""
    N = 16

    def setUp(self):
        if False:
            while True:
                i = 10
        self.s = serial.serial_for_url(PORT, BAUDRATE, timeout=10)

    def tearDown(self):
        if False:
            return 10
        self.s.close()

    def test0_WriteReadLoopback(self):
        if False:
            print('Hello World!')
        'Send big strings, write/read order.'
        for i in range(self.N):
            q = bytes_0to255
            self.s.write(q)
            self.assertEqual(self.s.read(len(q)), q)
        self.assertEqual(self.s.inWaiting(), 0)

    def test1_WriteWriteReadLoopback(self):
        if False:
            return 10
        'Send big strings, multiple write one read.'
        q = bytes_0to255
        for i in range(self.N):
            self.s.write(q)
        read = self.s.read(len(q) * self.N)
        self.assertEqual(read, q * self.N, 'expected what was written before. got {} bytes, expected {}'.format(len(read), self.N * len(q)))
        self.assertEqual(self.s.inWaiting(), 0)
if __name__ == '__main__':
    import sys
    sys.stdout.write(__doc__)
    if len(sys.argv) > 1:
        PORT = sys.argv[1]
    sys.stdout.write('Testing port: {!r}\n'.format(PORT))
    sys.argv[1:] = ['-v']
    unittest.main()