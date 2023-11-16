"""
Test PTY related functionality.
"""
import os
import sys
try:
    import pty
except ImportError:
    pty = None
import unittest
import serial
DATA = b'Hello\n'

@unittest.skipIf(pty is None, 'pty module not supported on platform')
class Test_Pty_Serial_Open(unittest.TestCase):
    """Test PTY serial open"""

    def setUp(self):
        if False:
            while True:
                i = 10
        (self.master, self.slave) = pty.openpty()

    def test_pty_serial_open_slave(self):
        if False:
            for i in range(10):
                print('nop')
        with serial.Serial(os.ttyname(self.slave), timeout=1) as slave:
            pass

    def test_pty_serial_write(self):
        if False:
            print('Hello World!')
        with serial.Serial(os.ttyname(self.slave), timeout=1) as slave:
            with os.fdopen(self.master, 'wb') as fd:
                fd.write(DATA)
                fd.flush()
                out = slave.read(len(DATA))
                self.assertEqual(DATA, out)

    def test_pty_serial_read(self):
        if False:
            while True:
                i = 10
        with serial.Serial(os.ttyname(self.slave), timeout=1) as slave:
            with os.fdopen(self.master, 'rb') as fd:
                slave.write(DATA)
                slave.flush()
                out = fd.read(len(DATA))
                self.assertEqual(DATA, out)
if __name__ == '__main__':
    sys.stdout.write(__doc__)
    unittest.main()