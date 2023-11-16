import unittest
from test import support
from test.support import os_helper
import io
import _pyio as pyio
lengths = list(range(1, 257)) + [512, 1000, 1024, 2048, 4096, 8192, 10000, 16384, 32768, 65536, 1000000]

class BufferSizeTest:

    def try_one(self, s):
        if False:
            for i in range(10):
                print('nop')
        os_helper.unlink(os_helper.TESTFN)
        f = self.open(os_helper.TESTFN, 'wb')
        try:
            f.write(s)
            f.write(b'\n')
            f.write(s)
            f.close()
            f = open(os_helper.TESTFN, 'rb')
            line = f.readline()
            self.assertEqual(line, s + b'\n')
            line = f.readline()
            self.assertEqual(line, s)
            line = f.readline()
            self.assertFalse(line)
            f.close()
        finally:
            os_helper.unlink(os_helper.TESTFN)

    def drive_one(self, pattern):
        if False:
            return 10
        for length in lengths:
            (q, r) = divmod(length, len(pattern))
            teststring = pattern * q + pattern[:r]
            self.assertEqual(len(teststring), length)
            self.try_one(teststring)
            self.try_one(teststring + b'x')
            self.try_one(teststring[:-1])

    def test_primepat(self):
        if False:
            while True:
                i = 10
        self.drive_one(b'1234567890\x00\x01\x02\x03\x04\x05\x06')

    def test_nullpat(self):
        if False:
            while True:
                i = 10
        self.drive_one(b'\x00' * 1000)

class CBufferSizeTest(BufferSizeTest, unittest.TestCase):
    open = io.open

class PyBufferSizeTest(BufferSizeTest, unittest.TestCase):
    open = staticmethod(pyio.open)
if __name__ == '__main__':
    unittest.main()