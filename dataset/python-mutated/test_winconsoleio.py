"""Tests for WindowsConsoleIO
"""
import io
import os
import sys
import tempfile
import unittest
from test.support import os_helper
if sys.platform != 'win32':
    raise unittest.SkipTest('test only relevant on win32')
from _testconsole import write_input
ConIO = io._WindowsConsoleIO

class WindowsConsoleIOTests(unittest.TestCase):

    def test_abc(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(issubclass(ConIO, io.RawIOBase))
        self.assertFalse(issubclass(ConIO, io.BufferedIOBase))
        self.assertFalse(issubclass(ConIO, io.TextIOBase))

    def test_open_fd(self):
        if False:
            i = 10
            return i + 15
        self.assertRaisesRegex(ValueError, 'negative file descriptor', ConIO, -1)
        with tempfile.TemporaryFile() as tmpfile:
            fd = tmpfile.fileno()
            self.assertRaisesRegex(ValueError, 'Cannot open (console|non-console file)', ConIO, fd)
        try:
            f = ConIO(0)
        except ValueError:
            pass
        else:
            self.assertTrue(f.readable())
            self.assertFalse(f.writable())
            self.assertEqual(0, f.fileno())
            f.close()
            f.close()
        try:
            f = ConIO(1, 'w')
        except ValueError:
            pass
        else:
            self.assertFalse(f.readable())
            self.assertTrue(f.writable())
            self.assertEqual(1, f.fileno())
            f.close()
            f.close()
        try:
            f = ConIO(2, 'w')
        except ValueError:
            pass
        else:
            self.assertFalse(f.readable())
            self.assertTrue(f.writable())
            self.assertEqual(2, f.fileno())
            f.close()
            f.close()

    def test_open_name(self):
        if False:
            print('Hello World!')
        self.assertRaises(ValueError, ConIO, sys.executable)
        f = ConIO('CON')
        self.assertTrue(f.readable())
        self.assertFalse(f.writable())
        self.assertIsNotNone(f.fileno())
        f.close()
        f.close()
        f = ConIO('CONIN$')
        self.assertTrue(f.readable())
        self.assertFalse(f.writable())
        self.assertIsNotNone(f.fileno())
        f.close()
        f.close()
        f = ConIO('CONOUT$', 'w')
        self.assertFalse(f.readable())
        self.assertTrue(f.writable())
        self.assertIsNotNone(f.fileno())
        f.close()
        f.close()
        if sys.getwindowsversion()[:3] < (10, 0, 22000):
            f = open('C:/con', 'rb', buffering=0)
            self.assertIsInstance(f, ConIO)
            f.close()

    @unittest.skipIf(sys.getwindowsversion()[:2] <= (6, 1), 'test does not work on Windows 7 and earlier')
    def test_conin_conout_names(self):
        if False:
            return 10
        f = open('\\\\.\\conin$', 'rb', buffering=0)
        self.assertIsInstance(f, ConIO)
        f.close()
        f = open('//?/conout$', 'wb', buffering=0)
        self.assertIsInstance(f, ConIO)
        f.close()

    def test_conout_path(self):
        if False:
            return 10
        temp_path = tempfile.mkdtemp()
        self.addCleanup(os_helper.rmtree, temp_path)
        conout_path = os.path.join(temp_path, 'CONOUT$')
        with open(conout_path, 'wb', buffering=0) as f:
            if (6, 1) < sys.getwindowsversion()[:3] < (10, 0, 22000):
                self.assertIsInstance(f, ConIO)
            else:
                self.assertNotIsInstance(f, ConIO)

    def test_write_empty_data(self):
        if False:
            print('Hello World!')
        with ConIO('CONOUT$', 'w') as f:
            self.assertEqual(f.write(b''), 0)

    def assertStdinRoundTrip(self, text):
        if False:
            while True:
                i = 10
        stdin = open('CONIN$', 'r')
        old_stdin = sys.stdin
        try:
            sys.stdin = stdin
            write_input(stdin.buffer.raw, (text + '\r\n').encode('utf-16-le', 'surrogatepass'))
            actual = input()
        finally:
            sys.stdin = old_stdin
        self.assertEqual(actual, text)

    def test_input(self):
        if False:
            while True:
                i = 10
        self.assertStdinRoundTrip('abc123')
        self.assertStdinRoundTrip('ϼўТλФЙ')
        self.assertStdinRoundTrip('A͏B ﬖ̳AA̝')

    @unittest.skipIf(True, 'Handling Non-BMP characters is broken')
    def test_input_nonbmp(self):
        if False:
            while True:
                i = 10
        self.assertStdinRoundTrip('\U00100000\U0010ffff\U0010fffd')

    def test_partial_reads(self):
        if False:
            return 10
        source = 'ϼўТλФЙ\r\n'.encode('utf-16-le')
        expected = 'ϼўТλФЙ\r\n'.encode('utf-8')
        for read_count in range(1, 16):
            with open('CONIN$', 'rb', buffering=0) as stdin:
                write_input(stdin, source)
                actual = b''
                while not actual.endswith(b'\n'):
                    b = stdin.read(read_count)
                    actual += b
                self.assertEqual(actual, expected, 'stdin.read({})'.format(read_count))

    @unittest.skipIf(True, 'Handling Non-BMP characters is broken')
    def test_partial_surrogate_reads(self):
        if False:
            i = 10
            return i + 15
        source = '\U00101fff\U00101001\r\n'.encode('utf-16-le')
        expected = '\U00101fff\U00101001\r\n'.encode('utf-8')
        for read_count in range(1, 16):
            with open('CONIN$', 'rb', buffering=0) as stdin:
                write_input(stdin, source)
                actual = b''
                while not actual.endswith(b'\n'):
                    b = stdin.read(read_count)
                    actual += b
                self.assertEqual(actual, expected, 'stdin.read({})'.format(read_count))

    def test_ctrl_z(self):
        if False:
            while True:
                i = 10
        with open('CONIN$', 'rb', buffering=0) as stdin:
            source = 'Ä\x1a\r\n'.encode('utf-16-le')
            expected = 'Ä'.encode('utf-8')
            write_input(stdin, source)
            (a, b) = (stdin.read(1), stdin.readall())
            self.assertEqual(expected[0:1], a)
            self.assertEqual(expected[1:], b)
if __name__ == '__main__':
    unittest.main()