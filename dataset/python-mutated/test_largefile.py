"""Test largefile support on system where this makes sense.
"""
import os
import stat
import sys
import unittest
import socket
import shutil
import threading
from test.support import requires, bigmemtest
from test.support import SHORT_TIMEOUT
from test.support import socket_helper
from test.support.os_helper import TESTFN, unlink
import io
import _pyio as pyio
size = 2500000000
TESTFN2 = TESTFN + '2'

class LargeFileTest:

    def setUp(self):
        if False:
            while True:
                i = 10
        if os.path.exists(TESTFN):
            mode = 'r+b'
        else:
            mode = 'w+b'
        with self.open(TESTFN, mode) as f:
            current_size = os.fstat(f.fileno())[stat.ST_SIZE]
            if current_size == size + 1:
                return
            if current_size == 0:
                f.write(b'z')
            f.seek(0)
            f.seek(size)
            f.write(b'a')
            f.flush()
            self.assertEqual(os.fstat(f.fileno())[stat.ST_SIZE], size + 1)

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        with cls.open(TESTFN, 'wb'):
            pass
        if not os.stat(TESTFN)[stat.ST_SIZE] == 0:
            raise cls.failureException('File was not truncated by opening with mode "wb"')
        unlink(TESTFN2)

class TestFileMethods(LargeFileTest):
    """Test that each file function works as expected for large
    (i.e. > 2 GiB) files.
    """

    @bigmemtest(size=size, memuse=2, dry_run=False)
    def test_large_read(self, _size):
        if False:
            i = 10
            return i + 15
        with self.open(TESTFN, 'rb') as f:
            self.assertEqual(len(f.read()), size + 1)
            self.assertEqual(f.tell(), size + 1)

    def test_osstat(self):
        if False:
            print('Hello World!')
        self.assertEqual(os.stat(TESTFN)[stat.ST_SIZE], size + 1)

    def test_seek_read(self):
        if False:
            for i in range(10):
                print('nop')
        with self.open(TESTFN, 'rb') as f:
            self.assertEqual(f.tell(), 0)
            self.assertEqual(f.read(1), b'z')
            self.assertEqual(f.tell(), 1)
            f.seek(0)
            self.assertEqual(f.tell(), 0)
            f.seek(0, 0)
            self.assertEqual(f.tell(), 0)
            f.seek(42)
            self.assertEqual(f.tell(), 42)
            f.seek(42, 0)
            self.assertEqual(f.tell(), 42)
            f.seek(42, 1)
            self.assertEqual(f.tell(), 84)
            f.seek(0, 1)
            self.assertEqual(f.tell(), 84)
            f.seek(0, 2)
            self.assertEqual(f.tell(), size + 1 + 0)
            f.seek(-10, 2)
            self.assertEqual(f.tell(), size + 1 - 10)
            f.seek(-size - 1, 2)
            self.assertEqual(f.tell(), 0)
            f.seek(size)
            self.assertEqual(f.tell(), size)
            self.assertEqual(f.read(1), b'a')
            f.seek(-size - 1, 1)
            self.assertEqual(f.read(1), b'z')
            self.assertEqual(f.tell(), 1)

    def test_lseek(self):
        if False:
            print('Hello World!')
        with self.open(TESTFN, 'rb') as f:
            self.assertEqual(os.lseek(f.fileno(), 0, 0), 0)
            self.assertEqual(os.lseek(f.fileno(), 42, 0), 42)
            self.assertEqual(os.lseek(f.fileno(), 42, 1), 84)
            self.assertEqual(os.lseek(f.fileno(), 0, 1), 84)
            self.assertEqual(os.lseek(f.fileno(), 0, 2), size + 1 + 0)
            self.assertEqual(os.lseek(f.fileno(), -10, 2), size + 1 - 10)
            self.assertEqual(os.lseek(f.fileno(), -size - 1, 2), 0)
            self.assertEqual(os.lseek(f.fileno(), size, 0), size)
            self.assertEqual(f.read(1), b'a')

    def test_truncate(self):
        if False:
            print('Hello World!')
        with self.open(TESTFN, 'r+b') as f:
            if not hasattr(f, 'truncate'):
                raise unittest.SkipTest('open().truncate() not available on this system')
            f.seek(0, 2)
            self.assertEqual(f.tell(), size + 1)
            newsize = size - 10
            f.seek(newsize)
            f.truncate()
            self.assertEqual(f.tell(), newsize)
            f.seek(0, 2)
            self.assertEqual(f.tell(), newsize)
            newsize -= 1
            f.seek(42)
            f.truncate(newsize)
            self.assertEqual(f.tell(), 42)
            f.seek(0, 2)
            self.assertEqual(f.tell(), newsize)
            f.seek(0)
            f.truncate(1)
            self.assertEqual(f.tell(), 0)
            f.seek(0)
            self.assertEqual(len(f.read()), 1)

    def test_seekable(self):
        if False:
            for i in range(10):
                print('nop')
        for pos in (2 ** 31 - 1, 2 ** 31, 2 ** 31 + 1):
            with self.open(TESTFN, 'rb') as f:
                f.seek(pos)
                self.assertTrue(f.seekable())

def skip_no_disk_space(path, required):
    if False:
        while True:
            i = 10

    def decorator(fun):
        if False:
            return 10

        def wrapper(*args, **kwargs):
            if False:
                return 10
            if shutil.disk_usage(os.path.realpath(path)).free < required:
                hsize = int(required / 1024 / 1024)
                raise unittest.SkipTest(f'required {hsize} MiB of free disk space')
            return fun(*args, **kwargs)
        return wrapper
    return decorator

class TestCopyfile(LargeFileTest, unittest.TestCase):
    open = staticmethod(io.open)

    @skip_no_disk_space(TESTFN, size * 2.5)
    def test_it(self):
        if False:
            return 10
        size = os.path.getsize(TESTFN)
        shutil.copyfile(TESTFN, TESTFN2)
        self.assertEqual(os.path.getsize(TESTFN2), size)
        with open(TESTFN2, 'rb') as f:
            self.assertEqual(f.read(5), b'z\x00\x00\x00\x00')
            f.seek(size - 5)
            self.assertEqual(f.read(), b'\x00\x00\x00\x00a')

@unittest.skipIf(not hasattr(os, 'sendfile'), 'sendfile not supported')
class TestSocketSendfile(LargeFileTest, unittest.TestCase):
    open = staticmethod(io.open)
    timeout = SHORT_TIMEOUT

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.thread = None

    def tearDown(self):
        if False:
            return 10
        super().tearDown()
        if self.thread is not None:
            self.thread.join(self.timeout)
            self.thread = None

    def tcp_server(self, sock):
        if False:
            return 10

        def run(sock):
            if False:
                i = 10
                return i + 15
            with sock:
                (conn, _) = sock.accept()
                conn.settimeout(self.timeout)
                with conn, open(TESTFN2, 'wb') as f:
                    event.wait(self.timeout)
                    while True:
                        chunk = conn.recv(65536)
                        if not chunk:
                            return
                        f.write(chunk)
        event = threading.Event()
        sock.settimeout(self.timeout)
        self.thread = threading.Thread(target=run, args=(sock,))
        self.thread.start()
        event.set()

    @skip_no_disk_space(TESTFN, size * 2.5)
    def test_it(self):
        if False:
            while True:
                i = 10
        port = socket_helper.find_unused_port()
        with socket.create_server(('', port)) as sock:
            self.tcp_server(sock)
            with socket.create_connection(('127.0.0.1', port)) as client:
                with open(TESTFN, 'rb') as f:
                    client.sendfile(f)
        self.tearDown()
        size = os.path.getsize(TESTFN)
        self.assertEqual(os.path.getsize(TESTFN2), size)
        with open(TESTFN2, 'rb') as f:
            self.assertEqual(f.read(5), b'z\x00\x00\x00\x00')
            f.seek(size - 5)
            self.assertEqual(f.read(), b'\x00\x00\x00\x00a')

def setUpModule():
    if False:
        return 10
    try:
        import signal
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    except (ImportError, AttributeError):
        pass
    if sys.platform[:3] == 'win' or sys.platform == 'darwin':
        requires('largefile', 'test requires %s bytes and a long time to run' % str(size))
    else:
        f = open(TESTFN, 'wb', buffering=0)
        try:
            f.seek(2147483649)
            f.write(b'x')
            f.flush()
        except (OSError, OverflowError):
            raise unittest.SkipTest('filesystem does not have largefile support')
        finally:
            f.close()
            unlink(TESTFN)

class CLargeFileTest(TestFileMethods, unittest.TestCase):
    open = staticmethod(io.open)

class PyLargeFileTest(TestFileMethods, unittest.TestCase):
    open = staticmethod(pyio.open)

def tearDownModule():
    if False:
        i = 10
        return i + 15
    unlink(TESTFN)
    unlink(TESTFN2)
if __name__ == '__main__':
    unittest.main()