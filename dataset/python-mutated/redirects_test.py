import ctypes
import os
import shutil
import sys
import tempfile
import unittest
from torch.distributed.elastic.multiprocessing.redirects import redirect, redirect_stderr, redirect_stdout
libc = ctypes.CDLL('libc.so.6')
c_stderr = ctypes.c_void_p.in_dll(libc, 'stderr')

class RedirectsTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.test_dir = tempfile.mkdtemp(prefix=f'{self.__class__.__name__}_')

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        shutil.rmtree(self.test_dir)

    def test_redirect_invalid_std(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            with redirect('stdfoo', os.path.join(self.test_dir, 'stdfoo.log')):
                pass

    def test_redirect_stdout(self):
        if False:
            i = 10
            return i + 15
        stdout_log = os.path.join(self.test_dir, 'stdout.log')
        print('foo first from python')
        libc.printf(b'foo first from c\n')
        os.system('echo foo first from cmd')
        with redirect_stdout(stdout_log):
            print('foo from python')
            libc.printf(b'foo from c\n')
            os.system('echo foo from cmd')
        print('foo again from python')
        libc.printf(b'foo again from c\n')
        os.system('echo foo again from cmd')
        with open(stdout_log) as f:
            lines = set(f.readlines())
            self.assertEqual({'foo from python\n', 'foo from c\n', 'foo from cmd\n'}, lines)

    def test_redirect_stderr(self):
        if False:
            for i in range(10):
                print('nop')
        stderr_log = os.path.join(self.test_dir, 'stderr.log')
        print('bar first from python')
        libc.fprintf(c_stderr, b'bar first from c\n')
        os.system('echo bar first from cmd 1>&2')
        with redirect_stderr(stderr_log):
            print('bar from python', file=sys.stderr)
            libc.fprintf(c_stderr, b'bar from c\n')
            os.system('echo bar from cmd 1>&2')
        print('bar again from python')
        libc.fprintf(c_stderr, b'bar again from c\n')
        os.system('echo bar again from cmd 1>&2')
        with open(stderr_log) as f:
            lines = set(f.readlines())
            self.assertEqual({'bar from python\n', 'bar from c\n', 'bar from cmd\n'}, lines)

    def test_redirect_both(self):
        if False:
            while True:
                i = 10
        stdout_log = os.path.join(self.test_dir, 'stdout.log')
        stderr_log = os.path.join(self.test_dir, 'stderr.log')
        print('first stdout from python')
        libc.printf(b'first stdout from c\n')
        print('first stderr from python', file=sys.stderr)
        libc.fprintf(c_stderr, b'first stderr from c\n')
        with redirect_stdout(stdout_log), redirect_stderr(stderr_log):
            print('redir stdout from python')
            print('redir stderr from python', file=sys.stderr)
            libc.printf(b'redir stdout from c\n')
            libc.fprintf(c_stderr, b'redir stderr from c\n')
        print('again stdout from python')
        libc.fprintf(c_stderr, b'again stderr from c\n')
        with open(stdout_log) as f:
            lines = set(f.readlines())
            self.assertEqual({'redir stdout from python\n', 'redir stdout from c\n'}, lines)
        with open(stderr_log) as f:
            lines = set(f.readlines())
            self.assertEqual({'redir stderr from python\n', 'redir stderr from c\n'}, lines)

    def _redirect_large_buffer(self, print_fn, num_lines=500000):
        if False:
            while True:
                i = 10
        stdout_log = os.path.join(self.test_dir, 'stdout.log')
        with redirect_stdout(stdout_log):
            for i in range(num_lines):
                print_fn(i)
        with open(stdout_log) as fp:
            actual = {int(line.split(':')[1]) for line in fp.readlines()}
            expected = set(range(num_lines))
            self.assertSetEqual(expected, actual)

    def test_redirect_large_buffer_py(self):
        if False:
            while True:
                i = 10

        def py_print(i):
            if False:
                while True:
                    i = 10
            print(f'py:{i}')
        self._redirect_large_buffer(py_print)

    def test_redirect_large_buffer_c(self):
        if False:
            return 10

        def c_print(i):
            if False:
                return 10
            libc.printf(bytes(f'c:{i}\n', 'utf-8'))
        self._redirect_large_buffer(c_print)