"""Tests for testing utils (psutil.tests namespace)."""
import collections
import contextlib
import errno
import os
import socket
import stat
import subprocess
import unittest
import psutil
import psutil.tests
from psutil import FREEBSD
from psutil import NETBSD
from psutil import POSIX
from psutil._common import open_binary
from psutil._common import open_text
from psutil._common import supports_ipv6
from psutil.tests import CI_TESTING
from psutil.tests import COVERAGE
from psutil.tests import HAS_CONNECTIONS_UNIX
from psutil.tests import PYTHON_EXE
from psutil.tests import PYTHON_EXE_ENV
from psutil.tests import PsutilTestCase
from psutil.tests import TestMemoryLeak
from psutil.tests import bind_socket
from psutil.tests import bind_unix_socket
from psutil.tests import call_until
from psutil.tests import chdir
from psutil.tests import create_sockets
from psutil.tests import get_free_port
from psutil.tests import is_namedtuple
from psutil.tests import mock
from psutil.tests import process_namespace
from psutil.tests import reap_children
from psutil.tests import retry
from psutil.tests import retry_on_failure
from psutil.tests import safe_mkdir
from psutil.tests import safe_rmpath
from psutil.tests import serialrun
from psutil.tests import system_namespace
from psutil.tests import tcp_socketpair
from psutil.tests import terminate
from psutil.tests import unix_socketpair
from psutil.tests import wait_for_file
from psutil.tests import wait_for_pid

class TestRetryDecorator(PsutilTestCase):

    @mock.patch('time.sleep')
    def test_retry_success(self, sleep):
        if False:
            return 10

        @retry(retries=5, interval=1, logfun=None)
        def foo():
            if False:
                print('Hello World!')
            while queue:
                queue.pop()
                1 / 0
            return 1
        queue = list(range(3))
        self.assertEqual(foo(), 1)
        self.assertEqual(sleep.call_count, 3)

    @mock.patch('time.sleep')
    def test_retry_failure(self, sleep):
        if False:
            for i in range(10):
                print('nop')

        @retry(retries=5, interval=1, logfun=None)
        def foo():
            if False:
                return 10
            while queue:
                queue.pop()
                1 / 0
            return 1
        queue = list(range(6))
        self.assertRaises(ZeroDivisionError, foo)
        self.assertEqual(sleep.call_count, 5)

    @mock.patch('time.sleep')
    def test_exception_arg(self, sleep):
        if False:
            for i in range(10):
                print('nop')

        @retry(exception=ValueError, interval=1)
        def foo():
            if False:
                i = 10
                return i + 15
            raise TypeError
        self.assertRaises(TypeError, foo)
        self.assertEqual(sleep.call_count, 0)

    @mock.patch('time.sleep')
    def test_no_interval_arg(self, sleep):
        if False:
            return 10

        @retry(retries=5, interval=None, logfun=None)
        def foo():
            if False:
                for i in range(10):
                    print('nop')
            1 / 0
        self.assertRaises(ZeroDivisionError, foo)
        self.assertEqual(sleep.call_count, 0)

    @mock.patch('time.sleep')
    def test_retries_arg(self, sleep):
        if False:
            i = 10
            return i + 15

        @retry(retries=5, interval=1, logfun=None)
        def foo():
            if False:
                return 10
            1 / 0
        self.assertRaises(ZeroDivisionError, foo)
        self.assertEqual(sleep.call_count, 5)

    @mock.patch('time.sleep')
    def test_retries_and_timeout_args(self, sleep):
        if False:
            while True:
                i = 10
        self.assertRaises(ValueError, retry, retries=5, timeout=1)

class TestSyncTestUtils(PsutilTestCase):

    def test_wait_for_pid(self):
        if False:
            while True:
                i = 10
        wait_for_pid(os.getpid())
        nopid = max(psutil.pids()) + 99999
        with mock.patch('psutil.tests.retry.__iter__', return_value=iter([0])):
            self.assertRaises(psutil.NoSuchProcess, wait_for_pid, nopid)

    def test_wait_for_file(self):
        if False:
            while True:
                i = 10
        testfn = self.get_testfn()
        with open(testfn, 'w') as f:
            f.write('foo')
        wait_for_file(testfn)
        assert not os.path.exists(testfn)

    def test_wait_for_file_empty(self):
        if False:
            while True:
                i = 10
        testfn = self.get_testfn()
        with open(testfn, 'w'):
            pass
        wait_for_file(testfn, empty=True)
        assert not os.path.exists(testfn)

    def test_wait_for_file_no_file(self):
        if False:
            for i in range(10):
                print('nop')
        testfn = self.get_testfn()
        with mock.patch('psutil.tests.retry.__iter__', return_value=iter([0])):
            self.assertRaises(IOError, wait_for_file, testfn)

    def test_wait_for_file_no_delete(self):
        if False:
            for i in range(10):
                print('nop')
        testfn = self.get_testfn()
        with open(testfn, 'w') as f:
            f.write('foo')
        wait_for_file(testfn, delete=False)
        assert os.path.exists(testfn)

    def test_call_until(self):
        if False:
            while True:
                i = 10
        ret = call_until(lambda : 1, 'ret == 1')
        self.assertEqual(ret, 1)

class TestFSTestUtils(PsutilTestCase):

    def test_open_text(self):
        if False:
            while True:
                i = 10
        with open_text(__file__) as f:
            self.assertEqual(f.mode, 'r')

    def test_open_binary(self):
        if False:
            for i in range(10):
                print('nop')
        with open_binary(__file__) as f:
            self.assertEqual(f.mode, 'rb')

    def test_safe_mkdir(self):
        if False:
            while True:
                i = 10
        testfn = self.get_testfn()
        safe_mkdir(testfn)
        assert os.path.isdir(testfn)
        safe_mkdir(testfn)
        assert os.path.isdir(testfn)

    def test_safe_rmpath(self):
        if False:
            i = 10
            return i + 15
        testfn = self.get_testfn()
        open(testfn, 'w').close()
        safe_rmpath(testfn)
        assert not os.path.exists(testfn)
        safe_rmpath(testfn)
        os.mkdir(testfn)
        safe_rmpath(testfn)
        assert not os.path.exists(testfn)
        with mock.patch('psutil.tests.os.stat', side_effect=OSError(errno.EINVAL, '')) as m:
            with self.assertRaises(OSError):
                safe_rmpath(testfn)
            assert m.called

    def test_chdir(self):
        if False:
            i = 10
            return i + 15
        testfn = self.get_testfn()
        base = os.getcwd()
        os.mkdir(testfn)
        with chdir(testfn):
            self.assertEqual(os.getcwd(), os.path.join(base, testfn))
        self.assertEqual(os.getcwd(), base)

class TestProcessUtils(PsutilTestCase):

    def test_reap_children(self):
        if False:
            while True:
                i = 10
        subp = self.spawn_testproc()
        p = psutil.Process(subp.pid)
        assert p.is_running()
        reap_children()
        assert not p.is_running()
        assert not psutil.tests._pids_started
        assert not psutil.tests._subprocesses_started

    def test_spawn_children_pair(self):
        if False:
            i = 10
            return i + 15
        (child, grandchild) = self.spawn_children_pair()
        self.assertNotEqual(child.pid, grandchild.pid)
        assert child.is_running()
        assert grandchild.is_running()
        children = psutil.Process().children()
        self.assertEqual(children, [child])
        children = psutil.Process().children(recursive=True)
        self.assertEqual(len(children), 2)
        self.assertIn(child, children)
        self.assertIn(grandchild, children)
        self.assertEqual(child.ppid(), os.getpid())
        self.assertEqual(grandchild.ppid(), child.pid)
        terminate(child)
        assert not child.is_running()
        assert grandchild.is_running()
        terminate(grandchild)
        assert not grandchild.is_running()

    @unittest.skipIf(not POSIX, 'POSIX only')
    def test_spawn_zombie(self):
        if False:
            print('Hello World!')
        (parent, zombie) = self.spawn_zombie()
        self.assertEqual(zombie.status(), psutil.STATUS_ZOMBIE)

    def test_terminate(self):
        if False:
            for i in range(10):
                print('nop')
        p = self.spawn_testproc()
        terminate(p)
        self.assertPidGone(p.pid)
        terminate(p)
        p = psutil.Process(self.spawn_testproc().pid)
        terminate(p)
        self.assertPidGone(p.pid)
        terminate(p)
        cmd = [PYTHON_EXE, '-c', 'import time; time.sleep(60);']
        p = psutil.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=PYTHON_EXE_ENV)
        terminate(p)
        self.assertPidGone(p.pid)
        terminate(p)
        pid = self.spawn_testproc().pid
        terminate(pid)
        self.assertPidGone(p.pid)
        terminate(pid)
        if POSIX:
            (parent, zombie) = self.spawn_zombie()
            terminate(parent)
            terminate(zombie)
            self.assertPidGone(parent.pid)
            self.assertPidGone(zombie.pid)

class TestNetUtils(PsutilTestCase):

    def bind_socket(self):
        if False:
            return 10
        port = get_free_port()
        with contextlib.closing(bind_socket(addr=('', port))) as s:
            self.assertEqual(s.getsockname()[1], port)

    @unittest.skipIf(not POSIX, 'POSIX only')
    def test_bind_unix_socket(self):
        if False:
            print('Hello World!')
        name = self.get_testfn()
        sock = bind_unix_socket(name)
        with contextlib.closing(sock):
            self.assertEqual(sock.family, socket.AF_UNIX)
            self.assertEqual(sock.type, socket.SOCK_STREAM)
            self.assertEqual(sock.getsockname(), name)
            assert os.path.exists(name)
            assert stat.S_ISSOCK(os.stat(name).st_mode)
        name = self.get_testfn()
        sock = bind_unix_socket(name, type=socket.SOCK_DGRAM)
        with contextlib.closing(sock):
            self.assertEqual(sock.type, socket.SOCK_DGRAM)

    def tcp_tcp_socketpair(self):
        if False:
            for i in range(10):
                print('nop')
        addr = ('127.0.0.1', get_free_port())
        (server, client) = tcp_socketpair(socket.AF_INET, addr=addr)
        with contextlib.closing(server):
            with contextlib.closing(client):
                self.assertEqual(server.getsockname(), addr)
                self.assertEqual(client.getpeername(), addr)
                self.assertNotEqual(client.getsockname(), addr)

    @unittest.skipIf(not POSIX, 'POSIX only')
    @unittest.skipIf(NETBSD or FREEBSD, '/var/run/log UNIX socket opened by default')
    def test_unix_socketpair(self):
        if False:
            while True:
                i = 10
        p = psutil.Process()
        num_fds = p.num_fds()
        assert not p.connections(kind='unix')
        name = self.get_testfn()
        (server, client) = unix_socketpair(name)
        try:
            assert os.path.exists(name)
            assert stat.S_ISSOCK(os.stat(name).st_mode)
            self.assertEqual(p.num_fds() - num_fds, 2)
            self.assertEqual(len(p.connections(kind='unix')), 2)
            self.assertEqual(server.getsockname(), name)
            self.assertEqual(client.getpeername(), name)
        finally:
            client.close()
            server.close()

    def test_create_sockets(self):
        if False:
            for i in range(10):
                print('nop')
        with create_sockets() as socks:
            fams = collections.defaultdict(int)
            types = collections.defaultdict(int)
            for s in socks:
                fams[s.family] += 1
                types[s.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE)] += 1
            self.assertGreaterEqual(fams[socket.AF_INET], 2)
            if supports_ipv6():
                self.assertGreaterEqual(fams[socket.AF_INET6], 2)
            if POSIX and HAS_CONNECTIONS_UNIX:
                self.assertGreaterEqual(fams[socket.AF_UNIX], 2)
            self.assertGreaterEqual(types[socket.SOCK_STREAM], 2)
            self.assertGreaterEqual(types[socket.SOCK_DGRAM], 2)

@serialrun
class TestMemLeakClass(TestMemoryLeak):

    @retry_on_failure()
    def test_times(self):
        if False:
            for i in range(10):
                print('nop')

        def fun():
            if False:
                for i in range(10):
                    print('nop')
            cnt['cnt'] += 1
        cnt = {'cnt': 0}
        self.execute(fun, times=10, warmup_times=15)
        self.assertEqual(cnt['cnt'], 26)

    def test_param_err(self):
        if False:
            return 10
        self.assertRaises(ValueError, self.execute, lambda : 0, times=0)
        self.assertRaises(ValueError, self.execute, lambda : 0, times=-1)
        self.assertRaises(ValueError, self.execute, lambda : 0, warmup_times=-1)
        self.assertRaises(ValueError, self.execute, lambda : 0, tolerance=-1)
        self.assertRaises(ValueError, self.execute, lambda : 0, retries=-1)

    @retry_on_failure()
    @unittest.skipIf(CI_TESTING, 'skipped on CI')
    @unittest.skipIf(COVERAGE, 'skipped during test coverage')
    def test_leak_mem(self):
        if False:
            while True:
                i = 10
        ls = []

        def fun(ls=ls):
            if False:
                for i in range(10):
                    print('nop')
            ls.append('x' * 24 * 1024)
        try:
            self.assertRaisesRegex(AssertionError, 'extra-mem', self.execute, fun, times=50)
        finally:
            del ls

    def test_unclosed_files(self):
        if False:
            i = 10
            return i + 15

        def fun():
            if False:
                while True:
                    i = 10
            f = open(__file__)
            self.addCleanup(f.close)
            box.append(f)
        box = []
        kind = 'fd' if POSIX else 'handle'
        self.assertRaisesRegex(AssertionError, 'unclosed ' + kind, self.execute, fun)

    def test_tolerance(self):
        if False:
            i = 10
            return i + 15

        def fun():
            if False:
                for i in range(10):
                    print('nop')
            ls.append('x' * 24 * 1024)
        ls = []
        times = 100
        self.execute(fun, times=times, warmup_times=0, tolerance=200 * 1024 * 1024)
        self.assertEqual(len(ls), times + 1)

    def test_execute_w_exc(self):
        if False:
            print('Hello World!')

        def fun_1():
            if False:
                return 10
            1 / 0
        self.execute_w_exc(ZeroDivisionError, fun_1)
        with self.assertRaises(ZeroDivisionError):
            self.execute_w_exc(OSError, fun_1)

        def fun_2():
            if False:
                for i in range(10):
                    print('nop')
            pass
        with self.assertRaises(AssertionError):
            self.execute_w_exc(ZeroDivisionError, fun_2)

class TestTestingUtils(PsutilTestCase):

    def test_process_namespace(self):
        if False:
            while True:
                i = 10
        p = psutil.Process()
        ns = process_namespace(p)
        ns.test()
        fun = [x for x in ns.iter(ns.getters) if x[1] == 'ppid'][0][0]
        self.assertEqual(fun(), p.ppid())

    def test_system_namespace(self):
        if False:
            for i in range(10):
                print('nop')
        ns = system_namespace()
        fun = [x for x in ns.iter(ns.getters) if x[1] == 'net_if_addrs'][0][0]
        self.assertEqual(fun(), psutil.net_if_addrs())

class TestOtherUtils(PsutilTestCase):

    def test_is_namedtuple(self):
        if False:
            i = 10
            return i + 15
        assert is_namedtuple(collections.namedtuple('foo', 'a b c')(1, 2, 3))
        assert not is_namedtuple(tuple())
if __name__ == '__main__':
    from psutil.tests.runner import run_from_name
    run_from_name(__file__)