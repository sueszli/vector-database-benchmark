"""
This test suite exercises some system calls subject to interruption with EINTR,
to check that it is actually handled transparently.
It is intended to be run by the main test suite within a child process, to
ensure there is no background thread running (so that signals are delivered to
the correct thread).
Signals are generated in-process using setitimer(ITIMER_REAL), which allows
sub-second periodicity (contrarily to signal()).
"""
import contextlib
import faulthandler
import fcntl
import os
import platform
import select
import signal
import socket
import subprocess
import sys
import time
import unittest
from test import support
from test.support import os_helper
from test.support import socket_helper

@contextlib.contextmanager
def kill_on_error(proc):
    if False:
        return 10
    'Context manager killing the subprocess if a Python exception is raised.'
    with proc:
        try:
            yield proc
        except:
            proc.kill()
            raise

@unittest.skipUnless(hasattr(signal, 'setitimer'), 'requires setitimer()')
class EINTRBaseTest(unittest.TestCase):
    """ Base class for EINTR tests. """
    signal_delay = 0.1
    signal_period = 0.1
    sleep_time = 0.2

    def sighandler(self, signum, frame):
        if False:
            i = 10
            return i + 15
        self.signals += 1

    def setUp(self):
        if False:
            print('Hello World!')
        self.signals = 0
        self.orig_handler = signal.signal(signal.SIGALRM, self.sighandler)
        signal.setitimer(signal.ITIMER_REAL, self.signal_delay, self.signal_period)
        faulthandler.dump_traceback_later(10 * 60, exit=True, file=sys.__stderr__)

    @staticmethod
    def stop_alarm():
        if False:
            print('Hello World!')
        signal.setitimer(signal.ITIMER_REAL, 0, 0)

    def tearDown(self):
        if False:
            print('Hello World!')
        self.stop_alarm()
        signal.signal(signal.SIGALRM, self.orig_handler)
        faulthandler.cancel_dump_traceback_later()

    def subprocess(self, *args, **kw):
        if False:
            for i in range(10):
                print('nop')
        cmd_args = (sys.executable, '-c') + args
        return subprocess.Popen(cmd_args, **kw)

@unittest.skipUnless(hasattr(signal, 'setitimer'), 'requires setitimer()')
class OSEINTRTest(EINTRBaseTest):
    """ EINTR tests for the os module. """

    def new_sleep_process(self):
        if False:
            print('Hello World!')
        code = 'import time; time.sleep(%r)' % self.sleep_time
        return self.subprocess(code)

    def _test_wait_multiple(self, wait_func):
        if False:
            return 10
        num = 3
        processes = [self.new_sleep_process() for _ in range(num)]
        for _ in range(num):
            wait_func()
        for proc in processes:
            proc.wait()

    def test_wait(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_wait_multiple(os.wait)

    @unittest.skipUnless(hasattr(os, 'wait3'), 'requires wait3()')
    def test_wait3(self):
        if False:
            print('Hello World!')
        self._test_wait_multiple(lambda : os.wait3(0))

    def _test_wait_single(self, wait_func):
        if False:
            i = 10
            return i + 15
        proc = self.new_sleep_process()
        wait_func(proc.pid)
        proc.wait()

    def test_waitpid(self):
        if False:
            i = 10
            return i + 15
        self._test_wait_single(lambda pid: os.waitpid(pid, 0))

    @unittest.skipUnless(hasattr(os, 'wait4'), 'requires wait4()')
    def test_wait4(self):
        if False:
            return 10
        self._test_wait_single(lambda pid: os.wait4(pid, 0))

    def test_read(self):
        if False:
            for i in range(10):
                print('nop')
        (rd, wr) = os.pipe()
        self.addCleanup(os.close, rd)
        datas = [b'hello', b'world', b'spam']
        code = '\n'.join(('import os, sys, time', '', 'wr = int(sys.argv[1])', 'datas = %r' % datas, 'sleep_time = %r' % self.sleep_time, '', 'for data in datas:', '    # let the parent block on read()', '    time.sleep(sleep_time)', '    os.write(wr, data)'))
        proc = self.subprocess(code, str(wr), pass_fds=[wr])
        with kill_on_error(proc):
            os.close(wr)
            for data in datas:
                self.assertEqual(data, os.read(rd, len(data)))
            self.assertEqual(proc.wait(), 0)

    def test_write(self):
        if False:
            print('Hello World!')
        (rd, wr) = os.pipe()
        self.addCleanup(os.close, wr)
        data = b'x' * support.PIPE_MAX_SIZE
        code = '\n'.join(('import io, os, sys, time', '', 'rd = int(sys.argv[1])', 'sleep_time = %r' % self.sleep_time, 'data = b"x" * %s' % support.PIPE_MAX_SIZE, 'data_len = len(data)', '', '# let the parent block on write()', 'time.sleep(sleep_time)', '', 'read_data = io.BytesIO()', 'while len(read_data.getvalue()) < data_len:', '    chunk = os.read(rd, 2 * data_len)', '    read_data.write(chunk)', '', 'value = read_data.getvalue()', 'if value != data:', '    raise Exception("read error: %s vs %s bytes"', '                    % (len(value), data_len))'))
        proc = self.subprocess(code, str(rd), pass_fds=[rd])
        with kill_on_error(proc):
            os.close(rd)
            written = 0
            while written < len(data):
                written += os.write(wr, memoryview(data)[written:])
            self.assertEqual(proc.wait(), 0)

@unittest.skipUnless(hasattr(signal, 'setitimer'), 'requires setitimer()')
class SocketEINTRTest(EINTRBaseTest):
    """ EINTR tests for the socket module. """

    @unittest.skipUnless(hasattr(socket, 'socketpair'), 'needs socketpair()')
    def _test_recv(self, recv_func):
        if False:
            i = 10
            return i + 15
        (rd, wr) = socket.socketpair()
        self.addCleanup(rd.close)
        datas = [b'x', b'y', b'z']
        code = '\n'.join(('import os, socket, sys, time', '', 'fd = int(sys.argv[1])', 'family = %s' % int(wr.family), 'sock_type = %s' % int(wr.type), 'datas = %r' % datas, 'sleep_time = %r' % self.sleep_time, '', 'wr = socket.fromfd(fd, family, sock_type)', 'os.close(fd)', '', 'with wr:', '    for data in datas:', '        # let the parent block on recv()', '        time.sleep(sleep_time)', '        wr.sendall(data)'))
        fd = wr.fileno()
        proc = self.subprocess(code, str(fd), pass_fds=[fd])
        with kill_on_error(proc):
            wr.close()
            for data in datas:
                self.assertEqual(data, recv_func(rd, len(data)))
            self.assertEqual(proc.wait(), 0)

    def test_recv(self):
        if False:
            i = 10
            return i + 15
        self._test_recv(socket.socket.recv)

    @unittest.skipUnless(hasattr(socket.socket, 'recvmsg'), 'needs recvmsg()')
    def test_recvmsg(self):
        if False:
            print('Hello World!')
        self._test_recv(lambda sock, data: sock.recvmsg(data)[0])

    def _test_send(self, send_func):
        if False:
            i = 10
            return i + 15
        (rd, wr) = socket.socketpair()
        self.addCleanup(wr.close)
        data = b'xyz' * (support.SOCK_MAX_SIZE // 3)
        code = '\n'.join(('import os, socket, sys, time', '', 'fd = int(sys.argv[1])', 'family = %s' % int(rd.family), 'sock_type = %s' % int(rd.type), 'sleep_time = %r' % self.sleep_time, 'data = b"xyz" * %s' % (support.SOCK_MAX_SIZE // 3), 'data_len = len(data)', '', 'rd = socket.fromfd(fd, family, sock_type)', 'os.close(fd)', '', 'with rd:', '    # let the parent block on send()', '    time.sleep(sleep_time)', '', '    received_data = bytearray(data_len)', '    n = 0', '    while n < data_len:', '        n += rd.recv_into(memoryview(received_data)[n:])', '', 'if received_data != data:', '    raise Exception("recv error: %s vs %s bytes"', '                    % (len(received_data), data_len))'))
        fd = rd.fileno()
        proc = self.subprocess(code, str(fd), pass_fds=[fd])
        with kill_on_error(proc):
            rd.close()
            written = 0
            while written < len(data):
                sent = send_func(wr, memoryview(data)[written:])
                written += len(data) if sent is None else sent
            self.assertEqual(proc.wait(), 0)

    def test_send(self):
        if False:
            while True:
                i = 10
        self._test_send(socket.socket.send)

    def test_sendall(self):
        if False:
            return 10
        self._test_send(socket.socket.sendall)

    @unittest.skipUnless(hasattr(socket.socket, 'sendmsg'), 'needs sendmsg()')
    def test_sendmsg(self):
        if False:
            print('Hello World!')
        self._test_send(lambda sock, data: sock.sendmsg([data]))

    def test_accept(self):
        if False:
            while True:
                i = 10
        sock = socket.create_server((socket_helper.HOST, 0))
        self.addCleanup(sock.close)
        port = sock.getsockname()[1]
        code = '\n'.join(('import socket, time', '', 'host = %r' % socket_helper.HOST, 'port = %s' % port, 'sleep_time = %r' % self.sleep_time, '', '# let parent block on accept()', 'time.sleep(sleep_time)', 'with socket.create_connection((host, port)):', '    time.sleep(sleep_time)'))
        proc = self.subprocess(code)
        with kill_on_error(proc):
            (client_sock, _) = sock.accept()
            client_sock.close()
            self.assertEqual(proc.wait(), 0)

    @support.requires_freebsd_version(10, 3)
    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'needs mkfifo()')
    def _test_open(self, do_open_close_reader, do_open_close_writer):
        if False:
            return 10
        filename = os_helper.TESTFN
        os_helper.unlink(filename)
        try:
            os.mkfifo(filename)
        except PermissionError as e:
            self.skipTest('os.mkfifo(): %s' % e)
        self.addCleanup(os_helper.unlink, filename)
        code = '\n'.join(('import os, time', '', 'path = %a' % filename, 'sleep_time = %r' % self.sleep_time, '', '# let the parent block', 'time.sleep(sleep_time)', '', do_open_close_reader))
        proc = self.subprocess(code)
        with kill_on_error(proc):
            do_open_close_writer(filename)
            self.assertEqual(proc.wait(), 0)

    def python_open(self, path):
        if False:
            for i in range(10):
                print('nop')
        fp = open(path, 'w')
        fp.close()

    @unittest.skipIf(sys.platform == 'darwin', 'hangs under macOS; see bpo-25234, bpo-35363')
    def test_open(self):
        if False:
            i = 10
            return i + 15
        self._test_open("fp = open(path, 'r')\nfp.close()", self.python_open)

    def os_open(self, path):
        if False:
            i = 10
            return i + 15
        fd = os.open(path, os.O_WRONLY)
        os.close(fd)

    @unittest.skipIf(sys.platform == 'darwin', 'hangs under macOS; see bpo-25234, bpo-35363')
    def test_os_open(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_open('fd = os.open(path, os.O_RDONLY)\nos.close(fd)', self.os_open)

@unittest.skipUnless(hasattr(signal, 'setitimer'), 'requires setitimer()')
class TimeEINTRTest(EINTRBaseTest):
    """ EINTR tests for the time module. """

    def test_sleep(self):
        if False:
            return 10
        t0 = time.monotonic()
        time.sleep(self.sleep_time)
        self.stop_alarm()
        dt = time.monotonic() - t0
        self.assertGreaterEqual(dt, self.sleep_time)

@unittest.skipUnless(hasattr(signal, 'setitimer'), 'requires setitimer()')
@unittest.skipUnless(hasattr(signal, 'pthread_sigmask'), 'need signal.pthread_sigmask()')
class SignalEINTRTest(EINTRBaseTest):
    """ EINTR tests for the signal module. """

    def check_sigwait(self, wait_func):
        if False:
            return 10
        signum = signal.SIGUSR1
        pid = os.getpid()
        old_handler = signal.signal(signum, lambda *args: None)
        self.addCleanup(signal.signal, signum, old_handler)
        code = '\n'.join(('import os, time', 'pid = %s' % os.getpid(), 'signum = %s' % int(signum), 'sleep_time = %r' % self.sleep_time, 'time.sleep(sleep_time)', 'os.kill(pid, signum)'))
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, [signum])
        self.addCleanup(signal.pthread_sigmask, signal.SIG_UNBLOCK, [signum])
        t0 = time.monotonic()
        proc = self.subprocess(code)
        with kill_on_error(proc):
            wait_func(signum)
            dt = time.monotonic() - t0
        self.assertEqual(proc.wait(), 0)

    @unittest.skipUnless(hasattr(signal, 'sigwaitinfo'), 'need signal.sigwaitinfo()')
    def test_sigwaitinfo(self):
        if False:
            print('Hello World!')

        def wait_func(signum):
            if False:
                i = 10
                return i + 15
            signal.sigwaitinfo([signum])
        self.check_sigwait(wait_func)

    @unittest.skipUnless(hasattr(signal, 'sigtimedwait'), 'need signal.sigwaitinfo()')
    def test_sigtimedwait(self):
        if False:
            print('Hello World!')

        def wait_func(signum):
            if False:
                return 10
            signal.sigtimedwait([signum], 120.0)
        self.check_sigwait(wait_func)

@unittest.skipUnless(hasattr(signal, 'setitimer'), 'requires setitimer()')
class SelectEINTRTest(EINTRBaseTest):
    """ EINTR tests for the select module. """

    def test_select(self):
        if False:
            for i in range(10):
                print('nop')
        t0 = time.monotonic()
        select.select([], [], [], self.sleep_time)
        dt = time.monotonic() - t0
        self.stop_alarm()
        self.assertGreaterEqual(dt, self.sleep_time)

    @unittest.skipIf(sys.platform == 'darwin', 'poll may fail on macOS; see issue #28087')
    @unittest.skipUnless(hasattr(select, 'poll'), 'need select.poll')
    def test_poll(self):
        if False:
            while True:
                i = 10
        poller = select.poll()
        t0 = time.monotonic()
        poller.poll(self.sleep_time * 1000.0)
        dt = time.monotonic() - t0
        self.stop_alarm()
        self.assertGreaterEqual(dt, self.sleep_time)

    @unittest.skipUnless(hasattr(select, 'epoll'), 'need select.epoll')
    def test_epoll(self):
        if False:
            while True:
                i = 10
        poller = select.epoll()
        self.addCleanup(poller.close)
        t0 = time.monotonic()
        poller.poll(self.sleep_time)
        dt = time.monotonic() - t0
        self.stop_alarm()
        self.assertGreaterEqual(dt, self.sleep_time)

    @unittest.skipUnless(hasattr(select, 'kqueue'), 'need select.kqueue')
    def test_kqueue(self):
        if False:
            for i in range(10):
                print('nop')
        kqueue = select.kqueue()
        self.addCleanup(kqueue.close)
        t0 = time.monotonic()
        kqueue.control(None, 1, self.sleep_time)
        dt = time.monotonic() - t0
        self.stop_alarm()
        self.assertGreaterEqual(dt, self.sleep_time)

    @unittest.skipUnless(hasattr(select, 'devpoll'), 'need select.devpoll')
    def test_devpoll(self):
        if False:
            print('Hello World!')
        poller = select.devpoll()
        self.addCleanup(poller.close)
        t0 = time.monotonic()
        poller.poll(self.sleep_time * 1000.0)
        dt = time.monotonic() - t0
        self.stop_alarm()
        self.assertGreaterEqual(dt, self.sleep_time)

class FNTLEINTRTest(EINTRBaseTest):

    def _lock(self, lock_func, lock_name):
        if False:
            print('Hello World!')
        self.addCleanup(os_helper.unlink, os_helper.TESTFN)
        code = '\n'.join(('import fcntl, time', "with open('%s', 'wb') as f:" % os_helper.TESTFN, '   fcntl.%s(f, fcntl.LOCK_EX)' % lock_name, '   time.sleep(%s)' % self.sleep_time))
        start_time = time.monotonic()
        proc = self.subprocess(code)
        with kill_on_error(proc):
            with open(os_helper.TESTFN, 'wb') as f:
                while True:
                    dt = time.monotonic() - start_time
                    if dt > 60.0:
                        raise Exception('failed to sync child in %.1f sec' % dt)
                    try:
                        lock_func(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        lock_func(f, fcntl.LOCK_UN)
                        time.sleep(0.01)
                    except BlockingIOError:
                        break
                lock_func(f, fcntl.LOCK_EX)
                dt = time.monotonic() - start_time
                self.assertGreaterEqual(dt, self.sleep_time)
                self.stop_alarm()
            proc.wait()

    @unittest.skipIf(platform.system() == 'AIX', 'AIX returns PermissionError')
    def test_lockf(self):
        if False:
            for i in range(10):
                print('nop')
        self._lock(fcntl.lockf, 'lockf')

    def test_flock(self):
        if False:
            i = 10
            return i + 15
        self._lock(fcntl.flock, 'flock')
if __name__ == '__main__':
    unittest.main()