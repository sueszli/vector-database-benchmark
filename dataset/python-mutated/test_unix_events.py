"""Tests for unix_events.py."""
import contextlib
import errno
import io
import os
import pathlib
import signal
import socket
import stat
import sys
import tempfile
import threading
import unittest
from unittest import mock
from test.support import os_helper
from test.support import socket_helper
if sys.platform == 'win32':
    raise unittest.SkipTest('UNIX only')
import asyncio
from asyncio import log
from asyncio import unix_events
from test.test_asyncio import utils as test_utils

def tearDownModule():
    if False:
        i = 10
        return i + 15
    asyncio.set_event_loop_policy(None)
MOCK_ANY = mock.ANY

def EXITCODE(exitcode):
    if False:
        print('Hello World!')
    return 32768 + exitcode

def SIGNAL(signum):
    if False:
        for i in range(10):
            print('nop')
    if not 1 <= signum <= 68:
        raise AssertionError(f'invalid signum {signum}')
    return 32768 - signum

def close_pipe_transport(transport):
    if False:
        for i in range(10):
            print('nop')
    if transport._pipe is None:
        return
    transport._pipe.close()
    transport._pipe = None

@unittest.skipUnless(signal, 'Signals are not supported')
class SelectorEventLoopSignalTests(test_utils.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.loop = asyncio.SelectorEventLoop()
        self.set_event_loop(self.loop)

    def test_check_signal(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, self.loop._check_signal, '1')
        self.assertRaises(ValueError, self.loop._check_signal, signal.NSIG + 1)

    def test_handle_signal_no_handler(self):
        if False:
            i = 10
            return i + 15
        self.loop._handle_signal(signal.NSIG + 1)

    def test_handle_signal_cancelled_handler(self):
        if False:
            i = 10
            return i + 15
        h = asyncio.Handle(mock.Mock(), (), loop=mock.Mock())
        h.cancel()
        self.loop._signal_handlers[signal.NSIG + 1] = h
        self.loop.remove_signal_handler = mock.Mock()
        self.loop._handle_signal(signal.NSIG + 1)
        self.loop.remove_signal_handler.assert_called_with(signal.NSIG + 1)

    @mock.patch('asyncio.unix_events.signal')
    def test_add_signal_handler_setup_error(self, m_signal):
        if False:
            i = 10
            return i + 15
        m_signal.NSIG = signal.NSIG
        m_signal.valid_signals = signal.valid_signals
        m_signal.set_wakeup_fd.side_effect = ValueError
        self.assertRaises(RuntimeError, self.loop.add_signal_handler, signal.SIGINT, lambda : True)

    @mock.patch('asyncio.unix_events.signal')
    def test_add_signal_handler_coroutine_error(self, m_signal):
        if False:
            for i in range(10):
                print('nop')
        m_signal.NSIG = signal.NSIG

        async def simple_coroutine():
            pass
        coro_func = simple_coroutine
        coro_obj = coro_func()
        self.addCleanup(coro_obj.close)
        for func in (coro_func, coro_obj):
            self.assertRaisesRegex(TypeError, 'coroutines cannot be used with add_signal_handler', self.loop.add_signal_handler, signal.SIGINT, func)

    @mock.patch('asyncio.unix_events.signal')
    def test_add_signal_handler(self, m_signal):
        if False:
            print('Hello World!')
        m_signal.NSIG = signal.NSIG
        m_signal.valid_signals = signal.valid_signals
        cb = lambda : True
        self.loop.add_signal_handler(signal.SIGHUP, cb)
        h = self.loop._signal_handlers.get(signal.SIGHUP)
        self.assertIsInstance(h, asyncio.Handle)
        self.assertEqual(h._callback, cb)

    @mock.patch('asyncio.unix_events.signal')
    def test_add_signal_handler_install_error(self, m_signal):
        if False:
            for i in range(10):
                print('nop')
        m_signal.NSIG = signal.NSIG
        m_signal.valid_signals = signal.valid_signals

        def set_wakeup_fd(fd):
            if False:
                for i in range(10):
                    print('nop')
            if fd == -1:
                raise ValueError()
        m_signal.set_wakeup_fd = set_wakeup_fd

        class Err(OSError):
            errno = errno.EFAULT
        m_signal.signal.side_effect = Err
        self.assertRaises(Err, self.loop.add_signal_handler, signal.SIGINT, lambda : True)

    @mock.patch('asyncio.unix_events.signal')
    @mock.patch('asyncio.base_events.logger')
    def test_add_signal_handler_install_error2(self, m_logging, m_signal):
        if False:
            return 10
        m_signal.NSIG = signal.NSIG
        m_signal.valid_signals = signal.valid_signals

        class Err(OSError):
            errno = errno.EINVAL
        m_signal.signal.side_effect = Err
        self.loop._signal_handlers[signal.SIGHUP] = lambda : True
        self.assertRaises(RuntimeError, self.loop.add_signal_handler, signal.SIGINT, lambda : True)
        self.assertFalse(m_logging.info.called)
        self.assertEqual(1, m_signal.set_wakeup_fd.call_count)

    @mock.patch('asyncio.unix_events.signal')
    @mock.patch('asyncio.base_events.logger')
    def test_add_signal_handler_install_error3(self, m_logging, m_signal):
        if False:
            while True:
                i = 10

        class Err(OSError):
            errno = errno.EINVAL
        m_signal.signal.side_effect = Err
        m_signal.NSIG = signal.NSIG
        m_signal.valid_signals = signal.valid_signals
        self.assertRaises(RuntimeError, self.loop.add_signal_handler, signal.SIGINT, lambda : True)
        self.assertFalse(m_logging.info.called)
        self.assertEqual(2, m_signal.set_wakeup_fd.call_count)

    @mock.patch('asyncio.unix_events.signal')
    def test_remove_signal_handler(self, m_signal):
        if False:
            for i in range(10):
                print('nop')
        m_signal.NSIG = signal.NSIG
        m_signal.valid_signals = signal.valid_signals
        self.loop.add_signal_handler(signal.SIGHUP, lambda : True)
        self.assertTrue(self.loop.remove_signal_handler(signal.SIGHUP))
        self.assertTrue(m_signal.set_wakeup_fd.called)
        self.assertTrue(m_signal.signal.called)
        self.assertEqual((signal.SIGHUP, m_signal.SIG_DFL), m_signal.signal.call_args[0])

    @mock.patch('asyncio.unix_events.signal')
    def test_remove_signal_handler_2(self, m_signal):
        if False:
            while True:
                i = 10
        m_signal.NSIG = signal.NSIG
        m_signal.SIGINT = signal.SIGINT
        m_signal.valid_signals = signal.valid_signals
        self.loop.add_signal_handler(signal.SIGINT, lambda : True)
        self.loop._signal_handlers[signal.SIGHUP] = object()
        m_signal.set_wakeup_fd.reset_mock()
        self.assertTrue(self.loop.remove_signal_handler(signal.SIGINT))
        self.assertFalse(m_signal.set_wakeup_fd.called)
        self.assertTrue(m_signal.signal.called)
        self.assertEqual((signal.SIGINT, m_signal.default_int_handler), m_signal.signal.call_args[0])

    @mock.patch('asyncio.unix_events.signal')
    @mock.patch('asyncio.base_events.logger')
    def test_remove_signal_handler_cleanup_error(self, m_logging, m_signal):
        if False:
            for i in range(10):
                print('nop')
        m_signal.NSIG = signal.NSIG
        m_signal.valid_signals = signal.valid_signals
        self.loop.add_signal_handler(signal.SIGHUP, lambda : True)
        m_signal.set_wakeup_fd.side_effect = ValueError
        self.loop.remove_signal_handler(signal.SIGHUP)
        self.assertTrue(m_logging.info)

    @mock.patch('asyncio.unix_events.signal')
    def test_remove_signal_handler_error(self, m_signal):
        if False:
            print('Hello World!')
        m_signal.NSIG = signal.NSIG
        m_signal.valid_signals = signal.valid_signals
        self.loop.add_signal_handler(signal.SIGHUP, lambda : True)
        m_signal.signal.side_effect = OSError
        self.assertRaises(OSError, self.loop.remove_signal_handler, signal.SIGHUP)

    @mock.patch('asyncio.unix_events.signal')
    def test_remove_signal_handler_error2(self, m_signal):
        if False:
            return 10
        m_signal.NSIG = signal.NSIG
        m_signal.valid_signals = signal.valid_signals
        self.loop.add_signal_handler(signal.SIGHUP, lambda : True)

        class Err(OSError):
            errno = errno.EINVAL
        m_signal.signal.side_effect = Err
        self.assertRaises(RuntimeError, self.loop.remove_signal_handler, signal.SIGHUP)

    @mock.patch('asyncio.unix_events.signal')
    def test_close(self, m_signal):
        if False:
            i = 10
            return i + 15
        m_signal.NSIG = signal.NSIG
        m_signal.valid_signals = signal.valid_signals
        self.loop.add_signal_handler(signal.SIGHUP, lambda : True)
        self.loop.add_signal_handler(signal.SIGCHLD, lambda : True)
        self.assertEqual(len(self.loop._signal_handlers), 2)
        m_signal.set_wakeup_fd.reset_mock()
        self.loop.close()
        self.assertEqual(len(self.loop._signal_handlers), 0)
        m_signal.set_wakeup_fd.assert_called_once_with(-1)

    @mock.patch('asyncio.unix_events.sys')
    @mock.patch('asyncio.unix_events.signal')
    def test_close_on_finalizing(self, m_signal, m_sys):
        if False:
            while True:
                i = 10
        m_signal.NSIG = signal.NSIG
        m_signal.valid_signals = signal.valid_signals
        self.loop.add_signal_handler(signal.SIGHUP, lambda : True)
        self.assertEqual(len(self.loop._signal_handlers), 1)
        m_sys.is_finalizing.return_value = True
        m_signal.signal.reset_mock()
        with self.assertWarnsRegex(ResourceWarning, 'skipping signal handlers removal'):
            self.loop.close()
        self.assertEqual(len(self.loop._signal_handlers), 0)
        self.assertFalse(m_signal.signal.called)

@unittest.skipUnless(hasattr(socket, 'AF_UNIX'), 'UNIX Sockets are not supported')
class SelectorEventLoopUnixSocketTests(test_utils.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.loop = asyncio.SelectorEventLoop()
        self.set_event_loop(self.loop)

    @socket_helper.skip_unless_bind_unix_socket
    def test_create_unix_server_existing_path_sock(self):
        if False:
            return 10
        with test_utils.unix_socket_path() as path:
            sock = socket.socket(socket.AF_UNIX)
            sock.bind(path)
            sock.listen(1)
            sock.close()
            coro = self.loop.create_unix_server(lambda : None, path)
            srv = self.loop.run_until_complete(coro)
            srv.close()
            self.loop.run_until_complete(srv.wait_closed())

    @socket_helper.skip_unless_bind_unix_socket
    def test_create_unix_server_pathlib(self):
        if False:
            print('Hello World!')
        with test_utils.unix_socket_path() as path:
            path = pathlib.Path(path)
            srv_coro = self.loop.create_unix_server(lambda : None, path)
            srv = self.loop.run_until_complete(srv_coro)
            srv.close()
            self.loop.run_until_complete(srv.wait_closed())

    def test_create_unix_connection_pathlib(self):
        if False:
            for i in range(10):
                print('nop')
        with test_utils.unix_socket_path() as path:
            path = pathlib.Path(path)
            coro = self.loop.create_unix_connection(lambda : None, path)
            with self.assertRaises(FileNotFoundError):
                self.loop.run_until_complete(coro)

    def test_create_unix_server_existing_path_nonsock(self):
        if False:
            i = 10
            return i + 15
        with tempfile.NamedTemporaryFile() as file:
            coro = self.loop.create_unix_server(lambda : None, file.name)
            with self.assertRaisesRegex(OSError, 'Address.*is already in use'):
                self.loop.run_until_complete(coro)

    def test_create_unix_server_ssl_bool(self):
        if False:
            print('Hello World!')
        coro = self.loop.create_unix_server(lambda : None, path='spam', ssl=True)
        with self.assertRaisesRegex(TypeError, 'ssl argument must be an SSLContext'):
            self.loop.run_until_complete(coro)

    def test_create_unix_server_nopath_nosock(self):
        if False:
            i = 10
            return i + 15
        coro = self.loop.create_unix_server(lambda : None, path=None)
        with self.assertRaisesRegex(ValueError, 'path was not specified, and no sock'):
            self.loop.run_until_complete(coro)

    def test_create_unix_server_path_inetsock(self):
        if False:
            while True:
                i = 10
        sock = socket.socket()
        with sock:
            coro = self.loop.create_unix_server(lambda : None, path=None, sock=sock)
            with self.assertRaisesRegex(ValueError, 'A UNIX Domain Stream.*was expected'):
                self.loop.run_until_complete(coro)

    def test_create_unix_server_path_dgram(self):
        if False:
            for i in range(10):
                print('nop')
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        with sock:
            coro = self.loop.create_unix_server(lambda : None, path=None, sock=sock)
            with self.assertRaisesRegex(ValueError, 'A UNIX Domain Stream.*was expected'):
                self.loop.run_until_complete(coro)

    @unittest.skipUnless(hasattr(socket, 'SOCK_NONBLOCK'), 'no socket.SOCK_NONBLOCK (linux only)')
    @socket_helper.skip_unless_bind_unix_socket
    def test_create_unix_server_path_stream_bittype(self):
        if False:
            for i in range(10):
                print('nop')
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_NONBLOCK)
        with tempfile.NamedTemporaryFile() as file:
            fn = file.name
        try:
            with sock:
                sock.bind(fn)
                coro = self.loop.create_unix_server(lambda : None, path=None, sock=sock)
                srv = self.loop.run_until_complete(coro)
                srv.close()
                self.loop.run_until_complete(srv.wait_closed())
        finally:
            os.unlink(fn)

    def test_create_unix_server_ssl_timeout_with_plain_sock(self):
        if False:
            return 10
        coro = self.loop.create_unix_server(lambda : None, path='spam', ssl_handshake_timeout=1)
        with self.assertRaisesRegex(ValueError, 'ssl_handshake_timeout is only meaningful with ssl'):
            self.loop.run_until_complete(coro)

    def test_create_unix_connection_path_inetsock(self):
        if False:
            i = 10
            return i + 15
        sock = socket.socket()
        with sock:
            coro = self.loop.create_unix_connection(lambda : None, sock=sock)
            with self.assertRaisesRegex(ValueError, 'A UNIX Domain Stream.*was expected'):
                self.loop.run_until_complete(coro)

    @mock.patch('asyncio.unix_events.socket')
    def test_create_unix_server_bind_error(self, m_socket):
        if False:
            while True:
                i = 10
        sock = mock.Mock()
        m_socket.socket.return_value = sock
        sock.bind.side_effect = OSError
        coro = self.loop.create_unix_server(lambda : None, path='/test')
        with self.assertRaises(OSError):
            self.loop.run_until_complete(coro)
        self.assertTrue(sock.close.called)
        sock.bind.side_effect = MemoryError
        coro = self.loop.create_unix_server(lambda : None, path='/test')
        with self.assertRaises(MemoryError):
            self.loop.run_until_complete(coro)
        self.assertTrue(sock.close.called)

    def test_create_unix_connection_path_sock(self):
        if False:
            i = 10
            return i + 15
        coro = self.loop.create_unix_connection(lambda : None, os.devnull, sock=object())
        with self.assertRaisesRegex(ValueError, 'path and sock can not be'):
            self.loop.run_until_complete(coro)

    def test_create_unix_connection_nopath_nosock(self):
        if False:
            while True:
                i = 10
        coro = self.loop.create_unix_connection(lambda : None, None)
        with self.assertRaisesRegex(ValueError, 'no path and sock were specified'):
            self.loop.run_until_complete(coro)

    def test_create_unix_connection_nossl_serverhost(self):
        if False:
            for i in range(10):
                print('nop')
        coro = self.loop.create_unix_connection(lambda : None, os.devnull, server_hostname='spam')
        with self.assertRaisesRegex(ValueError, 'server_hostname is only meaningful'):
            self.loop.run_until_complete(coro)

    def test_create_unix_connection_ssl_noserverhost(self):
        if False:
            for i in range(10):
                print('nop')
        coro = self.loop.create_unix_connection(lambda : None, os.devnull, ssl=True)
        with self.assertRaisesRegex(ValueError, 'you have to pass server_hostname when using ssl'):
            self.loop.run_until_complete(coro)

    def test_create_unix_connection_ssl_timeout_with_plain_sock(self):
        if False:
            print('Hello World!')
        coro = self.loop.create_unix_connection(lambda : None, path='spam', ssl_handshake_timeout=1)
        with self.assertRaisesRegex(ValueError, 'ssl_handshake_timeout is only meaningful with ssl'):
            self.loop.run_until_complete(coro)

@unittest.skipUnless(hasattr(os, 'sendfile'), 'sendfile is not supported')
class SelectorEventLoopUnixSockSendfileTests(test_utils.TestCase):
    DATA = b'12345abcde' * 16 * 1024

    class MyProto(asyncio.Protocol):

        def __init__(self, loop):
            if False:
                return 10
            self.started = False
            self.closed = False
            self.data = bytearray()
            self.fut = loop.create_future()
            self.transport = None
            self._ready = loop.create_future()

        def connection_made(self, transport):
            if False:
                for i in range(10):
                    print('nop')
            self.started = True
            self.transport = transport
            self._ready.set_result(None)

        def data_received(self, data):
            if False:
                return 10
            self.data.extend(data)

        def connection_lost(self, exc):
            if False:
                for i in range(10):
                    print('nop')
            self.closed = True
            self.fut.set_result(None)

        async def wait_closed(self):
            await self.fut

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        with open(os_helper.TESTFN, 'wb') as fp:
            fp.write(cls.DATA)
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        os_helper.unlink(os_helper.TESTFN)
        super().tearDownClass()

    def setUp(self):
        if False:
            print('Hello World!')
        self.loop = asyncio.new_event_loop()
        self.set_event_loop(self.loop)
        self.file = open(os_helper.TESTFN, 'rb')
        self.addCleanup(self.file.close)
        super().setUp()

    def make_socket(self, cleanup=True):
        if False:
            i = 10
            return i + 15
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setblocking(False)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        if cleanup:
            self.addCleanup(sock.close)
        return sock

    def run_loop(self, coro):
        if False:
            while True:
                i = 10
        return self.loop.run_until_complete(coro)

    def prepare(self):
        if False:
            for i in range(10):
                print('nop')
        sock = self.make_socket()
        proto = self.MyProto(self.loop)
        port = socket_helper.find_unused_port()
        srv_sock = self.make_socket(cleanup=False)
        srv_sock.bind((socket_helper.HOST, port))
        server = self.run_loop(self.loop.create_server(lambda : proto, sock=srv_sock))
        self.run_loop(self.loop.sock_connect(sock, (socket_helper.HOST, port)))
        self.run_loop(proto._ready)

        def cleanup():
            if False:
                print('Hello World!')
            proto.transport.close()
            self.run_loop(proto.wait_closed())
            server.close()
            self.run_loop(server.wait_closed())
        self.addCleanup(cleanup)
        return (sock, proto)

    def test_sock_sendfile_not_available(self):
        if False:
            print('Hello World!')
        (sock, proto) = self.prepare()
        with mock.patch('asyncio.unix_events.os', spec=[]):
            with self.assertRaisesRegex(asyncio.SendfileNotAvailableError, 'os[.]sendfile[(][)] is not available'):
                self.run_loop(self.loop._sock_sendfile_native(sock, self.file, 0, None))
        self.assertEqual(self.file.tell(), 0)

    def test_sock_sendfile_not_a_file(self):
        if False:
            print('Hello World!')
        (sock, proto) = self.prepare()
        f = object()
        with self.assertRaisesRegex(asyncio.SendfileNotAvailableError, 'not a regular file'):
            self.run_loop(self.loop._sock_sendfile_native(sock, f, 0, None))
        self.assertEqual(self.file.tell(), 0)

    def test_sock_sendfile_iobuffer(self):
        if False:
            print('Hello World!')
        (sock, proto) = self.prepare()
        f = io.BytesIO()
        with self.assertRaisesRegex(asyncio.SendfileNotAvailableError, 'not a regular file'):
            self.run_loop(self.loop._sock_sendfile_native(sock, f, 0, None))
        self.assertEqual(self.file.tell(), 0)

    def test_sock_sendfile_not_regular_file(self):
        if False:
            for i in range(10):
                print('nop')
        (sock, proto) = self.prepare()
        f = mock.Mock()
        f.fileno.return_value = -1
        with self.assertRaisesRegex(asyncio.SendfileNotAvailableError, 'not a regular file'):
            self.run_loop(self.loop._sock_sendfile_native(sock, f, 0, None))
        self.assertEqual(self.file.tell(), 0)

    def test_sock_sendfile_cancel1(self):
        if False:
            i = 10
            return i + 15
        (sock, proto) = self.prepare()
        fut = self.loop.create_future()
        fileno = self.file.fileno()
        self.loop._sock_sendfile_native_impl(fut, None, sock, fileno, 0, None, len(self.DATA), 0)
        fut.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            self.run_loop(fut)
        with self.assertRaises(KeyError):
            self.loop._selector.get_key(sock)

    def test_sock_sendfile_cancel2(self):
        if False:
            for i in range(10):
                print('nop')
        (sock, proto) = self.prepare()
        fut = self.loop.create_future()
        fileno = self.file.fileno()
        self.loop._sock_sendfile_native_impl(fut, None, sock, fileno, 0, None, len(self.DATA), 0)
        fut.cancel()
        self.loop._sock_sendfile_native_impl(fut, sock.fileno(), sock, fileno, 0, None, len(self.DATA), 0)
        with self.assertRaises(KeyError):
            self.loop._selector.get_key(sock)

    def test_sock_sendfile_blocking_error(self):
        if False:
            while True:
                i = 10
        (sock, proto) = self.prepare()
        fileno = self.file.fileno()
        fut = mock.Mock()
        fut.cancelled.return_value = False
        with mock.patch('os.sendfile', side_effect=BlockingIOError()):
            self.loop._sock_sendfile_native_impl(fut, None, sock, fileno, 0, None, len(self.DATA), 0)
        key = self.loop._selector.get_key(sock)
        self.assertIsNotNone(key)
        fut.add_done_callback.assert_called_once_with(mock.ANY)

    def test_sock_sendfile_os_error_first_call(self):
        if False:
            while True:
                i = 10
        (sock, proto) = self.prepare()
        fileno = self.file.fileno()
        fut = self.loop.create_future()
        with mock.patch('os.sendfile', side_effect=OSError()):
            self.loop._sock_sendfile_native_impl(fut, None, sock, fileno, 0, None, len(self.DATA), 0)
        with self.assertRaises(KeyError):
            self.loop._selector.get_key(sock)
        exc = fut.exception()
        self.assertIsInstance(exc, asyncio.SendfileNotAvailableError)
        self.assertEqual(0, self.file.tell())

    def test_sock_sendfile_os_error_next_call(self):
        if False:
            return 10
        (sock, proto) = self.prepare()
        fileno = self.file.fileno()
        fut = self.loop.create_future()
        err = OSError()
        with mock.patch('os.sendfile', side_effect=err):
            self.loop._sock_sendfile_native_impl(fut, sock.fileno(), sock, fileno, 1000, None, len(self.DATA), 1000)
        with self.assertRaises(KeyError):
            self.loop._selector.get_key(sock)
        exc = fut.exception()
        self.assertIs(exc, err)
        self.assertEqual(1000, self.file.tell())

    def test_sock_sendfile_exception(self):
        if False:
            i = 10
            return i + 15
        (sock, proto) = self.prepare()
        fileno = self.file.fileno()
        fut = self.loop.create_future()
        err = asyncio.SendfileNotAvailableError()
        with mock.patch('os.sendfile', side_effect=err):
            self.loop._sock_sendfile_native_impl(fut, sock.fileno(), sock, fileno, 1000, None, len(self.DATA), 1000)
        with self.assertRaises(KeyError):
            self.loop._selector.get_key(sock)
        exc = fut.exception()
        self.assertIs(exc, err)
        self.assertEqual(1000, self.file.tell())

class UnixReadPipeTransportTests(test_utils.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.loop = self.new_test_loop()
        self.protocol = test_utils.make_test_protocol(asyncio.Protocol)
        self.pipe = mock.Mock(spec_set=io.RawIOBase)
        self.pipe.fileno.return_value = 5
        blocking_patcher = mock.patch('os.set_blocking')
        blocking_patcher.start()
        self.addCleanup(blocking_patcher.stop)
        fstat_patcher = mock.patch('os.fstat')
        m_fstat = fstat_patcher.start()
        st = mock.Mock()
        st.st_mode = stat.S_IFIFO
        m_fstat.return_value = st
        self.addCleanup(fstat_patcher.stop)

    def read_pipe_transport(self, waiter=None):
        if False:
            i = 10
            return i + 15
        transport = unix_events._UnixReadPipeTransport(self.loop, self.pipe, self.protocol, waiter=waiter)
        self.addCleanup(close_pipe_transport, transport)
        return transport

    def test_ctor(self):
        if False:
            print('Hello World!')
        waiter = self.loop.create_future()
        tr = self.read_pipe_transport(waiter=waiter)
        self.loop.run_until_complete(waiter)
        self.protocol.connection_made.assert_called_with(tr)
        self.loop.assert_reader(5, tr._read_ready)
        self.assertIsNone(waiter.result())

    @mock.patch('os.read')
    def test__read_ready(self, m_read):
        if False:
            while True:
                i = 10
        tr = self.read_pipe_transport()
        m_read.return_value = b'data'
        tr._read_ready()
        m_read.assert_called_with(5, tr.max_size)
        self.protocol.data_received.assert_called_with(b'data')

    @mock.patch('os.read')
    def test__read_ready_eof(self, m_read):
        if False:
            print('Hello World!')
        tr = self.read_pipe_transport()
        m_read.return_value = b''
        tr._read_ready()
        m_read.assert_called_with(5, tr.max_size)
        self.assertFalse(self.loop.readers)
        test_utils.run_briefly(self.loop)
        self.protocol.eof_received.assert_called_with()
        self.protocol.connection_lost.assert_called_with(None)

    @mock.patch('os.read')
    def test__read_ready_blocked(self, m_read):
        if False:
            while True:
                i = 10
        tr = self.read_pipe_transport()
        m_read.side_effect = BlockingIOError
        tr._read_ready()
        m_read.assert_called_with(5, tr.max_size)
        test_utils.run_briefly(self.loop)
        self.assertFalse(self.protocol.data_received.called)

    @mock.patch('asyncio.log.logger.error')
    @mock.patch('os.read')
    def test__read_ready_error(self, m_read, m_logexc):
        if False:
            i = 10
            return i + 15
        tr = self.read_pipe_transport()
        err = OSError()
        m_read.side_effect = err
        tr._close = mock.Mock()
        tr._read_ready()
        m_read.assert_called_with(5, tr.max_size)
        tr._close.assert_called_with(err)
        m_logexc.assert_called_with(test_utils.MockPattern('Fatal read error on pipe transport\nprotocol:.*\ntransport:.*'), exc_info=(OSError, MOCK_ANY, MOCK_ANY))

    @mock.patch('os.read')
    def test_pause_reading(self, m_read):
        if False:
            return 10
        tr = self.read_pipe_transport()
        m = mock.Mock()
        self.loop.add_reader(5, m)
        tr.pause_reading()
        self.assertFalse(self.loop.readers)

    @mock.patch('os.read')
    def test_resume_reading(self, m_read):
        if False:
            return 10
        tr = self.read_pipe_transport()
        tr.pause_reading()
        tr.resume_reading()
        self.loop.assert_reader(5, tr._read_ready)

    @mock.patch('os.read')
    def test_close(self, m_read):
        if False:
            for i in range(10):
                print('nop')
        tr = self.read_pipe_transport()
        tr._close = mock.Mock()
        tr.close()
        tr._close.assert_called_with(None)

    @mock.patch('os.read')
    def test_close_already_closing(self, m_read):
        if False:
            for i in range(10):
                print('nop')
        tr = self.read_pipe_transport()
        tr._closing = True
        tr._close = mock.Mock()
        tr.close()
        self.assertFalse(tr._close.called)

    @mock.patch('os.read')
    def test__close(self, m_read):
        if False:
            print('Hello World!')
        tr = self.read_pipe_transport()
        err = object()
        tr._close(err)
        self.assertTrue(tr.is_closing())
        self.assertFalse(self.loop.readers)
        test_utils.run_briefly(self.loop)
        self.protocol.connection_lost.assert_called_with(err)

    def test__call_connection_lost(self):
        if False:
            i = 10
            return i + 15
        tr = self.read_pipe_transport()
        self.assertIsNotNone(tr._protocol)
        self.assertIsNotNone(tr._loop)
        err = None
        tr._call_connection_lost(err)
        self.protocol.connection_lost.assert_called_with(err)
        self.pipe.close.assert_called_with()
        self.assertIsNone(tr._protocol)
        self.assertIsNone(tr._loop)

    def test__call_connection_lost_with_err(self):
        if False:
            for i in range(10):
                print('nop')
        tr = self.read_pipe_transport()
        self.assertIsNotNone(tr._protocol)
        self.assertIsNotNone(tr._loop)
        err = OSError()
        tr._call_connection_lost(err)
        self.protocol.connection_lost.assert_called_with(err)
        self.pipe.close.assert_called_with()
        self.assertIsNone(tr._protocol)
        self.assertIsNone(tr._loop)

    def test_pause_reading_on_closed_pipe(self):
        if False:
            i = 10
            return i + 15
        tr = self.read_pipe_transport()
        tr.close()
        test_utils.run_briefly(self.loop)
        self.assertIsNone(tr._loop)
        tr.pause_reading()

    def test_pause_reading_on_paused_pipe(self):
        if False:
            return 10
        tr = self.read_pipe_transport()
        tr.pause_reading()
        tr.pause_reading()

    def test_resume_reading_on_closed_pipe(self):
        if False:
            print('Hello World!')
        tr = self.read_pipe_transport()
        tr.close()
        test_utils.run_briefly(self.loop)
        self.assertIsNone(tr._loop)
        tr.resume_reading()

    def test_resume_reading_on_paused_pipe(self):
        if False:
            return 10
        tr = self.read_pipe_transport()
        tr.resume_reading()

class UnixWritePipeTransportTests(test_utils.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.loop = self.new_test_loop()
        self.protocol = test_utils.make_test_protocol(asyncio.BaseProtocol)
        self.pipe = mock.Mock(spec_set=io.RawIOBase)
        self.pipe.fileno.return_value = 5
        blocking_patcher = mock.patch('os.set_blocking')
        blocking_patcher.start()
        self.addCleanup(blocking_patcher.stop)
        fstat_patcher = mock.patch('os.fstat')
        m_fstat = fstat_patcher.start()
        st = mock.Mock()
        st.st_mode = stat.S_IFSOCK
        m_fstat.return_value = st
        self.addCleanup(fstat_patcher.stop)

    def write_pipe_transport(self, waiter=None):
        if False:
            i = 10
            return i + 15
        transport = unix_events._UnixWritePipeTransport(self.loop, self.pipe, self.protocol, waiter=waiter)
        self.addCleanup(close_pipe_transport, transport)
        return transport

    def test_ctor(self):
        if False:
            for i in range(10):
                print('nop')
        waiter = self.loop.create_future()
        tr = self.write_pipe_transport(waiter=waiter)
        self.loop.run_until_complete(waiter)
        self.protocol.connection_made.assert_called_with(tr)
        self.loop.assert_reader(5, tr._read_ready)
        self.assertEqual(None, waiter.result())

    def test_can_write_eof(self):
        if False:
            print('Hello World!')
        tr = self.write_pipe_transport()
        self.assertTrue(tr.can_write_eof())

    @mock.patch('os.write')
    def test_write(self, m_write):
        if False:
            for i in range(10):
                print('nop')
        tr = self.write_pipe_transport()
        m_write.return_value = 4
        tr.write(b'data')
        m_write.assert_called_with(5, b'data')
        self.assertFalse(self.loop.writers)
        self.assertEqual(bytearray(), tr._buffer)

    @mock.patch('os.write')
    def test_write_no_data(self, m_write):
        if False:
            i = 10
            return i + 15
        tr = self.write_pipe_transport()
        tr.write(b'')
        self.assertFalse(m_write.called)
        self.assertFalse(self.loop.writers)
        self.assertEqual(bytearray(b''), tr._buffer)

    @mock.patch('os.write')
    def test_write_partial(self, m_write):
        if False:
            for i in range(10):
                print('nop')
        tr = self.write_pipe_transport()
        m_write.return_value = 2
        tr.write(b'data')
        self.loop.assert_writer(5, tr._write_ready)
        self.assertEqual(bytearray(b'ta'), tr._buffer)

    @mock.patch('os.write')
    def test_write_buffer(self, m_write):
        if False:
            while True:
                i = 10
        tr = self.write_pipe_transport()
        self.loop.add_writer(5, tr._write_ready)
        tr._buffer = bytearray(b'previous')
        tr.write(b'data')
        self.assertFalse(m_write.called)
        self.loop.assert_writer(5, tr._write_ready)
        self.assertEqual(bytearray(b'previousdata'), tr._buffer)

    @mock.patch('os.write')
    def test_write_again(self, m_write):
        if False:
            while True:
                i = 10
        tr = self.write_pipe_transport()
        m_write.side_effect = BlockingIOError()
        tr.write(b'data')
        m_write.assert_called_with(5, bytearray(b'data'))
        self.loop.assert_writer(5, tr._write_ready)
        self.assertEqual(bytearray(b'data'), tr._buffer)

    @mock.patch('asyncio.unix_events.logger')
    @mock.patch('os.write')
    def test_write_err(self, m_write, m_log):
        if False:
            while True:
                i = 10
        tr = self.write_pipe_transport()
        err = OSError()
        m_write.side_effect = err
        tr._fatal_error = mock.Mock()
        tr.write(b'data')
        m_write.assert_called_with(5, b'data')
        self.assertFalse(self.loop.writers)
        self.assertEqual(bytearray(), tr._buffer)
        tr._fatal_error.assert_called_with(err, 'Fatal write error on pipe transport')
        self.assertEqual(1, tr._conn_lost)
        tr.write(b'data')
        self.assertEqual(2, tr._conn_lost)
        tr.write(b'data')
        tr.write(b'data')
        tr.write(b'data')
        tr.write(b'data')
        m_log.warning.assert_called_with('pipe closed by peer or os.write(pipe, data) raised exception.')
        tr.close()

    @mock.patch('os.write')
    def test_write_close(self, m_write):
        if False:
            for i in range(10):
                print('nop')
        tr = self.write_pipe_transport()
        tr._read_ready()
        tr.write(b'data')
        self.assertEqual(tr._conn_lost, 1)
        tr.write(b'data')
        self.assertEqual(tr._conn_lost, 2)

    def test__read_ready(self):
        if False:
            print('Hello World!')
        tr = self.write_pipe_transport()
        tr._read_ready()
        self.assertFalse(self.loop.readers)
        self.assertFalse(self.loop.writers)
        self.assertTrue(tr.is_closing())
        test_utils.run_briefly(self.loop)
        self.protocol.connection_lost.assert_called_with(None)

    @mock.patch('os.write')
    def test__write_ready(self, m_write):
        if False:
            return 10
        tr = self.write_pipe_transport()
        self.loop.add_writer(5, tr._write_ready)
        tr._buffer = bytearray(b'data')
        m_write.return_value = 4
        tr._write_ready()
        self.assertFalse(self.loop.writers)
        self.assertEqual(bytearray(), tr._buffer)

    @mock.patch('os.write')
    def test__write_ready_partial(self, m_write):
        if False:
            i = 10
            return i + 15
        tr = self.write_pipe_transport()
        self.loop.add_writer(5, tr._write_ready)
        tr._buffer = bytearray(b'data')
        m_write.return_value = 3
        tr._write_ready()
        self.loop.assert_writer(5, tr._write_ready)
        self.assertEqual(bytearray(b'a'), tr._buffer)

    @mock.patch('os.write')
    def test__write_ready_again(self, m_write):
        if False:
            while True:
                i = 10
        tr = self.write_pipe_transport()
        self.loop.add_writer(5, tr._write_ready)
        tr._buffer = bytearray(b'data')
        m_write.side_effect = BlockingIOError()
        tr._write_ready()
        m_write.assert_called_with(5, bytearray(b'data'))
        self.loop.assert_writer(5, tr._write_ready)
        self.assertEqual(bytearray(b'data'), tr._buffer)

    @mock.patch('os.write')
    def test__write_ready_empty(self, m_write):
        if False:
            return 10
        tr = self.write_pipe_transport()
        self.loop.add_writer(5, tr._write_ready)
        tr._buffer = bytearray(b'data')
        m_write.return_value = 0
        tr._write_ready()
        m_write.assert_called_with(5, bytearray(b'data'))
        self.loop.assert_writer(5, tr._write_ready)
        self.assertEqual(bytearray(b'data'), tr._buffer)

    @mock.patch('asyncio.log.logger.error')
    @mock.patch('os.write')
    def test__write_ready_err(self, m_write, m_logexc):
        if False:
            print('Hello World!')
        tr = self.write_pipe_transport()
        self.loop.add_writer(5, tr._write_ready)
        tr._buffer = bytearray(b'data')
        m_write.side_effect = err = OSError()
        tr._write_ready()
        self.assertFalse(self.loop.writers)
        self.assertFalse(self.loop.readers)
        self.assertEqual(bytearray(), tr._buffer)
        self.assertTrue(tr.is_closing())
        m_logexc.assert_not_called()
        self.assertEqual(1, tr._conn_lost)
        test_utils.run_briefly(self.loop)
        self.protocol.connection_lost.assert_called_with(err)

    @mock.patch('os.write')
    def test__write_ready_closing(self, m_write):
        if False:
            i = 10
            return i + 15
        tr = self.write_pipe_transport()
        self.loop.add_writer(5, tr._write_ready)
        tr._closing = True
        tr._buffer = bytearray(b'data')
        m_write.return_value = 4
        tr._write_ready()
        self.assertFalse(self.loop.writers)
        self.assertFalse(self.loop.readers)
        self.assertEqual(bytearray(), tr._buffer)
        self.protocol.connection_lost.assert_called_with(None)
        self.pipe.close.assert_called_with()

    @mock.patch('os.write')
    def test_abort(self, m_write):
        if False:
            while True:
                i = 10
        tr = self.write_pipe_transport()
        self.loop.add_writer(5, tr._write_ready)
        self.loop.add_reader(5, tr._read_ready)
        tr._buffer = [b'da', b'ta']
        tr.abort()
        self.assertFalse(m_write.called)
        self.assertFalse(self.loop.readers)
        self.assertFalse(self.loop.writers)
        self.assertEqual([], tr._buffer)
        self.assertTrue(tr.is_closing())
        test_utils.run_briefly(self.loop)
        self.protocol.connection_lost.assert_called_with(None)

    def test__call_connection_lost(self):
        if False:
            i = 10
            return i + 15
        tr = self.write_pipe_transport()
        self.assertIsNotNone(tr._protocol)
        self.assertIsNotNone(tr._loop)
        err = None
        tr._call_connection_lost(err)
        self.protocol.connection_lost.assert_called_with(err)
        self.pipe.close.assert_called_with()
        self.assertIsNone(tr._protocol)
        self.assertIsNone(tr._loop)

    def test__call_connection_lost_with_err(self):
        if False:
            for i in range(10):
                print('nop')
        tr = self.write_pipe_transport()
        self.assertIsNotNone(tr._protocol)
        self.assertIsNotNone(tr._loop)
        err = OSError()
        tr._call_connection_lost(err)
        self.protocol.connection_lost.assert_called_with(err)
        self.pipe.close.assert_called_with()
        self.assertIsNone(tr._protocol)
        self.assertIsNone(tr._loop)

    def test_close(self):
        if False:
            return 10
        tr = self.write_pipe_transport()
        tr.write_eof = mock.Mock()
        tr.close()
        tr.write_eof.assert_called_with()
        tr.close()

    def test_close_closing(self):
        if False:
            print('Hello World!')
        tr = self.write_pipe_transport()
        tr.write_eof = mock.Mock()
        tr._closing = True
        tr.close()
        self.assertFalse(tr.write_eof.called)

    def test_write_eof(self):
        if False:
            i = 10
            return i + 15
        tr = self.write_pipe_transport()
        tr.write_eof()
        self.assertTrue(tr.is_closing())
        self.assertFalse(self.loop.readers)
        test_utils.run_briefly(self.loop)
        self.protocol.connection_lost.assert_called_with(None)

    def test_write_eof_pending(self):
        if False:
            print('Hello World!')
        tr = self.write_pipe_transport()
        tr._buffer = [b'data']
        tr.write_eof()
        self.assertTrue(tr.is_closing())
        self.assertFalse(self.protocol.connection_lost.called)

class AbstractChildWatcherTests(unittest.TestCase):

    def test_not_implemented(self):
        if False:
            print('Hello World!')
        f = mock.Mock()
        watcher = asyncio.AbstractChildWatcher()
        self.assertRaises(NotImplementedError, watcher.add_child_handler, f, f)
        self.assertRaises(NotImplementedError, watcher.remove_child_handler, f)
        self.assertRaises(NotImplementedError, watcher.attach_loop, f)
        self.assertRaises(NotImplementedError, watcher.close)
        self.assertRaises(NotImplementedError, watcher.is_active)
        self.assertRaises(NotImplementedError, watcher.__enter__)
        self.assertRaises(NotImplementedError, watcher.__exit__, f, f, f)

class BaseChildWatcherTests(unittest.TestCase):

    def test_not_implemented(self):
        if False:
            print('Hello World!')
        f = mock.Mock()
        watcher = unix_events.BaseChildWatcher()
        self.assertRaises(NotImplementedError, watcher._do_waitpid, f)

class ChildWatcherTestsMixin:
    ignore_warnings = mock.patch.object(log.logger, 'warning')

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.loop = self.new_test_loop()
        self.running = False
        self.zombies = {}
        with mock.patch.object(self.loop, 'add_signal_handler') as self.m_add_signal_handler:
            self.watcher = self.create_watcher()
            self.watcher.attach_loop(self.loop)

    def waitpid(self, pid, flags):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.watcher, asyncio.SafeChildWatcher) or pid != -1:
            self.assertGreater(pid, 0)
        try:
            if pid < 0:
                return self.zombies.popitem()
            else:
                return (pid, self.zombies.pop(pid))
        except KeyError:
            pass
        if self.running:
            return (0, 0)
        else:
            raise ChildProcessError()

    def add_zombie(self, pid, status):
        if False:
            while True:
                i = 10
        self.zombies[pid] = status

    def waitstatus_to_exitcode(self, status):
        if False:
            while True:
                i = 10
        if status > 32768:
            return status - 32768
        elif 32700 < status < 32768:
            return status - 32768
        else:
            return status

    def test_create_watcher(self):
        if False:
            i = 10
            return i + 15
        self.m_add_signal_handler.assert_called_once_with(signal.SIGCHLD, self.watcher._sig_chld)

    def waitpid_mocks(func):
        if False:
            while True:
                i = 10

        def wrapped_func(self):
            if False:
                return 10

            def patch(target, wrapper):
                if False:
                    print('Hello World!')
                return mock.patch(target, wraps=wrapper, new_callable=mock.Mock)
            with patch('asyncio.unix_events.waitstatus_to_exitcode', self.waitstatus_to_exitcode), patch('os.waitpid', self.waitpid) as m_waitpid:
                func(self, m_waitpid)
        return wrapped_func

    @waitpid_mocks
    def test_sigchld(self, m_waitpid):
        if False:
            return 10
        callback = mock.Mock()
        with self.watcher:
            self.running = True
            self.watcher.add_child_handler(42, callback, 9, 10, 14)
        self.assertFalse(callback.called)
        self.watcher._sig_chld()
        self.assertFalse(callback.called)
        self.running = False
        self.add_zombie(42, EXITCODE(12))
        self.watcher._sig_chld()
        callback.assert_called_once_with(42, 12, 9, 10, 14)
        callback.reset_mock()
        self.add_zombie(42, EXITCODE(13))
        with self.ignore_warnings:
            self.watcher._sig_chld()
        self.assertFalse(callback.called)
        self.zombies.clear()
        self.watcher._sig_chld()
        self.assertFalse(callback.called)

    @waitpid_mocks
    def test_sigchld_two_children(self, m_waitpid):
        if False:
            for i in range(10):
                print('nop')
        callback1 = mock.Mock()
        callback2 = mock.Mock()
        with self.watcher:
            self.running = True
            self.watcher.add_child_handler(43, callback1, 7, 8)
        self.assertFalse(callback1.called)
        self.assertFalse(callback2.called)
        with self.watcher:
            self.watcher.add_child_handler(44, callback2, 147, 18)
        self.assertFalse(callback1.called)
        self.assertFalse(callback2.called)
        self.watcher._sig_chld()
        self.assertFalse(callback1.called)
        self.assertFalse(callback2.called)
        self.add_zombie(43, SIGNAL(3))
        self.watcher._sig_chld()
        callback1.assert_called_once_with(43, -3, 7, 8)
        self.assertFalse(callback2.called)
        callback1.reset_mock()
        self.watcher._sig_chld()
        self.assertFalse(callback1.called)
        self.assertFalse(callback2.called)
        self.add_zombie(44, EXITCODE(108))
        self.running = False
        self.watcher._sig_chld()
        callback2.assert_called_once_with(44, 108, 147, 18)
        self.assertFalse(callback1.called)
        callback2.reset_mock()
        self.add_zombie(43, EXITCODE(14))
        self.add_zombie(44, EXITCODE(15))
        with self.ignore_warnings:
            self.watcher._sig_chld()
        self.assertFalse(callback1.called)
        self.assertFalse(callback2.called)
        self.zombies.clear()
        self.watcher._sig_chld()
        self.assertFalse(callback1.called)
        self.assertFalse(callback2.called)

    @waitpid_mocks
    def test_sigchld_two_children_terminating_together(self, m_waitpid):
        if False:
            while True:
                i = 10
        callback1 = mock.Mock()
        callback2 = mock.Mock()
        with self.watcher:
            self.running = True
            self.watcher.add_child_handler(45, callback1, 17, 8)
        self.assertFalse(callback1.called)
        self.assertFalse(callback2.called)
        with self.watcher:
            self.watcher.add_child_handler(46, callback2, 1147, 18)
        self.assertFalse(callback1.called)
        self.assertFalse(callback2.called)
        self.watcher._sig_chld()
        self.assertFalse(callback1.called)
        self.assertFalse(callback2.called)
        self.add_zombie(45, EXITCODE(78))
        self.add_zombie(46, SIGNAL(5))
        self.running = False
        self.watcher._sig_chld()
        callback1.assert_called_once_with(45, 78, 17, 8)
        callback2.assert_called_once_with(46, -5, 1147, 18)
        callback1.reset_mock()
        callback2.reset_mock()
        self.add_zombie(45, EXITCODE(14))
        self.add_zombie(46, EXITCODE(15))
        with self.ignore_warnings:
            self.watcher._sig_chld()
        self.assertFalse(callback1.called)
        self.assertFalse(callback2.called)

    @waitpid_mocks
    def test_sigchld_race_condition(self, m_waitpid):
        if False:
            while True:
                i = 10
        callback = mock.Mock()
        with self.watcher:
            self.add_zombie(50, EXITCODE(4))
            self.watcher._sig_chld()
            self.watcher.add_child_handler(50, callback, 1, 12)
        callback.assert_called_once_with(50, 4, 1, 12)
        callback.reset_mock()
        self.add_zombie(50, SIGNAL(1))
        with self.ignore_warnings:
            self.watcher._sig_chld()
        self.assertFalse(callback.called)

    @waitpid_mocks
    def test_sigchld_replace_handler(self, m_waitpid):
        if False:
            for i in range(10):
                print('nop')
        callback1 = mock.Mock()
        callback2 = mock.Mock()
        with self.watcher:
            self.running = True
            self.watcher.add_child_handler(51, callback1, 19)
        self.assertFalse(callback1.called)
        self.assertFalse(callback2.called)
        with self.watcher:
            self.watcher.add_child_handler(51, callback2, 21)
        self.assertFalse(callback1.called)
        self.assertFalse(callback2.called)
        self.running = False
        self.add_zombie(51, SIGNAL(8))
        self.watcher._sig_chld()
        callback2.assert_called_once_with(51, -8, 21)
        self.assertFalse(callback1.called)
        callback2.reset_mock()
        self.add_zombie(51, EXITCODE(13))
        with self.ignore_warnings:
            self.watcher._sig_chld()
        self.assertFalse(callback1.called)
        self.assertFalse(callback2.called)

    @waitpid_mocks
    def test_sigchld_remove_handler(self, m_waitpid):
        if False:
            return 10
        callback = mock.Mock()
        with self.watcher:
            self.running = True
            self.watcher.add_child_handler(52, callback, 1984)
        self.assertFalse(callback.called)
        self.watcher.remove_child_handler(52)
        self.assertFalse(callback.called)
        self.running = False
        self.add_zombie(52, EXITCODE(99))
        with self.ignore_warnings:
            self.watcher._sig_chld()
        self.assertFalse(callback.called)

    @waitpid_mocks
    def test_sigchld_unknown_status(self, m_waitpid):
        if False:
            i = 10
            return i + 15
        callback = mock.Mock()
        with self.watcher:
            self.running = True
            self.watcher.add_child_handler(53, callback, -19)
        self.assertFalse(callback.called)
        self.zombies[53] = 1178
        self.running = False
        self.watcher._sig_chld()
        callback.assert_called_once_with(53, 1178, -19)
        callback.reset_mock()
        self.add_zombie(53, EXITCODE(101))
        with self.ignore_warnings:
            self.watcher._sig_chld()
        self.assertFalse(callback.called)

    @waitpid_mocks
    def test_remove_child_handler(self, m_waitpid):
        if False:
            return 10
        callback1 = mock.Mock()
        callback2 = mock.Mock()
        callback3 = mock.Mock()
        with self.watcher:
            self.running = True
            self.watcher.add_child_handler(54, callback1, 1)
            self.watcher.add_child_handler(55, callback2, 2)
            self.watcher.add_child_handler(56, callback3, 3)
        self.assertTrue(self.watcher.remove_child_handler(54))
        self.assertTrue(self.watcher.remove_child_handler(55))
        self.assertFalse(self.watcher.remove_child_handler(55))
        self.assertFalse(self.watcher.remove_child_handler(55))
        self.add_zombie(54, EXITCODE(0))
        self.add_zombie(55, EXITCODE(1))
        self.add_zombie(56, EXITCODE(2))
        self.running = False
        with self.ignore_warnings:
            self.watcher._sig_chld()
        self.assertFalse(callback1.called)
        self.assertFalse(callback2.called)
        callback3.assert_called_once_with(56, 2, 3)

    @waitpid_mocks
    def test_sigchld_unhandled_exception(self, m_waitpid):
        if False:
            i = 10
            return i + 15
        callback = mock.Mock()
        with self.watcher:
            self.running = True
            self.watcher.add_child_handler(57, callback)
        m_waitpid.side_effect = ValueError
        with mock.patch.object(log.logger, 'error') as m_error:
            self.assertEqual(self.watcher._sig_chld(), None)
            self.assertTrue(m_error.called)

    @waitpid_mocks
    def test_sigchld_child_reaped_elsewhere(self, m_waitpid):
        if False:
            i = 10
            return i + 15
        callback = mock.Mock()
        with self.watcher:
            self.running = True
            self.watcher.add_child_handler(58, callback)
        self.assertFalse(callback.called)
        self.running = False
        self.add_zombie(58, EXITCODE(4))
        os.waitpid(58, os.WNOHANG)
        m_waitpid.reset_mock()
        with self.ignore_warnings:
            self.watcher._sig_chld()
        if isinstance(self.watcher, asyncio.FastChildWatcher):
            self.assertFalse(callback.called)
        else:
            callback.assert_called_once_with(58, 255)

    @waitpid_mocks
    def test_sigchld_unknown_pid_during_registration(self, m_waitpid):
        if False:
            for i in range(10):
                print('nop')
        callback1 = mock.Mock()
        callback2 = mock.Mock()
        with self.ignore_warnings, self.watcher:
            self.running = True
            self.add_zombie(591, EXITCODE(7))
            self.add_zombie(593, EXITCODE(17))
            self.watcher._sig_chld()
            self.watcher.add_child_handler(591, callback1)
            self.watcher.add_child_handler(592, callback2)
        callback1.assert_called_once_with(591, 7)
        self.assertFalse(callback2.called)

    @waitpid_mocks
    def test_set_loop(self, m_waitpid):
        if False:
            while True:
                i = 10
        callback = mock.Mock()
        with self.watcher:
            self.running = True
            self.watcher.add_child_handler(60, callback)
        old_loop = self.loop
        self.loop = self.new_test_loop()
        patch = mock.patch.object
        with patch(old_loop, 'remove_signal_handler') as m_old_remove, patch(self.loop, 'add_signal_handler') as m_new_add:
            self.watcher.attach_loop(self.loop)
            m_old_remove.assert_called_once_with(signal.SIGCHLD)
            m_new_add.assert_called_once_with(signal.SIGCHLD, self.watcher._sig_chld)
        self.running = False
        self.add_zombie(60, EXITCODE(9))
        self.watcher._sig_chld()
        callback.assert_called_once_with(60, 9)

    @waitpid_mocks
    def test_set_loop_race_condition(self, m_waitpid):
        if False:
            i = 10
            return i + 15
        callback1 = mock.Mock()
        callback2 = mock.Mock()
        callback3 = mock.Mock()
        with self.watcher:
            self.running = True
            self.watcher.add_child_handler(61, callback1)
            self.watcher.add_child_handler(62, callback2)
            self.watcher.add_child_handler(622, callback3)
        old_loop = self.loop
        self.loop = None
        with mock.patch.object(old_loop, 'remove_signal_handler') as m_remove_signal_handler:
            with self.assertWarnsRegex(RuntimeWarning, 'A loop is being detached'):
                self.watcher.attach_loop(None)
            m_remove_signal_handler.assert_called_once_with(signal.SIGCHLD)
        self.add_zombie(61, EXITCODE(11))
        self.add_zombie(62, SIGNAL(5))
        self.assertFalse(callback1.called)
        self.assertFalse(callback2.called)
        self.assertFalse(callback3.called)
        self.loop = self.new_test_loop()
        with mock.patch.object(self.loop, 'add_signal_handler') as m_add_signal_handler:
            self.watcher.attach_loop(self.loop)
            m_add_signal_handler.assert_called_once_with(signal.SIGCHLD, self.watcher._sig_chld)
            callback1.assert_called_once_with(61, 11)
            callback2.assert_called_once_with(62, -5)
            self.assertFalse(callback3.called)
        callback1.reset_mock()
        callback2.reset_mock()
        self.running = False
        self.add_zombie(622, EXITCODE(19))
        self.watcher._sig_chld()
        self.assertFalse(callback1.called)
        self.assertFalse(callback2.called)
        callback3.assert_called_once_with(622, 19)

    @waitpid_mocks
    def test_close(self, m_waitpid):
        if False:
            i = 10
            return i + 15
        callback1 = mock.Mock()
        with self.watcher:
            self.running = True
            self.add_zombie(63, EXITCODE(9))
            self.add_zombie(65, EXITCODE(18))
            self.watcher._sig_chld()
            self.watcher.add_child_handler(63, callback1)
            self.watcher.add_child_handler(64, callback1)
            self.assertEqual(len(self.watcher._callbacks), 1)
            if isinstance(self.watcher, asyncio.FastChildWatcher):
                self.assertEqual(len(self.watcher._zombies), 1)
            with mock.patch.object(self.loop, 'remove_signal_handler') as m_remove_signal_handler:
                self.watcher.close()
                m_remove_signal_handler.assert_called_once_with(signal.SIGCHLD)
                self.assertFalse(self.watcher._callbacks)
                if isinstance(self.watcher, asyncio.FastChildWatcher):
                    self.assertFalse(self.watcher._zombies)

class SafeChildWatcherTests(ChildWatcherTestsMixin, test_utils.TestCase):

    def create_watcher(self):
        if False:
            i = 10
            return i + 15
        return asyncio.SafeChildWatcher()

class FastChildWatcherTests(ChildWatcherTestsMixin, test_utils.TestCase):

    def create_watcher(self):
        if False:
            while True:
                i = 10
        return asyncio.FastChildWatcher()

class PolicyTests(unittest.TestCase):

    def create_policy(self):
        if False:
            i = 10
            return i + 15
        return asyncio.DefaultEventLoopPolicy()

    def test_get_default_child_watcher(self):
        if False:
            for i in range(10):
                print('nop')
        policy = self.create_policy()
        self.assertIsNone(policy._watcher)
        watcher = policy.get_child_watcher()
        self.assertIsInstance(watcher, asyncio.ThreadedChildWatcher)
        self.assertIs(policy._watcher, watcher)
        self.assertIs(watcher, policy.get_child_watcher())

    def test_get_child_watcher_after_set(self):
        if False:
            while True:
                i = 10
        policy = self.create_policy()
        watcher = asyncio.FastChildWatcher()
        policy.set_child_watcher(watcher)
        self.assertIs(policy._watcher, watcher)
        self.assertIs(watcher, policy.get_child_watcher())

    def test_get_child_watcher_thread(self):
        if False:
            i = 10
            return i + 15

        def f():
            if False:
                i = 10
                return i + 15
            policy.set_event_loop(policy.new_event_loop())
            self.assertIsInstance(policy.get_event_loop(), asyncio.AbstractEventLoop)
            watcher = policy.get_child_watcher()
            self.assertIsInstance(watcher, asyncio.SafeChildWatcher)
            self.assertIsNone(watcher._loop)
            policy.get_event_loop().close()
        policy = self.create_policy()
        policy.set_child_watcher(asyncio.SafeChildWatcher())
        th = threading.Thread(target=f)
        th.start()
        th.join()

    def test_child_watcher_replace_mainloop_existing(self):
        if False:
            print('Hello World!')
        policy = self.create_policy()
        loop = policy.get_event_loop()
        watcher = asyncio.SafeChildWatcher()
        policy.set_child_watcher(watcher)
        watcher.attach_loop(loop)
        self.assertIs(watcher._loop, loop)
        new_loop = policy.new_event_loop()
        policy.set_event_loop(new_loop)
        self.assertIs(watcher._loop, new_loop)
        policy.set_event_loop(None)
        self.assertIs(watcher._loop, None)
        loop.close()
        new_loop.close()

class TestFunctional(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        if False:
            print('Hello World!')
        self.loop.close()
        asyncio.set_event_loop(None)

    def test_add_reader_invalid_argument(self):
        if False:
            return 10

        def assert_raises():
            if False:
                print('Hello World!')
            return self.assertRaisesRegex(ValueError, 'Invalid file object')
        cb = lambda : None
        with assert_raises():
            self.loop.add_reader(object(), cb)
        with assert_raises():
            self.loop.add_writer(object(), cb)
        with assert_raises():
            self.loop.remove_reader(object())
        with assert_raises():
            self.loop.remove_writer(object())

    def test_add_reader_or_writer_transport_fd(self):
        if False:
            for i in range(10):
                print('nop')

        def assert_raises():
            if False:
                for i in range(10):
                    print('nop')
            return self.assertRaisesRegex(RuntimeError, 'File descriptor .* is used by transport')

        async def runner():
            (tr, pr) = await self.loop.create_connection(lambda : asyncio.Protocol(), sock=rsock)
            try:
                cb = lambda : None
                with assert_raises():
                    self.loop.add_reader(rsock, cb)
                with assert_raises():
                    self.loop.add_reader(rsock.fileno(), cb)
                with assert_raises():
                    self.loop.remove_reader(rsock)
                with assert_raises():
                    self.loop.remove_reader(rsock.fileno())
                with assert_raises():
                    self.loop.add_writer(rsock, cb)
                with assert_raises():
                    self.loop.add_writer(rsock.fileno(), cb)
                with assert_raises():
                    self.loop.remove_writer(rsock)
                with assert_raises():
                    self.loop.remove_writer(rsock.fileno())
            finally:
                tr.close()
        (rsock, wsock) = socket.socketpair()
        try:
            self.loop.run_until_complete(runner())
        finally:
            rsock.close()
            wsock.close()
if __name__ == '__main__':
    unittest.main()