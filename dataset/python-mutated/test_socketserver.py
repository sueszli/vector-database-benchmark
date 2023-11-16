"""
Test suite for socketserver.
"""
import contextlib
import io
import os
import select
import signal
import socket
import tempfile
import threading
import unittest
import socketserver
import test.support
from test.support import reap_children, verbose
from test.support import os_helper
from test.support import socket_helper
from test.support import threading_helper
test.support.requires('network')
TEST_STR = b'hello world\n'
HOST = socket_helper.HOST
HAVE_UNIX_SOCKETS = hasattr(socket, 'AF_UNIX')
requires_unix_sockets = unittest.skipUnless(HAVE_UNIX_SOCKETS, 'requires Unix sockets')
HAVE_FORKING = hasattr(os, 'fork')
requires_forking = unittest.skipUnless(HAVE_FORKING, 'requires forking')

def signal_alarm(n):
    if False:
        i = 10
        return i + 15
    'Call signal.alarm when it exists (i.e. not on Windows).'
    if hasattr(signal, 'alarm'):
        signal.alarm(n)
_real_select = select.select

def receive(sock, n, timeout=test.support.SHORT_TIMEOUT):
    if False:
        for i in range(10):
            print('nop')
    (r, w, x) = _real_select([sock], [], [], timeout)
    if sock in r:
        return sock.recv(n)
    else:
        raise RuntimeError('timed out on %r' % (sock,))
if HAVE_UNIX_SOCKETS and HAVE_FORKING:

    class ForkingUnixStreamServer(socketserver.ForkingMixIn, socketserver.UnixStreamServer):
        pass

    class ForkingUnixDatagramServer(socketserver.ForkingMixIn, socketserver.UnixDatagramServer):
        pass

@contextlib.contextmanager
def simple_subprocess(testcase):
    if False:
        while True:
            i = 10
    'Tests that a custom child process is not waited on (Issue 1540386)'
    pid = os.fork()
    if pid == 0:
        os._exit(72)
    try:
        yield None
    except:
        raise
    finally:
        test.support.wait_process(pid, exitcode=72)

class SocketServerTest(unittest.TestCase):
    """Test all socket servers."""

    def setUp(self):
        if False:
            return 10
        signal_alarm(60)
        self.port_seed = 0
        self.test_files = []

    def tearDown(self):
        if False:
            while True:
                i = 10
        signal_alarm(0)
        reap_children()
        for fn in self.test_files:
            try:
                os.remove(fn)
            except OSError:
                pass
        self.test_files[:] = []

    def pickaddr(self, proto):
        if False:
            return 10
        if proto == socket.AF_INET:
            return (HOST, 0)
        else:
            dir = None
            fn = tempfile.mktemp(prefix='unix_socket.', dir=dir)
            self.test_files.append(fn)
            return fn

    def make_server(self, addr, svrcls, hdlrbase):
        if False:
            print('Hello World!')

        class MyServer(svrcls):

            def handle_error(self, request, client_address):
                if False:
                    for i in range(10):
                        print('nop')
                self.close_request(request)
                raise

        class MyHandler(hdlrbase):

            def handle(self):
                if False:
                    return 10
                line = self.rfile.readline()
                self.wfile.write(line)
        if verbose:
            print('creating server')
        try:
            server = MyServer(addr, MyHandler)
        except PermissionError as e:
            self.skipTest('Cannot create server (%s, %s): %s' % (svrcls, addr, e))
        self.assertEqual(server.server_address, server.socket.getsockname())
        return server

    @threading_helper.reap_threads
    def run_server(self, svrcls, hdlrbase, testfunc):
        if False:
            for i in range(10):
                print('nop')
        server = self.make_server(self.pickaddr(svrcls.address_family), svrcls, hdlrbase)
        addr = server.server_address
        if verbose:
            print('ADDR =', addr)
            print('CLASS =', svrcls)
        t = threading.Thread(name='%s serving' % svrcls, target=server.serve_forever, kwargs={'poll_interval': 0.01})
        t.daemon = True
        t.start()
        if verbose:
            print('server running')
        for i in range(3):
            if verbose:
                print('test client', i)
            testfunc(svrcls.address_family, addr)
        if verbose:
            print('waiting for server')
        server.shutdown()
        t.join()
        server.server_close()
        self.assertEqual(-1, server.socket.fileno())
        if HAVE_FORKING and isinstance(server, socketserver.ForkingMixIn):
            self.assertFalse(server.active_children)
        if verbose:
            print('done')

    def stream_examine(self, proto, addr):
        if False:
            while True:
                i = 10
        with socket.socket(proto, socket.SOCK_STREAM) as s:
            s.connect(addr)
            s.sendall(TEST_STR)
            buf = data = receive(s, 100)
            while data and b'\n' not in buf:
                data = receive(s, 100)
                buf += data
            self.assertEqual(buf, TEST_STR)

    def dgram_examine(self, proto, addr):
        if False:
            i = 10
            return i + 15
        with socket.socket(proto, socket.SOCK_DGRAM) as s:
            if HAVE_UNIX_SOCKETS and proto == socket.AF_UNIX:
                s.bind(self.pickaddr(proto))
            s.sendto(TEST_STR, addr)
            buf = data = receive(s, 100)
            while data and b'\n' not in buf:
                data = receive(s, 100)
                buf += data
            self.assertEqual(buf, TEST_STR)

    def test_TCPServer(self):
        if False:
            i = 10
            return i + 15
        self.run_server(socketserver.TCPServer, socketserver.StreamRequestHandler, self.stream_examine)

    def test_ThreadingTCPServer(self):
        if False:
            i = 10
            return i + 15
        self.run_server(socketserver.ThreadingTCPServer, socketserver.StreamRequestHandler, self.stream_examine)

    @requires_forking
    def test_ForkingTCPServer(self):
        if False:
            i = 10
            return i + 15
        with simple_subprocess(self):
            self.run_server(socketserver.ForkingTCPServer, socketserver.StreamRequestHandler, self.stream_examine)

    @requires_unix_sockets
    def test_UnixStreamServer(self):
        if False:
            i = 10
            return i + 15
        self.run_server(socketserver.UnixStreamServer, socketserver.StreamRequestHandler, self.stream_examine)

    @requires_unix_sockets
    def test_ThreadingUnixStreamServer(self):
        if False:
            while True:
                i = 10
        self.run_server(socketserver.ThreadingUnixStreamServer, socketserver.StreamRequestHandler, self.stream_examine)

    @requires_unix_sockets
    @requires_forking
    def test_ForkingUnixStreamServer(self):
        if False:
            for i in range(10):
                print('nop')
        with simple_subprocess(self):
            self.run_server(ForkingUnixStreamServer, socketserver.StreamRequestHandler, self.stream_examine)

    def test_UDPServer(self):
        if False:
            print('Hello World!')
        self.run_server(socketserver.UDPServer, socketserver.DatagramRequestHandler, self.dgram_examine)

    def test_ThreadingUDPServer(self):
        if False:
            return 10
        self.run_server(socketserver.ThreadingUDPServer, socketserver.DatagramRequestHandler, self.dgram_examine)

    @requires_forking
    def test_ForkingUDPServer(self):
        if False:
            while True:
                i = 10
        with simple_subprocess(self):
            self.run_server(socketserver.ForkingUDPServer, socketserver.DatagramRequestHandler, self.dgram_examine)

    @requires_unix_sockets
    def test_UnixDatagramServer(self):
        if False:
            print('Hello World!')
        self.run_server(socketserver.UnixDatagramServer, socketserver.DatagramRequestHandler, self.dgram_examine)

    @requires_unix_sockets
    def test_ThreadingUnixDatagramServer(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_server(socketserver.ThreadingUnixDatagramServer, socketserver.DatagramRequestHandler, self.dgram_examine)

    @requires_unix_sockets
    @requires_forking
    def test_ForkingUnixDatagramServer(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_server(ForkingUnixDatagramServer, socketserver.DatagramRequestHandler, self.dgram_examine)

    @threading_helper.reap_threads
    def test_shutdown(self):
        if False:
            i = 10
            return i + 15

        class MyServer(socketserver.TCPServer):
            pass

        class MyHandler(socketserver.StreamRequestHandler):
            pass
        threads = []
        for i in range(20):
            s = MyServer((HOST, 0), MyHandler)
            t = threading.Thread(name='MyServer serving', target=s.serve_forever, kwargs={'poll_interval': 0.01})
            t.daemon = True
            threads.append((t, s))
        for (t, s) in threads:
            t.start()
            s.shutdown()
        for (t, s) in threads:
            t.join()
            s.server_close()

    def test_close_immediately(self):
        if False:
            for i in range(10):
                print('nop')

        class MyServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            pass
        server = MyServer((HOST, 0), lambda : None)
        server.server_close()

    def test_tcpserver_bind_leak(self):
        if False:
            return 10
        for i in range(1024):
            with self.assertRaises(OverflowError):
                socketserver.TCPServer((HOST, -1), socketserver.StreamRequestHandler)

    def test_context_manager(self):
        if False:
            for i in range(10):
                print('nop')
        with socketserver.TCPServer((HOST, 0), socketserver.StreamRequestHandler) as server:
            pass
        self.assertEqual(-1, server.socket.fileno())

class ErrorHandlerTest(unittest.TestCase):
    """Test that the servers pass normal exceptions from the handler to
    handle_error(), and that exiting exceptions like SystemExit and
    KeyboardInterrupt are not passed."""

    def tearDown(self):
        if False:
            while True:
                i = 10
        os_helper.unlink(os_helper.TESTFN)

    def test_sync_handled(self):
        if False:
            for i in range(10):
                print('nop')
        BaseErrorTestServer(ValueError)
        self.check_result(handled=True)

    def test_sync_not_handled(self):
        if False:
            return 10
        with self.assertRaises(SystemExit):
            BaseErrorTestServer(SystemExit)
        self.check_result(handled=False)

    def test_threading_handled(self):
        if False:
            i = 10
            return i + 15
        ThreadingErrorTestServer(ValueError)
        self.check_result(handled=True)

    def test_threading_not_handled(self):
        if False:
            while True:
                i = 10
        with threading_helper.catch_threading_exception() as cm:
            ThreadingErrorTestServer(SystemExit)
            self.check_result(handled=False)
            self.assertIs(cm.exc_type, SystemExit)

    @requires_forking
    def test_forking_handled(self):
        if False:
            print('Hello World!')
        ForkingErrorTestServer(ValueError)
        self.check_result(handled=True)

    @requires_forking
    def test_forking_not_handled(self):
        if False:
            for i in range(10):
                print('nop')
        ForkingErrorTestServer(SystemExit)
        self.check_result(handled=False)

    def check_result(self, handled):
        if False:
            print('Hello World!')
        with open(os_helper.TESTFN) as log:
            expected = 'Handler called\n' + 'Error handled\n' * handled
            self.assertEqual(log.read(), expected)

class BaseErrorTestServer(socketserver.TCPServer):

    def __init__(self, exception):
        if False:
            return 10
        self.exception = exception
        super().__init__((HOST, 0), BadHandler)
        with socket.create_connection(self.server_address):
            pass
        try:
            self.handle_request()
        finally:
            self.server_close()
        self.wait_done()

    def handle_error(self, request, client_address):
        if False:
            i = 10
            return i + 15
        with open(os_helper.TESTFN, 'a') as log:
            log.write('Error handled\n')

    def wait_done(self):
        if False:
            print('Hello World!')
        pass

class BadHandler(socketserver.BaseRequestHandler):

    def handle(self):
        if False:
            i = 10
            return i + 15
        with open(os_helper.TESTFN, 'a') as log:
            log.write('Handler called\n')
        raise self.server.exception('Test error')

class ThreadingErrorTestServer(socketserver.ThreadingMixIn, BaseErrorTestServer):

    def __init__(self, *pos, **kw):
        if False:
            for i in range(10):
                print('nop')
        self.done = threading.Event()
        super().__init__(*pos, **kw)

    def shutdown_request(self, *pos, **kw):
        if False:
            return 10
        super().shutdown_request(*pos, **kw)
        self.done.set()

    def wait_done(self):
        if False:
            while True:
                i = 10
        self.done.wait()
if HAVE_FORKING:

    class ForkingErrorTestServer(socketserver.ForkingMixIn, BaseErrorTestServer):
        pass

class SocketWriterTest(unittest.TestCase):

    def test_basics(self):
        if False:
            i = 10
            return i + 15

        class Handler(socketserver.StreamRequestHandler):

            def handle(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.server.wfile = self.wfile
                self.server.wfile_fileno = self.wfile.fileno()
                self.server.request_fileno = self.request.fileno()
        server = socketserver.TCPServer((HOST, 0), Handler)
        self.addCleanup(server.server_close)
        s = socket.socket(server.address_family, socket.SOCK_STREAM, socket.IPPROTO_TCP)
        with s:
            s.connect(server.server_address)
        server.handle_request()
        self.assertIsInstance(server.wfile, io.BufferedIOBase)
        self.assertEqual(server.wfile_fileno, server.request_fileno)

    def test_write(self):
        if False:
            for i in range(10):
                print('nop')
        pthread_kill = test.support.get_attribute(signal, 'pthread_kill')

        class Handler(socketserver.StreamRequestHandler):

            def handle(self):
                if False:
                    while True:
                        i = 10
                self.server.sent1 = self.wfile.write(b'write data\n')
                self.server.received = self.rfile.readline()
                big_chunk = b'\x00' * test.support.SOCK_MAX_SIZE
                self.server.sent2 = self.wfile.write(big_chunk)
        server = socketserver.TCPServer((HOST, 0), Handler)
        self.addCleanup(server.server_close)
        interrupted = threading.Event()

        def signal_handler(signum, frame):
            if False:
                i = 10
                return i + 15
            interrupted.set()
        original = signal.signal(signal.SIGUSR1, signal_handler)
        self.addCleanup(signal.signal, signal.SIGUSR1, original)
        response1 = None
        received2 = None
        main_thread = threading.get_ident()

        def run_client():
            if False:
                i = 10
                return i + 15
            s = socket.socket(server.address_family, socket.SOCK_STREAM, socket.IPPROTO_TCP)
            with s, s.makefile('rb') as reader:
                s.connect(server.server_address)
                nonlocal response1
                response1 = reader.readline()
                s.sendall(b'client response\n')
                reader.read(100)
                while True:
                    pthread_kill(main_thread, signal.SIGUSR1)
                    if interrupted.wait(timeout=float(1)):
                        break
                nonlocal received2
                received2 = len(reader.read())
        background = threading.Thread(target=run_client)
        background.start()
        server.handle_request()
        background.join()
        self.assertEqual(server.sent1, len(response1))
        self.assertEqual(response1, b'write data\n')
        self.assertEqual(server.received, b'client response\n')
        self.assertEqual(server.sent2, test.support.SOCK_MAX_SIZE)
        self.assertEqual(received2, test.support.SOCK_MAX_SIZE - 100)

class MiscTestCase(unittest.TestCase):

    def test_all(self):
        if False:
            for i in range(10):
                print('nop')
        expected = []
        for name in dir(socketserver):
            if not name.startswith('_'):
                mod_object = getattr(socketserver, name)
                if getattr(mod_object, '__module__', None) == 'socketserver':
                    expected.append(name)
        self.assertCountEqual(socketserver.__all__, expected)

    def test_shutdown_request_called_if_verify_request_false(self):
        if False:
            i = 10
            return i + 15

        class MyServer(socketserver.TCPServer):

            def verify_request(self, request, client_address):
                if False:
                    print('Hello World!')
                return False
            shutdown_called = 0

            def shutdown_request(self, request):
                if False:
                    for i in range(10):
                        print('nop')
                self.shutdown_called += 1
                socketserver.TCPServer.shutdown_request(self, request)
        server = MyServer((HOST, 0), socketserver.StreamRequestHandler)
        s = socket.socket(server.address_family, socket.SOCK_STREAM)
        s.connect(server.server_address)
        s.close()
        server.handle_request()
        self.assertEqual(server.shutdown_called, 1)
        server.server_close()

    def test_threads_reaped(self):
        if False:
            while True:
                i = 10
        '\n        In #37193, users reported a memory leak\n        due to the saving of every request thread. Ensure that\n        not all threads are kept forever.\n        '

        class MyServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            pass
        server = MyServer((HOST, 0), socketserver.StreamRequestHandler)
        for n in range(10):
            with socket.create_connection(server.server_address):
                server.handle_request()
        self.assertLess(len(server._threads), 10)
        server.server_close()
if __name__ == '__main__':
    unittest.main()