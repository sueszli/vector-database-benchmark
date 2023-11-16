from __future__ import print_function
from __future__ import absolute_import
from gevent import monkey
monkey.patch_all()
import sys
import array
import socket
import time
import unittest
from functools import wraps
import gevent
from gevent._compat import reraise
import gevent.testing as greentest
from gevent.testing import six
from gevent.testing import LARGE_TIMEOUT
from gevent.testing import support
from gevent.testing import params
from gevent.testing.sockets import tcp_listener
from gevent.testing.skipping import skipWithoutExternalNetwork
from gevent.testing.skipping import skipOnMacOnCI
from threading import Thread as _Thread
from threading import Event
errno_types = int

class BaseThread(object):
    terminal_exc = None

    def __init__(self, target):
        if False:
            return 10

        @wraps(target)
        def errors_are_fatal(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            try:
                return target(*args, **kwargs)
            except:
                self.terminal_exc = sys.exc_info()
                raise
        self.target = errors_are_fatal

class GreenletThread(BaseThread):

    def __init__(self, target=None, args=()):
        if False:
            return 10
        BaseThread.__init__(self, target)
        self.glet = gevent.spawn(self.target, *args)

    def join(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return self.glet.join(*args, **kwargs)

    def is_alive(self):
        if False:
            for i in range(10):
                print('nop')
        return not self.glet.ready()
if not monkey.is_module_patched('threading'):

    class ThreadThread(BaseThread, _Thread):

        def __init__(self, **kwargs):
            if False:
                return 10
            target = kwargs.pop('target')
            BaseThread.__init__(self, target)
            _Thread.__init__(self, target=self.target, **kwargs)
            self.start()
    Thread = ThreadThread
else:
    Thread = GreenletThread

class TestTCP(greentest.TestCase):
    __timeout__ = None
    TIMEOUT_ERROR = socket.timeout
    long_data = ', '.join([str(x) for x in range(20000)])
    if not isinstance(long_data, bytes):
        long_data = long_data.encode('ascii')

    def setUp(self):
        if False:
            return 10
        super(TestTCP, self).setUp()
        if '-v' in sys.argv:
            printed = []
            try:
                from time import perf_counter as now
            except ImportError:
                from time import time as now

            def log(*args):
                if False:
                    while True:
                        i = 10
                if not printed:
                    print()
                    printed.append(1)
                print('\t -> %0.6f' % now(), *args)
            orig_cot = self._close_on_teardown

            def cot(o):
                if False:
                    return 10
                log('Registering for teardown', o)

                def c(o=o):
                    if False:
                        return 10
                    log('Closing on teardown', o)
                    o.close()
                    o = None
                orig_cot(c)
                return o
            self._close_on_teardown = cot
        else:

            def log(*_args):
                if False:
                    for i in range(10):
                        print('nop')
                'Does nothing'
        self.log = log
        self.listener = self._close_on_teardown(self._setup_listener())
        self.port = self.listener.getsockname()[1]

    def _setup_listener(self):
        if False:
            i = 10
            return i + 15
        return tcp_listener()

    def create_connection(self, host=None, port=None, timeout=None, blocking=None):
        if False:
            for i in range(10):
                print('nop')
        sock = self._close_on_teardown(socket.socket())
        sock.connect((host or params.DEFAULT_CONNECT, port or self.port))
        if timeout is not None:
            sock.settimeout(timeout)
        if blocking is not None:
            sock.setblocking(blocking)
        return sock

    def _test_sendall(self, data, match_data=None, client_method='sendall', **client_args):
        if False:
            for i in range(10):
                print('nop')
        log = self.log
        log('test_sendall using method', client_method)
        read_data = []
        accepted_event = Event()

        def accept_and_read():
            if False:
                for i in range(10):
                    print('nop')
            log('\taccepting', self.listener)
            (conn, _) = self.listener.accept()
            try:
                with conn.makefile(mode='rb') as r:
                    log('\taccepted on server; client conn is', conn, 'file is', r)
                    accepted_event.set()
                    log('\treading')
                    read_data.append(r.read())
                    log('\tdone reading', r, 'got bytes', len(read_data[0]))
                del r
            finally:
                conn.close()
                del conn
        server = Thread(target=accept_and_read)
        try:
            log('creating client connection')
            client = self.create_connection(**client_args)
            accepted_event.wait()
            log('Client got accepted event from server', client, '; sending data', len(data))
            try:
                x = getattr(client, client_method)(data)
                log('Client sent data: result from method', x)
            finally:
                log('Client will unwrap and shutdown')
                if hasattr(client, 'unwrap'):
                    try:
                        client = client.unwrap()
                    except (ValueError, OSError):
                        pass
                try:
                    client.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                log('Client will close')
                client.close()
        finally:
            server.join(10)
            assert not server.is_alive()
        if server.terminal_exc:
            reraise(*server.terminal_exc)
        if match_data is None:
            match_data = self.long_data
        read_data = read_data[0].split(b',')
        match_data = match_data.split(b',')
        self.assertEqual(read_data[0], match_data[0])
        self.assertEqual(len(read_data), len(match_data))
        self.assertEqual(read_data, match_data)

    def test_sendall_str(self):
        if False:
            print('Hello World!')
        self._test_sendall(self.long_data)
    if six.PY2:

        def test_sendall_unicode(self):
            if False:
                print('Hello World!')
            self._test_sendall(six.text_type(self.long_data))

    @skipOnMacOnCI('Sometimes fails for no apparent reason (buffering?)')
    def test_sendall_array(self):
        if False:
            while True:
                i = 10
        data = array.array('B', self.long_data)
        self._test_sendall(data)

    def test_sendall_empty(self):
        if False:
            return 10
        data = b''
        self._test_sendall(data, data)

    def test_sendall_empty_with_timeout(self):
        if False:
            print('Hello World!')
        data = b''
        self._test_sendall(data, data, timeout=10)

    def test_sendall_nonblocking(self):
        if False:
            while True:
                i = 10
        data = b'hi\n'
        self._test_sendall(data, data, blocking=False)

    def test_empty_send(self):
        if False:
            return 10
        data = b''
        self._test_sendall(data, data, client_method='send')

    def test_fullduplex(self):
        if False:
            for i in range(10):
                print('nop')
        N = 100000

        def server():
            if False:
                return 10
            (remote_client, _) = self.listener.accept()
            self._close_on_teardown(remote_client)
            sender = Thread(target=remote_client.sendall, args=(b't' * N,))
            try:
                result = remote_client.recv(1000)
                self.assertEqual(result, b'hello world')
            finally:
                sender.join()
        server_thread = Thread(target=server)
        client = self.create_connection()
        client_file = self._close_on_teardown(client.makefile())
        client_reader = Thread(target=client_file.read, args=(N,))
        time.sleep(0.1)
        client.sendall(b'hello world')
        time.sleep(0.1)
        client_file.close()
        client.close()
        server_thread.join()
        client_reader.join()

    def test_recv_timeout(self):
        if False:
            return 10

        def accept():
            if False:
                return 10
            (conn, _) = self.listener.accept()
            self._close_on_teardown(conn)
        acceptor = Thread(target=accept)
        client = self.create_connection()
        try:
            client.settimeout(1)
            start = time.time()
            with self.assertRaises(self.TIMEOUT_ERROR):
                client.recv(1024)
            took = time.time() - start
            self.assertTimeWithinRange(took, 1 - 0.1, 1 + 0.1)
        finally:
            acceptor.join()
    _test_sendall_timeout_check_time = True
    _test_sendall_data = b'hello' * 100000000

    @greentest.skipOnWindows('On Windows send() accepts whatever is thrown at it')
    def test_sendall_timeout(self):
        if False:
            while True:
                i = 10
        client_sock = []
        acceptor = Thread(target=lambda : client_sock.append(self.listener.accept()))
        client = self.create_connection()
        time.sleep(0.1)
        assert client_sock
        client.settimeout(0.1)
        start = time.time()
        try:
            with self.assertRaises(self.TIMEOUT_ERROR):
                client.sendall(self._test_sendall_data)
            if self._test_sendall_timeout_check_time:
                took = time.time() - start
                self.assertTimeWithinRange(took, 0.09, 0.2)
        finally:
            acceptor.join()
            client.close()
            client_sock[0][0].close()

    def test_makefile(self):
        if False:
            while True:
                i = 10

        def accept_once():
            if False:
                return 10
            (conn, _) = self.listener.accept()
            fd = conn.makefile(mode='wb')
            fd.write(b'hello\n')
            fd.flush()
            fd.close()
            conn.close()
        acceptor = Thread(target=accept_once)
        try:
            client = self.create_connection()
            client_file = client.makefile(mode='rb')
            client.close()
            line = client_file.readline()
            self.assertEqual(line, b'hello\n')
            self.assertEqual(client_file.read(), b'')
            client_file.close()
        finally:
            acceptor.join()

    def test_makefile_timeout(self):
        if False:
            return 10

        def accept_once():
            if False:
                return 10
            (conn, _) = self.listener.accept()
            try:
                time.sleep(0.3)
            finally:
                conn.close()
        acceptor = Thread(target=accept_once)
        try:
            client = self.create_connection()
            client.settimeout(0.1)
            fd = client.makefile(mode='rb')
            self.assertRaises(self.TIMEOUT_ERROR, fd.readline)
            client.close()
            fd.close()
        finally:
            acceptor.join()

    def test_attributes(self):
        if False:
            i = 10
            return i + 15
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, 0)
        self.assertIs(s.family, socket.AF_INET)
        self.assertEqual(s.type, socket.SOCK_DGRAM)
        self.assertEqual(0, s.proto)
        if hasattr(socket, 'SOCK_NONBLOCK'):
            s.settimeout(1)
            self.assertIs(s.family, socket.AF_INET)
            s.setblocking(0)
            std_socket = monkey.get_original('socket', 'socket')(socket.AF_INET, socket.SOCK_DGRAM, 0)
            try:
                std_socket.setblocking(0)
                self.assertEqual(std_socket.type, s.type)
            finally:
                std_socket.close()
        s.close()

    def test_connect_ex_nonblocking_bad_connection(self):
        if False:
            i = 10
            return i + 15
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setblocking(False)
            ret = s.connect_ex((greentest.DEFAULT_LOCAL_HOST_ADDR, support.find_unused_port()))
            self.assertIsInstance(ret, errno_types)
        finally:
            s.close()

    @skipWithoutExternalNetwork('Tries to resolve hostname')
    def test_connect_ex_gaierror(self):
        if False:
            for i in range(10):
                print('nop')
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with self.assertRaises(socket.gaierror):
                s.connect_ex(('foo.bar.fizzbuzz', support.find_unused_port()))
        finally:
            s.close()

    @skipWithoutExternalNetwork('Tries to resolve hostname')
    def test_connect_ex_not_call_connect(self):
        if False:
            while True:
                i = 10

        def do_it(sock):
            if False:
                while True:
                    i = 10
            try:
                with self.assertRaises(socket.gaierror):
                    sock.connect_ex(('foo.bar.fizzbuzz', support.find_unused_port()))
            finally:
                sock.close()
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        with self.assertRaises(AttributeError):
            s.connect = None
        s.close()

        class S(socket.socket):

            def connect(self, *args):
                if False:
                    i = 10
                    return i + 15
                raise AssertionError('Should not be called')
        s = S(socket.AF_INET, socket.SOCK_STREAM)
        do_it(s)

    def test_connect_ex_nonblocking_overflow(self):
        if False:
            while True:
                i = 10
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.setblocking(False)
            with self.assertRaises(OverflowError):
                s.connect_ex((greentest.DEFAULT_LOCAL_HOST_ADDR, 65539))
        finally:
            s.close()

    @unittest.skipUnless(hasattr(socket, 'SOCK_CLOEXEC'), 'Requires SOCK_CLOEXEC')
    def test_connect_with_type_flags_ignored(self):
        if False:
            print('Hello World!')
        SOCK_CLOEXEC = socket.SOCK_CLOEXEC
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM | SOCK_CLOEXEC)

        def accept_once():
            if False:
                print('Hello World!')
            (conn, _) = self.listener.accept()
            fd = conn.makefile(mode='wb')
            fd.write(b'hello\n')
            fd.close()
            conn.close()
        acceptor = Thread(target=accept_once)
        try:
            s.connect((params.DEFAULT_CONNECT, self.port))
            fd = s.makefile(mode='rb')
            self.assertEqual(fd.readline(), b'hello\n')
            fd.close()
            s.close()
        finally:
            acceptor.join()

class TestCreateConnection(greentest.TestCase):
    __timeout__ = LARGE_TIMEOUT

    def test_refuses(self, **conn_args):
        if False:
            while True:
                i = 10
        connect_port = support.find_unused_port()
        with self.assertRaisesRegex(socket.error, 'refused|not known|already in use|assign|not available'):
            socket.create_connection((greentest.DEFAULT_BIND_ADDR, connect_port), timeout=30, **conn_args)

    def test_refuses_from_port(self):
        if False:
            while True:
                i = 10
        source_port = support.find_unused_port()
        self.test_refuses(source_address=('', source_port))

    @greentest.ignores_leakcheck
    @skipWithoutExternalNetwork('Tries to resolve hostname')
    def test_base_exception(self):
        if False:
            for i in range(10):
                print('nop')

        class E(BaseException):
            pass

        class MockSocket(object):
            created = ()
            closed = False

            def __init__(self, *_):
                if False:
                    return 10
                MockSocket.created += (self,)

            def connect(self, _):
                if False:
                    i = 10
                    return i + 15
                raise E(_)

            def close(self):
                if False:
                    print('Hello World!')
                self.closed = True

        def mockgetaddrinfo(*_):
            if False:
                return 10
            return [(1, 2, 3, 3, 5)]
        import gevent.socket as gsocket
        self.assertEqual(gsocket.create_connection, socket.create_connection)
        orig_socket = gsocket.socket
        orig_getaddrinfo = gsocket.getaddrinfo
        try:
            gsocket.socket = MockSocket
            gsocket.getaddrinfo = mockgetaddrinfo
            with self.assertRaises(E):
                socket.create_connection(('host', 'port'))
            self.assertEqual(1, len(MockSocket.created))
            self.assertTrue(MockSocket.created[0].closed)
        finally:
            MockSocket.created = ()
            gsocket.socket = orig_socket
            gsocket.getaddrinfo = orig_getaddrinfo

class TestFunctions(greentest.TestCase):

    @greentest.ignores_leakcheck
    def test_wait_timeout(self):
        if False:
            while True:
                i = 10
        from gevent import socket as gsocket

        class io(object):
            callback = None

            def start(self, *_args):
                if False:
                    while True:
                        i = 10
                gevent.sleep(10)
        with self.assertRaises(gsocket.timeout):
            gsocket.wait(io(), timeout=0.01)

    def test_signatures(self):
        if False:
            for i in range(10):
                print('nop')
        exclude = []
        if greentest.PYPY:
            exclude.append('gethostbyname')
            exclude.append('gethostbyname_ex')
            exclude.append('gethostbyaddr')
        if sys.version_info[:2] < (3, 11):
            exclude.append('create_connection')
        self.assertMonkeyPatchedFuncSignatures('socket', exclude=exclude)

    def test_resolve_ipv6_scope_id(self):
        if False:
            while True:
                i = 10
        from gevent import _socketcommon as SC
        if not SC.__socket__.has_ipv6:
            self.skipTest('Needs IPv6')
        if not hasattr(SC.__socket__, 'inet_pton'):
            self.skipTest('Needs inet_pton')
        addr = ('2607:f8b0:4000:80e::200e', 80, 0, 9)

        class sock(object):
            family = SC.AF_INET6
        self.assertIs(addr, SC._resolve_addr(sock, addr))

class TestSocket(greentest.TestCase):

    def test_shutdown_when_closed(self):
        if False:
            while True:
                i = 10
        s = socket.socket()
        s.close()
        with self.assertRaises(socket.error):
            s.shutdown(socket.SHUT_RDWR)

    def test_can_be_weak_ref(self):
        if False:
            print('Hello World!')
        import weakref
        s = socket.socket()
        try:
            w = weakref.ref(s)
            self.assertIsNotNone(w)
        finally:
            s.close()

    def test_has_no_dict(self):
        if False:
            print('Hello World!')
        s = socket.socket()
        try:
            with self.assertRaises(AttributeError):
                getattr(s, '__dict__')
        finally:
            s.close()
if __name__ == '__main__':
    greentest.main()