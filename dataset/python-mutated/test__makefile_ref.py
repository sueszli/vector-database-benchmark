from __future__ import print_function
import os
from gevent import monkey
monkey.patch_all()
import socket
import ssl
import threading
import errno
import weakref
import gevent.testing as greentest
from gevent.testing.params import DEFAULT_BIND_ADDR_TUPLE
from gevent.testing.params import DEFAULT_CONNECT
from gevent.testing.sockets import tcp_listener
dirname = os.path.dirname(os.path.abspath(__file__))
CERTFILE = os.path.join(dirname, '2_7_keycert.pem')
pid = os.getpid()
PY3 = greentest.PY3
PYPY = greentest.PYPY
CPYTHON = not PYPY
PY2 = not PY3
fd_types = int
if PY3:
    long = int
fd_types = (int, long)
WIN = greentest.WIN
from gevent.testing import get_open_files
try:
    import psutil
except ImportError:
    psutil = None

class Test(greentest.TestCase):
    extra_allowed_open_states = ()

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.extra_allowed_open_states = ()
        super(Test, self).tearDown()

    def assert_raises_EBADF(self, func):
        if False:
            print('Hello World!')
        try:
            result = func()
        except OSError as ex:
            if ex.args[0] == errno.EBADF:
                return
            if WIN and ex.args[0] == 10038:
                return
            raise
        raise AssertionError('NOT RAISED EBADF: %r() returned %r' % (func, result))
    if WIN or (PYPY and greentest.LINUX):

        def __assert_fd_open(self, fileno):
            if False:
                for i in range(10):
                    print('nop')
            pass
    else:

        def __assert_fd_open(self, fileno):
            if False:
                while True:
                    i = 10
            assert isinstance(fileno, fd_types)
            open_files = get_open_files()
            if fileno not in open_files:
                raise AssertionError('%r is not open:\n%s' % (fileno, open_files['data']))

    def assert_fd_closed(self, fileno):
        if False:
            return 10
        assert isinstance(fileno, fd_types), repr(fileno)
        assert fileno > 0, fileno
        open_files = get_open_files(count_closing_as_open=False)
        if fileno in open_files:
            raise AssertionError('%r is not closed:\n%s' % (fileno, open_files['data']))

    def _assert_sock_open(self, sock):
        if False:
            while True:
                i = 10
        open_files = get_open_files()
        sockname = sock.getsockname()
        for x in open_files['data']:
            if getattr(x, 'laddr', None) == sockname:
                assert x.status in (psutil.CONN_LISTEN, psutil.CONN_ESTABLISHED) + self.extra_allowed_open_states, x.status
                return
        raise AssertionError('%r is not open:\n%s' % (sock, open_files['data']))

    def assert_open(self, sock, *rest):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(sock, fd_types):
            self.__assert_fd_open(sock)
        else:
            fileno = sock.fileno()
            assert isinstance(fileno, fd_types), fileno
            sockname = sock.getsockname()
            assert isinstance(sockname, tuple), sockname
            if not WIN:
                self.__assert_fd_open(fileno)
            else:
                self._assert_sock_open(sock)
        if rest:
            self.assert_open(rest[0], *rest[1:])

    def assert_closed(self, sock, *rest):
        if False:
            while True:
                i = 10
        if isinstance(sock, fd_types):
            self.assert_fd_closed(sock)
        else:
            if PY3:
                self.assertEqual(sock.fileno(), -1)
            else:
                self.assert_raises_EBADF(sock.fileno)
            self.assert_raises_EBADF(sock.getsockname)
            self.assert_raises_EBADF(sock.accept)
        if rest:
            self.assert_closed(rest[0], *rest[1:])

    def make_open_socket(self):
        if False:
            return 10
        s = socket.socket()
        try:
            s.bind(DEFAULT_BIND_ADDR_TUPLE)
            if WIN or greentest.LINUX:
                s.listen(1)
            self.assert_open(s, s.fileno())
        except:
            s.close()
            s = None
            raise
        return s

@greentest.skipOnAppVeyor('This sometimes times out for no apparent reason.')
class TestSocket(Test):

    def test_simple_close(self):
        if False:
            while True:
                i = 10
        with Closing() as closer:
            s = closer(self.make_open_socket())
            fileno = s.fileno()
            s.close()
        self.assert_closed(s, fileno)

    def test_makefile1(self):
        if False:
            print('Hello World!')
        with Closing() as closer:
            s = closer(self.make_open_socket())
            fileno = s.fileno()
            f = closer(s.makefile())
            self.assert_open(s, fileno)
            s.close()
            if PY3:
                self.assert_open(s, fileno)
            else:
                self.assert_closed(s)
                self.assert_open(fileno)
            f.close()
            self.assert_closed(s)
            self.assert_closed(fileno)

    def test_makefile2(self):
        if False:
            print('Hello World!')
        with Closing() as closer:
            s = closer(self.make_open_socket())
            fileno = s.fileno()
            self.assert_open(s, fileno)
            f = closer(s.makefile())
            self.assert_open(s)
            self.assert_open(s, fileno)
            f.close()
            self.assert_open(s, fileno)
            s.close()
            self.assert_closed(s, fileno)

    def test_server_simple(self):
        if False:
            return 10
        with Closing() as closer:
            listener = closer(tcp_listener(backlog=1))
            port = listener.getsockname()[1]
            connector = closer(socket.socket())

            def connect():
                if False:
                    while True:
                        i = 10
                connector.connect((DEFAULT_CONNECT, port))
            closer.running_task(threading.Thread(target=connect))
            client_socket = closer.accept(listener)
            fileno = client_socket.fileno()
            self.assert_open(client_socket, fileno)
            client_socket.close()
            self.assert_closed(client_socket)

    def test_server_makefile1(self):
        if False:
            while True:
                i = 10
        with Closing() as closer:
            listener = closer(tcp_listener(backlog=1))
            port = listener.getsockname()[1]
            connector = closer(socket.socket())

            def connect():
                if False:
                    print('Hello World!')
                connector.connect((DEFAULT_CONNECT, port))
            closer.running_task(threading.Thread(target=connect))
            client_socket = closer.accept(listener)
            fileno = client_socket.fileno()
            f = closer(client_socket.makefile())
            self.assert_open(client_socket, fileno)
            client_socket.close()
            if PY3:
                self.assert_open(client_socket, fileno)
            else:
                self.assert_closed(client_socket)
                self.assert_open(fileno)
            f.close()
            self.assert_closed(client_socket, fileno)

    def test_server_makefile2(self):
        if False:
            i = 10
            return i + 15
        with Closing() as closer:
            listener = closer(tcp_listener(backlog=1))
            port = listener.getsockname()[1]
            connector = closer(socket.socket())

            def connect():
                if False:
                    for i in range(10):
                        print('nop')
                connector.connect((DEFAULT_CONNECT, port))
            closer.running_task(threading.Thread(target=connect))
            client_socket = closer.accept(listener)
            fileno = client_socket.fileno()
            f = closer(client_socket.makefile())
            self.assert_open(client_socket, fileno)
            f.close()
            self.assert_open(client_socket, fileno)
            client_socket.close()
            self.assert_closed(client_socket, fileno)

@greentest.skipOnAppVeyor('This sometimes times out for no apparent reason.')
class TestSSL(Test):

    def _ssl_connect_task(self, connector, port, accepted_event):
        if False:
            for i in range(10):
                print('nop')
        connector.connect((DEFAULT_CONNECT, port))
        try:
            x = ssl.SSLContext().wrap_socket(connector)
            accepted_event.wait()
        except socket.error:
            pass
        else:
            x.close()

    def _make_ssl_connect_task(self, connector, port):
        if False:
            i = 10
            return i + 15
        accepted_event = threading.Event()
        t = threading.Thread(target=self._ssl_connect_task, args=(connector, port, accepted_event))
        t.daemon = True
        t.accepted_event = accepted_event
        return t

    def test_simple_close(self):
        if False:
            i = 10
            return i + 15
        with Closing() as closer:
            s = closer(self.make_open_socket())
            fileno = s.fileno()
            s = closer(ssl.SSLContext().wrap_socket(s))
            fileno = s.fileno()
            self.assert_open(s, fileno)
            s.close()
            self.assert_closed(s, fileno)

    def test_makefile1(self):
        if False:
            print('Hello World!')
        with Closing() as closer:
            raw_s = closer(self.make_open_socket())
            s = closer(ssl.SSLContext().wrap_socket(raw_s))
            fileno = s.fileno()
            self.assert_open(s, fileno)
            f = closer(s.makefile())
            self.assert_open(s, fileno)
            s.close()
            self.assert_open(s, fileno)
            f.close()
            raw_s.close()
            self.assert_closed(s, fileno)

    def test_makefile2(self):
        if False:
            for i in range(10):
                print('nop')
        with Closing() as closer:
            s = closer(self.make_open_socket())
            fileno = s.fileno()
            s = closer(ssl.SSLContext().wrap_socket(s))
            fileno = s.fileno()
            self.assert_open(s, fileno)
            f = closer(s.makefile())
            self.assert_open(s, fileno)
            f.close()
            self.assert_open(s, fileno)
            s.close()
            self.assert_closed(s, fileno)

    def _wrap_socket(self, sock, *, keyfile, certfile, server_side=False):
        if False:
            while True:
                i = 10
        context = ssl.SSLContext()
        context.load_cert_chain(certfile=certfile, keyfile=keyfile)
        return context.wrap_socket(sock, server_side=server_side)

    def test_server_simple(self):
        if False:
            while True:
                i = 10
        with Closing() as closer:
            listener = closer(tcp_listener(backlog=1))
            port = listener.getsockname()[1]
            connector = closer(socket.socket())
            t = self._make_ssl_connect_task(connector, port)
            closer.running_task(t)
            client_socket = closer.accept(listener)
            t.accepted_event.set()
            client_socket = closer(self._wrap_socket(client_socket, keyfile=CERTFILE, certfile=CERTFILE, server_side=True))
            fileno = client_socket.fileno()
            self.assert_open(client_socket, fileno)
            client_socket.close()
            self.assert_closed(client_socket, fileno)

    def test_server_makefile1(self):
        if False:
            print('Hello World!')
        with Closing() as closer:
            listener = closer(tcp_listener(backlog=1))
            port = listener.getsockname()[1]
            connector = closer(socket.socket())
            t = self._make_ssl_connect_task(connector, port)
            closer.running_task(t)
            client_socket = closer.accept(listener)
            t.accepted_event.set()
            client_socket = closer(self._wrap_socket(client_socket, keyfile=CERTFILE, certfile=CERTFILE, server_side=True))
            fileno = client_socket.fileno()
            self.assert_open(client_socket, fileno)
            f = client_socket.makefile()
            self.assert_open(client_socket, fileno)
            client_socket.close()
            self.assert_open(client_socket, fileno)
            f.close()
            self.assert_closed(client_socket, fileno)

    def test_server_makefile2(self):
        if False:
            print('Hello World!')
        with Closing() as closer:
            listener = closer(tcp_listener(backlog=1))
            port = listener.getsockname()[1]
            connector = closer(socket.socket())
            t = self._make_ssl_connect_task(connector, port)
            closer.running_task(t)
            t.accepted_event.set()
            client_socket = closer.accept(listener)
            client_socket = closer(self._wrap_socket(client_socket, keyfile=CERTFILE, certfile=CERTFILE, server_side=True))
            fileno = client_socket.fileno()
            self.assert_open(client_socket, fileno)
            f = client_socket.makefile()
            self.assert_open(client_socket, fileno)
            f.close()
            self.assert_open(client_socket, fileno)
            client_socket.close()
            self.assert_closed(client_socket, fileno)

    def test_serverssl_makefile1(self):
        if False:
            return 10
        raw_listener = tcp_listener(backlog=1)
        fileno = raw_listener.fileno()
        port = raw_listener.getsockname()[1]
        listener = self._wrap_socket(raw_listener, keyfile=CERTFILE, certfile=CERTFILE)
        connector = socket.socket()
        t = self._make_ssl_connect_task(connector, port)
        t.start()
        with CleaningUp(t, listener, raw_listener, connector) as client_socket:
            t.accepted_event.set()
            fileno = client_socket.fileno()
            self.assert_open(client_socket, fileno)
            f = client_socket.makefile()
            self.assert_open(client_socket, fileno)
            client_socket.close()
            self.assert_open(client_socket, fileno)
            f.close()
            self.assert_closed(client_socket, fileno)

    def test_serverssl_makefile2(self):
        if False:
            for i in range(10):
                print('nop')
        raw_listener = tcp_listener(backlog=1)
        port = raw_listener.getsockname()[1]
        listener = self._wrap_socket(raw_listener, keyfile=CERTFILE, certfile=CERTFILE)
        accepted_event = threading.Event()

        def connect(connector=socket.socket()):
            if False:
                while True:
                    i = 10
            try:
                connector.connect((DEFAULT_CONNECT, port))
                s = ssl.SSLContext().wrap_socket(connector)
                accepted_event.wait()
                s.sendall(b'test_serverssl_makefile2')
                s.shutdown(socket.SHUT_RDWR)
                s.close()
            finally:
                connector.close()
        t = threading.Thread(target=connect)
        t.daemon = True
        t.start()
        client_socket = None
        with CleaningUp(t, listener, raw_listener) as client_socket:
            accepted_event.set()
            fileno = client_socket.fileno()
            self.assert_open(client_socket, fileno)
            f = client_socket.makefile()
            self.assert_open(client_socket, fileno)
            self.assertEqual(f.read(), 'test_serverssl_makefile2')
            self.assertEqual(f.read(), '')
            f.close()
            if WIN and psutil:
                self.extra_allowed_open_states = (psutil.CONN_CLOSE_WAIT,)
            self.assert_open(client_socket, fileno)
            client_socket.close()
            self.assert_closed(client_socket, fileno)

class Closing(object):

    def __init__(self, *init):
        if False:
            while True:
                i = 10
        self._objects = []
        for i in init:
            self.closing(i)
        self.task = None

    def accept(self, listener):
        if False:
            print('Hello World!')
        (client_socket, _addr) = listener.accept()
        return self.closing(client_socket)

    def __enter__(self):
        if False:
            while True:
                i = 10
        o = self.objects()
        if len(o) == 1:
            return o[0]
        return self
    if PY2 and CPYTHON:

        def closing(self, o):
            if False:
                return 10
            self._objects.append(weakref.ref(o))
            return o

        def objects(self):
            if False:
                return 10
            return [r() for r in self._objects if r() is not None]
    else:

        def objects(self):
            if False:
                print('Hello World!')
            return list(reversed(self._objects))

        def closing(self, o):
            if False:
                i = 10
                return i + 15
            self._objects.append(o)
            return o
    __call__ = closing

    def running_task(self, thread):
        if False:
            i = 10
            return i + 15
        assert self.task is None
        self.task = thread
        self.task.start()
        return self.task

    def __exit__(self, t, v, tb):
        if False:
            print('Hello World!')
        try:
            if self.task is not None:
                self.task.join()
        finally:
            self.task = None
            for o in self.objects():
                try:
                    o.close()
                except Exception:
                    pass
        self._objects = ()

class CleaningUp(Closing):

    def __init__(self, task, listener, *other_sockets):
        if False:
            i = 10
            return i + 15
        super(CleaningUp, self).__init__(listener, *other_sockets)
        self.task = task
        self.listener = listener

    def __enter__(self):
        if False:
            return 10
        return self.accept(self.listener)

    def __exit__(self, t, v, tb):
        if False:
            for i in range(10):
                print('nop')
        try:
            Closing.__exit__(self, t, v, tb)
        finally:
            self.listener = None
if __name__ == '__main__':
    greentest.main()