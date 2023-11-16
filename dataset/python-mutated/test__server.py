from __future__ import print_function, division
from contextlib import contextmanager
import unittest
import errno
import os
import gevent.testing as greentest
from gevent.testing import PY3
from gevent.testing import sysinfo
from gevent.testing import DEFAULT_SOCKET_TIMEOUT as _DEFAULT_SOCKET_TIMEOUT
from gevent.testing.timing import SMALLEST_RELIABLE_DELAY
from gevent.testing.sockets import tcp_listener
from gevent.testing import WIN
from gevent import socket
import gevent
from gevent.server import StreamServer
from gevent.exceptions import LoopExit

class SimpleStreamServer(StreamServer):

    def handle(self, client_socket, _address):
        if False:
            i = 10
            return i + 15
        fd = client_socket.makefile()
        try:
            request_line = fd.readline()
            if not request_line:
                return
            try:
                (_method, path, _rest) = request_line.split(' ', 3)
            except Exception:
                print('Failed to parse request line: %r' % (request_line,))
                raise
            if path == '/ping':
                client_socket.sendall(b'HTTP/1.0 200 OK\r\n\r\nPONG')
            elif path in ['/long', '/short']:
                client_socket.sendall(b'hello')
                while True:
                    data = client_socket.recv(1)
                    if not data:
                        break
            else:
                client_socket.sendall(b'HTTP/1.0 404 WTF?\r\n\r\n')
        finally:
            fd.close()

def sleep_to_clear_old_sockets(*_args):
    if False:
        print('Hello World!')
    try:
        gevent.sleep(0 if not WIN else SMALLEST_RELIABLE_DELAY)
    except Exception:
        pass

class Settings(object):
    ServerClass = StreamServer
    ServerSubClass = SimpleStreamServer
    restartable = True
    close_socket_detected = True

    @staticmethod
    def assertAcceptedConnectionError(inst):
        if False:
            return 10
        with inst.makefile() as conn:
            try:
                result = conn.read()
            except socket.timeout:
                result = None
        inst.assertFalse(result)
    assert500 = assertAcceptedConnectionError

    @staticmethod
    def assert503(inst):
        if False:
            i = 10
            return i + 15
        inst.assert500()
        try:
            inst.send_request()
        except socket.error as ex:
            if ex.args[0] not in greentest.CONN_ABORTED_ERRORS:
                raise

    @staticmethod
    def assertPoolFull(inst):
        if False:
            for i in range(10):
                print('nop')
        with inst.assertRaises(socket.timeout):
            inst.assertRequestSucceeded(timeout=0.01)

    @staticmethod
    def fill_default_server_args(inst, kwargs):
        if False:
            i = 10
            return i + 15
        kwargs.setdefault('spawn', inst.get_spawn())
        return kwargs

class TestCase(greentest.TestCase):
    __timeout__ = greentest.LARGE_TIMEOUT
    Settings = Settings
    server = None

    def cleanup(self):
        if False:
            while True:
                i = 10
        if getattr(self, 'server', None) is not None:
            self.server.stop()
            self.server = None
        sleep_to_clear_old_sockets()

    def get_listener(self):
        if False:
            for i in range(10):
                print('nop')
        return self._close_on_teardown(tcp_listener(backlog=5))

    def get_server_host_port_family(self):
        if False:
            print('Hello World!')
        server_host = self.server.server_host
        if not server_host:
            server_host = greentest.DEFAULT_LOCAL_HOST_ADDR
        elif server_host == '::':
            server_host = greentest.DEFAULT_LOCAL_HOST_ADDR6
        try:
            family = self.server.socket.family
        except AttributeError:
            family = socket.AF_INET
        return (server_host, self.server.server_port, family)

    @contextmanager
    def makefile(self, timeout=_DEFAULT_SOCKET_TIMEOUT, bufsize=1, include_raw_socket=False):
        if False:
            for i in range(10):
                print('nop')
        (server_host, server_port, family) = self.get_server_host_port_family()
        bufarg = 'buffering' if PY3 else 'bufsize'
        makefile_kwargs = {bufarg: bufsize}
        if PY3:
            makefile_kwargs['mode'] = 'rwb'
        with socket.socket(family=family) as sock:
            rconn = None
            sock.connect((server_host, server_port))
            sock.settimeout(timeout)
            with sock.makefile(**makefile_kwargs) as rconn:
                result = rconn if not include_raw_socket else (rconn, sock)
                yield result

    def send_request(self, url='/', timeout=_DEFAULT_SOCKET_TIMEOUT, bufsize=1):
        if False:
            i = 10
            return i + 15
        with self.makefile(timeout=timeout, bufsize=bufsize) as conn:
            self.send_request_to_fd(conn, url)

    def send_request_to_fd(self, fd, url='/'):
        if False:
            return 10
        fd.write(('GET %s HTTP/1.0\r\n\r\n' % url).encode('latin-1'))
        fd.flush()
    LOCAL_CONN_REFUSED_ERRORS = ()
    if greentest.OSX:
        LOCAL_CONN_REFUSED_ERRORS = (errno.EPROTOTYPE,)
    elif greentest.WIN and greentest.PYPY3:
        LOCAL_CONN_REFUSED_ERRORS = (10049,)

    def assertConnectionRefused(self, in_proc_server=True):
        if False:
            for i in range(10):
                print('nop')
        try:
            with self.assertRaises(socket.error) as exc:
                with self.makefile() as conn:
                    conn.close()
        except LoopExit:
            if not in_proc_server:
                raise
            return
        ex = exc.exception
        self.assertIn(ex.args[0], (errno.ECONNREFUSED, errno.EADDRNOTAVAIL, errno.ECONNRESET, errno.ECONNABORTED) + self.LOCAL_CONN_REFUSED_ERRORS, (ex, ex.args))

    def assert500(self):
        if False:
            while True:
                i = 10
        self.Settings.assert500(self)

    def assert503(self):
        if False:
            for i in range(10):
                print('nop')
        self.Settings.assert503(self)

    def assertAcceptedConnectionError(self):
        if False:
            while True:
                i = 10
        self.Settings.assertAcceptedConnectionError(self)

    def assertPoolFull(self):
        if False:
            for i in range(10):
                print('nop')
        self.Settings.assertPoolFull(self)

    def assertNotAccepted(self):
        if False:
            while True:
                i = 10
        try:
            with self.makefile(include_raw_socket=True) as (conn, sock):
                conn.write(b'GET / HTTP/1.0\r\n\r\n')
                conn.flush()
                result = b''
                try:
                    while True:
                        data = sock.recv(1)
                        if not data:
                            break
                        result += data
                except socket.timeout:
                    self.assertFalse(result)
                    return
        except LoopExit:
            return
        self.assertTrue(result.startswith(b'HTTP/1.0 500 Internal Server Error'), repr(result))

    def assertRequestSucceeded(self, timeout=_DEFAULT_SOCKET_TIMEOUT):
        if False:
            for i in range(10):
                print('nop')
        with self.makefile(timeout=timeout) as conn:
            conn.write(b'GET /ping HTTP/1.0\r\n\r\n')
            result = conn.read()
        self.assertTrue(result.endswith(b'\r\n\r\nPONG'), repr(result))

    def start_server(self):
        if False:
            return 10
        self.server.start()
        self.assertRequestSucceeded()
        self.assertRequestSucceeded()

    def stop_server(self):
        if False:
            print('Hello World!')
        self.server.stop()
        self.assertConnectionRefused()

    def report_netstat(self, _msg):
        if False:
            i = 10
            return i + 15
        return

    def _create_server(self, *args, **kwargs):
        if False:
            print('Hello World!')
        kind = kwargs.pop('server_kind', self.ServerSubClass)
        addr = kwargs.pop('server_listen_addr', (greentest.DEFAULT_BIND_ADDR, 0))
        return kind(addr, *args, **kwargs)

    def init_server(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.server = self._create_server(*args, **kwargs)
        self.server.start()
        sleep_to_clear_old_sockets()

    @property
    def socket(self):
        if False:
            while True:
                i = 10
        return self.server.socket

    def _test_invalid_callback(self):
        if False:
            return 10
        if sysinfo.RUNNING_ON_APPVEYOR:
            self.skipTest('Sometimes misses the error')
        try:
            self.init_server(lambda : None)
            self.expect_one_error()
            self.assert500()
            self.assert_error(TypeError)
        finally:
            self.server.stop()
            import gc
            gc.collect()

    def fill_default_server_args(self, kwargs):
        if False:
            print('Hello World!')
        return self.Settings.fill_default_server_args(self, kwargs)

    def ServerClass(self, *args, **kwargs):
        if False:
            return 10
        return self.Settings.ServerClass(*args, **self.fill_default_server_args(kwargs))

    def ServerSubClass(self, *args, **kwargs):
        if False:
            return 10
        return self.Settings.ServerSubClass(*args, **self.fill_default_server_args(kwargs))

    def get_spawn(self):
        if False:
            for i in range(10):
                print('nop')
        return None

class TestDefaultSpawn(TestCase):

    def get_spawn(self):
        if False:
            i = 10
            return i + 15
        return gevent.spawn

    def _test_server_start_stop(self, restartable):
        if False:
            return 10
        self.report_netstat('before start')
        self.start_server()
        self.report_netstat('after start')
        if restartable and self.Settings.restartable:
            self.server.stop_accepting()
            self.report_netstat('after stop_accepting')
            self.assertNotAccepted()
            self.server.start_accepting()
            self.report_netstat('after start_accepting')
            sleep_to_clear_old_sockets()
            self.assertRequestSucceeded()
        self.stop_server()
        self.report_netstat('after stop')

    def test_backlog_is_not_accepted_for_socket(self):
        if False:
            i = 10
            return i + 15
        self.switch_expected = False
        with self.assertRaises(TypeError):
            self.ServerClass(self.get_listener(), backlog=25)

    @greentest.skipOnLibuvOnCIOnPyPy('Sometimes times out')
    @greentest.skipOnAppVeyor('Sometimes times out.')
    def test_backlog_is_accepted_for_address(self):
        if False:
            return 10
        self.server = self.ServerSubClass((greentest.DEFAULT_BIND_ADDR, 0), backlog=25)
        self.assertConnectionRefused()
        self._test_server_start_stop(restartable=False)

    def test_subclass_just_create(self):
        if False:
            i = 10
            return i + 15
        self.server = self.ServerSubClass(self.get_listener())
        self.assertNotAccepted()

    @greentest.skipOnAppVeyor('Sometimes times out.')
    def test_subclass_with_socket(self):
        if False:
            for i in range(10):
                print('nop')
        self.server = self.ServerSubClass(self.get_listener())
        self.assertNotAccepted()
        self._test_server_start_stop(restartable=True)

    def test_subclass_with_address(self):
        if False:
            while True:
                i = 10
        self.server = self.ServerSubClass((greentest.DEFAULT_BIND_ADDR, 0))
        self.assertConnectionRefused()
        self._test_server_start_stop(restartable=True)

    def test_invalid_callback(self):
        if False:
            i = 10
            return i + 15
        self._test_invalid_callback()

    @greentest.reraises_flaky_timeout(socket.timeout)
    def _test_serve_forever(self):
        if False:
            for i in range(10):
                print('nop')
        g = gevent.spawn(self.server.serve_forever)
        try:
            sleep_to_clear_old_sockets()
            self.assertRequestSucceeded()
            self.server.stop()
            self.assertFalse(self.server.started)
            self.assertConnectionRefused()
        finally:
            g.kill()
            g.get()
            self.server.stop()

    def test_serve_forever(self):
        if False:
            while True:
                i = 10
        self.server = self.ServerSubClass((greentest.DEFAULT_BIND_ADDR, 0))
        self.assertFalse(self.server.started)
        self.assertConnectionRefused()
        self._test_serve_forever()

    def test_serve_forever_after_start(self):
        if False:
            i = 10
            return i + 15
        self.server = self.ServerSubClass((greentest.DEFAULT_BIND_ADDR, 0))
        self.assertConnectionRefused()
        self.assertFalse(self.server.started)
        self.server.start()
        self.assertTrue(self.server.started)
        self._test_serve_forever()

    @greentest.skipIf(greentest.EXPECT_POOR_TIMER_RESOLUTION, 'Sometimes spuriously fails')
    def test_server_closes_client_sockets(self):
        if False:
            print('Hello World!')
        self.server = self.ServerClass((greentest.DEFAULT_BIND_ADDR, 0), lambda *args: [])
        self.server.start()
        sleep_to_clear_old_sockets()
        with self.makefile() as conn:
            self.send_request_to_fd(conn)
            with gevent.Timeout._start_new_or_dummy(1):
                try:
                    result = conn.read()
                    if result:
                        assert result.startswith('HTTP/1.0 500 Internal Server Error'), repr(result)
                except socket.timeout:
                    pass
                except socket.error as ex:
                    if ex.args[0] == 10053:
                        pass
                    elif ex.args[0] == errno.ECONNRESET:
                        pass
                    else:
                        raise
        self.stop_server()

    @property
    def socket(self):
        if False:
            for i in range(10):
                print('nop')
        return self.server.socket

    def test_error_in_spawn(self):
        if False:
            return 10
        self.init_server()
        self.assertTrue(self.server.started)
        error = ExpectedError('test_error_in_spawn')

        def _spawn(*_args):
            if False:
                print('Hello World!')
            gevent.getcurrent().throw(error)
        self.server._spawn = _spawn
        self.expect_one_error()
        self.assertAcceptedConnectionError()
        self.assert_error(ExpectedError, error)

    def test_server_repr_when_handle_is_instancemethod(self):
        if False:
            return 10
        self.init_server()
        assert self.server.started
        self.assertIn('Server', repr(self.server))
        self.server.set_handle(self.server.handle)
        self.assertIn('handle=<bound method', repr(self.server))
        self.assertIn('of self>', repr(self.server))
        self.server.set_handle(self.test_server_repr_when_handle_is_instancemethod)
        self.assertIn('test_server_repr_when_handle_is_instancemethod', repr(self.server))

        def handle():
            if False:
                print('Hello World!')
            pass
        self.server.set_handle(handle)
        self.assertIn('handle=<function', repr(self.server))

class TestRawSpawn(TestDefaultSpawn):

    def get_spawn(self):
        if False:
            print('Hello World!')
        return gevent.spawn_raw

class TestPoolSpawn(TestDefaultSpawn):

    def get_spawn(self):
        if False:
            return 10
        return 2

    @greentest.skipIf(greentest.EXPECT_POOR_TIMER_RESOLUTION, 'If we have bad timer resolution and hence increase timeouts, it can be hard to sleep for a correct amount of time that lets requests in the pool be full.')
    def test_pool_full(self):
        if False:
            while True:
                i = 10
        self.init_server()
        with self.makefile() as long_request:
            with self.makefile() as short_request:
                self.send_request_to_fd(short_request, '/short')
                self.send_request_to_fd(long_request, '/long')
                gevent.get_hub().loop.update_now()
                gevent.sleep(_DEFAULT_SOCKET_TIMEOUT / 10.0)
                self.assertPoolFull()
                self.assertPoolFull()
                self.assertPoolFull()
        gevent.sleep(_DEFAULT_SOCKET_TIMEOUT)
        try:
            self.assertRequestSucceeded()
        except socket.timeout:
            greentest.reraiseFlakyTestTimeout()
    test_pool_full.error_fatal = False

class TestNoneSpawn(TestCase):

    def get_spawn(self):
        if False:
            print('Hello World!')
        return None

    def test_invalid_callback(self):
        if False:
            while True:
                i = 10
        self._test_invalid_callback()

    @greentest.skipOnAppVeyor("Sometimes doesn't get the error.")
    def test_assertion_in_blocking_func(self):
        if False:
            i = 10
            return i + 15

        def sleep(*_args):
            if False:
                return 10
            gevent.sleep(SMALLEST_RELIABLE_DELAY)
        self.init_server(sleep, server_kind=self.ServerSubClass, spawn=None)
        self.expect_one_error()
        self.assert500()
        self.assert_error(AssertionError, 'Impossible to call blocking function in the event loop callback')

class ExpectedError(Exception):
    pass

class TestSSLSocketNotAllowed(TestCase):
    switch_expected = False

    def get_spawn(self):
        if False:
            i = 10
            return i + 15
        return gevent.spawn

    @unittest.skipUnless(hasattr(socket, 'ssl'), 'Uses socket.ssl')
    def test(self):
        if False:
            for i in range(10):
                print('nop')
        from gevent.socket import ssl
        listener = self._close_on_teardown(tcp_listener(backlog=5))
        listener = ssl(listener)
        self.assertRaises(TypeError, self.ServerSubClass, listener)

def _file(name, here=os.path.dirname(__file__)):
    if False:
        i = 10
        return i + 15
    return os.path.abspath(os.path.join(here, name))

class BadWrapException(BaseException):
    pass

class TestSSLGetCertificate(TestCase):

    def _create_server(self):
        if False:
            for i in range(10):
                print('nop')
        return self.ServerSubClass((greentest.DEFAULT_BIND_ADDR, 0), keyfile=_file('server.key'), certfile=_file('server.crt'))

    def get_spawn(self):
        if False:
            i = 10
            return i + 15
        return gevent.spawn

    def test_certificate(self):
        if False:
            return 10
        from gevent import monkey, ssl
        self.assertFalse(monkey.is_module_patched('ssl'))
        self.assertFalse(monkey.is_module_patched('socket'))
        self.init_server()
        (server_host, server_port, _family) = self.get_server_host_port_family()
        ssl.get_server_certificate((server_host, server_port))

    def test_wrap_socket_and_handle_wrap_failure(self):
        if False:
            print('Hello World!')
        self.init_server()

        def bad_wrap(_client_socket, **_wrap_args):
            if False:
                i = 10
                return i + 15
            raise BadWrapException()
        self.server.wrap_socket = bad_wrap
        with self.assertRaises(BadWrapException):
            self.server._handle(None, None)
if __name__ == '__main__':
    greentest.main()