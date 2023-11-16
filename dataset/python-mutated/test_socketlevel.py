from __future__ import annotations
import contextlib
import errno
import io
import os
import os.path
import select
import shutil
import socket
import ssl
import tempfile
import time
import typing
import zlib
from collections import OrderedDict
from pathlib import Path
from test import LONG_TIMEOUT, SHORT_TIMEOUT, notWindows, resolvesLocalhostFQDN
from threading import Event
from unittest import mock
import pytest
import trustme
from dummyserver.testcase import SocketDummyServerTestCase, consume_socket
from dummyserver.tornadoserver import DEFAULT_CA, DEFAULT_CERTS, encrypt_key_pem, get_unreachable_address
from urllib3 import HTTPConnectionPool, HTTPSConnectionPool, ProxyManager, util
from urllib3._collections import HTTPHeaderDict
from urllib3.connection import HTTPConnection, _get_default_user_agent
from urllib3.connectionpool import _url_from_pool
from urllib3.exceptions import InsecureRequestWarning, MaxRetryError, ProtocolError, ProxyError, ReadTimeoutError, SSLError
from urllib3.poolmanager import proxy_from_url
from urllib3.util import ssl_, ssl_wrap_socket
from urllib3.util.retry import Retry
from urllib3.util.timeout import Timeout
from .. import LogRecorder, has_alpn
if typing.TYPE_CHECKING:
    from _typeshed import StrOrBytesPath
else:
    StrOrBytesPath = object

class TestCookies(SocketDummyServerTestCase):

    def test_multi_setcookie(self) -> None:
        if False:
            return 10

        def multicookie_response_handler(listener: socket.socket) -> None:
            if False:
                print('Hello World!')
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            sock.send(b'HTTP/1.1 200 OK\r\nSet-Cookie: foo=1\r\nSet-Cookie: bar=1\r\n\r\n')
            sock.close()
        self._start_server(multicookie_response_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('GET', '/', retries=0)
            assert r.headers == {'set-cookie': 'foo=1, bar=1'}
            assert r.headers.getlist('set-cookie') == ['foo=1', 'bar=1']

class TestSNI(SocketDummyServerTestCase):

    def test_hostname_in_first_request_packet(self) -> None:
        if False:
            i = 10
            return i + 15
        done_receiving = Event()
        self.buf = b''

        def socket_handler(listener: socket.socket) -> None:
            if False:
                while True:
                    i = 10
            sock = listener.accept()[0]
            self.buf = sock.recv(65536)
            done_receiving.set()
            sock.close()
        self._start_server(socket_handler)
        with HTTPSConnectionPool(self.host, self.port) as pool:
            try:
                pool.request('GET', '/', retries=0)
            except MaxRetryError:
                pass
            successful = done_receiving.wait(LONG_TIMEOUT)
            assert successful, 'Timed out waiting for connection accept'
            assert self.host.encode('ascii') in self.buf, 'missing hostname in SSL handshake'

class TestALPN(SocketDummyServerTestCase):

    def test_alpn_protocol_in_first_request_packet(self) -> None:
        if False:
            print('Hello World!')
        if not has_alpn():
            pytest.skip('ALPN-support not available')
        done_receiving = Event()
        self.buf = b''

        def socket_handler(listener: socket.socket) -> None:
            if False:
                return 10
            sock = listener.accept()[0]
            self.buf = sock.recv(65536)
            done_receiving.set()
            sock.close()
        self._start_server(socket_handler)
        with HTTPSConnectionPool(self.host, self.port) as pool:
            try:
                pool.request('GET', '/', retries=0)
            except MaxRetryError:
                pass
            successful = done_receiving.wait(LONG_TIMEOUT)
            assert successful, 'Timed out waiting for connection accept'
            for protocol in util.ALPN_PROTOCOLS:
                assert protocol.encode('ascii') in self.buf, 'missing ALPN protocol in SSL handshake'

def original_ssl_wrap_socket(sock: socket.socket, keyfile: StrOrBytesPath | None=None, certfile: StrOrBytesPath | None=None, server_side: bool=False, cert_reqs: ssl.VerifyMode=ssl.CERT_NONE, ssl_version: int=ssl.PROTOCOL_TLS, ca_certs: str | None=None, do_handshake_on_connect: bool=True, suppress_ragged_eofs: bool=True, ciphers: str | None=None) -> ssl.SSLSocket:
    if False:
        while True:
            i = 10
    if server_side and (not certfile):
        raise ValueError('certfile must be specified for server-side operations')
    if keyfile and (not certfile):
        raise ValueError('certfile must be specified')
    context = ssl.SSLContext(ssl_version)
    context.verify_mode = cert_reqs
    if ca_certs:
        context.load_verify_locations(ca_certs)
    if certfile:
        context.load_cert_chain(certfile, keyfile)
    if ciphers:
        context.set_ciphers(ciphers)
    return context.wrap_socket(sock=sock, server_side=server_side, do_handshake_on_connect=do_handshake_on_connect, suppress_ragged_eofs=suppress_ragged_eofs)

class TestClientCerts(SocketDummyServerTestCase):
    """
    Tests for client certificate support.
    """

    @classmethod
    def setup_class(cls) -> None:
        if False:
            while True:
                i = 10
        cls.tmpdir = tempfile.mkdtemp()
        ca = trustme.CA()
        cert = ca.issue_cert('localhost')
        encrypted_key = encrypt_key_pem(cert.private_key_pem, b'letmein')
        cls.ca_path = os.path.join(cls.tmpdir, 'ca.pem')
        cls.cert_combined_path = os.path.join(cls.tmpdir, 'server.combined.pem')
        cls.cert_path = os.path.join(cls.tmpdir, 'server.pem')
        cls.key_path = os.path.join(cls.tmpdir, 'key.pem')
        cls.password_key_path = os.path.join(cls.tmpdir, 'password_key.pem')
        ca.cert_pem.write_to_path(cls.ca_path)
        cert.private_key_and_cert_chain_pem.write_to_path(cls.cert_combined_path)
        cert.cert_chain_pems[0].write_to_path(cls.cert_path)
        cert.private_key_pem.write_to_path(cls.key_path)
        encrypted_key.write_to_path(cls.password_key_path)

    @classmethod
    def teardown_class(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        shutil.rmtree(cls.tmpdir)

    def _wrap_in_ssl(self, sock: socket.socket) -> ssl.SSLSocket:
        if False:
            print('Hello World!')
        '\n        Given a single socket, wraps it in TLS.\n        '
        return original_ssl_wrap_socket(sock, ssl_version=ssl.PROTOCOL_SSLv23, cert_reqs=ssl.CERT_REQUIRED, ca_certs=self.ca_path, certfile=self.cert_path, keyfile=self.key_path, server_side=True)

    def test_client_certs_two_files(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Having a client cert in a separate file to its associated key works\n        properly.\n        '
        done_receiving = Event()
        client_certs = []

        def socket_handler(listener: socket.socket) -> None:
            if False:
                print('Hello World!')
            sock = listener.accept()[0]
            sock = self._wrap_in_ssl(sock)
            client_certs.append(sock.getpeercert())
            data = b''
            while not data.endswith(b'\r\n\r\n'):
                data += sock.recv(8192)
            sock.sendall(b'HTTP/1.1 200 OK\r\nServer: testsocket\r\nConnection: close\r\nContent-Length: 6\r\n\r\nValid!')
            done_receiving.wait(5)
            sock.close()
        self._start_server(socket_handler)
        with HTTPSConnectionPool(self.host, self.port, cert_file=self.cert_path, key_file=self.key_path, cert_reqs='REQUIRED', ca_certs=self.ca_path) as pool:
            pool.request('GET', '/', retries=0)
            done_receiving.set()
            assert len(client_certs) == 1

    def test_client_certs_one_file(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Having a client cert and its associated private key in just one file\n        works properly.\n        '
        done_receiving = Event()
        client_certs = []

        def socket_handler(listener: socket.socket) -> None:
            if False:
                print('Hello World!')
            sock = listener.accept()[0]
            sock = self._wrap_in_ssl(sock)
            client_certs.append(sock.getpeercert())
            data = b''
            while not data.endswith(b'\r\n\r\n'):
                data += sock.recv(8192)
            sock.sendall(b'HTTP/1.1 200 OK\r\nServer: testsocket\r\nConnection: close\r\nContent-Length: 6\r\n\r\nValid!')
            done_receiving.wait(5)
            sock.close()
        self._start_server(socket_handler)
        with HTTPSConnectionPool(self.host, self.port, cert_file=self.cert_combined_path, cert_reqs='REQUIRED', ca_certs=self.ca_path) as pool:
            pool.request('GET', '/', retries=0)
            done_receiving.set()
            assert len(client_certs) == 1

    def test_missing_client_certs_raises_error(self) -> None:
        if False:
            return 10
        '\n        Having client certs not be present causes an error.\n        '
        done_receiving = Event()

        def socket_handler(listener: socket.socket) -> None:
            if False:
                while True:
                    i = 10
            sock = listener.accept()[0]
            try:
                self._wrap_in_ssl(sock)
            except ssl.SSLError:
                pass
            done_receiving.wait(5)
            sock.close()
        self._start_server(socket_handler)
        with HTTPSConnectionPool(self.host, self.port, cert_reqs='REQUIRED', ca_certs=self.ca_path) as pool:
            with pytest.raises(MaxRetryError):
                pool.request('GET', '/', retries=0)
                done_receiving.set()
            done_receiving.set()

    def test_client_cert_with_string_password(self) -> None:
        if False:
            i = 10
            return i + 15
        self.run_client_cert_with_password_test('letmein')

    def test_client_cert_with_bytes_password(self) -> None:
        if False:
            while True:
                i = 10
        self.run_client_cert_with_password_test(b'letmein')

    def run_client_cert_with_password_test(self, password: bytes | str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Tests client certificate password functionality\n        '
        done_receiving = Event()
        client_certs = []

        def socket_handler(listener: socket.socket) -> None:
            if False:
                i = 10
                return i + 15
            sock = listener.accept()[0]
            sock = self._wrap_in_ssl(sock)
            client_certs.append(sock.getpeercert())
            data = b''
            while not data.endswith(b'\r\n\r\n'):
                data += sock.recv(8192)
            sock.sendall(b'HTTP/1.1 200 OK\r\nServer: testsocket\r\nConnection: close\r\nContent-Length: 6\r\n\r\nValid!')
            done_receiving.wait(5)
            sock.close()
        self._start_server(socket_handler)
        assert ssl_.SSLContext is not None
        ssl_context = ssl_.SSLContext(ssl_.PROTOCOL_SSLv23)
        ssl_context.load_cert_chain(certfile=self.cert_path, keyfile=self.password_key_path, password=password)
        with HTTPSConnectionPool(self.host, self.port, ssl_context=ssl_context, cert_reqs='REQUIRED', ca_certs=self.ca_path) as pool:
            pool.request('GET', '/', retries=0)
            done_receiving.set()
            assert len(client_certs) == 1

    def test_load_keyfile_with_invalid_password(self) -> None:
        if False:
            print('Hello World!')
        assert ssl_.SSLContext is not None
        context = ssl_.SSLContext(ssl_.PROTOCOL_SSLv23)
        with pytest.raises(ssl.SSLError):
            context.load_cert_chain(certfile=self.cert_path, keyfile=self.password_key_path, password=b'letmei')

    def test_load_invalid_cert_file(self) -> None:
        if False:
            print('Hello World!')
        assert ssl_.SSLContext is not None
        context = ssl_.SSLContext(ssl_.PROTOCOL_SSLv23)
        with pytest.raises(ssl.SSLError):
            context.load_cert_chain(certfile=self.password_key_path)

class TestSocketClosing(SocketDummyServerTestCase):

    def test_recovery_when_server_closes_connection(self) -> None:
        if False:
            while True:
                i = 10
        done_closing = Event()

        def socket_handler(listener: socket.socket) -> None:
            if False:
                while True:
                    i = 10
            for i in (0, 1):
                sock = listener.accept()[0]
                buf = b''
                while not buf.endswith(b'\r\n\r\n'):
                    buf = sock.recv(65536)
                body = f'Response {int(i)}'
                sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n%s' % (len(body), body)).encode('utf-8'))
                sock.close()
                done_closing.set()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            response = pool.request('GET', '/', retries=0)
            assert response.status == 200
            assert response.data == b'Response 0'
            done_closing.wait()
            response = pool.request('GET', '/', retries=0)
            assert response.status == 200
            assert response.data == b'Response 1'

    def test_connection_refused(self) -> None:
        if False:
            while True:
                i = 10
        (host, port) = get_unreachable_address()
        with HTTPConnectionPool(host, port, maxsize=3, block=True) as http:
            with pytest.raises(MaxRetryError):
                http.request('GET', '/', retries=0, release_conn=False)
            assert http.pool is not None
            assert http.pool.qsize() == http.pool.maxsize

    def test_connection_read_timeout(self) -> None:
        if False:
            return 10
        timed_out = Event()

        def socket_handler(listener: socket.socket) -> None:
            if False:
                while True:
                    i = 10
            sock = listener.accept()[0]
            while not sock.recv(65536).endswith(b'\r\n\r\n'):
                pass
            timed_out.wait()
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port, timeout=SHORT_TIMEOUT, retries=False, maxsize=3, block=True) as http:
            try:
                with pytest.raises(ReadTimeoutError):
                    http.request('GET', '/', release_conn=False)
            finally:
                timed_out.set()
            assert http.pool is not None
            assert http.pool.qsize() == http.pool.maxsize

    def test_read_timeout_dont_retry_method_not_in_allowlist(self) -> None:
        if False:
            while True:
                i = 10
        timed_out = Event()

        def socket_handler(listener: socket.socket) -> None:
            if False:
                for i in range(10):
                    print('nop')
            sock = listener.accept()[0]
            sock.recv(65536)
            timed_out.wait()
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port, timeout=LONG_TIMEOUT, retries=True) as pool:
            try:
                with pytest.raises(ReadTimeoutError):
                    pool.request('POST', '/')
            finally:
                timed_out.set()

    def test_https_connection_read_timeout(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Handshake timeouts should fail with a Timeout'
        timed_out = Event()

        def socket_handler(listener: socket.socket) -> None:
            if False:
                while True:
                    i = 10
            sock = listener.accept()[0]
            while not sock.recv(65536):
                pass
            timed_out.wait()
            sock.close()
        self._start_server(socket_handler)
        with HTTPSConnectionPool(self.host, self.port, timeout=LONG_TIMEOUT, retries=False) as pool:
            try:
                with pytest.raises(ReadTimeoutError):
                    pool.request('GET', '/')
            finally:
                timed_out.set()
        with HTTPSConnectionPool(host=self.host):
            err = OSError()
            err.errno = errno.EAGAIN
            with pytest.raises(ReadTimeoutError):
                pool._raise_timeout(err, '', 0)

    def test_timeout_errors_cause_retries(self) -> None:
        if False:
            while True:
                i = 10

        def socket_handler(listener: socket.socket) -> None:
            if False:
                print('Hello World!')
            sock_timeout = listener.accept()[0]
            sock = listener.accept()[0]
            sock_timeout.close()
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            body = 'Response 2'
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n%s' % (len(body), body)).encode('utf-8'))
            sock.close()
        default_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(1)
        try:
            self._start_server(socket_handler)
            t = Timeout(connect=LONG_TIMEOUT, read=LONG_TIMEOUT)
            with HTTPConnectionPool(self.host, self.port, timeout=t) as pool:
                response = pool.request('GET', '/', retries=1)
                assert response.status == 200
                assert response.data == b'Response 2'
        finally:
            socket.setdefaulttimeout(default_timeout)

    def test_delayed_body_read_timeout(self) -> None:
        if False:
            return 10
        timed_out = Event()

        def socket_handler(listener: socket.socket) -> None:
            if False:
                return 10
            sock = listener.accept()[0]
            buf = b''
            body = 'Hi'
            while not buf.endswith(b'\r\n\r\n'):
                buf = sock.recv(65536)
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n' % len(body)).encode('utf-8'))
            timed_out.wait()
            sock.send(body.encode('utf-8'))
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            response = pool.urlopen('GET', '/', retries=0, preload_content=False, timeout=Timeout(connect=1, read=LONG_TIMEOUT))
            try:
                with pytest.raises(ReadTimeoutError):
                    response.read()
            finally:
                timed_out.set()

    def test_delayed_body_read_timeout_with_preload(self) -> None:
        if False:
            while True:
                i = 10
        timed_out = Event()

        def socket_handler(listener: socket.socket) -> None:
            if False:
                while True:
                    i = 10
            sock = listener.accept()[0]
            buf = b''
            body = 'Hi'
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n' % len(body)).encode('utf-8'))
            timed_out.wait(5)
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            try:
                with pytest.raises(ReadTimeoutError):
                    timeout = Timeout(connect=LONG_TIMEOUT, read=SHORT_TIMEOUT)
                    pool.urlopen('GET', '/', retries=False, timeout=timeout)
            finally:
                timed_out.set()

    def test_incomplete_response(self) -> None:
        if False:
            return 10
        body = 'Response'
        partial_body = body[:2]

        def socket_handler(listener: socket.socket) -> None:
            if False:
                for i in range(10):
                    print('nop')
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf = sock.recv(65536)
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n%s' % (len(body), partial_body)).encode('utf-8'))
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            response = pool.request('GET', '/', retries=0, preload_content=False)
            with pytest.raises(ProtocolError):
                response.read()

    def test_retry_weird_http_version(self) -> None:
        if False:
            i = 10
            return i + 15
        'Retry class should handle httplib.BadStatusLine errors properly'

        def socket_handler(listener: socket.socket) -> None:
            if False:
                print('Hello World!')
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            body = 'bad http 0.5 response'
            sock.send(('HTTP/0.5 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n%s' % (len(body), body)).encode('utf-8'))
            sock.close()
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\nfoo' % len('foo')).encode('utf-8'))
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            retry = Retry(read=1)
            response = pool.request('GET', '/', retries=retry)
            assert response.status == 200
            assert response.data == b'foo'

    def test_connection_cleanup_on_read_timeout(self) -> None:
        if False:
            return 10
        timed_out = Event()

        def socket_handler(listener: socket.socket) -> None:
            if False:
                for i in range(10):
                    print('nop')
            sock = listener.accept()[0]
            buf = b''
            body = 'Hi'
            while not buf.endswith(b'\r\n\r\n'):
                buf = sock.recv(65536)
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n' % len(body)).encode('utf-8'))
            timed_out.wait()
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            assert pool.pool is not None
            poolsize = pool.pool.qsize()
            response = pool.urlopen('GET', '/', retries=0, preload_content=False, timeout=LONG_TIMEOUT)
            try:
                with pytest.raises(ReadTimeoutError):
                    response.read()
                assert poolsize == pool.pool.qsize()
            finally:
                timed_out.set()

    def test_connection_cleanup_on_protocol_error_during_read(self) -> None:
        if False:
            i = 10
            return i + 15
        body = 'Response'
        partial_body = body[:2]

        def socket_handler(listener: socket.socket) -> None:
            if False:
                while True:
                    i = 10
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf = sock.recv(65536)
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n%s' % (len(body), partial_body)).encode('utf-8'))
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            assert pool.pool is not None
            poolsize = pool.pool.qsize()
            response = pool.request('GET', '/', retries=0, preload_content=False)
            with pytest.raises(ProtocolError):
                response.read()
            assert poolsize == pool.pool.qsize()

    def test_connection_closed_on_read_timeout_preload_false(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        timed_out = Event()

        def socket_handler(listener: socket.socket) -> None:
            if False:
                return 10
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf = sock.recv(65535)
            sock.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nTransfer-Encoding: chunked\r\n\r\n8\r\n12345678\r\n')
            timed_out.wait(5)
            (rlist, _, _) = select.select([listener], [], [], 1)
            assert rlist
            new_sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf = new_sock.recv(65535)
            new_sock.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nTransfer-Encoding: chunked\r\n\r\n8\r\n12345678\r\n0\r\n\r\n')
            new_sock.close()
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            response = pool.urlopen('GET', '/', retries=0, preload_content=False, timeout=LONG_TIMEOUT)
            try:
                with pytest.raises(ReadTimeoutError):
                    response.read()
            finally:
                timed_out.set()
            response = pool.urlopen('GET', '/', retries=0, preload_content=False, timeout=LONG_TIMEOUT)
            assert len(response.read()) == 8

    def test_closing_response_actually_closes_connection(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        done_closing = Event()
        complete = Event()

        def socket_handler(listener: socket.socket) -> None:
            if False:
                for i in range(10):
                    print('nop')
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf = sock.recv(65536)
            sock.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 0\r\n\r\n')
            done_closing.wait(timeout=LONG_TIMEOUT)
            sock.settimeout(LONG_TIMEOUT)
            new_data = sock.recv(65536)
            assert not new_data
            sock.close()
            complete.set()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            response = pool.request('GET', '/', retries=0, preload_content=False)
            assert response.status == 200
            response.close()
            done_closing.set()
            successful = complete.wait(timeout=LONG_TIMEOUT)
            assert successful, 'Timed out waiting for connection close'

    def test_release_conn_param_is_respected_after_timeout_retry(self) -> None:
        if False:
            i = 10
            return i + 15
        "For successful ```urlopen(release_conn=False)```,\n        the connection isn't released, even after a retry.\n\n        This test allows a retry: one request fails, the next request succeeds.\n\n        This is a regression test for issue #651 [1], where the connection\n        would be released if the initial request failed, even if a retry\n        succeeded.\n\n        [1] <https://github.com/urllib3/urllib3/issues/651>\n        "

        def socket_handler(listener: socket.socket) -> None:
            if False:
                return 10
            sock = listener.accept()[0]
            consume_socket(sock)
            sock.close()
            (rlist, _, _) = select.select([listener], [], [], 5)
            assert rlist
            sock = listener.accept()[0]
            consume_socket(sock)
            sock.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nTransfer-Encoding: chunked\r\n\r\n8\r\n12345678\r\n0\r\n\r\n')
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port, maxsize=1) as pool:
            response = pool.urlopen('GET', '/', retries=1, release_conn=False, preload_content=False, timeout=LONG_TIMEOUT)
            assert pool.num_connections == 2
            assert pool.pool is not None
            assert pool.pool.qsize() == 0
            assert response.connection is not None
            response.read()
            assert pool.pool.qsize() == 1
            assert response.connection is None

    def test_socket_close_socket_then_file(self) -> None:
        if False:
            i = 10
            return i + 15

        def consume_ssl_socket(listener: socket.socket) -> None:
            if False:
                print('Hello World!')
            try:
                with listener.accept()[0] as sock, original_ssl_wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'], ca_certs=DEFAULT_CA) as ssl_sock:
                    consume_socket(ssl_sock)
            except (ConnectionResetError, ConnectionAbortedError, OSError):
                pass
        self._start_server(consume_ssl_socket)
        with socket.create_connection((self.host, self.port)) as sock, contextlib.closing(ssl_wrap_socket(sock, server_hostname=self.host, ca_certs=DEFAULT_CA)) as ssl_sock, ssl_sock.makefile('rb') as f:
            ssl_sock.close()
            f.close()
            with pytest.raises(OSError):
                ssl_sock.sendall(b'hello')
            assert ssl_sock.fileno() == -1

    def test_socket_close_stays_open_with_makefile_open(self) -> None:
        if False:
            i = 10
            return i + 15

        def consume_ssl_socket(listener: socket.socket) -> None:
            if False:
                i = 10
                return i + 15
            try:
                with listener.accept()[0] as sock, original_ssl_wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'], ca_certs=DEFAULT_CA) as ssl_sock:
                    consume_socket(ssl_sock)
            except (ConnectionResetError, ConnectionAbortedError, OSError):
                pass
        self._start_server(consume_ssl_socket)
        with socket.create_connection((self.host, self.port)) as sock, contextlib.closing(ssl_wrap_socket(sock, server_hostname=self.host, ca_certs=DEFAULT_CA)) as ssl_sock, ssl_sock.makefile('rb'):
            ssl_sock.close()
            ssl_sock.close()
            ssl_sock.sendall(b'hello')
            assert ssl_sock.fileno() > 0

class TestProxyManager(SocketDummyServerTestCase):

    def test_simple(self) -> None:
        if False:
            i = 10
            return i + 15

        def echo_socket_handler(listener: socket.socket) -> None:
            if False:
                print('Hello World!')
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n%s' % (len(buf), buf.decode('utf-8'))).encode('utf-8'))
            sock.close()
        self._start_server(echo_socket_handler)
        base_url = f'http://{self.host}:{self.port}'
        with proxy_from_url(base_url) as proxy:
            r = proxy.request('GET', 'http://google.com/')
            assert r.status == 200
            assert sorted(r.data.split(b'\r\n')) == sorted([b'GET http://google.com/ HTTP/1.1', b'Host: google.com', b'Accept-Encoding: identity', b'Accept: */*', b'User-Agent: ' + _get_default_user_agent().encode('utf-8'), b'', b''])

    def test_headers(self) -> None:
        if False:
            i = 10
            return i + 15

        def echo_socket_handler(listener: socket.socket) -> None:
            if False:
                for i in range(10):
                    print('nop')
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n%s' % (len(buf), buf.decode('utf-8'))).encode('utf-8'))
            sock.close()
        self._start_server(echo_socket_handler)
        base_url = f'http://{self.host}:{self.port}'
        proxy_headers = HTTPHeaderDict({'For The Proxy': 'YEAH!'})
        with proxy_from_url(base_url, proxy_headers=proxy_headers) as proxy:
            conn = proxy.connection_from_url('http://www.google.com/')
            r = conn.urlopen('GET', 'http://www.google.com/', assert_same_host=False)
            assert r.status == 200
            assert b'For The Proxy: YEAH!\r\n' in r.data

    def test_retries(self) -> None:
        if False:
            print('Hello World!')
        close_event = Event()

        def echo_socket_handler(listener: socket.socket) -> None:
            if False:
                while True:
                    i = 10
            sock = listener.accept()[0]
            sock.close()
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            sock.send(('HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n%s' % (len(buf), buf.decode('utf-8'))).encode('utf-8'))
            sock.close()
            close_event.set()
        self._start_server(echo_socket_handler)
        base_url = f'http://{self.host}:{self.port}'
        with proxy_from_url(base_url) as proxy:
            conn = proxy.connection_from_url('http://www.google.com')
            r = conn.urlopen('GET', 'http://www.google.com', assert_same_host=False, retries=1)
            assert r.status == 200
            close_event.wait(timeout=LONG_TIMEOUT)
            with pytest.raises(ProxyError):
                conn.urlopen('GET', 'http://www.google.com', assert_same_host=False, retries=False)

    def test_connect_reconn(self) -> None:
        if False:
            while True:
                i = 10

        def proxy_ssl_one(listener: socket.socket) -> None:
            if False:
                print('Hello World!')
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            s = buf.decode('utf-8')
            if not s.startswith('CONNECT '):
                sock.send(b'HTTP/1.1 405 Method not allowed\r\nAllow: CONNECT\r\n\r\n')
                sock.close()
                return
            if not s.startswith(f'CONNECT {self.host}:443'):
                sock.send(b'HTTP/1.1 403 Forbidden\r\n\r\n')
                sock.close()
                return
            sock.send(b'HTTP/1.1 200 Connection Established\r\n\r\n')
            ssl_sock = original_ssl_wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'], ca_certs=DEFAULT_CA)
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += ssl_sock.recv(65536)
            ssl_sock.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\nHi')
            ssl_sock.close()

        def echo_socket_handler(listener: socket.socket) -> None:
            if False:
                for i in range(10):
                    print('nop')
            proxy_ssl_one(listener)
            proxy_ssl_one(listener)
        self._start_server(echo_socket_handler)
        base_url = f'http://{self.host}:{self.port}'
        with proxy_from_url(base_url, ca_certs=DEFAULT_CA) as proxy:
            url = f'https://{self.host}'
            conn = proxy.connection_from_url(url)
            r = conn.urlopen('GET', url, retries=0)
            assert r.status == 200
            r = conn.urlopen('GET', url, retries=0)
            assert r.status == 200

    def test_connect_ipv6_addr(self) -> None:
        if False:
            i = 10
            return i + 15
        ipv6_addr = '2001:4998:c:a06::2:4008'

        def echo_socket_handler(listener: socket.socket) -> None:
            if False:
                return 10
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            s = buf.decode('utf-8')
            if s.startswith(f'CONNECT [{ipv6_addr}]:443'):
                sock.send(b'HTTP/1.1 200 Connection Established\r\n\r\n')
                ssl_sock = original_ssl_wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'])
                buf = b''
                while not buf.endswith(b'\r\n\r\n'):
                    buf += ssl_sock.recv(65536)
                ssl_sock.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\nHi')
                ssl_sock.close()
            else:
                sock.close()
        self._start_server(echo_socket_handler)
        base_url = f'http://{self.host}:{self.port}'
        with proxy_from_url(base_url, cert_reqs='NONE') as proxy:
            url = f'https://[{ipv6_addr}]'
            conn = proxy.connection_from_url(url)
            try:
                with pytest.warns(InsecureRequestWarning):
                    r = conn.urlopen('GET', url, retries=0)
                assert r.status == 200
            except MaxRetryError:
                pytest.fail('Invalid IPv6 format in HTTP CONNECT request')

    @pytest.mark.parametrize('target_scheme', ['http', 'https'])
    def test_https_proxymanager_connected_to_http_proxy(self, target_scheme: str) -> None:
        if False:
            i = 10
            return i + 15
        errored = Event()

        def http_socket_handler(listener: socket.socket) -> None:
            if False:
                while True:
                    i = 10
            sock = listener.accept()[0]
            sock.send(b'HTTP/1.0 501 Not Implemented\r\nConnection: close\r\n\r\n')
            errored.wait()
            sock.close()
        self._start_server(http_socket_handler)
        base_url = f'https://{self.host}:{self.port}'
        with ProxyManager(base_url, cert_reqs='NONE') as proxy:
            with pytest.raises(MaxRetryError) as e:
                proxy.request('GET', f'{target_scheme}://example.com', retries=0)
            errored.set()
            assert type(e.value.reason) is ProxyError
            assert 'Your proxy appears to only use HTTP and not HTTPS' in str(e.value.reason)

class TestSSL(SocketDummyServerTestCase):

    def test_ssl_failure_midway_through_conn(self) -> None:
        if False:
            print('Hello World!')

        def socket_handler(listener: socket.socket) -> None:
            if False:
                return 10
            sock = listener.accept()[0]
            sock2 = sock.dup()
            ssl_sock = original_ssl_wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'], ca_certs=DEFAULT_CA)
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += ssl_sock.recv(65536)
            sock2.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\n\r\nHi')
            sock2.close()
            ssl_sock.close()
        self._start_server(socket_handler)
        with HTTPSConnectionPool(self.host, self.port, ca_certs=DEFAULT_CA) as pool:
            with pytest.raises(SSLError, match='(wrong version number|record overflow)'):
                pool.request('GET', '/', retries=False)

    def test_ssl_read_timeout(self) -> None:
        if False:
            print('Hello World!')
        timed_out = Event()

        def socket_handler(listener: socket.socket) -> None:
            if False:
                while True:
                    i = 10
            sock = listener.accept()[0]
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
            ssl_sock = original_ssl_wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'])
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += ssl_sock.recv(65536)
            ssl_sock.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 10\r\n\r\nHi-')
            timed_out.wait()
            sock.close()
            ssl_sock.close()
        self._start_server(socket_handler)
        with HTTPSConnectionPool(self.host, self.port, ca_certs=DEFAULT_CA) as pool:
            response = pool.urlopen('GET', '/', retries=0, preload_content=False, timeout=LONG_TIMEOUT)
            try:
                with pytest.raises(ReadTimeoutError):
                    response.read()
            finally:
                timed_out.set()

    def test_ssl_failed_fingerprint_verification(self) -> None:
        if False:
            i = 10
            return i + 15

        def socket_handler(listener: socket.socket) -> None:
            if False:
                i = 10
                return i + 15
            for i in range(2):
                sock = listener.accept()[0]
                try:
                    ssl_sock = original_ssl_wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'], ca_certs=DEFAULT_CA)
                except (ssl.SSLError, ConnectionResetError):
                    if i == 1:
                        raise
                    return
                ssl_sock.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5\r\n\r\nHello')
                ssl_sock.close()
                sock.close()
        self._start_server(socket_handler)
        fingerprint = 'A0:C4:A7:46:00:ED:A7:2D:C0:BE:CB:9A:8C:B6:07:CA:58:EE:74:5E'

        def request() -> None:
            if False:
                return 10
            pool = HTTPSConnectionPool(self.host, self.port, assert_fingerprint=fingerprint)
            try:
                timeout = Timeout(connect=LONG_TIMEOUT, read=SHORT_TIMEOUT)
                response = pool.urlopen('GET', '/', preload_content=False, retries=0, timeout=timeout)
                response.read()
            finally:
                pool.close()
        with pytest.raises(MaxRetryError) as cm:
            request()
        assert type(cm.value.reason) is SSLError
        with pytest.raises(MaxRetryError):
            request()

    def test_retry_ssl_error(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def socket_handler(listener: socket.socket) -> None:
            if False:
                return 10
            sock = listener.accept()[0]
            sock2 = sock.dup()
            ssl_sock = original_ssl_wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'])
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += ssl_sock.recv(65536)
            sock2.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 4\r\n\r\nFail')
            sock2.close()
            ssl_sock.close()
            sock = listener.accept()[0]
            ssl_sock = original_ssl_wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'])
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += ssl_sock.recv(65536)
            ssl_sock.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 7\r\n\r\nSuccess')
            ssl_sock.close()
        self._start_server(socket_handler)
        with HTTPSConnectionPool(self.host, self.port, ca_certs=DEFAULT_CA) as pool:
            response = pool.urlopen('GET', '/', retries=1)
            assert response.data == b'Success'

    def test_ssl_load_default_certs_when_empty(self) -> None:
        if False:
            print('Hello World!')

        def socket_handler(listener: socket.socket) -> None:
            if False:
                for i in range(10):
                    print('nop')
            sock = listener.accept()[0]
            try:
                ssl_sock = original_ssl_wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'], ca_certs=DEFAULT_CA)
            except (ssl.SSLError, OSError):
                return
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += ssl_sock.recv(65536)
            ssl_sock.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5\r\n\r\nHello')
            ssl_sock.close()
            sock.close()
        context = mock.create_autospec(ssl_.SSLContext)
        context.load_default_certs = mock.Mock()
        context.options = 0
        with mock.patch('urllib3.util.ssl_.SSLContext', lambda *_, **__: context):
            self._start_server(socket_handler)
            with HTTPSConnectionPool(self.host, self.port) as pool:
                with pytest.raises(Exception):
                    pool.request('GET', '/', timeout=SHORT_TIMEOUT)
                context.load_default_certs.assert_called_with()

    def test_ssl_dont_load_default_certs_when_given(self) -> None:
        if False:
            for i in range(10):
                print('nop')

        def socket_handler(listener: socket.socket) -> None:
            if False:
                for i in range(10):
                    print('nop')
            sock = listener.accept()[0]
            try:
                ssl_sock = original_ssl_wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'], ca_certs=DEFAULT_CA)
            except (ssl.SSLError, OSError):
                return
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += ssl_sock.recv(65536)
            ssl_sock.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5\r\n\r\nHello')
            ssl_sock.close()
            sock.close()
        context = mock.create_autospec(ssl_.SSLContext)
        context.load_default_certs = mock.Mock()
        context.options = 0
        with mock.patch('urllib3.util.ssl_.SSLContext', lambda *_, **__: context):
            for kwargs in [{'ca_certs': '/a'}, {'ca_cert_dir': '/a'}, {'ca_certs': 'a', 'ca_cert_dir': 'a'}, {'ssl_context': context}]:
                self._start_server(socket_handler)
                with HTTPSConnectionPool(self.host, self.port, **kwargs) as pool:
                    with pytest.raises(Exception):
                        pool.request('GET', '/', timeout=SHORT_TIMEOUT)
                    context.load_default_certs.assert_not_called()

    def test_load_verify_locations_exception(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensure that load_verify_locations raises SSLError for all backends\n        '
        with pytest.raises(SSLError):
            ssl_wrap_socket(None, ca_certs='/tmp/fake-file')

    def test_ssl_custom_validation_failure_terminates(self, tmpdir: Path) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensure that the underlying socket is terminated if custom validation fails.\n        '
        server_closed = Event()

        def is_closed_socket(sock: socket.socket) -> bool:
            if False:
                i = 10
                return i + 15
            try:
                sock.settimeout(SHORT_TIMEOUT)
            except OSError:
                return True
            return False

        def socket_handler(listener: socket.socket) -> None:
            if False:
                i = 10
                return i + 15
            sock = listener.accept()[0]
            try:
                _ = original_ssl_wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'], ca_certs=DEFAULT_CA)
            except ConnectionResetError:
                return
            except ssl.SSLError as e:
                assert 'alert unknown ca' in str(e)
                if is_closed_socket(sock):
                    server_closed.set()
        self._start_server(socket_handler)
        other_ca = trustme.CA()
        other_ca_path = str(tmpdir / 'ca.pem')
        other_ca.cert_pem.write_to_path(other_ca_path)
        with HTTPSConnectionPool(self.host, self.port, cert_reqs='REQUIRED', ca_certs=other_ca_path) as pool:
            with pytest.raises(SSLError):
                pool.request('GET', '/', retries=False, timeout=LONG_TIMEOUT)
        assert server_closed.wait(LONG_TIMEOUT), 'The socket was not terminated'

    @pytest.mark.integration
    @pytest.mark.parametrize('preload_content,read_amt', [(True, None), (False, None), (False, 2 ** 31)])
    def test_requesting_large_resources_via_ssl(self, preload_content: bool, read_amt: int | None) -> None:
        if False:
            print('Hello World!')
        '\n        Ensure that it is possible to read 2 GiB or more via an SSL\n        socket.\n        https://github.com/urllib3/urllib3/issues/2513\n        '
        content_length = 2 ** 31
        ssl_ready = Event()

        def socket_handler(listener: socket.socket) -> None:
            if False:
                return 10
            sock = listener.accept()[0]
            ssl_sock = original_ssl_wrap_socket(sock, server_side=True, keyfile=DEFAULT_CERTS['keyfile'], certfile=DEFAULT_CERTS['certfile'], ca_certs=DEFAULT_CA)
            ssl_ready.set()
            while not ssl_sock.recv(65536).endswith(b'\r\n\r\n'):
                continue
            ssl_sock.send(b'HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n' % content_length)
            chunks = 2
            for i in range(chunks):
                ssl_sock.sendall(bytes(content_length // chunks))
            ssl_sock.close()
            sock.close()
        self._start_server(socket_handler)
        ssl_ready.wait(5)
        with HTTPSConnectionPool(self.host, self.port, ca_certs=DEFAULT_CA, retries=False) as pool:
            response = pool.request('GET', '/', preload_content=preload_content)
            data = response.data if preload_content else response.read(read_amt)
            assert len(data) == content_length

class TestErrorWrapping(SocketDummyServerTestCase):

    def test_bad_statusline(self) -> None:
        if False:
            print('Hello World!')
        self.start_response_handler(b'HTTP/1.1 Omg What Is This?\r\nContent-Length: 0\r\n\r\n')
        with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
            with pytest.raises(ProtocolError):
                pool.request('GET', '/')

    def test_unknown_protocol(self) -> None:
        if False:
            while True:
                i = 10
        self.start_response_handler(b'HTTP/1000 200 OK\r\nContent-Length: 0\r\n\r\n')
        with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
            with pytest.raises(ProtocolError):
                pool.request('GET', '/')

class TestHeaders(SocketDummyServerTestCase):

    def test_httplib_headers_case_insensitive(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.start_response_handler(b'HTTP/1.1 200 OK\r\nContent-Length: 0\r\nContent-type: text/plain\r\n\r\n')
        with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
            HEADERS = {'Content-Length': '0', 'Content-type': 'text/plain'}
            r = pool.request('GET', '/')
            assert HEADERS == dict(r.headers.items())

    def start_parsing_handler(self) -> None:
        if False:
            i = 10
            return i + 15
        self.parsed_headers: typing.OrderedDict[str, str] = OrderedDict()
        self.received_headers: list[bytes] = []

        def socket_handler(listener: socket.socket) -> None:
            if False:
                while True:
                    i = 10
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            self.received_headers = [header for header in buf.split(b'\r\n')[1:] if header]
            for header in self.received_headers:
                (key, value) = header.split(b': ')
                self.parsed_headers[key.decode('ascii')] = value.decode('ascii')
            sock.send(b'HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n')
            sock.close()
        self._start_server(socket_handler)

    def test_headers_are_sent_with_the_original_case(self) -> None:
        if False:
            i = 10
            return i + 15
        headers = {'foo': 'bar', 'bAz': 'quux'}
        self.start_parsing_handler()
        expected_headers = {'Accept-Encoding': 'identity', 'Host': f'{self.host}:{self.port}', 'User-Agent': _get_default_user_agent()}
        expected_headers.update(headers)
        with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
            pool.request('GET', '/', headers=HTTPHeaderDict(headers))
            assert expected_headers == self.parsed_headers

    def test_ua_header_can_be_overridden(self) -> None:
        if False:
            while True:
                i = 10
        headers = {'uSeR-AgENt': 'Definitely not urllib3!'}
        self.start_parsing_handler()
        expected_headers = {'Accept-Encoding': 'identity', 'Host': f'{self.host}:{self.port}'}
        expected_headers.update(headers)
        with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
            pool.request('GET', '/', headers=HTTPHeaderDict(headers))
            assert expected_headers == self.parsed_headers

    def test_request_headers_are_sent_in_the_original_order(self) -> None:
        if False:
            print('Hello World!')
        K = 16
        expected_request_headers = [(f'X-Header-{int(i)}', str(i)) for i in reversed(range(K))]

        def filter_non_x_headers(d: typing.OrderedDict[str, str]) -> list[tuple[str, str]]:
            if False:
                print('Hello World!')
            return [(k, v) for (k, v) in d.items() if k.startswith('X-Header-')]
        self.start_parsing_handler()
        with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
            pool.request('GET', '/', headers=OrderedDict(expected_request_headers))
            assert expected_request_headers == filter_non_x_headers(self.parsed_headers)

    @resolvesLocalhostFQDN()
    def test_request_host_header_ignores_fqdn_dot(self) -> None:
        if False:
            print('Hello World!')
        self.start_parsing_handler()
        with HTTPConnectionPool(self.host + '.', self.port, retries=False) as pool:
            pool.request('GET', '/')
            self.assert_header_received(self.received_headers, 'Host', f'{self.host}:{self.port}')

    def test_response_headers_are_returned_in_the_original_order(self) -> None:
        if False:
            return 10
        K = 16
        expected_response_headers = [(f'X-Header-{int(i)}', str(i)) for i in reversed(range(K))]

        def socket_handler(listener: socket.socket) -> None:
            if False:
                i = 10
                return i + 15
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            sock.send(b'HTTP/1.1 200 OK\r\n' + b'\r\n'.join([k.encode('utf8') + b': ' + v.encode('utf8') for (k, v) in expected_response_headers]) + b'\r\n')
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('GET', '/', retries=0)
            actual_response_headers = [(k, v) for (k, v) in r.headers.items() if k.startswith('X-Header-')]
            assert expected_response_headers == actual_response_headers

    @pytest.mark.parametrize('method_type, body_type', [('GET', None), ('POST', None), ('POST', 'bytes'), ('POST', 'bytes-io')])
    def test_headers_sent_with_add(self, method_type: str, body_type: str | None) -> None:
        if False:
            while True:
                i = 10
        '\n        Confirm that when adding headers with combine=True that we simply append to the\n        most recent value, rather than create a new header line.\n        '
        body: None | bytes | io.BytesIO
        if body_type is None:
            body = None
        elif body_type == 'bytes':
            body = b'my-body'
        elif body_type == 'bytes-io':
            body = io.BytesIO(b'bytes-io-body')
            body.seek(0, 0)
        else:
            raise ValueError('Unknonw body type')
        buffer: bytes = b''

        def socket_handler(listener: socket.socket) -> None:
            if False:
                while True:
                    i = 10
            nonlocal buffer
            sock = listener.accept()[0]
            sock.settimeout(0)
            start = time.time()
            while time.time() - start < LONG_TIMEOUT / 2:
                try:
                    buffer += sock.recv(65536)
                except OSError:
                    continue
            sock.sendall(b'HTTP/1.1 200 OK\r\nServer: example.com\r\nContent-Length: 0\r\n\r\n')
            sock.close()
        self._start_server(socket_handler)
        headers = HTTPHeaderDict()
        headers.add('A', '1')
        headers.add('C', '3')
        headers.add('B', '2')
        headers.add('B', '3')
        headers.add('A', '4', combine=False)
        headers.add('C', '5', combine=True)
        headers.add('C', '6')
        with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
            r = pool.request(method_type, '/', body=body, headers=headers)
            assert r.status == 200
            assert b'A: 1\r\nA: 4\r\nC: 3, 5\r\nC: 6\r\nB: 2\r\nB: 3' in buffer

class TestBrokenHeaders(SocketDummyServerTestCase):

    def _test_broken_header_parsing(self, headers: list[bytes], unparsed_data_check: str | None=None) -> None:
        if False:
            i = 10
            return i + 15
        self.start_response_handler(b'HTTP/1.1 200 OK\r\nContent-Length: 0\r\nContent-type: text/plain\r\n' + b'\r\n'.join(headers) + b'\r\n\r\n')
        with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
            with LogRecorder() as logs:
                pool.request('GET', '/')
            for record in logs:
                if 'Failed to parse headers' in record.msg and type(record.args) is tuple and (_url_from_pool(pool, '/') == record.args[0]):
                    if unparsed_data_check is None or unparsed_data_check in record.getMessage():
                        return
            pytest.fail('Missing log about unparsed headers')

    def test_header_without_name(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._test_broken_header_parsing([b': Value', b'Another: Header'])

    def test_header_without_name_or_value(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._test_broken_header_parsing([b':', b'Another: Header'])

    def test_header_without_colon_or_value(self) -> None:
        if False:
            return 10
        self._test_broken_header_parsing([b'Broken Header', b'Another: Header'], 'Broken Header')

class TestHeaderParsingContentType(SocketDummyServerTestCase):

    def _test_okay_header_parsing(self, header: bytes) -> None:
        if False:
            print('Hello World!')
        self.start_response_handler(b'HTTP/1.1 200 OK\r\nContent-Length: 0\r\n' + header + b'\r\n\r\n')
        with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
            with LogRecorder() as logs:
                pool.request('GET', '/')
            for record in logs:
                assert 'Failed to parse headers' not in record.msg

    def test_header_text_plain(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._test_okay_header_parsing(b'Content-type: text/plain')

    def test_header_message_rfc822(self) -> None:
        if False:
            while True:
                i = 10
        self._test_okay_header_parsing(b'Content-type: message/rfc822')

class TestHEAD(SocketDummyServerTestCase):

    def test_chunked_head_response_does_not_hang(self) -> None:
        if False:
            i = 10
            return i + 15
        self.start_response_handler(b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-type: text/plain\r\n\r\n')
        with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
            r = pool.request('HEAD', '/', timeout=LONG_TIMEOUT, preload_content=False)
            assert [] == list(r.stream())

    def test_empty_head_response_does_not_hang(self) -> None:
        if False:
            while True:
                i = 10
        self.start_response_handler(b'HTTP/1.1 200 OK\r\nContent-Length: 256\r\nContent-type: text/plain\r\n\r\n')
        with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
            r = pool.request('HEAD', '/', timeout=LONG_TIMEOUT, preload_content=False)
            assert [] == list(r.stream())

class TestStream(SocketDummyServerTestCase):

    def test_stream_none_unchunked_response_does_not_hang(self) -> None:
        if False:
            while True:
                i = 10
        done_event = Event()

        def socket_handler(listener: socket.socket) -> None:
            if False:
                return 10
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            sock.send(b'HTTP/1.1 200 OK\r\nContent-Length: 12\r\nContent-type: text/plain\r\n\r\nhello, world')
            done_event.wait(5)
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
            r = pool.request('GET', '/', timeout=LONG_TIMEOUT, preload_content=False)
            assert [b'hello, world'] == list(r.stream(None))
            done_event.set()

    def test_large_compressed_stream(self) -> None:
        if False:
            while True:
                i = 10
        done_event = Event()
        expected_total_length = 296085

        def socket_handler(listener: socket.socket) -> None:
            if False:
                print('Hello World!')
            compress = zlib.compressobj(6, zlib.DEFLATED, 16 + zlib.MAX_WBITS)
            data = compress.compress(b'x' * expected_total_length)
            data += compress.flush()
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            sock.sendall(b'HTTP/1.1 200 OK\r\nContent-Length: %d\r\nContent-Encoding: gzip\r\n\r\n' % (len(data),) + data)
            done_event.wait(5)
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port, retries=False) as pool:
            r = pool.request('GET', '/', timeout=LONG_TIMEOUT, preload_content=False)
            total_length = 0
            chunks_smaller_than_10240 = 0
            for chunk in r.stream(10240, decode_content=True):
                assert 0 < len(chunk) <= 10240
                if len(chunk) < 10240:
                    chunks_smaller_than_10240 += 1
                else:
                    assert chunks_smaller_than_10240 == 0
                total_length += len(chunk)
            assert chunks_smaller_than_10240 == 1
            assert expected_total_length == total_length
            done_event.set()

class TestBadContentLength(SocketDummyServerTestCase):

    def test_enforce_content_length_get(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        done_event = Event()

        def socket_handler(listener: socket.socket) -> None:
            if False:
                return 10
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            sock.send(b'HTTP/1.1 200 OK\r\nContent-Length: 22\r\nContent-type: text/plain\r\n\r\nhello, world')
            done_event.wait(LONG_TIMEOUT)
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port, maxsize=1) as conn:
            get_response = conn.request('GET', url='/', preload_content=False, enforce_content_length=True)
            data = get_response.stream(100)
            with pytest.raises(ProtocolError, match='12 bytes read, 10 more expected'):
                next(data)
            done_event.set()

    def test_enforce_content_length_no_body(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        done_event = Event()

        def socket_handler(listener: socket.socket) -> None:
            if False:
                for i in range(10):
                    print('nop')
            sock = listener.accept()[0]
            buf = b''
            while not buf.endswith(b'\r\n\r\n'):
                buf += sock.recv(65536)
            sock.send(b'HTTP/1.1 200 OK\r\nContent-Length: 22\r\nContent-type: text/plain\r\n\r\n')
            done_event.wait(1)
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port, maxsize=1) as conn:
            head_response = conn.request('HEAD', url='/', preload_content=False, enforce_content_length=True)
            data = [chunk for chunk in head_response.stream(1)]
            assert len(data) == 0
            done_event.set()

class TestRetryPoolSizeDrainFail(SocketDummyServerTestCase):

    def test_pool_size_retry_drain_fail(self) -> None:
        if False:
            return 10

        def socket_handler(listener: socket.socket) -> None:
            if False:
                i = 10
                return i + 15
            for _ in range(2):
                sock = listener.accept()[0]
                while not sock.recv(65536).endswith(b'\r\n\r\n'):
                    pass
                sock.send(b'HTTP/1.1 404 NOT FOUND\r\nContent-Length: 1000\r\nContent-Type: text/plain\r\n\r\n')
                sock.close()
        self._start_server(socket_handler)
        retries = Retry(total=1, raise_on_status=False, status_forcelist=[404])
        with HTTPConnectionPool(self.host, self.port, maxsize=10, retries=retries, block=True) as pool:
            pool.urlopen('GET', '/not_found', preload_content=False)
            assert pool.num_connections == 1

class TestBrokenPipe(SocketDummyServerTestCase):

    @notWindows()
    def test_ignore_broken_pipe_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        if False:
            return 10
        sock_shut = Event()
        orig_connect = HTTPConnection.connect
        buf = 'a' * 1024 * 1024 * 4

        def connect_and_wait(*args: typing.Any, **kw: typing.Any) -> None:
            if False:
                while True:
                    i = 10
            ret = orig_connect(*args, **kw)
            assert sock_shut.wait(5)
            return ret

        def socket_handler(listener: socket.socket) -> None:
            if False:
                return 10
            for i in range(2):
                sock = listener.accept()[0]
                sock.send(b'HTTP/1.1 404 Not Found\r\nConnection: close\r\nContent-Length: 10\r\n\r\nxxxxxxxxxx')
                sock.shutdown(socket.SHUT_RDWR)
                sock_shut.set()
                sock.close()
        monkeypatch.setattr(HTTPConnection, 'connect', connect_and_wait)
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port) as pool:
            r = pool.request('POST', '/', body=buf)
            assert r.status == 404
            assert r.headers['content-length'] == '10'
            assert r.data == b'xxxxxxxxxx'
            r = pool.request('POST', '/admin', chunked=True, body=buf)
            assert r.status == 404
            assert r.headers['content-length'] == '10'
            assert r.data == b'xxxxxxxxxx'

class TestMultipartResponse(SocketDummyServerTestCase):

    def test_multipart_assert_header_parsing_no_defects(self) -> None:
        if False:
            while True:
                i = 10

        def socket_handler(listener: socket.socket) -> None:
            if False:
                return 10
            for _ in range(2):
                sock = listener.accept()[0]
                while not sock.recv(65536).endswith(b'\r\n\r\n'):
                    pass
                sock.sendall(b'HTTP/1.1 404 Not Found\r\nServer: example.com\r\nContent-Type: multipart/mixed; boundary=36eeb8c4e26d842a\r\nContent-Length: 73\r\n\r\n--36eeb8c4e26d842a\r\nContent-Type: text/plain\r\n\r\n1\r\n--36eeb8c4e26d842a--\r\n')
                sock.close()
        self._start_server(socket_handler)
        from urllib3.connectionpool import log
        with mock.patch.object(log, 'warning') as log_warning:
            with HTTPConnectionPool(self.host, self.port, timeout=LONG_TIMEOUT) as pool:
                resp = pool.urlopen('GET', '/')
                assert resp.status == 404
                assert resp.headers['content-type'] == 'multipart/mixed; boundary=36eeb8c4e26d842a'
                assert len(resp.data) == 73
                log_warning.assert_not_called()

class TestContentFraming(SocketDummyServerTestCase):

    @pytest.mark.parametrize('content_length', [None, 0])
    @pytest.mark.parametrize('method', ['POST', 'PUT', 'PATCH'])
    def test_content_length_0_by_default(self, method: str, content_length: int | None) -> None:
        if False:
            while True:
                i = 10
        buffer = bytearray()

        def socket_handler(listener: socket.socket) -> None:
            if False:
                i = 10
                return i + 15
            nonlocal buffer
            sock = listener.accept()[0]
            while not buffer.endswith(b'\r\n\r\n'):
                buffer += sock.recv(65536)
            sock.sendall(b'HTTP/1.1 200 OK\r\nServer: example.com\r\nContent-Length: 0\r\n\r\n')
            sock.close()
        self._start_server(socket_handler)
        headers = {}
        if content_length is not None:
            headers['Content-Length'] = str(content_length)
        with HTTPConnectionPool(self.host, self.port, timeout=3) as pool:
            resp = pool.request(method, '/')
            assert resp.status == 200
        sent_bytes = bytes(buffer)
        assert b'Accept-Encoding: identity\r\n' in sent_bytes
        assert b'Content-Length: 0\r\n' in sent_bytes
        assert b'transfer-encoding' not in sent_bytes.lower()

    @pytest.mark.parametrize('chunked', [True, False])
    @pytest.mark.parametrize('method', ['POST', 'PUT', 'PATCH'])
    @pytest.mark.parametrize('body_type', ['file', 'generator', 'bytes'])
    def test_chunked_specified(self, method: str, chunked: bool, body_type: str) -> None:
        if False:
            i = 10
            return i + 15
        buffer = bytearray()

        def socket_handler(listener: socket.socket) -> None:
            if False:
                for i in range(10):
                    print('nop')
            nonlocal buffer
            sock = listener.accept()[0]
            sock.settimeout(0)
            start = time.time()
            while time.time() - start < LONG_TIMEOUT / 2:
                try:
                    buffer += sock.recv(65536)
                except OSError:
                    continue
            sock.sendall(b'HTTP/1.1 200 OK\r\nServer: example.com\r\nContent-Length: 0\r\n\r\n')
            sock.close()
        self._start_server(socket_handler)
        body: typing.Any
        if body_type == 'generator':

            def body_generator() -> typing.Generator[bytes, None, None]:
                if False:
                    print('Hello World!')
                yield (b'x' * 10)
            body = body_generator()
        elif body_type == 'file':
            body = io.BytesIO(b'x' * 10)
            body.seek(0, 0)
        else:
            if chunked is False:
                pytest.skip('urllib3 uses Content-Length in this case')
            body = b'x' * 10
        with HTTPConnectionPool(self.host, self.port, timeout=LONG_TIMEOUT, retries=False) as pool:
            resp = pool.request(method, '/', chunked=chunked, body=body)
            assert resp.status == 200
        sent_bytes = bytes(buffer)
        assert sent_bytes.count(b':') == 5
        assert b'Host: localhost:' in sent_bytes
        assert b'Accept-Encoding: identity\r\n' in sent_bytes
        assert b'Transfer-Encoding: chunked\r\n' in sent_bytes
        assert b'User-Agent: python-urllib3/' in sent_bytes
        assert b'content-length' not in sent_bytes.lower()
        assert b'\r\n\r\na\r\nxxxxxxxxxx\r\n0\r\n\r\n' in sent_bytes

    @pytest.mark.parametrize('method', ['POST', 'PUT', 'PATCH'])
    @pytest.mark.parametrize('body_type', ['file', 'generator', 'bytes', 'bytearray', 'file_text'])
    def test_chunked_not_specified(self, method: str, body_type: str) -> None:
        if False:
            return 10
        buffer = bytearray()

        def socket_handler(listener: socket.socket) -> None:
            if False:
                i = 10
                return i + 15
            nonlocal buffer
            sock = listener.accept()[0]
            sock.settimeout(0)
            start = time.time()
            while time.time() - start < LONG_TIMEOUT / 2:
                try:
                    buffer += sock.recv(65536)
                except OSError:
                    continue
            sock.sendall(b'HTTP/1.1 200 OK\r\nServer: example.com\r\nContent-Length: 0\r\n\r\n')
            sock.close()
        self._start_server(socket_handler)
        body: typing.Any
        if body_type == 'generator':

            def body_generator() -> typing.Generator[bytes, None, None]:
                if False:
                    return 10
                yield (b'x' * 10)
            body = body_generator()
            should_be_chunked = True
        elif body_type == 'file':
            body = io.BytesIO(b'x' * 10)
            body.seek(0, 0)
            should_be_chunked = True
        elif body_type == 'file_text':
            body = io.StringIO('x' * 10)
            body.seek(0, 0)
            should_be_chunked = True
        elif body_type == 'bytearray':
            body = bytearray(b'x' * 10)
            should_be_chunked = False
        else:
            body = b'x' * 10
            should_be_chunked = False
        with HTTPConnectionPool(self.host, self.port, timeout=LONG_TIMEOUT, retries=False) as pool:
            resp = pool.request(method, '/', body=body)
            assert resp.status == 200
        sent_bytes = bytes(buffer)
        assert sent_bytes.count(b':') == 5
        assert b'Host: localhost:' in sent_bytes
        assert b'Accept-Encoding: identity\r\n' in sent_bytes
        assert b'User-Agent: python-urllib3/' in sent_bytes
        if should_be_chunked:
            assert b'content-length' not in sent_bytes.lower()
            assert b'Transfer-Encoding: chunked\r\n' in sent_bytes
            assert b'\r\n\r\na\r\nxxxxxxxxxx\r\n0\r\n\r\n' in sent_bytes
        else:
            assert b'Content-Length: 10\r\n' in sent_bytes
            assert b'transfer-encoding' not in sent_bytes.lower()
            assert sent_bytes.endswith(b'\r\n\r\nxxxxxxxxxx')

    @pytest.mark.parametrize('header_transform', [str.lower, str.title, str.upper])
    @pytest.mark.parametrize(['header', 'header_value', 'expected'], [('content-length', '10', b': 10\r\n\r\nxxxxxxxx'), ('transfer-encoding', 'chunked', b': chunked\r\n\r\n8\r\nxxxxxxxx\r\n0\r\n\r\n')])
    def test_framing_set_via_headers(self, header_transform: typing.Callable[[str], str], header: str, header_value: str, expected: bytes) -> None:
        if False:
            for i in range(10):
                print('nop')
        buffer = bytearray()

        def socket_handler(listener: socket.socket) -> None:
            if False:
                for i in range(10):
                    print('nop')
            nonlocal buffer
            sock = listener.accept()[0]
            sock.settimeout(0)
            start = time.time()
            while time.time() - start < LONG_TIMEOUT / 2:
                try:
                    buffer += sock.recv(65536)
                except OSError:
                    continue
            sock.sendall(b'HTTP/1.1 200 OK\r\nServer: example.com\r\nContent-Length: 0\r\n\r\n')
            sock.close()
        self._start_server(socket_handler)
        with HTTPConnectionPool(self.host, self.port, timeout=LONG_TIMEOUT, retries=False) as pool:
            resp = pool.request('POST', '/', body=b'xxxxxxxx', headers={header_transform(header): header_value})
            assert resp.status == 200
            sent_bytes = bytes(buffer)
            assert sent_bytes.endswith(expected)