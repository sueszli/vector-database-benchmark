from __future__ import annotations
import platform
import select
import socket
import ssl
import typing
from unittest import mock
import pytest
from dummyserver.testcase import SocketDummyServerTestCase, consume_socket
from dummyserver.tornadoserver import DEFAULT_CA, DEFAULT_CERTS
from urllib3.util import ssl_
from urllib3.util.ssltransport import SSLTransport
if typing.TYPE_CHECKING:
    from typing import Literal
PER_TEST_TIMEOUT = 60

def server_client_ssl_contexts() -> tuple[ssl.SSLContext, ssl.SSLContext]:
    if False:
        return 10
    if hasattr(ssl, 'PROTOCOL_TLS_SERVER'):
        server_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_context.load_cert_chain(DEFAULT_CERTS['certfile'], DEFAULT_CERTS['keyfile'])
    if hasattr(ssl, 'PROTOCOL_TLS_CLIENT'):
        client_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    client_context.load_verify_locations(DEFAULT_CA)
    return (server_context, client_context)

@typing.overload
def sample_request(binary: Literal[True]=...) -> bytes:
    if False:
        while True:
            i = 10
    ...

@typing.overload
def sample_request(binary: Literal[False]) -> str:
    if False:
        i = 10
        return i + 15
    ...

def sample_request(binary: bool=True) -> bytes | str:
    if False:
        while True:
            i = 10
    request = b'GET http://www.testing.com/ HTTP/1.1\r\nHost: www.testing.com\r\nUser-Agent: awesome-test\r\n\r\n'
    return request if binary else request.decode('utf-8')

def validate_request(provided_request: bytearray, binary: Literal[False, True]=True) -> None:
    if False:
        i = 10
        return i + 15
    assert provided_request is not None
    expected_request = sample_request(binary)
    assert provided_request == expected_request

@typing.overload
def sample_response(binary: Literal[True]=...) -> bytes:
    if False:
        while True:
            i = 10
    ...

@typing.overload
def sample_response(binary: Literal[False]) -> str:
    if False:
        print('Hello World!')
    ...

@typing.overload
def sample_response(binary: bool=...) -> bytes | str:
    if False:
        print('Hello World!')
    ...

def sample_response(binary: bool=True) -> bytes | str:
    if False:
        for i in range(10):
            print('nop')
    response = b'HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n'
    return response if binary else response.decode('utf-8')

def validate_response(provided_response: bytes | bytearray | str, binary: bool=True) -> None:
    if False:
        i = 10
        return i + 15
    assert provided_response is not None
    expected_response = sample_response(binary)
    assert provided_response == expected_response

def validate_peercert(ssl_socket: SSLTransport) -> None:
    if False:
        while True:
            i = 10
    binary_cert = ssl_socket.getpeercert(binary_form=True)
    assert type(binary_cert) is bytes
    assert len(binary_cert) > 0
    cert = ssl_socket.getpeercert()
    assert type(cert) is dict
    assert 'serialNumber' in cert
    assert cert['serialNumber'] != ''

class SingleTLSLayerTestCase(SocketDummyServerTestCase):
    """
    Uses the SocketDummyServer to validate a single TLS layer can be
    established through the SSLTransport.
    """

    @classmethod
    def setup_class(cls) -> None:
        if False:
            print('Hello World!')
        (cls.server_context, cls.client_context) = server_client_ssl_contexts()

    def start_dummy_server(self, handler: typing.Callable[[socket.socket], None] | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')

        def socket_handler(listener: socket.socket) -> None:
            if False:
                return 10
            sock = listener.accept()[0]
            try:
                with self.server_context.wrap_socket(sock, server_side=True) as ssock:
                    request = consume_socket(ssock)
                    validate_request(request)
                    ssock.send(sample_response())
            except (ConnectionAbortedError, ConnectionResetError):
                return
        chosen_handler = handler if handler else socket_handler
        self._start_server(chosen_handler)

    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    def test_start_closed_socket(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Errors generated from an unconnected socket should bubble up.'
        sock = socket.socket(socket.AF_INET)
        context = ssl.create_default_context()
        sock.close()
        with pytest.raises(OSError):
            SSLTransport(sock, context)

    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    def test_close_after_handshake(self) -> None:
        if False:
            i = 10
            return i + 15
        'Socket errors should be bubbled up'
        self.start_dummy_server()
        sock = socket.create_connection((self.host, self.port))
        with SSLTransport(sock, self.client_context, server_hostname='localhost') as ssock:
            ssock.close()
            with pytest.raises(OSError):
                ssock.send(b'blaaargh')

    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    def test_wrap_existing_socket(self) -> None:
        if False:
            while True:
                i = 10
        'Validates a single TLS layer can be established.'
        self.start_dummy_server()
        sock = socket.create_connection((self.host, self.port))
        with SSLTransport(sock, self.client_context, server_hostname='localhost') as ssock:
            assert ssock.version() is not None
            ssock.send(sample_request())
            response = consume_socket(ssock)
            validate_response(response)

    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    def test_unbuffered_text_makefile(self) -> None:
        if False:
            while True:
                i = 10
        self.start_dummy_server()
        sock = socket.create_connection((self.host, self.port))
        with SSLTransport(sock, self.client_context, server_hostname='localhost') as ssock:
            with pytest.raises(ValueError):
                ssock.makefile('r', buffering=0)
            ssock.send(sample_request())
            response = consume_socket(ssock)
            validate_response(response)

    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    def test_unwrap_existing_socket(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Validates we can break up the TLS layer\n        A full request/response is sent over TLS, and later over plain text.\n        '

        def shutdown_handler(listener: socket.socket) -> None:
            if False:
                i = 10
                return i + 15
            sock = listener.accept()[0]
            ssl_sock = self.server_context.wrap_socket(sock, server_side=True)
            request = consume_socket(ssl_sock)
            validate_request(request)
            ssl_sock.sendall(sample_response())
            unwrapped_sock = ssl_sock.unwrap()
            request = consume_socket(unwrapped_sock)
            validate_request(request)
            unwrapped_sock.sendall(sample_response())
        self.start_dummy_server(shutdown_handler)
        sock = socket.create_connection((self.host, self.port))
        ssock = SSLTransport(sock, self.client_context, server_hostname='localhost')
        ssock.sendall(sample_request())
        response = consume_socket(ssock)
        validate_response(response)
        ssock.unwrap()
        sock.sendall(sample_request())
        response = consume_socket(sock)
        validate_response(response)

    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    def test_ssl_object_attributes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Ensures common ssl attributes are exposed'
        self.start_dummy_server()
        sock = socket.create_connection((self.host, self.port))
        with SSLTransport(sock, self.client_context, server_hostname='localhost') as ssock:
            cipher = ssock.cipher()
            assert type(cipher) is tuple
            assert ssock.selected_alpn_protocol() is None
            assert ssock.selected_npn_protocol() is None
            shared_ciphers = ssock.shared_ciphers()
            assert shared_ciphers is None or (type(shared_ciphers) is list and len(shared_ciphers) > 0)
            assert ssock.compression() is None
            validate_peercert(ssock)
            ssock.send(sample_request())
            response = consume_socket(ssock)
            validate_response(response)

    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    def test_socket_object_attributes(self) -> None:
        if False:
            print('Hello World!')
        'Ensures common socket attributes are exposed'
        self.start_dummy_server()
        sock = socket.create_connection((self.host, self.port))
        with SSLTransport(sock, self.client_context, server_hostname='localhost') as ssock:
            assert ssock.fileno() is not None
            test_timeout = 10
            ssock.settimeout(test_timeout)
            assert ssock.gettimeout() == test_timeout
            assert ssock.socket.gettimeout() == test_timeout
            ssock.send(sample_request())
            response = consume_socket(ssock)
            validate_response(response)

class SocketProxyDummyServer(SocketDummyServerTestCase):
    """
    Simulates a proxy that performs a simple I/O loop on client/server
    socket.
    """

    def __init__(self, destination_server_host: str, destination_server_port: int) -> None:
        if False:
            print('Hello World!')
        self.destination_server_host = destination_server_host
        self.destination_server_port = destination_server_port
        (self.server_ctx, _) = server_client_ssl_contexts()

    def start_proxy_handler(self) -> None:
        if False:
            return 10
        '\n        Socket handler for the proxy. Terminates the first TLS layer and tunnels\n        any bytes needed for client <-> server communicatin.\n        '

        def proxy_handler(listener: socket.socket) -> None:
            if False:
                i = 10
                return i + 15
            sock = listener.accept()[0]
            with self.server_ctx.wrap_socket(sock, server_side=True) as client_sock:
                upstream_sock = socket.create_connection((self.destination_server_host, self.destination_server_port))
                self._read_write_loop(client_sock, upstream_sock)
                upstream_sock.close()
                client_sock.close()
        self._start_server(proxy_handler)

    def _read_write_loop(self, client_sock: socket.socket, server_sock: socket.socket, chunks: int=65536) -> None:
        if False:
            i = 10
            return i + 15
        inputs = [client_sock, server_sock]
        output = [client_sock, server_sock]
        while inputs:
            (readable, writable, exception) = select.select(inputs, output, inputs)
            if exception:
                break
            for s in readable:
                (read_socket, write_socket) = (None, None)
                if s == client_sock:
                    read_socket = client_sock
                    write_socket = server_sock
                else:
                    read_socket = server_sock
                    write_socket = client_sock
                if write_socket in writable:
                    try:
                        b = read_socket.recv(chunks)
                        if len(b) == 0:
                            return
                        write_socket.send(b)
                    except ssl.SSLEOFError:
                        return

class TlsInTlsTestCase(SocketDummyServerTestCase):
    """
    Creates a TLS in TLS tunnel by chaining a 'SocketProxyDummyServer' and a
    `SocketDummyServerTestCase`.

    Client will first connect to the proxy, who will then proxy any bytes send
    to the destination server. First TLS layer terminates at the proxy, second
    TLS layer terminates at the destination server.
    """

    @classmethod
    def setup_class(cls) -> None:
        if False:
            i = 10
            return i + 15
        (cls.server_context, cls.client_context) = server_client_ssl_contexts()

    @classmethod
    def start_proxy_server(cls) -> None:
        if False:
            return 10
        cls.proxy_server = SocketProxyDummyServer(cls.host, cls.port)
        cls.proxy_server.start_proxy_handler()

    @classmethod
    def teardown_class(cls) -> None:
        if False:
            i = 10
            return i + 15
        if hasattr(cls, 'proxy_server'):
            cls.proxy_server.teardown_class()
        super().teardown_class()

    @classmethod
    def start_destination_server(cls) -> None:
        if False:
            return 10
        '\n        Socket handler for the destination_server. Terminates the second TLS\n        layer and send a basic HTTP response.\n        '

        def socket_handler(listener: socket.socket) -> None:
            if False:
                while True:
                    i = 10
            sock = listener.accept()[0]
            try:
                with cls.server_context.wrap_socket(sock, server_side=True) as ssock:
                    request = consume_socket(ssock)
                    validate_request(request)
                    ssock.send(sample_response())
            except (ssl.SSLEOFError, ssl.SSLZeroReturnError, OSError):
                return
            sock.close()
        cls._start_server(socket_handler)

    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    def test_tls_in_tls_tunnel(self) -> None:
        if False:
            print('Hello World!')
        '\n        Basic communication over the TLS in TLS tunnel.\n        '
        self.start_destination_server()
        self.start_proxy_server()
        sock = socket.create_connection((self.proxy_server.host, self.proxy_server.port))
        with self.client_context.wrap_socket(sock, server_hostname='localhost') as proxy_sock:
            with SSLTransport(proxy_sock, self.client_context, server_hostname='localhost') as destination_sock:
                assert destination_sock.version() is not None
                destination_sock.send(sample_request())
                response = consume_socket(destination_sock)
                validate_response(response)

    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    def test_wrong_sni_hint(self) -> None:
        if False:
            print('Hello World!')
        '\n        Provides a wrong sni hint to validate an exception is thrown.\n        '
        self.start_destination_server()
        self.start_proxy_server()
        sock = socket.create_connection((self.proxy_server.host, self.proxy_server.port))
        with self.client_context.wrap_socket(sock, server_hostname='localhost') as proxy_sock:
            with pytest.raises(ssl.SSLCertVerificationError):
                SSLTransport(proxy_sock, self.client_context, server_hostname='veryverywrong')

    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    @pytest.mark.parametrize('buffering', [None, 0])
    def test_tls_in_tls_makefile_raw_rw_binary(self, buffering: int | None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Uses makefile with read, write and binary modes without buffering.\n        '
        self.start_destination_server()
        self.start_proxy_server()
        sock = socket.create_connection((self.proxy_server.host, self.proxy_server.port))
        with self.client_context.wrap_socket(sock, server_hostname='localhost') as proxy_sock:
            with SSLTransport(proxy_sock, self.client_context, server_hostname='localhost') as destination_sock:
                file = destination_sock.makefile('rwb', buffering)
                file.write(sample_request())
                file.flush()
                response = bytearray(65536)
                wrote = file.readinto(response)
                assert wrote is not None
                str_response = response.decode('utf-8').rstrip('\x00')
                validate_response(str_response, binary=False)
                file.close()

    @pytest.mark.skipif(platform.system() == 'Windows', reason='Skipping windows due to text makefile support')
    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    def test_tls_in_tls_makefile_rw_text(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Creates a separate buffer for reading and writing using text mode and\n        utf-8 encoding.\n        '
        self.start_destination_server()
        self.start_proxy_server()
        sock = socket.create_connection((self.proxy_server.host, self.proxy_server.port))
        with self.client_context.wrap_socket(sock, server_hostname='localhost') as proxy_sock:
            with SSLTransport(proxy_sock, self.client_context, server_hostname='localhost') as destination_sock:
                read = destination_sock.makefile('r', encoding='utf-8')
                write = destination_sock.makefile('w', encoding='utf-8')
                write.write(sample_request(binary=False))
                write.flush()
                response = read.read()
                assert type(response) is str
                if '\r' not in response:
                    assert type(response) is str
                    response = response.replace('\n', '\r\n')
                validate_response(response, binary=False)

    @pytest.mark.timeout(PER_TEST_TIMEOUT)
    def test_tls_in_tls_recv_into_sendall(self) -> None:
        if False:
            print('Hello World!')
        '\n        Valides recv_into and sendall also work as expected. Other tests are\n        using recv/send.\n        '
        self.start_destination_server()
        self.start_proxy_server()
        sock = socket.create_connection((self.proxy_server.host, self.proxy_server.port))
        with self.client_context.wrap_socket(sock, server_hostname='localhost') as proxy_sock:
            with SSLTransport(proxy_sock, self.client_context, server_hostname='localhost') as destination_sock:
                destination_sock.sendall(sample_request())
                response = bytearray(65536)
                destination_sock.recv_into(response)
                str_response = response.decode('utf-8').rstrip('\x00')
                validate_response(str_response, binary=False)

class TestSSLTransportWithMock:

    def test_constructor_params(self) -> None:
        if False:
            return 10
        server_hostname = 'example-domain.com'
        sock = mock.Mock()
        context = mock.create_autospec(ssl_.SSLContext)
        ssl_transport = SSLTransport(sock, context, server_hostname=server_hostname, suppress_ragged_eofs=False)
        context.wrap_bio.assert_called_with(mock.ANY, mock.ANY, server_hostname=server_hostname)
        assert not ssl_transport.suppress_ragged_eofs

    def test_various_flags_errors(self) -> None:
        if False:
            while True:
                i = 10
        server_hostname = 'example-domain.com'
        sock = mock.Mock()
        context = mock.create_autospec(ssl_.SSLContext)
        ssl_transport = SSLTransport(sock, context, server_hostname=server_hostname, suppress_ragged_eofs=False)
        with pytest.raises(ValueError):
            ssl_transport.recv(flags=1)
        with pytest.raises(ValueError):
            ssl_transport.recv_into(bytearray(), flags=1)
        with pytest.raises(ValueError):
            ssl_transport.sendall(bytearray(), flags=1)
        with pytest.raises(ValueError):
            ssl_transport.send(None, flags=1)

    def test_makefile_wrong_mode_error(self) -> None:
        if False:
            return 10
        server_hostname = 'example-domain.com'
        sock = mock.Mock()
        context = mock.create_autospec(ssl_.SSLContext)
        ssl_transport = SSLTransport(sock, context, server_hostname=server_hostname, suppress_ragged_eofs=False)
        with pytest.raises(ValueError):
            ssl_transport.makefile(mode='x')

    def test_wrap_ssl_read_error(self) -> None:
        if False:
            return 10
        server_hostname = 'example-domain.com'
        sock = mock.Mock()
        context = mock.create_autospec(ssl_.SSLContext)
        ssl_transport = SSLTransport(sock, context, server_hostname=server_hostname, suppress_ragged_eofs=False)
        with mock.patch.object(ssl_transport, '_ssl_io_loop') as _ssl_io_loop:
            _ssl_io_loop.side_effect = ssl.SSLError()
            with pytest.raises(ssl.SSLError):
                ssl_transport._wrap_ssl_read(1)