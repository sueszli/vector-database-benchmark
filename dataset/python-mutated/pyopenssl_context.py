"""A CPython compatible SSLContext implementation wrapping PyOpenSSL's
context.
"""
from __future__ import annotations
import socket as _socket
import ssl as _stdlibssl
import sys as _sys
import time as _time
from errno import EINTR as _EINTR
from ipaddress import ip_address as _ip_address
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar, Union
from cryptography.x509 import load_der_x509_certificate as _load_der_x509_certificate
from OpenSSL import SSL as _SSL
from OpenSSL import crypto as _crypto
from service_identity import CertificateError as _SICertificateError
from service_identity import VerificationError as _SIVerificationError
from service_identity.pyopenssl import verify_hostname as _verify_hostname
from service_identity.pyopenssl import verify_ip_address as _verify_ip_address
from pymongo.errors import ConfigurationError as _ConfigurationError
from pymongo.errors import _CertificateError
from pymongo.ocsp_cache import _OCSPCache
from pymongo.ocsp_support import _load_trusted_ca_certs, _ocsp_callback
from pymongo.socket_checker import SocketChecker as _SocketChecker
from pymongo.socket_checker import _errno_from_exception
from pymongo.write_concern import validate_boolean
if TYPE_CHECKING:
    from ssl import VerifyMode
    from cryptography.x509 import Certificate
_T = TypeVar('_T')
try:
    import certifi
    _HAVE_CERTIFI = True
except ImportError:
    _HAVE_CERTIFI = False
PROTOCOL_SSLv23 = _SSL.SSLv23_METHOD
OP_NO_SSLv2 = _SSL.OP_NO_SSLv2
OP_NO_SSLv3 = _SSL.OP_NO_SSLv3
OP_NO_COMPRESSION = _SSL.OP_NO_COMPRESSION
OP_NO_RENEGOTIATION = getattr(_SSL, 'OP_NO_RENEGOTIATION', 0)
HAS_SNI = True
IS_PYOPENSSL = True
SSLError = _SSL.Error
_VERIFY_MAP = {_stdlibssl.CERT_NONE: _SSL.VERIFY_NONE, _stdlibssl.CERT_OPTIONAL: _SSL.VERIFY_PEER, _stdlibssl.CERT_REQUIRED: _SSL.VERIFY_PEER | _SSL.VERIFY_FAIL_IF_NO_PEER_CERT}
_REVERSE_VERIFY_MAP = {value: key for (key, value) in _VERIFY_MAP.items()}

def _is_ip_address(address: Any) -> bool:
    if False:
        while True:
            i = 10
    try:
        _ip_address(address)
        return True
    except (ValueError, UnicodeError):
        return False
BLOCKING_IO_ERRORS = (_SSL.WantReadError, _SSL.WantWriteError, _SSL.WantX509LookupError)

def _ragged_eof(exc: BaseException) -> bool:
    if False:
        i = 10
        return i + 15
    'Return True if the OpenSSL.SSL.SysCallError is a ragged EOF.'
    return exc.args == (-1, 'Unexpected EOF')

class _sslConn(_SSL.Connection):

    def __init__(self, ctx: _SSL.Context, sock: Optional[_socket.socket], suppress_ragged_eofs: bool):
        if False:
            while True:
                i = 10
        self.socket_checker = _SocketChecker()
        self.suppress_ragged_eofs = suppress_ragged_eofs
        super().__init__(ctx, sock)

    def _call(self, call: Callable[..., _T], *args: Any, **kwargs: Any) -> _T:
        if False:
            i = 10
            return i + 15
        timeout = self.gettimeout()
        if timeout:
            start = _time.monotonic()
        while True:
            try:
                return call(*args, **kwargs)
            except BLOCKING_IO_ERRORS as exc:
                if self.fileno() == -1:
                    if timeout and _time.monotonic() - start > timeout:
                        raise _socket.timeout('timed out') from None
                    raise SSLError('Underlying socket has been closed') from None
                if isinstance(exc, _SSL.WantReadError):
                    want_read = True
                    want_write = False
                elif isinstance(exc, _SSL.WantWriteError):
                    want_read = False
                    want_write = True
                else:
                    want_read = True
                    want_write = True
                self.socket_checker.select(self, want_read, want_write, timeout)
                if timeout and _time.monotonic() - start > timeout:
                    raise _socket.timeout('timed out') from None
                continue

    def do_handshake(self, *args: Any, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        return self._call(super().do_handshake, *args, **kwargs)

    def recv(self, *args: Any, **kwargs: Any) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._call(super().recv, *args, **kwargs)
        except _SSL.SysCallError as exc:
            if self.suppress_ragged_eofs and _ragged_eof(exc):
                return b''
            raise

    def recv_into(self, *args: Any, **kwargs: Any) -> int:
        if False:
            i = 10
            return i + 15
        try:
            return self._call(super().recv_into, *args, **kwargs)
        except _SSL.SysCallError as exc:
            if self.suppress_ragged_eofs and _ragged_eof(exc):
                return 0
            raise

    def sendall(self, buf: bytes, flags: int=0) -> None:
        if False:
            for i in range(10):
                print('nop')
        view = memoryview(buf)
        total_length = len(buf)
        total_sent = 0
        while total_sent < total_length:
            try:
                sent = self._call(super().send, view[total_sent:], flags)
            except OSError as exc:
                if _errno_from_exception(exc) == _EINTR:
                    continue
                raise
            if sent <= 0:
                raise OSError('connection closed')
            total_sent += sent

class _CallbackData:
    """Data class which is passed to the OCSP callback."""

    def __init__(self) -> None:
        if False:
            return 10
        self.trusted_ca_certs: Optional[list[Certificate]] = None
        self.check_ocsp_endpoint: Optional[bool] = None
        self.ocsp_response_cache = _OCSPCache()

class SSLContext:
    """A CPython compatible SSLContext implementation wrapping PyOpenSSL's
    context.
    """
    __slots__ = ('_protocol', '_ctx', '_callback_data', '_check_hostname')

    def __init__(self, protocol: int):
        if False:
            for i in range(10):
                print('nop')
        self._protocol = protocol
        self._ctx = _SSL.Context(self._protocol)
        self._callback_data = _CallbackData()
        self._check_hostname = True
        self._callback_data.check_ocsp_endpoint = True
        self._ctx.set_ocsp_client_callback(callback=_ocsp_callback, data=self._callback_data)

    @property
    def protocol(self) -> int:
        if False:
            return 10
        'The protocol version chosen when constructing the context.\n        This attribute is read-only.\n        '
        return self._protocol

    def __get_verify_mode(self) -> VerifyMode:
        if False:
            while True:
                i = 10
        "Whether to try to verify other peers' certificates and how to\n        behave if verification fails. This attribute must be one of\n        ssl.CERT_NONE, ssl.CERT_OPTIONAL or ssl.CERT_REQUIRED.\n        "
        return _REVERSE_VERIFY_MAP[self._ctx.get_verify_mode()]

    def __set_verify_mode(self, value: VerifyMode) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Setter for verify_mode.'

        def _cb(_connobj: _SSL.Connection, _x509obj: _crypto.X509, _errnum: int, _errdepth: int, retcode: int) -> bool:
            if False:
                i = 10
                return i + 15
            return bool(retcode)
        self._ctx.set_verify(_VERIFY_MAP[value], _cb)
    verify_mode = property(__get_verify_mode, __set_verify_mode)

    def __get_check_hostname(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self._check_hostname

    def __set_check_hostname(self, value: Any) -> None:
        if False:
            while True:
                i = 10
        validate_boolean('check_hostname', value)
        self._check_hostname = value
    check_hostname = property(__get_check_hostname, __set_check_hostname)

    def __get_check_ocsp_endpoint(self) -> Optional[bool]:
        if False:
            while True:
                i = 10
        return self._callback_data.check_ocsp_endpoint

    def __set_check_ocsp_endpoint(self, value: bool) -> None:
        if False:
            while True:
                i = 10
        validate_boolean('check_ocsp', value)
        self._callback_data.check_ocsp_endpoint = value
    check_ocsp_endpoint = property(__get_check_ocsp_endpoint, __set_check_ocsp_endpoint)

    def __get_options(self) -> None:
        if False:
            print('Hello World!')
        return self._ctx.set_options(0)

    def __set_options(self, value: int) -> None:
        if False:
            return 10
        self._ctx.set_options(int(value))
    options = property(__get_options, __set_options)

    def load_cert_chain(self, certfile: Union[str, bytes], keyfile: Union[str, bytes, None]=None, password: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        "Load a private key and the corresponding certificate. The certfile\n        string must be the path to a single file in PEM format containing the\n        certificate as well as any number of CA certificates needed to\n        establish the certificate's authenticity. The keyfile string, if\n        present, must point to a file containing the private key. Otherwise\n        the private key will be taken from certfile as well.\n        "
        if password:

            def _pwcb(_max_length: int, _prompt_twice: bool, _user_data: bytes) -> bytes:
                if False:
                    return 10
                assert password is not None
                return password.encode('utf-8')
            self._ctx.set_passwd_cb(_pwcb)
        self._ctx.use_certificate_chain_file(certfile)
        self._ctx.use_privatekey_file(keyfile or certfile)
        self._ctx.check_privatekey()

    def load_verify_locations(self, cafile: Optional[str]=None, capath: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        'Load a set of "certification authority"(CA) certificates used to\n        validate other peers\' certificates when `~verify_mode` is other than\n        ssl.CERT_NONE.\n        '
        self._ctx.load_verify_locations(cafile, capath)
        if not hasattr(_SSL.Connection, 'get_verified_chain'):
            assert cafile is not None
            self._callback_data.trusted_ca_certs = _load_trusted_ca_certs(cafile)

    def _load_certifi(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Attempt to load CA certs from certifi.'
        if _HAVE_CERTIFI:
            self.load_verify_locations(certifi.where())
        else:
            raise _ConfigurationError('tlsAllowInvalidCertificates is False but no system CA certificates could be loaded. Please install the certifi package, or provide a path to a CA file using the tlsCAFile option')

    def _load_wincerts(self, store: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Attempt to load CA certs from Windows trust store.'
        cert_store = self._ctx.get_cert_store()
        oid = _stdlibssl.Purpose.SERVER_AUTH.oid
        for (cert, encoding, trust) in _stdlibssl.enum_certificates(store):
            if encoding == 'x509_asn':
                if trust is True or oid in trust:
                    cert_store.add_cert(_crypto.X509.from_cryptography(_load_der_x509_certificate(cert)))

    def load_default_certs(self) -> None:
        if False:
            return 10
        'A PyOpenSSL version of load_default_certs from CPython.'
        if _sys.platform == 'win32':
            try:
                for storename in ('CA', 'ROOT'):
                    self._load_wincerts(storename)
            except PermissionError:
                self._load_certifi()
        elif _sys.platform == 'darwin':
            self._load_certifi()
        self._ctx.set_default_verify_paths()

    def set_default_verify_paths(self) -> None:
        if False:
            i = 10
            return i + 15
        'Specify that the platform provided CA certificates are to be used\n        for verification purposes.\n        '
        self._ctx.set_default_verify_paths()

    def wrap_socket(self, sock: _socket.socket, server_side: bool=False, do_handshake_on_connect: bool=True, suppress_ragged_eofs: bool=True, server_hostname: Optional[str]=None, session: Optional[_SSL.Session]=None) -> _sslConn:
        if False:
            i = 10
            return i + 15
        'Wrap an existing Python socket connection and return a TLS socket\n        object.\n        '
        ssl_conn = _sslConn(self._ctx, sock, suppress_ragged_eofs)
        if session:
            ssl_conn.set_session(session)
        if server_side is True:
            ssl_conn.set_accept_state()
        else:
            if server_hostname and (not _is_ip_address(server_hostname)):
                ssl_conn.set_tlsext_host_name(server_hostname.encode('idna'))
            if self.verify_mode != _stdlibssl.CERT_NONE:
                ssl_conn.request_ocsp()
            ssl_conn.set_connect_state()
        if do_handshake_on_connect:
            ssl_conn.do_handshake()
            if self.check_hostname and server_hostname is not None:
                try:
                    if _is_ip_address(server_hostname):
                        _verify_ip_address(ssl_conn, server_hostname)
                    else:
                        _verify_hostname(ssl_conn, server_hostname)
                except (_SICertificateError, _SIVerificationError) as exc:
                    raise _CertificateError(str(exc)) from None
        return ssl_conn