"""
Module for using pyOpenSSL as a TLS backend. This module was relevant before
the standard library ``ssl`` module supported SNI, but now that we've dropped
support for Python 2.7 all relevant Python versions support SNI so
**this module is no longer recommended**.

This needs the following packages installed:

* `pyOpenSSL`_ (tested with 16.0.0)
* `cryptography`_ (minimum 1.3.4, from pyopenssl)
* `idna`_ (minimum 2.0)

However, pyOpenSSL depends on cryptography, so while we use all three directly here we
end up having relatively few packages required.

You can install them with the following command:

.. code-block:: bash

    $ python -m pip install pyopenssl cryptography idna

To activate certificate checking, call
:func:`~urllib3.contrib.pyopenssl.inject_into_urllib3` from your Python code
before you begin making HTTP requests. This can be done in a ``sitecustomize``
module, or at any other time before your application begins using ``urllib3``,
like this:

.. code-block:: python

    try:
        import urllib3.contrib.pyopenssl
        urllib3.contrib.pyopenssl.inject_into_urllib3()
    except ImportError:
        pass

.. _pyopenssl: https://www.pyopenssl.org
.. _cryptography: https://cryptography.io
.. _idna: https://github.com/kjd/idna
"""
from __future__ import annotations
import OpenSSL.SSL
from cryptography import x509
try:
    from cryptography.x509 import UnsupportedExtension
except ImportError:

    class UnsupportedExtension(Exception):
        pass
import logging
import ssl
import typing
from io import BytesIO
from socket import socket as socket_cls
from socket import timeout
from .. import util
if typing.TYPE_CHECKING:
    from OpenSSL.crypto import X509
__all__ = ['inject_into_urllib3', 'extract_from_urllib3']
_openssl_versions = {util.ssl_.PROTOCOL_TLS: OpenSSL.SSL.SSLv23_METHOD, util.ssl_.PROTOCOL_TLS_CLIENT: OpenSSL.SSL.SSLv23_METHOD, ssl.PROTOCOL_TLSv1: OpenSSL.SSL.TLSv1_METHOD}
if hasattr(ssl, 'PROTOCOL_TLSv1_1') and hasattr(OpenSSL.SSL, 'TLSv1_1_METHOD'):
    _openssl_versions[ssl.PROTOCOL_TLSv1_1] = OpenSSL.SSL.TLSv1_1_METHOD
if hasattr(ssl, 'PROTOCOL_TLSv1_2') and hasattr(OpenSSL.SSL, 'TLSv1_2_METHOD'):
    _openssl_versions[ssl.PROTOCOL_TLSv1_2] = OpenSSL.SSL.TLSv1_2_METHOD
_stdlib_to_openssl_verify = {ssl.CERT_NONE: OpenSSL.SSL.VERIFY_NONE, ssl.CERT_OPTIONAL: OpenSSL.SSL.VERIFY_PEER, ssl.CERT_REQUIRED: OpenSSL.SSL.VERIFY_PEER + OpenSSL.SSL.VERIFY_FAIL_IF_NO_PEER_CERT}
_openssl_to_stdlib_verify = {v: k for (k, v) in _stdlib_to_openssl_verify.items()}
_OP_NO_SSLv2_OR_SSLv3: int = getattr(OpenSSL.SSL, 'OP_NO_SSLv2', 0) | getattr(OpenSSL.SSL, 'OP_NO_SSLv3', 0)
_OP_NO_TLSv1: int = getattr(OpenSSL.SSL, 'OP_NO_TLSv1', 0)
_OP_NO_TLSv1_1: int = getattr(OpenSSL.SSL, 'OP_NO_TLSv1_1', 0)
_OP_NO_TLSv1_2: int = getattr(OpenSSL.SSL, 'OP_NO_TLSv1_2', 0)
_OP_NO_TLSv1_3: int = getattr(OpenSSL.SSL, 'OP_NO_TLSv1_3', 0)
_openssl_to_ssl_minimum_version: dict[int, int] = {ssl.TLSVersion.MINIMUM_SUPPORTED: _OP_NO_SSLv2_OR_SSLv3, ssl.TLSVersion.TLSv1: _OP_NO_SSLv2_OR_SSLv3, ssl.TLSVersion.TLSv1_1: _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1, ssl.TLSVersion.TLSv1_2: _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1 | _OP_NO_TLSv1_1, ssl.TLSVersion.TLSv1_3: _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1 | _OP_NO_TLSv1_1 | _OP_NO_TLSv1_2, ssl.TLSVersion.MAXIMUM_SUPPORTED: _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1 | _OP_NO_TLSv1_1 | _OP_NO_TLSv1_2}
_openssl_to_ssl_maximum_version: dict[int, int] = {ssl.TLSVersion.MINIMUM_SUPPORTED: _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1 | _OP_NO_TLSv1_1 | _OP_NO_TLSv1_2 | _OP_NO_TLSv1_3, ssl.TLSVersion.TLSv1: _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1_1 | _OP_NO_TLSv1_2 | _OP_NO_TLSv1_3, ssl.TLSVersion.TLSv1_1: _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1_2 | _OP_NO_TLSv1_3, ssl.TLSVersion.TLSv1_2: _OP_NO_SSLv2_OR_SSLv3 | _OP_NO_TLSv1_3, ssl.TLSVersion.TLSv1_3: _OP_NO_SSLv2_OR_SSLv3, ssl.TLSVersion.MAXIMUM_SUPPORTED: _OP_NO_SSLv2_OR_SSLv3}
SSL_WRITE_BLOCKSIZE = 16384
orig_util_SSLContext = util.ssl_.SSLContext
log = logging.getLogger(__name__)

def inject_into_urllib3() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Monkey-patch urllib3 with PyOpenSSL-backed SSL-support.'
    _validate_dependencies_met()
    util.SSLContext = PyOpenSSLContext
    util.ssl_.SSLContext = PyOpenSSLContext
    util.IS_PYOPENSSL = True
    util.ssl_.IS_PYOPENSSL = True

def extract_from_urllib3() -> None:
    if False:
        i = 10
        return i + 15
    'Undo monkey-patching by :func:`inject_into_urllib3`.'
    util.SSLContext = orig_util_SSLContext
    util.ssl_.SSLContext = orig_util_SSLContext
    util.IS_PYOPENSSL = False
    util.ssl_.IS_PYOPENSSL = False

def _validate_dependencies_met() -> None:
    if False:
        print('Hello World!')
    "\n    Verifies that PyOpenSSL's package-level dependencies have been met.\n    Throws `ImportError` if they are not met.\n    "
    from cryptography.x509.extensions import Extensions
    if getattr(Extensions, 'get_extension_for_class', None) is None:
        raise ImportError("'cryptography' module missing required functionality.  Try upgrading to v1.3.4 or newer.")
    from OpenSSL.crypto import X509
    x509 = X509()
    if getattr(x509, '_x509', None) is None:
        raise ImportError("'pyOpenSSL' module missing required functionality. Try upgrading to v0.14 or newer.")

def _dnsname_to_stdlib(name: str) -> str | None:
    if False:
        i = 10
        return i + 15
    '\n    Converts a dNSName SubjectAlternativeName field to the form used by the\n    standard library on the given Python version.\n\n    Cryptography produces a dNSName as a unicode string that was idna-decoded\n    from ASCII bytes. We need to idna-encode that string to get it back, and\n    then on Python 3 we also need to convert to unicode via UTF-8 (the stdlib\n    uses PyUnicode_FromStringAndSize on it, which decodes via UTF-8).\n\n    If the name cannot be idna-encoded then we return None signalling that\n    the name given should be skipped.\n    '

    def idna_encode(name: str) -> bytes | None:
        if False:
            for i in range(10):
                print('nop')
        "\n        Borrowed wholesale from the Python Cryptography Project. It turns out\n        that we can't just safely call `idna.encode`: it can explode for\n        wildcard names. This avoids that problem.\n        "
        import idna
        try:
            for prefix in ['*.', '.']:
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    return prefix.encode('ascii') + idna.encode(name)
            return idna.encode(name)
        except idna.core.IDNAError:
            return None
    if ':' in name:
        return name
    encoded_name = idna_encode(name)
    if encoded_name is None:
        return None
    return encoded_name.decode('utf-8')

def get_subj_alt_name(peer_cert: X509) -> list[tuple[str, str]]:
    if False:
        i = 10
        return i + 15
    '\n    Given an PyOpenSSL certificate, provides all the subject alternative names.\n    '
    cert = peer_cert.to_cryptography()
    try:
        ext = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    except x509.ExtensionNotFound:
        return []
    except (x509.DuplicateExtension, UnsupportedExtension, x509.UnsupportedGeneralNameType, UnicodeError) as e:
        log.warning('A problem was encountered with the certificate that prevented urllib3 from finding the SubjectAlternativeName field. This can affect certificate validation. The error was %s', e)
        return []
    names = [('DNS', name) for name in map(_dnsname_to_stdlib, ext.get_values_for_type(x509.DNSName)) if name is not None]
    names.extend((('IP Address', str(name)) for name in ext.get_values_for_type(x509.IPAddress)))
    return names

class WrappedSocket:
    """API-compatibility wrapper for Python OpenSSL's Connection-class."""

    def __init__(self, connection: OpenSSL.SSL.Connection, socket: socket_cls, suppress_ragged_eofs: bool=True) -> None:
        if False:
            print('Hello World!')
        self.connection = connection
        self.socket = socket
        self.suppress_ragged_eofs = suppress_ragged_eofs
        self._io_refs = 0
        self._closed = False

    def fileno(self) -> int:
        if False:
            print('Hello World!')
        return self.socket.fileno()

    def _decref_socketios(self) -> None:
        if False:
            print('Hello World!')
        if self._io_refs > 0:
            self._io_refs -= 1
        if self._closed:
            self.close()

    def recv(self, *args: typing.Any, **kwargs: typing.Any) -> bytes:
        if False:
            print('Hello World!')
        try:
            data = self.connection.recv(*args, **kwargs)
        except OpenSSL.SSL.SysCallError as e:
            if self.suppress_ragged_eofs and e.args == (-1, 'Unexpected EOF'):
                return b''
            else:
                raise OSError(e.args[0], str(e)) from e
        except OpenSSL.SSL.ZeroReturnError:
            if self.connection.get_shutdown() == OpenSSL.SSL.RECEIVED_SHUTDOWN:
                return b''
            else:
                raise
        except OpenSSL.SSL.WantReadError as e:
            if not util.wait_for_read(self.socket, self.socket.gettimeout()):
                raise timeout('The read operation timed out') from e
            else:
                return self.recv(*args, **kwargs)
        except OpenSSL.SSL.Error as e:
            raise ssl.SSLError(f'read error: {e!r}') from e
        else:
            return data

    def recv_into(self, *args: typing.Any, **kwargs: typing.Any) -> int:
        if False:
            i = 10
            return i + 15
        try:
            return self.connection.recv_into(*args, **kwargs)
        except OpenSSL.SSL.SysCallError as e:
            if self.suppress_ragged_eofs and e.args == (-1, 'Unexpected EOF'):
                return 0
            else:
                raise OSError(e.args[0], str(e)) from e
        except OpenSSL.SSL.ZeroReturnError:
            if self.connection.get_shutdown() == OpenSSL.SSL.RECEIVED_SHUTDOWN:
                return 0
            else:
                raise
        except OpenSSL.SSL.WantReadError as e:
            if not util.wait_for_read(self.socket, self.socket.gettimeout()):
                raise timeout('The read operation timed out') from e
            else:
                return self.recv_into(*args, **kwargs)
        except OpenSSL.SSL.Error as e:
            raise ssl.SSLError(f'read error: {e!r}') from e

    def settimeout(self, timeout: float) -> None:
        if False:
            i = 10
            return i + 15
        return self.socket.settimeout(timeout)

    def _send_until_done(self, data: bytes) -> int:
        if False:
            i = 10
            return i + 15
        while True:
            try:
                return self.connection.send(data)
            except OpenSSL.SSL.WantWriteError as e:
                if not util.wait_for_write(self.socket, self.socket.gettimeout()):
                    raise timeout() from e
                continue
            except OpenSSL.SSL.SysCallError as e:
                raise OSError(e.args[0], str(e)) from e

    def sendall(self, data: bytes) -> None:
        if False:
            for i in range(10):
                print('nop')
        total_sent = 0
        while total_sent < len(data):
            sent = self._send_until_done(data[total_sent:total_sent + SSL_WRITE_BLOCKSIZE])
            total_sent += sent

    def shutdown(self) -> None:
        if False:
            print('Hello World!')
        self.connection.shutdown()

    def close(self) -> None:
        if False:
            while True:
                i = 10
        self._closed = True
        if self._io_refs <= 0:
            self._real_close()

    def _real_close(self) -> None:
        if False:
            return 10
        try:
            return self.connection.close()
        except OpenSSL.SSL.Error:
            return

    def getpeercert(self, binary_form: bool=False) -> dict[str, list[typing.Any]] | None:
        if False:
            for i in range(10):
                print('nop')
        x509 = self.connection.get_peer_certificate()
        if not x509:
            return x509
        if binary_form:
            return OpenSSL.crypto.dump_certificate(OpenSSL.crypto.FILETYPE_ASN1, x509)
        return {'subject': ((('commonName', x509.get_subject().CN),),), 'subjectAltName': get_subj_alt_name(x509)}

    def version(self) -> str:
        if False:
            while True:
                i = 10
        return self.connection.get_protocol_version_name()
WrappedSocket.makefile = socket_cls.makefile

class PyOpenSSLContext:
    """
    I am a wrapper class for the PyOpenSSL ``Context`` object. I am responsible
    for translating the interface of the standard library ``SSLContext`` object
    to calls into PyOpenSSL.
    """

    def __init__(self, protocol: int) -> None:
        if False:
            return 10
        self.protocol = _openssl_versions[protocol]
        self._ctx = OpenSSL.SSL.Context(self.protocol)
        self._options = 0
        self.check_hostname = False
        self._minimum_version: int = ssl.TLSVersion.MINIMUM_SUPPORTED
        self._maximum_version: int = ssl.TLSVersion.MAXIMUM_SUPPORTED

    @property
    def options(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self._options

    @options.setter
    def options(self, value: int) -> None:
        if False:
            return 10
        self._options = value
        self._set_ctx_options()

    @property
    def verify_mode(self) -> int:
        if False:
            return 10
        return _openssl_to_stdlib_verify[self._ctx.get_verify_mode()]

    @verify_mode.setter
    def verify_mode(self, value: ssl.VerifyMode) -> None:
        if False:
            print('Hello World!')
        self._ctx.set_verify(_stdlib_to_openssl_verify[value], _verify_callback)

    def set_default_verify_paths(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._ctx.set_default_verify_paths()

    def set_ciphers(self, ciphers: bytes | str) -> None:
        if False:
            i = 10
            return i + 15
        if isinstance(ciphers, str):
            ciphers = ciphers.encode('utf-8')
        self._ctx.set_cipher_list(ciphers)

    def load_verify_locations(self, cafile: str | None=None, capath: str | None=None, cadata: bytes | None=None) -> None:
        if False:
            while True:
                i = 10
        if cafile is not None:
            cafile = cafile.encode('utf-8')
        if capath is not None:
            capath = capath.encode('utf-8')
        try:
            self._ctx.load_verify_locations(cafile, capath)
            if cadata is not None:
                self._ctx.load_verify_locations(BytesIO(cadata))
        except OpenSSL.SSL.Error as e:
            raise ssl.SSLError(f'unable to load trusted certificates: {e!r}') from e

    def load_cert_chain(self, certfile: str, keyfile: str | None=None, password: str | None=None) -> None:
        if False:
            print('Hello World!')
        try:
            self._ctx.use_certificate_chain_file(certfile)
            if password is not None:
                if not isinstance(password, bytes):
                    password = password.encode('utf-8')
                self._ctx.set_passwd_cb(lambda *_: password)
            self._ctx.use_privatekey_file(keyfile or certfile)
        except OpenSSL.SSL.Error as e:
            raise ssl.SSLError(f'Unable to load certificate chain: {e!r}') from e

    def set_alpn_protocols(self, protocols: list[bytes | str]) -> None:
        if False:
            print('Hello World!')
        protocols = [util.util.to_bytes(p, 'ascii') for p in protocols]
        return self._ctx.set_alpn_protos(protocols)

    def wrap_socket(self, sock: socket_cls, server_side: bool=False, do_handshake_on_connect: bool=True, suppress_ragged_eofs: bool=True, server_hostname: bytes | str | None=None) -> WrappedSocket:
        if False:
            i = 10
            return i + 15
        cnx = OpenSSL.SSL.Connection(self._ctx, sock)
        if server_hostname and (not util.ssl_.is_ipaddress(server_hostname)):
            if isinstance(server_hostname, str):
                server_hostname = server_hostname.encode('utf-8')
            cnx.set_tlsext_host_name(server_hostname)
        cnx.set_connect_state()
        while True:
            try:
                cnx.do_handshake()
            except OpenSSL.SSL.WantReadError as e:
                if not util.wait_for_read(sock, sock.gettimeout()):
                    raise timeout('select timed out') from e
                continue
            except OpenSSL.SSL.Error as e:
                raise ssl.SSLError(f'bad handshake: {e!r}') from e
            break
        return WrappedSocket(cnx, sock)

    def _set_ctx_options(self) -> None:
        if False:
            while True:
                i = 10
        self._ctx.set_options(self._options | _openssl_to_ssl_minimum_version[self._minimum_version] | _openssl_to_ssl_maximum_version[self._maximum_version])

    @property
    def minimum_version(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._minimum_version

    @minimum_version.setter
    def minimum_version(self, minimum_version: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._minimum_version = minimum_version
        self._set_ctx_options()

    @property
    def maximum_version(self) -> int:
        if False:
            print('Hello World!')
        return self._maximum_version

    @maximum_version.setter
    def maximum_version(self, maximum_version: int) -> None:
        if False:
            while True:
                i = 10
        self._maximum_version = maximum_version
        self._set_ctx_options()

def _verify_callback(cnx: OpenSSL.SSL.Connection, x509: X509, err_no: int, err_depth: int, return_code: int) -> bool:
    if False:
        i = 10
        return i + 15
    return err_no == 0