import os
import platform
import socket
import ssl
import typing
import _ssl
from ._ssl_constants import _original_SSLContext, _original_super_SSLContext, _truststore_SSLContext_dunder_class, _truststore_SSLContext_super_class
if platform.system() == 'Windows':
    from ._windows import _configure_context, _verify_peercerts_impl
elif platform.system() == 'Darwin':
    from ._macos import _configure_context, _verify_peercerts_impl
else:
    from ._openssl import _configure_context, _verify_peercerts_impl
if typing.TYPE_CHECKING:
    from pip._vendor.typing_extensions import Buffer
_StrOrBytesPath: typing.TypeAlias = str | bytes | os.PathLike[str] | os.PathLike[bytes]
_PasswordType: typing.TypeAlias = str | bytes | typing.Callable[[], str | bytes]

def inject_into_ssl() -> None:
    if False:
        i = 10
        return i + 15
    'Injects the :class:`truststore.SSLContext` into the ``ssl``\n    module by replacing :class:`ssl.SSLContext`.\n    '
    setattr(ssl, 'SSLContext', SSLContext)
    try:
        import pip._vendor.urllib3.util.ssl_ as urllib3_ssl
        setattr(urllib3_ssl, 'SSLContext', SSLContext)
    except ImportError:
        pass

def extract_from_ssl() -> None:
    if False:
        return 10
    'Restores the :class:`ssl.SSLContext` class to its original state'
    setattr(ssl, 'SSLContext', _original_SSLContext)
    try:
        import pip._vendor.urllib3.util.ssl_ as urllib3_ssl
        urllib3_ssl.SSLContext = _original_SSLContext
    except ImportError:
        pass

class SSLContext(_truststore_SSLContext_super_class):
    """SSLContext API that uses system certificates on all platforms"""

    @property
    def __class__(self) -> type:
        if False:
            while True:
                i = 10
        return _truststore_SSLContext_dunder_class or SSLContext

    def __init__(self, protocol: int=None) -> None:
        if False:
            i = 10
            return i + 15
        self._ctx = _original_SSLContext(protocol)

        class TruststoreSSLObject(ssl.SSLObject):

            def do_handshake(self) -> None:
                if False:
                    while True:
                        i = 10
                ret = super().do_handshake()
                _verify_peercerts(self, server_hostname=self.server_hostname)
                return ret
        self._ctx.sslobject_class = TruststoreSSLObject

    def wrap_socket(self, sock: socket.socket, server_side: bool=False, do_handshake_on_connect: bool=True, suppress_ragged_eofs: bool=True, server_hostname: str | None=None, session: ssl.SSLSession | None=None) -> ssl.SSLSocket:
        if False:
            i = 10
            return i + 15
        with _configure_context(self._ctx):
            ssl_sock = self._ctx.wrap_socket(sock, server_side=server_side, server_hostname=server_hostname, do_handshake_on_connect=do_handshake_on_connect, suppress_ragged_eofs=suppress_ragged_eofs, session=session)
        try:
            _verify_peercerts(ssl_sock, server_hostname=server_hostname)
        except Exception:
            ssl_sock.close()
            raise
        return ssl_sock

    def wrap_bio(self, incoming: ssl.MemoryBIO, outgoing: ssl.MemoryBIO, server_side: bool=False, server_hostname: str | None=None, session: ssl.SSLSession | None=None) -> ssl.SSLObject:
        if False:
            for i in range(10):
                print('nop')
        with _configure_context(self._ctx):
            ssl_obj = self._ctx.wrap_bio(incoming, outgoing, server_hostname=server_hostname, server_side=server_side, session=session)
        return ssl_obj

    def load_verify_locations(self, cafile: str | bytes | os.PathLike[str] | os.PathLike[bytes] | None=None, capath: str | bytes | os.PathLike[str] | os.PathLike[bytes] | None=None, cadata: typing.Union[str, 'Buffer', None]=None) -> None:
        if False:
            i = 10
            return i + 15
        return self._ctx.load_verify_locations(cafile=cafile, capath=capath, cadata=cadata)

    def load_cert_chain(self, certfile: _StrOrBytesPath, keyfile: _StrOrBytesPath | None=None, password: _PasswordType | None=None) -> None:
        if False:
            print('Hello World!')
        return self._ctx.load_cert_chain(certfile=certfile, keyfile=keyfile, password=password)

    def load_default_certs(self, purpose: ssl.Purpose=ssl.Purpose.SERVER_AUTH) -> None:
        if False:
            print('Hello World!')
        return self._ctx.load_default_certs(purpose)

    def set_alpn_protocols(self, alpn_protocols: typing.Iterable[str]) -> None:
        if False:
            print('Hello World!')
        return self._ctx.set_alpn_protocols(alpn_protocols)

    def set_npn_protocols(self, npn_protocols: typing.Iterable[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        return self._ctx.set_npn_protocols(npn_protocols)

    def set_ciphers(self, __cipherlist: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        return self._ctx.set_ciphers(__cipherlist)

    def get_ciphers(self) -> typing.Any:
        if False:
            for i in range(10):
                print('nop')
        return self._ctx.get_ciphers()

    def session_stats(self) -> dict[str, int]:
        if False:
            print('Hello World!')
        return self._ctx.session_stats()

    def cert_store_stats(self) -> dict[str, int]:
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @typing.overload
    def get_ca_certs(self, binary_form: typing.Literal[False]=...) -> list[typing.Any]:
        if False:
            i = 10
            return i + 15
        ...

    @typing.overload
    def get_ca_certs(self, binary_form: typing.Literal[True]=...) -> list[bytes]:
        if False:
            return 10
        ...

    @typing.overload
    def get_ca_certs(self, binary_form: bool=...) -> typing.Any:
        if False:
            for i in range(10):
                print('nop')
        ...

    def get_ca_certs(self, binary_form: bool=False) -> list[typing.Any] | list[bytes]:
        if False:
            return 10
        raise NotImplementedError()

    @property
    def check_hostname(self) -> bool:
        if False:
            return 10
        return self._ctx.check_hostname

    @check_hostname.setter
    def check_hostname(self, value: bool) -> None:
        if False:
            while True:
                i = 10
        self._ctx.check_hostname = value

    @property
    def hostname_checks_common_name(self) -> bool:
        if False:
            print('Hello World!')
        return self._ctx.hostname_checks_common_name

    @hostname_checks_common_name.setter
    def hostname_checks_common_name(self, value: bool) -> None:
        if False:
            while True:
                i = 10
        self._ctx.hostname_checks_common_name = value

    @property
    def keylog_filename(self) -> str:
        if False:
            i = 10
            return i + 15
        return self._ctx.keylog_filename

    @keylog_filename.setter
    def keylog_filename(self, value: str) -> None:
        if False:
            print('Hello World!')
        self._ctx.keylog_filename = value

    @property
    def maximum_version(self) -> ssl.TLSVersion:
        if False:
            while True:
                i = 10
        return self._ctx.maximum_version

    @maximum_version.setter
    def maximum_version(self, value: ssl.TLSVersion) -> None:
        if False:
            i = 10
            return i + 15
        _original_super_SSLContext.maximum_version.__set__(self._ctx, value)

    @property
    def minimum_version(self) -> ssl.TLSVersion:
        if False:
            return 10
        return self._ctx.minimum_version

    @minimum_version.setter
    def minimum_version(self, value: ssl.TLSVersion) -> None:
        if False:
            while True:
                i = 10
        _original_super_SSLContext.minimum_version.__set__(self._ctx, value)

    @property
    def options(self) -> ssl.Options:
        if False:
            print('Hello World!')
        return self._ctx.options

    @options.setter
    def options(self, value: ssl.Options) -> None:
        if False:
            while True:
                i = 10
        _original_super_SSLContext.options.__set__(self._ctx, value)

    @property
    def post_handshake_auth(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self._ctx.post_handshake_auth

    @post_handshake_auth.setter
    def post_handshake_auth(self, value: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._ctx.post_handshake_auth = value

    @property
    def protocol(self) -> ssl._SSLMethod:
        if False:
            while True:
                i = 10
        return self._ctx.protocol

    @property
    def security_level(self) -> int:
        if False:
            while True:
                i = 10
        return self._ctx.security_level

    @property
    def verify_flags(self) -> ssl.VerifyFlags:
        if False:
            i = 10
            return i + 15
        return self._ctx.verify_flags

    @verify_flags.setter
    def verify_flags(self, value: ssl.VerifyFlags) -> None:
        if False:
            while True:
                i = 10
        _original_super_SSLContext.verify_flags.__set__(self._ctx, value)

    @property
    def verify_mode(self) -> ssl.VerifyMode:
        if False:
            for i in range(10):
                print('nop')
        return self._ctx.verify_mode

    @verify_mode.setter
    def verify_mode(self, value: ssl.VerifyMode) -> None:
        if False:
            print('Hello World!')
        _original_super_SSLContext.verify_mode.__set__(self._ctx, value)

def _verify_peercerts(sock_or_sslobj: ssl.SSLSocket | ssl.SSLObject, server_hostname: str | None) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Verifies the peer certificates from an SSLSocket or SSLObject\n    against the certificates in the OS trust store.\n    '
    sslobj: ssl.SSLObject = sock_or_sslobj
    try:
        while not hasattr(sslobj, 'get_unverified_chain'):
            sslobj = sslobj._sslobj
    except AttributeError:
        pass
    unverified_chain: typing.Sequence[_ssl.Certificate] = sslobj.get_unverified_chain() or ()
    cert_bytes = [cert.public_bytes(_ssl.ENCODING_DER) for cert in unverified_chain]
    _verify_peercerts_impl(sock_or_sslobj.context, cert_bytes, server_hostname=server_hostname)