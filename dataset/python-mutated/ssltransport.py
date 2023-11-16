from __future__ import annotations
import io
import socket
import ssl
import typing
from ..exceptions import ProxySchemeUnsupported
if typing.TYPE_CHECKING:
    from typing import Literal
    from .ssl_ import _TYPE_PEER_CERT_RET, _TYPE_PEER_CERT_RET_DICT
_SelfT = typing.TypeVar('_SelfT', bound='SSLTransport')
_WriteBuffer = typing.Union[bytearray, memoryview]
_ReturnValue = typing.TypeVar('_ReturnValue')
SSL_BLOCKSIZE = 16384

class SSLTransport:
    """
    The SSLTransport wraps an existing socket and establishes an SSL connection.

    Contrary to Python's implementation of SSLSocket, it allows you to chain
    multiple TLS connections together. It's particularly useful if you need to
    implement TLS within TLS.

    The class supports most of the socket API operations.
    """

    @staticmethod
    def _validate_ssl_context_for_tls_in_tls(ssl_context: ssl.SSLContext) -> None:
        if False:
            print('Hello World!')
        "\n        Raises a ProxySchemeUnsupported if the provided ssl_context can't be used\n        for TLS in TLS.\n\n        The only requirement is that the ssl_context provides the 'wrap_bio'\n        methods.\n        "
        if not hasattr(ssl_context, 'wrap_bio'):
            raise ProxySchemeUnsupported("TLS in TLS requires SSLContext.wrap_bio() which isn't available on non-native SSLContext")

    def __init__(self, socket: socket.socket, ssl_context: ssl.SSLContext, server_hostname: str | None=None, suppress_ragged_eofs: bool=True) -> None:
        if False:
            print('Hello World!')
        '\n        Create an SSLTransport around socket using the provided ssl_context.\n        '
        self.incoming = ssl.MemoryBIO()
        self.outgoing = ssl.MemoryBIO()
        self.suppress_ragged_eofs = suppress_ragged_eofs
        self.socket = socket
        self.sslobj = ssl_context.wrap_bio(self.incoming, self.outgoing, server_hostname=server_hostname)
        self._ssl_io_loop(self.sslobj.do_handshake)

    def __enter__(self: _SelfT) -> _SelfT:
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, *_: typing.Any) -> None:
        if False:
            i = 10
            return i + 15
        self.close()

    def fileno(self) -> int:
        if False:
            i = 10
            return i + 15
        return self.socket.fileno()

    def read(self, len: int=1024, buffer: typing.Any | None=None) -> int | bytes:
        if False:
            print('Hello World!')
        return self._wrap_ssl_read(len, buffer)

    def recv(self, buflen: int=1024, flags: int=0) -> int | bytes:
        if False:
            while True:
                i = 10
        if flags != 0:
            raise ValueError('non-zero flags not allowed in calls to recv')
        return self._wrap_ssl_read(buflen)

    def recv_into(self, buffer: _WriteBuffer, nbytes: int | None=None, flags: int=0) -> None | int | bytes:
        if False:
            i = 10
            return i + 15
        if flags != 0:
            raise ValueError('non-zero flags not allowed in calls to recv_into')
        if nbytes is None:
            nbytes = len(buffer)
        return self.read(nbytes, buffer)

    def sendall(self, data: bytes, flags: int=0) -> None:
        if False:
            for i in range(10):
                print('nop')
        if flags != 0:
            raise ValueError('non-zero flags not allowed in calls to sendall')
        count = 0
        with memoryview(data) as view, view.cast('B') as byte_view:
            amount = len(byte_view)
            while count < amount:
                v = self.send(byte_view[count:])
                count += v

    def send(self, data: bytes, flags: int=0) -> int:
        if False:
            i = 10
            return i + 15
        if flags != 0:
            raise ValueError('non-zero flags not allowed in calls to send')
        return self._ssl_io_loop(self.sslobj.write, data)

    def makefile(self, mode: str, buffering: int | None=None, *, encoding: str | None=None, errors: str | None=None, newline: str | None=None) -> typing.BinaryIO | typing.TextIO | socket.SocketIO:
        if False:
            print('Hello World!')
        "\n        Python's httpclient uses makefile and buffered io when reading HTTP\n        messages and we need to support it.\n\n        This is unfortunately a copy and paste of socket.py makefile with small\n        changes to point to the socket directly.\n        "
        if not set(mode) <= {'r', 'w', 'b'}:
            raise ValueError(f'invalid mode {mode!r} (only r, w, b allowed)')
        writing = 'w' in mode
        reading = 'r' in mode or not writing
        assert reading or writing
        binary = 'b' in mode
        rawmode = ''
        if reading:
            rawmode += 'r'
        if writing:
            rawmode += 'w'
        raw = socket.SocketIO(self, rawmode)
        self.socket._io_refs += 1
        if buffering is None:
            buffering = -1
        if buffering < 0:
            buffering = io.DEFAULT_BUFFER_SIZE
        if buffering == 0:
            if not binary:
                raise ValueError('unbuffered streams must be binary')
            return raw
        buffer: typing.BinaryIO
        if reading and writing:
            buffer = io.BufferedRWPair(raw, raw, buffering)
        elif reading:
            buffer = io.BufferedReader(raw, buffering)
        else:
            assert writing
            buffer = io.BufferedWriter(raw, buffering)
        if binary:
            return buffer
        text = io.TextIOWrapper(buffer, encoding, errors, newline)
        text.mode = mode
        return text

    def unwrap(self) -> None:
        if False:
            while True:
                i = 10
        self._ssl_io_loop(self.sslobj.unwrap)

    def close(self) -> None:
        if False:
            return 10
        self.socket.close()

    @typing.overload
    def getpeercert(self, binary_form: Literal[False]=...) -> _TYPE_PEER_CERT_RET_DICT | None:
        if False:
            for i in range(10):
                print('nop')
        ...

    @typing.overload
    def getpeercert(self, binary_form: Literal[True]) -> bytes | None:
        if False:
            print('Hello World!')
        ...

    def getpeercert(self, binary_form: bool=False) -> _TYPE_PEER_CERT_RET:
        if False:
            i = 10
            return i + 15
        return self.sslobj.getpeercert(binary_form)

    def version(self) -> str | None:
        if False:
            i = 10
            return i + 15
        return self.sslobj.version()

    def cipher(self) -> tuple[str, str, int] | None:
        if False:
            i = 10
            return i + 15
        return self.sslobj.cipher()

    def selected_alpn_protocol(self) -> str | None:
        if False:
            while True:
                i = 10
        return self.sslobj.selected_alpn_protocol()

    def selected_npn_protocol(self) -> str | None:
        if False:
            while True:
                i = 10
        return self.sslobj.selected_npn_protocol()

    def shared_ciphers(self) -> list[tuple[str, str, int]] | None:
        if False:
            for i in range(10):
                print('nop')
        return self.sslobj.shared_ciphers()

    def compression(self) -> str | None:
        if False:
            print('Hello World!')
        return self.sslobj.compression()

    def settimeout(self, value: float | None) -> None:
        if False:
            while True:
                i = 10
        self.socket.settimeout(value)

    def gettimeout(self) -> float | None:
        if False:
            i = 10
            return i + 15
        return self.socket.gettimeout()

    def _decref_socketios(self) -> None:
        if False:
            while True:
                i = 10
        self.socket._decref_socketios()

    def _wrap_ssl_read(self, len: int, buffer: bytearray | None=None) -> int | bytes:
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._ssl_io_loop(self.sslobj.read, len, buffer)
        except ssl.SSLError as e:
            if e.errno == ssl.SSL_ERROR_EOF and self.suppress_ragged_eofs:
                return 0
            else:
                raise

    @typing.overload
    def _ssl_io_loop(self, func: typing.Callable[[], None]) -> None:
        if False:
            i = 10
            return i + 15
        ...

    @typing.overload
    def _ssl_io_loop(self, func: typing.Callable[[bytes], int], arg1: bytes) -> int:
        if False:
            i = 10
            return i + 15
        ...

    @typing.overload
    def _ssl_io_loop(self, func: typing.Callable[[int, bytearray | None], bytes], arg1: int, arg2: bytearray | None) -> bytes:
        if False:
            return 10
        ...

    def _ssl_io_loop(self, func: typing.Callable[..., _ReturnValue], arg1: None | bytes | int=None, arg2: bytearray | None=None) -> _ReturnValue:
        if False:
            while True:
                i = 10
        'Performs an I/O loop between incoming/outgoing and the socket.'
        should_loop = True
        ret = None
        while should_loop:
            errno = None
            try:
                if arg1 is None and arg2 is None:
                    ret = func()
                elif arg2 is None:
                    ret = func(arg1)
                else:
                    ret = func(arg1, arg2)
            except ssl.SSLError as e:
                if e.errno not in (ssl.SSL_ERROR_WANT_READ, ssl.SSL_ERROR_WANT_WRITE):
                    raise e
                errno = e.errno
            buf = self.outgoing.read()
            self.socket.sendall(buf)
            if errno is None:
                should_loop = False
            elif errno == ssl.SSL_ERROR_WANT_READ:
                buf = self.socket.recv(SSL_BLOCKSIZE)
                if buf:
                    self.incoming.write(buf)
                else:
                    self.incoming.write_eof()
        return typing.cast(_ReturnValue, ret)