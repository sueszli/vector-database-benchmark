import asyncio
import socket
import sys
from typing import Callable, List, Optional, Union
if sys.version_info.major >= 3 and sys.version_info.minor >= 11:
    from asyncio import timeout as async_timeout
else:
    from async_timeout import timeout as async_timeout
from redis.compat import TypedDict
from ..exceptions import ConnectionError, InvalidResponse, RedisError
from ..typing import EncodableT
from ..utils import HIREDIS_AVAILABLE
from .base import AsyncBaseParser, BaseParser
from .socket import NONBLOCKING_EXCEPTION_ERROR_NUMBERS, NONBLOCKING_EXCEPTIONS, SENTINEL, SERVER_CLOSED_CONNECTION_ERROR

class _HiredisReaderArgs(TypedDict, total=False):
    protocolError: Callable[[str], Exception]
    replyError: Callable[[str], Exception]
    encoding: Optional[str]
    errors: Optional[str]

class _HiredisParser(BaseParser):
    """Parser class for connections using Hiredis"""

    def __init__(self, socket_read_size):
        if False:
            i = 10
            return i + 15
        if not HIREDIS_AVAILABLE:
            raise RedisError('Hiredis is not installed')
        self.socket_read_size = socket_read_size
        self._buffer = bytearray(socket_read_size)

    def __del__(self):
        if False:
            i = 10
            return i + 15
        try:
            self.on_disconnect()
        except Exception:
            pass

    def on_connect(self, connection, **kwargs):
        if False:
            return 10
        import hiredis
        self._sock = connection._sock
        self._socket_timeout = connection.socket_timeout
        kwargs = {'protocolError': InvalidResponse, 'replyError': self.parse_error, 'errors': connection.encoder.encoding_errors}
        if connection.encoder.decode_responses:
            kwargs['encoding'] = connection.encoder.encoding
        self._reader = hiredis.Reader(**kwargs)
        self._next_response = False

    def on_disconnect(self):
        if False:
            while True:
                i = 10
        self._sock = None
        self._reader = None
        self._next_response = False

    def can_read(self, timeout):
        if False:
            i = 10
            return i + 15
        if not self._reader:
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)
        if self._next_response is False:
            self._next_response = self._reader.gets()
            if self._next_response is False:
                return self.read_from_socket(timeout=timeout, raise_on_timeout=False)
        return True

    def read_from_socket(self, timeout=SENTINEL, raise_on_timeout=True):
        if False:
            print('Hello World!')
        sock = self._sock
        custom_timeout = timeout is not SENTINEL
        try:
            if custom_timeout:
                sock.settimeout(timeout)
            bufflen = self._sock.recv_into(self._buffer)
            if bufflen == 0:
                raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)
            self._reader.feed(self._buffer, 0, bufflen)
            return True
        except socket.timeout:
            if raise_on_timeout:
                raise TimeoutError('Timeout reading from socket')
            return False
        except NONBLOCKING_EXCEPTIONS as ex:
            allowed = NONBLOCKING_EXCEPTION_ERROR_NUMBERS.get(ex.__class__, -1)
            if not raise_on_timeout and ex.errno == allowed:
                return False
            raise ConnectionError(f'Error while reading from socket: {ex.args}')
        finally:
            if custom_timeout:
                sock.settimeout(self._socket_timeout)

    def read_response(self, disable_decoding=False):
        if False:
            while True:
                i = 10
        if not self._reader:
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)
        if self._next_response is not False:
            response = self._next_response
            self._next_response = False
            return response
        if disable_decoding:
            response = self._reader.gets(False)
        else:
            response = self._reader.gets()
        while response is False:
            self.read_from_socket()
            if disable_decoding:
                response = self._reader.gets(False)
            else:
                response = self._reader.gets()
        if isinstance(response, ConnectionError):
            raise response
        elif isinstance(response, list) and response and isinstance(response[0], ConnectionError):
            raise response[0]
        return response

class _AsyncHiredisParser(AsyncBaseParser):
    """Async implementation of parser class for connections using Hiredis"""
    __slots__ = ('_reader',)

    def __init__(self, socket_read_size: int):
        if False:
            while True:
                i = 10
        if not HIREDIS_AVAILABLE:
            raise RedisError('Hiredis is not available.')
        super().__init__(socket_read_size=socket_read_size)
        self._reader = None

    def on_connect(self, connection):
        if False:
            print('Hello World!')
        import hiredis
        self._stream = connection._reader
        kwargs: _HiredisReaderArgs = {'protocolError': InvalidResponse, 'replyError': self.parse_error}
        if connection.encoder.decode_responses:
            kwargs['encoding'] = connection.encoder.encoding
            kwargs['errors'] = connection.encoder.encoding_errors
        self._reader = hiredis.Reader(**kwargs)
        self._connected = True

    def on_disconnect(self):
        if False:
            i = 10
            return i + 15
        self._connected = False

    async def can_read_destructive(self):
        if not self._connected:
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)
        if self._reader.gets():
            return True
        try:
            async with async_timeout(0):
                return await self.read_from_socket()
        except asyncio.TimeoutError:
            return False

    async def read_from_socket(self):
        buffer = await self._stream.read(self._read_size)
        if not buffer or not isinstance(buffer, bytes):
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR) from None
        self._reader.feed(buffer)
        return True

    async def read_response(self, disable_decoding: bool=False) -> Union[EncodableT, List[EncodableT]]:
        if not self._connected:
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR) from None
        response = self._reader.gets()
        while response is False:
            await self.read_from_socket()
            response = self._reader.gets()
        if isinstance(response, ConnectionError):
            raise response
        elif isinstance(response, list) and response and isinstance(response[0], ConnectionError):
            raise response[0]
        return response