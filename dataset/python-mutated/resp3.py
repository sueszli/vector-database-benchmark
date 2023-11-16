from logging import getLogger
from typing import Any, Union
from ..exceptions import ConnectionError, InvalidResponse, ResponseError
from ..typing import EncodableT
from .base import _AsyncRESPBase, _RESPBase
from .socket import SERVER_CLOSED_CONNECTION_ERROR

class _RESP3Parser(_RESPBase):
    """RESP3 protocol implementation"""

    def __init__(self, socket_read_size):
        if False:
            print('Hello World!')
        super().__init__(socket_read_size)
        self.push_handler_func = self.handle_push_response

    def handle_push_response(self, response):
        if False:
            print('Hello World!')
        logger = getLogger('push_response')
        logger.info('Push response: ' + str(response))
        return response

    def read_response(self, disable_decoding=False, push_request=False):
        if False:
            print('Hello World!')
        pos = self._buffer.get_pos() if self._buffer else None
        try:
            result = self._read_response(disable_decoding=disable_decoding, push_request=push_request)
        except BaseException:
            if self._buffer:
                self._buffer.rewind(pos)
            raise
        else:
            self._buffer.purge()
            return result

    def _read_response(self, disable_decoding=False, push_request=False):
        if False:
            print('Hello World!')
        raw = self._buffer.readline()
        if not raw:
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)
        (byte, response) = (raw[:1], raw[1:])
        if byte in (b'-', b'!'):
            if byte == b'!':
                response = self._buffer.read(int(response))
            response = response.decode('utf-8', errors='replace')
            error = self.parse_error(response)
            if isinstance(error, ConnectionError):
                raise error
            return error
        elif byte == b'+':
            pass
        elif byte == b'_':
            return None
        elif byte in (b':', b'('):
            return int(response)
        elif byte == b',':
            return float(response)
        elif byte == b'#':
            return response == b't'
        elif byte == b'$':
            response = self._buffer.read(int(response))
        elif byte == b'=':
            response = self._buffer.read(int(response))[4:]
        elif byte == b'*':
            response = [self._read_response(disable_decoding=disable_decoding) for _ in range(int(response))]
        elif byte == b'~':
            response = [self._read_response(disable_decoding=disable_decoding) for _ in range(int(response))]
            try:
                response = set(response)
            except TypeError:
                pass
        elif byte == b'%':
            resp_dict = {}
            for _ in range(int(response)):
                key = self._read_response(disable_decoding=disable_decoding)
                resp_dict[key] = self._read_response(disable_decoding=disable_decoding, push_request=push_request)
            response = resp_dict
        elif byte == b'>':
            response = [self._read_response(disable_decoding=disable_decoding, push_request=push_request) for _ in range(int(response))]
            res = self.push_handler_func(response)
            if not push_request:
                return self._read_response(disable_decoding=disable_decoding, push_request=push_request)
            else:
                return res
        else:
            raise InvalidResponse(f'Protocol Error: {raw!r}')
        if isinstance(response, bytes) and disable_decoding is False:
            response = self.encoder.decode(response)
        return response

    def set_push_handler(self, push_handler_func):
        if False:
            for i in range(10):
                print('nop')
        self.push_handler_func = push_handler_func

class _AsyncRESP3Parser(_AsyncRESPBase):

    def __init__(self, socket_read_size):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(socket_read_size)
        self.push_handler_func = self.handle_push_response

    def handle_push_response(self, response):
        if False:
            print('Hello World!')
        logger = getLogger('push_response')
        logger.info('Push response: ' + str(response))
        return response

    async def read_response(self, disable_decoding: bool=False, push_request: bool=False):
        if self._chunks:
            self._buffer += b''.join(self._chunks)
            self._chunks.clear()
        self._pos = 0
        response = await self._read_response(disable_decoding=disable_decoding, push_request=push_request)
        self._clear()
        return response

    async def _read_response(self, disable_decoding: bool=False, push_request: bool=False) -> Union[EncodableT, ResponseError, None]:
        if not self._stream or not self.encoder:
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)
        raw = await self._readline()
        response: Any
        (byte, response) = (raw[:1], raw[1:])
        if byte in (b'-', b'!'):
            if byte == b'!':
                response = await self._read(int(response))
            response = response.decode('utf-8', errors='replace')
            error = self.parse_error(response)
            if isinstance(error, ConnectionError):
                self._clear()
                raise error
            return error
        elif byte == b'+':
            pass
        elif byte == b'_':
            return None
        elif byte in (b':', b'('):
            return int(response)
        elif byte == b',':
            return float(response)
        elif byte == b'#':
            return response == b't'
        elif byte == b'$':
            response = await self._read(int(response))
        elif byte == b'=':
            response = (await self._read(int(response)))[4:]
        elif byte == b'*':
            response = [await self._read_response(disable_decoding=disable_decoding) for _ in range(int(response))]
        elif byte == b'~':
            response = [await self._read_response(disable_decoding=disable_decoding) for _ in range(int(response))]
            try:
                response = set(response)
            except TypeError:
                pass
        elif byte == b'%':
            resp_dict = {}
            for _ in range(int(response)):
                key = await self._read_response(disable_decoding=disable_decoding)
                resp_dict[key] = await self._read_response(disable_decoding=disable_decoding, push_request=push_request)
            response = resp_dict
        elif byte == b'>':
            response = [await self._read_response(disable_decoding=disable_decoding, push_request=push_request) for _ in range(int(response))]
            res = self.push_handler_func(response)
            if not push_request:
                return await self._read_response(disable_decoding=disable_decoding, push_request=push_request)
            else:
                return res
        else:
            raise InvalidResponse(f'Protocol Error: {raw!r}')
        if isinstance(response, bytes) and disable_decoding is False:
            response = self.encoder.decode(response)
        return response

    def set_push_handler(self, push_handler_func):
        if False:
            print('Hello World!')
        self.push_handler_func = push_handler_func