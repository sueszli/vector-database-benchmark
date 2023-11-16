from typing import Any, Union
from ..exceptions import ConnectionError, InvalidResponse, ResponseError
from ..typing import EncodableT
from .base import _AsyncRESPBase, _RESPBase
from .socket import SERVER_CLOSED_CONNECTION_ERROR

class _RESP2Parser(_RESPBase):
    """RESP2 protocol implementation"""

    def read_response(self, disable_decoding=False):
        if False:
            return 10
        pos = self._buffer.get_pos() if self._buffer else None
        try:
            result = self._read_response(disable_decoding=disable_decoding)
        except BaseException:
            if self._buffer:
                self._buffer.rewind(pos)
            raise
        else:
            self._buffer.purge()
            return result

    def _read_response(self, disable_decoding=False):
        if False:
            print('Hello World!')
        raw = self._buffer.readline()
        if not raw:
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)
        (byte, response) = (raw[:1], raw[1:])
        if byte == b'-':
            response = response.decode('utf-8', errors='replace')
            error = self.parse_error(response)
            if isinstance(error, ConnectionError):
                raise error
            return error
        elif byte == b'+':
            pass
        elif byte == b':':
            return int(response)
        elif byte == b'$' and response == b'-1':
            return None
        elif byte == b'$':
            response = self._buffer.read(int(response))
        elif byte == b'*' and response == b'-1':
            return None
        elif byte == b'*':
            response = [self._read_response(disable_decoding=disable_decoding) for i in range(int(response))]
        else:
            raise InvalidResponse(f'Protocol Error: {raw!r}')
        if disable_decoding is False:
            response = self.encoder.decode(response)
        return response

class _AsyncRESP2Parser(_AsyncRESPBase):
    """Async class for the RESP2 protocol"""

    async def read_response(self, disable_decoding: bool=False):
        if not self._connected:
            raise ConnectionError(SERVER_CLOSED_CONNECTION_ERROR)
        if self._chunks:
            self._buffer += b''.join(self._chunks)
            self._chunks.clear()
        self._pos = 0
        response = await self._read_response(disable_decoding=disable_decoding)
        self._clear()
        return response

    async def _read_response(self, disable_decoding: bool=False) -> Union[EncodableT, ResponseError, None]:
        raw = await self._readline()
        response: Any
        (byte, response) = (raw[:1], raw[1:])
        if byte == b'-':
            response = response.decode('utf-8', errors='replace')
            error = self.parse_error(response)
            if isinstance(error, ConnectionError):
                self._clear()
                raise error
            return error
        elif byte == b'+':
            pass
        elif byte == b':':
            return int(response)
        elif byte == b'$' and response == b'-1':
            return None
        elif byte == b'$':
            response = await self._read(int(response))
        elif byte == b'*' and response == b'-1':
            return None
        elif byte == b'*':
            response = [await self._read_response(disable_decoding) for _ in range(int(response))]
        else:
            raise InvalidResponse(f'Protocol Error: {raw!r}')
        if disable_decoding is False:
            response = self.encoder.decode(response)
        return response