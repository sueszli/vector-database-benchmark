from __future__ import annotations
import json
from typing import TYPE_CHECKING, Any, Generic, Iterator, AsyncIterator
from typing_extensions import override
import httpx
from ._types import ResponseT
from ._utils import is_mapping
from ._exceptions import APIError
if TYPE_CHECKING:
    from ._base_client import SyncAPIClient, AsyncAPIClient

class Stream(Generic[ResponseT]):
    """Provides the core interface to iterate over a synchronous stream response."""
    response: httpx.Response

    def __init__(self, *, cast_to: type[ResponseT], response: httpx.Response, client: SyncAPIClient) -> None:
        if False:
            return 10
        self.response = response
        self._cast_to = cast_to
        self._client = client
        self._decoder = SSEDecoder()
        self._iterator = self.__stream__()

    def __next__(self) -> ResponseT:
        if False:
            return 10
        return self._iterator.__next__()

    def __iter__(self) -> Iterator[ResponseT]:
        if False:
            i = 10
            return i + 15
        for item in self._iterator:
            yield item

    def _iter_events(self) -> Iterator[ServerSentEvent]:
        if False:
            return 10
        yield from self._decoder.iter(self.response.iter_lines())

    def __stream__(self) -> Iterator[ResponseT]:
        if False:
            i = 10
            return i + 15
        cast_to = self._cast_to
        response = self.response
        process_data = self._client._process_response_data
        iterator = self._iter_events()
        for sse in iterator:
            if sse.data.startswith('[DONE]'):
                break
            if sse.event is None:
                data = sse.json()
                if is_mapping(data) and data.get('error'):
                    raise APIError(message='An error ocurred during streaming', request=self.response.request, body=data['error'])
                yield process_data(data=data, cast_to=cast_to, response=response)
        for sse in iterator:
            ...

class AsyncStream(Generic[ResponseT]):
    """Provides the core interface to iterate over an asynchronous stream response."""
    response: httpx.Response

    def __init__(self, *, cast_to: type[ResponseT], response: httpx.Response, client: AsyncAPIClient) -> None:
        if False:
            i = 10
            return i + 15
        self.response = response
        self._cast_to = cast_to
        self._client = client
        self._decoder = SSEDecoder()
        self._iterator = self.__stream__()

    async def __anext__(self) -> ResponseT:
        return await self._iterator.__anext__()

    async def __aiter__(self) -> AsyncIterator[ResponseT]:
        async for item in self._iterator:
            yield item

    async def _iter_events(self) -> AsyncIterator[ServerSentEvent]:
        async for sse in self._decoder.aiter(self.response.aiter_lines()):
            yield sse

    async def __stream__(self) -> AsyncIterator[ResponseT]:
        cast_to = self._cast_to
        response = self.response
        process_data = self._client._process_response_data
        iterator = self._iter_events()
        async for sse in iterator:
            if sse.data.startswith('[DONE]'):
                break
            if sse.event is None:
                data = sse.json()
                if is_mapping(data) and data.get('error'):
                    raise APIError(message='An error ocurred during streaming', request=self.response.request, body=data['error'])
                yield process_data(data=data, cast_to=cast_to, response=response)
        async for sse in iterator:
            ...

class ServerSentEvent:

    def __init__(self, *, event: str | None=None, data: str | None=None, id: str | None=None, retry: int | None=None) -> None:
        if False:
            print('Hello World!')
        if data is None:
            data = ''
        self._id = id
        self._data = data
        self._event = event or None
        self._retry = retry

    @property
    def event(self) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        return self._event

    @property
    def id(self) -> str | None:
        if False:
            print('Hello World!')
        return self._id

    @property
    def retry(self) -> int | None:
        if False:
            for i in range(10):
                print('nop')
        return self._retry

    @property
    def data(self) -> str:
        if False:
            while True:
                i = 10
        return self._data

    def json(self) -> Any:
        if False:
            print('Hello World!')
        return json.loads(self.data)

    @override
    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'ServerSentEvent(event={self.event}, data={self.data}, id={self.id}, retry={self.retry})'

class SSEDecoder:
    _data: list[str]
    _event: str | None
    _retry: int | None
    _last_event_id: str | None

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self._event = None
        self._data = []
        self._last_event_id = None
        self._retry = None

    def iter(self, iterator: Iterator[str]) -> Iterator[ServerSentEvent]:
        if False:
            return 10
        'Given an iterator that yields lines, iterate over it & yield every event encountered'
        for line in iterator:
            line = line.rstrip('\n')
            sse = self.decode(line)
            if sse is not None:
                yield sse

    async def aiter(self, iterator: AsyncIterator[str]) -> AsyncIterator[ServerSentEvent]:
        """Given an async iterator that yields lines, iterate over it & yield every event encountered"""
        async for line in iterator:
            line = line.rstrip('\n')
            sse = self.decode(line)
            if sse is not None:
                yield sse

    def decode(self, line: str) -> ServerSentEvent | None:
        if False:
            i = 10
            return i + 15
        if not line:
            if not self._event and (not self._data) and (not self._last_event_id) and (self._retry is None):
                return None
            sse = ServerSentEvent(event=self._event, data='\n'.join(self._data), id=self._last_event_id, retry=self._retry)
            self._event = None
            self._data = []
            self._retry = None
            return sse
        if line.startswith(':'):
            return None
        (fieldname, _, value) = line.partition(':')
        if value.startswith(' '):
            value = value[1:]
        if fieldname == 'event':
            self._event = value
        elif fieldname == 'data':
            self._data.append(value)
        elif fieldname == 'id':
            if '\x00' in value:
                pass
            else:
                self._last_event_id = value
        elif fieldname == 'retry':
            try:
                self._retry = int(value)
            except (TypeError, ValueError):
                pass
        else:
            pass
        return None