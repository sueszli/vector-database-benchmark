import collections.abc
import asyncio
from itertools import groupby
from typing import Iterator, cast
from multidict import CIMultiDict
from ._http_response_impl_async import AsyncHttpResponseImpl, AsyncHttpResponseBackcompatMixin
from ..pipeline.transport._aiohttp import AioHttpStreamDownloadGenerator
from ..utils._pipeline_transport_rest_shared import _pad_attr_name, _aiohttp_body_helper
from ..exceptions import ResponseNotReadError

class _ItemsView(collections.abc.ItemsView):

    def __init__(self, ref):
        if False:
            while True:
                i = 10
        super().__init__(ref)
        self._ref = ref

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        for (key, groups) in groupby(self._ref.__iter__(), lambda x: x[0]):
            yield tuple([key, ', '.join((group[1] for group in groups))])

    def __contains__(self, item):
        if False:
            for i in range(10):
                print('nop')
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            return False
        for (k, v) in self.__iter__():
            if item[0].lower() == k.lower() and item[1] == v:
                return True
        return False

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'dict_items({list(self.__iter__())})'

class _KeysView(collections.abc.KeysView):

    def __init__(self, items):
        if False:
            i = 10
            return i + 15
        super().__init__(items)
        self._items = items

    def __iter__(self) -> Iterator[str]:
        if False:
            for i in range(10):
                print('nop')
        for (key, _) in self._items:
            yield key

    def __contains__(self, key):
        if False:
            i = 10
            return i + 15
        try:
            for k in self.__iter__():
                if cast(str, key).lower() == k.lower():
                    return True
        except AttributeError:
            pass
        return False

    def __repr__(self) -> str:
        if False:
            return 10
        return f'dict_keys({list(self.__iter__())})'

class _ValuesView(collections.abc.ValuesView):

    def __init__(self, items):
        if False:
            print('Hello World!')
        super().__init__(items)
        self._items = items

    def __iter__(self):
        if False:
            return 10
        for (_, value) in self._items:
            yield value

    def __contains__(self, value):
        if False:
            i = 10
            return i + 15
        for v in self.__iter__():
            if value == v:
                return True
        return False

    def __repr__(self):
        if False:
            print('Hello World!')
        return f'dict_values({list(self.__iter__())})'

class _CIMultiDict(CIMultiDict):
    """Dictionary with the support for duplicate case-insensitive keys."""

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self.keys())

    def keys(self):
        if False:
            while True:
                i = 10
        "Return a new view of the dictionary's keys.\n\n        :return: A new view of the dictionary's keys\n        :rtype: ~collections.abc.KeysView\n        "
        return _KeysView(self.items())

    def items(self):
        if False:
            print('Hello World!')
        "Return a new view of the dictionary's items.\n\n        :return: A new view of the dictionary's items\n        :rtype: ~collections.abc.ItemsView\n        "
        return _ItemsView(super().items())

    def values(self):
        if False:
            print('Hello World!')
        "Return a new view of the dictionary's values.\n\n        :return: A new view of the dictionary's values\n        :rtype: ~collections.abc.ValuesView\n        "
        return _ValuesView(self.items())

    def __getitem__(self, key: str) -> str:
        if False:
            return 10
        return ', '.join(self.getall(key, []))

    def get(self, key, default=None):
        if False:
            return 10
        values = self.getall(key, None)
        if values:
            values = ', '.join(values)
        return values or default

class _RestAioHttpTransportResponseBackcompatMixin(AsyncHttpResponseBackcompatMixin):
    """Backcompat mixin for aiohttp responses.

    Need to add it's own mixin because it has function load_body, which other
    transport responses don't have, and also because we need to synchronously
    decompress the body if users call .body()
    """

    def body(self) -> bytes:
        if False:
            while True:
                i = 10
        "Return the whole body as bytes in memory.\n\n        Have to modify the default behavior here. In AioHttp, we do decompression\n        when accessing the body method. The behavior here is the same as if the\n        caller did an async read of the response first. But for backcompat reasons,\n        we need to support this decompression within the synchronous body method.\n\n        :return: The response's bytes\n        :rtype: bytes\n        "
        return _aiohttp_body_helper(self)

    async def _load_body(self) -> None:
        """Load in memory the body, so it could be accessible from sync methods."""
        self._content = await self.read()

    def __getattr__(self, attr):
        if False:
            i = 10
            return i + 15
        backcompat_attrs = ['load_body']
        attr = _pad_attr_name(attr, backcompat_attrs)
        return super().__getattr__(attr)

class RestAioHttpTransportResponse(AsyncHttpResponseImpl, _RestAioHttpTransportResponseBackcompatMixin):

    def __init__(self, *, internal_response, decompress: bool=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        headers = _CIMultiDict(internal_response.headers)
        super().__init__(internal_response=internal_response, status_code=internal_response.status, headers=headers, content_type=headers.get('content-type'), reason=internal_response.reason, stream_download_generator=AioHttpStreamDownloadGenerator, content=None, **kwargs)
        self._decompress = decompress
        self._decompressed_content = False

    def __getstate__(self):
        if False:
            return 10
        state = self.__dict__.copy()
        state['_internal_response'] = None
        state['headers'] = CIMultiDict(self.headers)
        return state

    @property
    def content(self) -> bytes:
        if False:
            i = 10
            return i + 15
        "Return the response's content in bytes.\n\n        :return: The response's content in bytes\n        :rtype: bytes\n        "
        if self._content is None:
            raise ResponseNotReadError(self)
        return _aiohttp_body_helper(self)

    async def read(self) -> bytes:
        """Read the response's bytes into memory.

        :return: The response's bytes
        :rtype: bytes
        """
        if not self._content:
            self._stream_download_check()
            self._content = await self._internal_response.read()
        await self._set_read_checks()
        return _aiohttp_body_helper(self)

    async def close(self) -> None:
        """Close the response.

        :return: None
        :rtype: None
        """
        if not self.is_closed:
            self._is_closed = True
            self._internal_response.close()
            await asyncio.sleep(0)