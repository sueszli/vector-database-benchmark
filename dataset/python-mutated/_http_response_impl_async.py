from typing import Any, AsyncIterator, Optional, Type
from types import TracebackType
from ._rest_py3 import AsyncHttpResponse as _AsyncHttpResponse
from ._http_response_impl import _HttpResponseBaseImpl, _HttpResponseBackcompatMixinBase, _RestHttpClientTransportResponseBase
from ..utils._pipeline_transport_rest_shared import _pad_attr_name
from ..utils._pipeline_transport_rest_shared_async import _PartGenerator

class AsyncHttpResponseBackcompatMixin(_HttpResponseBackcompatMixinBase):
    """Backcompat mixin for async responses"""

    def __getattr__(self, attr):
        if False:
            while True:
                i = 10
        backcompat_attrs = ['parts']
        attr = _pad_attr_name(attr, backcompat_attrs)
        return super().__getattr__(attr)

    def parts(self):
        if False:
            for i in range(10):
                print('nop')
        'DEPRECATED: Assuming the content-type is multipart/mixed, will return the parts as an async iterator.\n        This is deprecated and will be removed in a later release.\n        :rtype: AsyncIterator\n        :return: The parts of the response\n        :raises ValueError: If the content is not multipart/mixed\n        '
        if not self.content_type or not self.content_type.startswith('multipart/mixed'):
            raise ValueError("You can't get parts if the response is not multipart/mixed")
        return _PartGenerator(self, default_http_response_type=RestAsyncHttpClientTransportResponse)

class AsyncHttpResponseImpl(_HttpResponseBaseImpl, _AsyncHttpResponse, AsyncHttpResponseBackcompatMixin):
    """AsyncHttpResponseImpl built on top of our HttpResponse protocol class.

    Since ~azure.core.rest.AsyncHttpResponse is an abstract base class, we need to
    implement HttpResponse for each of our transports. This is an implementation
    that each of the sync transport responses can be built on.

    :keyword request: The request that led to the response
    :type request: ~azure.core.rest.HttpRequest
    :keyword any internal_response: The response we get directly from the transport. For example, for our requests
     transport, this will be a requests.Response.
    :keyword optional[int] block_size: The block size we are using in our transport
    :keyword int status_code: The status code of the response
    :keyword str reason: The HTTP reason
    :keyword str content_type: The content type of the response
    :keyword MutableMapping[str, str] headers: The response headers
    :keyword Callable stream_download_generator: The stream download generator that we use to stream the response.
    """

    async def _set_read_checks(self):
        self._is_stream_consumed = True
        await self.close()

    async def read(self) -> bytes:
        """Read the response's bytes into memory.

        :return: The response's bytes
        :rtype: bytes
        """
        if self._content is None:
            parts = []
            async for part in self.iter_bytes():
                parts.append(part)
            self._content = b''.join(parts)
        await self._set_read_checks()
        return self._content

    async def iter_raw(self, **kwargs: Any) -> AsyncIterator[bytes]:
        """Asynchronously iterates over the response's bytes. Will not decompress in the process
        :return: An async iterator of bytes from the response
        :rtype: AsyncIterator[bytes]
        """
        self._stream_download_check()
        async for part in self._stream_download_generator(response=self, pipeline=None, decompress=False):
            yield part
        await self.close()

    async def iter_bytes(self, **kwargs: Any) -> AsyncIterator[bytes]:
        """Asynchronously iterates over the response's bytes. Will decompress in the process
        :return: An async iterator of bytes from the response
        :rtype: AsyncIterator[bytes]
        """
        if self._content is not None:
            for i in range(0, len(self.content), self._block_size):
                yield self.content[i:i + self._block_size]
        else:
            self._stream_download_check()
            async for part in self._stream_download_generator(response=self, pipeline=None, decompress=True):
                yield part
            await self.close()

    async def close(self) -> None:
        """Close the response.

        :return: None
        :rtype: None
        """
        if not self.is_closed:
            self._is_closed = True
            await self._internal_response.close()

    async def __aexit__(self, exc_type: Optional[Type[BaseException]]=None, exc_value: Optional[BaseException]=None, traceback: Optional[TracebackType]=None) -> None:
        await self.close()

    def __repr__(self) -> str:
        if False:
            return 10
        content_type_str = ', Content-Type: {}'.format(self.content_type) if self.content_type else ''
        return '<AsyncHttpResponse: {} {}{}>'.format(self.status_code, self.reason, content_type_str)

class RestAsyncHttpClientTransportResponse(_RestHttpClientTransportResponseBase, AsyncHttpResponseImpl):
    """Create a Rest HTTPResponse from an http.client response."""

    async def iter_bytes(self, **kwargs):
        raise TypeError('We do not support iter_bytes for this transport response')

    async def iter_raw(self, **kwargs):
        raise TypeError('We do not support iter_raw for this transport response')

    async def read(self):
        if self._content is None:
            self._content = self._internal_response.read()
        return self._content