from collections.abc import AsyncIterator
from io import BytesIO
import js
from pyodide import JsException
from pyodide.http import pyfetch, FetchResponse
from azure.core.exceptions import HttpResponseError
from azure.core.pipeline import Pipeline
from azure.core.utils import CaseInsensitiveDict
from azure.core.rest._http_response_impl_async import AsyncHttpResponseImpl
from azure.core.pipeline.transport import HttpRequest, AsyncioRequestsTransport

class PyodideTransportResponse(AsyncHttpResponseImpl):
    """Async response object for the `PyodideTransport`."""

    def _js_stream(self) -> FetchResponse:
        if False:
            return 10
        'So we get a fresh stream every time.\n\n        :return: The stream stream\n        :rtype: ~pyodide.http.FetchResponse\n        '
        return self._internal_response.clone().js_response.body

    async def close(self) -> None:
        """We don't actually have control over closing connections in the browser, so we just pretend
        to close.

        :return: None
        :rtype: None
        """
        self._is_closed = True

    def body(self) -> bytes:
        if False:
            return 10
        'The body is just the content.\n\n        :return: The body of the response.\n        :rtype: bytes\n        '
        return self.content

    async def load_body(self) -> None:
        """Backcompat"""
        if self._content is None:
            self._content = await self._internal_response.clone().bytes()

class PyodideStreamDownloadGenerator(AsyncIterator):
    """Simple stream download generator that returns the contents of
    a request.

    :param pipeline: The pipeline object
    :type pipeline: ~azure.core.pipeline.Pipeline
    :param response: The response object
    :type response: ~azure.core.experimental.transport.PyodideTransportResponse
    """

    def __init__(self, pipeline: Pipeline, response: PyodideTransportResponse, *_, **kwargs):
        if False:
            i = 10
            return i + 15
        self._block_size = response._block_size
        self.response = response
        if kwargs.pop('decompress', False) and self.response.headers.get('enc', None) in ('gzip', 'deflate'):
            self._js_reader = response._js_stream().pipeThrough(js.DecompressionStream.new('gzip')).getReader()
        else:
            self._js_reader = response._js_stream().getReader()
        self._stream = BytesIO()
        self._closed = False
        self._buffer_left = 0
        self.done = False

    async def __anext__(self) -> bytes:
        """Get the next block of bytes.

        :return: The next block of bytes.
        :rtype: bytes
        """
        if self._closed:
            raise StopAsyncIteration()
        start_pos = self._stream.tell()
        self._stream.read()
        while self._buffer_left < self._block_size:
            read = await self._js_reader.read()
            if read.done:
                self._closed = True
                break
            self._buffer_left += self._stream.write(bytes(read.value))
        self._stream.seek(start_pos)
        self._buffer_left -= self._block_size
        return self._stream.read(self._block_size)

class PyodideTransport(AsyncioRequestsTransport):
    """**This object is experimental**, meaning it may be changed in a future release
    or might break with a future Pyodide release. This transport was built with Pyodide
    version 0.20.0.

    Implements a basic HTTP sender using the Pyodide Javascript Fetch API.
    """

    async def send(self, request: HttpRequest, **kwargs) -> PyodideTransportResponse:
        """Send request object according to configuration.

        :param request: The request object to be sent.
        :type request: ~azure.core.pipeline.transport.HttpRequest
        :return: An HTTPResponse object.
        :rtype: PyodideResponseTransport
        """
        stream_response = kwargs.pop('stream_response', False)
        endpoint = request.url
        request_headers = dict(request.headers)
        init = {'method': request.method, 'headers': request_headers, 'body': request.data, 'files': request.files, 'verify': kwargs.pop('connection_verify', self.connection_config.verify), 'cert': kwargs.pop('connection_cert', self.connection_config.cert), 'allow_redirects': False, **kwargs}
        try:
            response = await pyfetch(endpoint, **init)
        except JsException as error:
            raise HttpResponseError(error, error=error) from error
        headers = CaseInsensitiveDict(response.js_response.headers)
        transport_response = PyodideTransportResponse(request=request, internal_response=response, block_size=self.connection_config.data_block_size, status_code=response.status, reason=response.status_text, content_type=headers.get('content-type'), headers=headers, stream_download_generator=PyodideStreamDownloadGenerator)
        if not stream_response:
            await transport_response.load_body()
        return transport_response