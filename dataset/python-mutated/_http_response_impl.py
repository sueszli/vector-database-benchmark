from json import loads
from typing import Any, Optional, Iterator, MutableMapping, Callable
from http.client import HTTPResponse as _HTTPResponse
from ._helpers import get_charset_encoding, decode_to_text
from ..exceptions import HttpResponseError, ResponseNotReadError, StreamConsumedError, StreamClosedError
from ._rest_py3 import _HttpResponseBase, HttpResponse as _HttpResponse, HttpRequest as _HttpRequest
from ..utils._utils import case_insensitive_dict
from ..utils._pipeline_transport_rest_shared import _pad_attr_name, BytesIOSocket, _decode_parts_helper, _get_raw_parts_helper, _parts_helper

class _HttpResponseBackcompatMixinBase:
    """Base Backcompat mixin for responses.

    This mixin is used by both sync and async HttpResponse
    backcompat mixins.
    """

    def __getattr__(self, attr):
        if False:
            i = 10
            return i + 15
        backcompat_attrs = ['body', 'internal_response', 'block_size', 'stream_download']
        attr = _pad_attr_name(attr, backcompat_attrs)
        return self.__getattribute__(attr)

    def __setattr__(self, attr, value):
        if False:
            i = 10
            return i + 15
        backcompat_attrs = ['block_size', 'internal_response', 'request', 'status_code', 'headers', 'reason', 'content_type', 'stream_download']
        attr = _pad_attr_name(attr, backcompat_attrs)
        super(_HttpResponseBackcompatMixinBase, self).__setattr__(attr, value)

    def _body(self):
        if False:
            for i in range(10):
                print('nop')
        'DEPRECATED: Get the response body.\n        This is deprecated and will be removed in a later release.\n        You should get it through the `content` property instead\n\n        :return: The response body.\n        :rtype: bytes\n        '
        self.read()
        return self.content

    def _decode_parts(self, message, http_response_type, requests):
        if False:
            while True:
                i = 10
        'Helper for _decode_parts.\n\n        Rebuild an HTTP response from pure string.\n\n        :param message: The body as an email.Message type\n        :type message: ~email.message.Message\n        :param http_response_type: The type of response to build\n        :type http_response_type: type\n        :param requests: A list of requests to process\n        :type requests: list[~azure.core.rest.HttpRequest]\n        :return: A list of responses\n        :rtype: list[~azure.core.rest.HttpResponse]\n        '

        def _deserialize_response(http_response_as_bytes, http_request, http_response_type):
            if False:
                for i in range(10):
                    print('nop')
            local_socket = BytesIOSocket(http_response_as_bytes)
            response = _HTTPResponse(local_socket, method=http_request.method)
            response.begin()
            return http_response_type(request=http_request, internal_response=response)
        return _decode_parts_helper(self, message, http_response_type or RestHttpClientTransportResponse, requests, _deserialize_response)

    def _get_raw_parts(self, http_response_type=None):
        if False:
            for i in range(10):
                print('nop')
        'Helper for get_raw_parts\n\n        Assuming this body is multipart, return the iterator or parts.\n\n        If parts are application/http use http_response_type or HttpClientTransportResponse\n        as envelope.\n\n        :param http_response_type: The type of response to build\n        :type http_response_type: type\n        :return: An iterator of responses\n        :rtype: Iterator[~azure.core.rest.HttpResponse]\n        '
        return _get_raw_parts_helper(self, http_response_type or RestHttpClientTransportResponse)

    def _stream_download(self, pipeline, **kwargs):
        if False:
            print('Hello World!')
        'DEPRECATED: Generator for streaming request body data.\n        This is deprecated and will be removed in a later release.\n        You should use `iter_bytes` or `iter_raw` instead.\n\n        :param pipeline: The pipeline object\n        :type pipeline: ~azure.core.pipeline.Pipeline\n        :return: An iterator for streaming request body data.\n        :rtype: iterator[bytes]\n        '
        return self._stream_download_generator(pipeline, self, **kwargs)

class HttpResponseBackcompatMixin(_HttpResponseBackcompatMixinBase):
    """Backcompat mixin for sync HttpResponses"""

    def __getattr__(self, attr):
        if False:
            return 10
        backcompat_attrs = ['parts']
        attr = _pad_attr_name(attr, backcompat_attrs)
        return super(HttpResponseBackcompatMixin, self).__getattr__(attr)

    def parts(self):
        if False:
            for i in range(10):
                print('nop')
        'DEPRECATED: Assuming the content-type is multipart/mixed, will return the parts as an async iterator.\n        This is deprecated and will be removed in a later release.\n\n        :rtype: Iterator\n        :return: The parts of the response\n        :raises ValueError: If the content is not multipart/mixed\n        '
        return _parts_helper(self)

class _HttpResponseBaseImpl(_HttpResponseBase, _HttpResponseBackcompatMixinBase):
    """Base Implementation class for azure.core.rest.HttpRespone and azure.core.rest.AsyncHttpResponse

    Since the rest responses are abstract base classes, we need to implement them for each of our transport
    responses. This is the base implementation class shared by HttpResponseImpl and AsyncHttpResponseImpl.
    The transport responses will be built on top of HttpResponseImpl and AsyncHttpResponseImpl

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

    def __init__(self, **kwargs) -> None:
        if False:
            return 10
        super(_HttpResponseBaseImpl, self).__init__()
        self._request = kwargs.pop('request')
        self._internal_response = kwargs.pop('internal_response')
        self._block_size: int = kwargs.pop('block_size', None) or 4096
        self._status_code: int = kwargs.pop('status_code')
        self._reason: str = kwargs.pop('reason')
        self._content_type: str = kwargs.pop('content_type')
        self._headers: MutableMapping[str, str] = kwargs.pop('headers')
        self._stream_download_generator: Callable = kwargs.pop('stream_download_generator')
        self._is_closed = False
        self._is_stream_consumed = False
        self._json = None
        self._content: Optional[bytes] = None
        self._text: Optional[str] = None

    @property
    def request(self) -> _HttpRequest:
        if False:
            print('Hello World!')
        'The request that resulted in this response.\n\n        :rtype: ~azure.core.rest.HttpRequest\n        :return: The request that resulted in this response.\n        '
        return self._request

    @property
    def url(self) -> str:
        if False:
            i = 10
            return i + 15
        'The URL that resulted in this response.\n\n        :rtype: str\n        :return: The URL that resulted in this response.\n        '
        return self.request.url

    @property
    def is_closed(self) -> bool:
        if False:
            print('Hello World!')
        'Whether the network connection has been closed yet.\n\n        :rtype: bool\n        :return: Whether the network connection has been closed yet.\n        '
        return self._is_closed

    @property
    def is_stream_consumed(self) -> bool:
        if False:
            return 10
        'Whether the stream has been consumed.\n\n        :rtype: bool\n        :return: Whether the stream has been consumed.\n        '
        return self._is_stream_consumed

    @property
    def status_code(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'The status code of this response.\n\n        :rtype: int\n        :return: The status code of this response.\n        '
        return self._status_code

    @property
    def headers(self) -> MutableMapping[str, str]:
        if False:
            for i in range(10):
                print('nop')
        'The response headers.\n\n        :rtype: MutableMapping[str, str]\n        :return: The response headers.\n        '
        return self._headers

    @property
    def content_type(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        'The content type of the response.\n\n        :rtype: optional[str]\n        :return: The content type of the response.\n        '
        return self._content_type

    @property
    def reason(self) -> str:
        if False:
            print('Hello World!')
        'The reason phrase for this response.\n\n        :rtype: str\n        :return: The reason phrase for this response.\n        '
        return self._reason

    @property
    def encoding(self) -> Optional[str]:
        if False:
            print('Hello World!')
        "Returns the response encoding.\n\n        :return: The response encoding. We either return the encoding set by the user,\n         or try extracting the encoding from the response's content type. If all fails,\n         we return `None`.\n        :rtype: optional[str]\n        "
        try:
            return self._encoding
        except AttributeError:
            self._encoding: Optional[str] = get_charset_encoding(self)
            return self._encoding

    @encoding.setter
    def encoding(self, value: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Sets the response encoding.\n\n        :param str value: Sets the response encoding.\n        '
        self._encoding = value
        self._text = None
        self._json = None

    def text(self, encoding: Optional[str]=None) -> str:
        if False:
            while True:
                i = 10
        "Returns the response body as a string\n\n        :param optional[str] encoding: The encoding you want to decode the text with. Can\n         also be set independently through our encoding property\n        :return: The response's content decoded as a string.\n        :rtype: str\n        "
        if encoding:
            return decode_to_text(encoding, self.content)
        if self._text:
            return self._text
        self._text = decode_to_text(self.encoding, self.content)
        return self._text

    def json(self) -> Any:
        if False:
            print('Hello World!')
        'Returns the whole body as a json object.\n\n        :return: The JSON deserialized response body\n        :rtype: any\n        :raises json.decoder.JSONDecodeError or ValueError (in python 2.7) if object is not JSON decodable:\n        '
        self.content
        if not self._json:
            self._json = loads(self.text())
        return self._json

    def _stream_download_check(self):
        if False:
            while True:
                i = 10
        if self.is_stream_consumed:
            raise StreamConsumedError(self)
        if self.is_closed:
            raise StreamClosedError(self)
        self._is_stream_consumed = True

    def raise_for_status(self) -> None:
        if False:
            while True:
                i = 10
        'Raises an HttpResponseError if the response has an error status code.\n\n        If response is good, does nothing.\n        '
        if self.status_code >= 400:
            raise HttpResponseError(response=self)

    @property
    def content(self) -> bytes:
        if False:
            return 10
        "Return the response's content in bytes.\n\n        :return: The response's content in bytes.\n        :rtype: bytes\n        "
        if self._content is None:
            raise ResponseNotReadError(self)
        return self._content

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        content_type_str = ', Content-Type: {}'.format(self.content_type) if self.content_type else ''
        return '<HttpResponse: {} {}{}>'.format(self.status_code, self.reason, content_type_str)

class HttpResponseImpl(_HttpResponseBaseImpl, _HttpResponse, HttpResponseBackcompatMixin):
    """HttpResponseImpl built on top of our HttpResponse protocol class.

    Since ~azure.core.rest.HttpResponse is an abstract base class, we need to
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

    def __enter__(self) -> 'HttpResponseImpl':
        if False:
            print('Hello World!')
        return self

    def close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        if not self.is_closed:
            self._is_closed = True
            self._internal_response.close()

    def __exit__(self, *args) -> None:
        if False:
            while True:
                i = 10
        self.close()

    def _set_read_checks(self):
        if False:
            i = 10
            return i + 15
        self._is_stream_consumed = True
        self.close()

    def read(self) -> bytes:
        if False:
            i = 10
            return i + 15
        "Read the response's bytes.\n\n        :return: The response's bytes\n        :rtype: bytes\n        "
        if self._content is None:
            self._content = b''.join(self.iter_bytes())
        self._set_read_checks()
        return self.content

    def iter_bytes(self, **kwargs) -> Iterator[bytes]:
        if False:
            while True:
                i = 10
        "Iterates over the response's bytes. Will decompress in the process.\n\n        :return: An iterator of bytes from the response\n        :rtype: Iterator[str]\n        "
        if self._content is not None:
            chunk_size = self._block_size
            for i in range(0, len(self.content), chunk_size):
                yield self.content[i:i + chunk_size]
        else:
            self._stream_download_check()
            for part in self._stream_download_generator(response=self, pipeline=None, decompress=True):
                yield part
        self.close()

    def iter_raw(self, **kwargs) -> Iterator[bytes]:
        if False:
            return 10
        "Iterates over the response's bytes. Will not decompress in the process.\n\n        :return: An iterator of bytes from the response\n        :rtype: Iterator[str]\n        "
        self._stream_download_check()
        for part in self._stream_download_generator(response=self, pipeline=None, decompress=False):
            yield part
        self.close()

class _RestHttpClientTransportResponseBackcompatBaseMixin(_HttpResponseBackcompatMixinBase):

    def body(self):
        if False:
            i = 10
            return i + 15
        if self._content is None:
            self._content = self.internal_response.read()
        return self.content

class _RestHttpClientTransportResponseBase(_HttpResponseBaseImpl, _RestHttpClientTransportResponseBackcompatBaseMixin):

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        internal_response = kwargs.pop('internal_response')
        headers = case_insensitive_dict(internal_response.getheaders())
        super(_RestHttpClientTransportResponseBase, self).__init__(internal_response=internal_response, status_code=internal_response.status, reason=internal_response.reason, headers=headers, content_type=headers.get('Content-Type'), stream_download_generator=None, **kwargs)

class RestHttpClientTransportResponse(_RestHttpClientTransportResponseBase, HttpResponseImpl):
    """Create a Rest HTTPResponse from an http.client response."""

    def iter_bytes(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        raise TypeError('We do not support iter_bytes for this transport response')

    def iter_raw(self, **kwargs):
        if False:
            return 10
        raise TypeError('We do not support iter_raw for this transport response')

    def read(self):
        if False:
            for i in range(10):
                print('nop')
        if self._content is None:
            self._content = self._internal_response.read()
        return self._content