import abc
import copy
from typing import Any, AsyncIterable, AsyncIterator, Iterable, Iterator, Optional, Union, MutableMapping, Dict, AsyncContextManager
from ..utils._utils import case_insensitive_dict
from ._helpers import ParamsType, FilesType, set_json_body, set_multipart_body, set_urlencoded_body, _format_parameters_helper, HttpRequestBackcompatMixin, set_content_body
ContentType = Union[str, bytes, Iterable[bytes], AsyncIterable[bytes]]

class HttpRequest(HttpRequestBackcompatMixin):
    """An HTTP request.

    It should be passed to your client's `send_request` method.

    >>> from azure.core.rest import HttpRequest
    >>> request = HttpRequest('GET', 'http://www.example.com')
    <HttpRequest [GET], url: 'http://www.example.com'>
    >>> response = client.send_request(request)
    <HttpResponse: 200 OK>

    :param str method: HTTP method (GET, HEAD, etc.)
    :param str url: The url for your request
    :keyword mapping params: Query parameters to be mapped into your URL. Your input
     should be a mapping of query name to query value(s).
    :keyword mapping headers: HTTP headers you want in your request. Your input should
     be a mapping of header name to header value.
    :keyword any json: A JSON serializable object. We handle JSON-serialization for your
     object, so use this for more complicated data structures than `data`.
    :keyword content: Content you want in your request body. Think of it as the kwarg you should input
     if your data doesn't fit into `json`, `data`, or `files`. Accepts a bytes type, or a generator
     that yields bytes.
    :paramtype content: str or bytes or iterable[bytes] or asynciterable[bytes]
    :keyword dict data: Form data you want in your request body. Use for form-encoded data, i.e.
     HTML forms.
    :keyword mapping files: Files you want to in your request body. Use for uploading files with
     multipart encoding. Your input should be a mapping of file name to file content.
     Use the `data` kwarg in addition if you want to include non-file data files as part of your request.
    :ivar str url: The URL this request is against.
    :ivar str method: The method type of this request.
    :ivar mapping headers: The HTTP headers you passed in to your request
    :ivar any content: The content passed in for the request
    """

    def __init__(self, method: str, url: str, *, params: Optional[ParamsType]=None, headers: Optional[MutableMapping[str, str]]=None, json: Any=None, content: Optional[ContentType]=None, data: Optional[Dict[str, Any]]=None, files: Optional[FilesType]=None, **kwargs: Any):
        if False:
            print('Hello World!')
        self.url = url
        self.method = method
        if params:
            _format_parameters_helper(self, params)
        self._files = None
        self._data: Any = None
        default_headers = self._set_body(content=content, data=data, files=files, json=json)
        self.headers: MutableMapping[str, str] = case_insensitive_dict(default_headers)
        self.headers.update(headers or {})
        if kwargs:
            raise TypeError("You have passed in kwargs '{}' that are not valid kwargs.".format("', '".join(list(kwargs.keys()))))

    def _set_body(self, content: Optional[ContentType]=None, data: Optional[Dict[str, Any]]=None, files: Optional[FilesType]=None, json: Any=None) -> MutableMapping[str, str]:
        if False:
            while True:
                i = 10
        'Sets the body of the request, and returns the default headers.\n\n        :param content: Content you want in your request body.\n        :type content: str or bytes or iterable[bytes] or asynciterable[bytes]\n        :param dict data: Form data you want in your request body.\n        :param dict files: Files you want to in your request body.\n        :param any json: A JSON serializable object.\n        :return: The default headers for the request\n        :rtype: MutableMapping[str, str]\n        '
        default_headers: MutableMapping[str, str] = {}
        if data is not None and (not isinstance(data, dict)):
            content = data
        if content is not None:
            (default_headers, self._data) = set_content_body(content)
            return default_headers
        if json is not None:
            (default_headers, self._data) = set_json_body(json)
            return default_headers
        if files:
            (default_headers, self._files) = set_multipart_body(files)
        if data:
            (default_headers, self._data) = set_urlencoded_body(data, has_files=bool(files))
        return default_headers

    @property
    def content(self) -> Any:
        if False:
            i = 10
            return i + 15
        "Get's the request's content\n\n        :return: The request's content\n        :rtype: any\n        "
        return self._data or self._files

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        return "<HttpRequest [{}], url: '{}'>".format(self.method, self.url)

    def __deepcopy__(self, memo: Optional[Dict[int, Any]]=None) -> 'HttpRequest':
        if False:
            for i in range(10):
                print('nop')
        try:
            request = HttpRequest(method=self.method, url=self.url, headers=self.headers)
            request._data = copy.deepcopy(self._data, memo)
            request._files = copy.deepcopy(self._files, memo)
            self._add_backcompat_properties(request, memo)
            return request
        except (ValueError, TypeError):
            return copy.copy(self)

class _HttpResponseBase(abc.ABC):
    """Base abstract base class for HttpResponses."""

    @property
    @abc.abstractmethod
    def request(self) -> HttpRequest:
        if False:
            while True:
                i = 10
        'The request that resulted in this response.\n\n        :rtype: ~azure.core.rest.HttpRequest\n        :return: The request that resulted in this response.\n        '

    @property
    @abc.abstractmethod
    def status_code(self) -> int:
        if False:
            i = 10
            return i + 15
        'The status code of this response.\n\n        :rtype: int\n        :return: The status code of this response.\n        '

    @property
    @abc.abstractmethod
    def headers(self) -> MutableMapping[str, str]:
        if False:
            print('Hello World!')
        'The response headers. Must be case-insensitive.\n\n        :rtype: MutableMapping[str, str]\n        :return: The response headers. Must be case-insensitive.\n        '

    @property
    @abc.abstractmethod
    def reason(self) -> str:
        if False:
            while True:
                i = 10
        'The reason phrase for this response.\n\n        :rtype: str\n        :return: The reason phrase for this response.\n        '

    @property
    @abc.abstractmethod
    def content_type(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        'The content type of the response.\n\n        :rtype: str\n        :return: The content type of the response.\n        '

    @property
    @abc.abstractmethod
    def is_closed(self) -> bool:
        if False:
            while True:
                i = 10
        'Whether the network connection has been closed yet.\n\n        :rtype: bool\n        :return: Whether the network connection has been closed yet.\n        '

    @property
    @abc.abstractmethod
    def is_stream_consumed(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Whether the stream has been consumed.\n\n        :rtype: bool\n        :return: Whether the stream has been consumed.\n        '

    @property
    @abc.abstractmethod
    def encoding(self) -> Optional[str]:
        if False:
            return 10
        "Returns the response encoding.\n\n        :return: The response encoding. We either return the encoding set by the user,\n         or try extracting the encoding from the response's content type. If all fails,\n         we return `None`.\n        :rtype: optional[str]\n        "

    @encoding.setter
    def encoding(self, value: Optional[str]) -> None:
        if False:
            return 10
        'Sets the response encoding.\n\n        :param optional[str] value: The encoding to set\n        '

    @property
    @abc.abstractmethod
    def url(self) -> str:
        if False:
            return 10
        'The URL that resulted in this response.\n\n        :rtype: str\n        :return: The URL that resulted in this response.\n        '

    @property
    @abc.abstractmethod
    def content(self) -> bytes:
        if False:
            return 10
        "Return the response's content in bytes.\n\n        :rtype: bytes\n        :return: The response's content in bytes.\n        "

    @abc.abstractmethod
    def text(self, encoding: Optional[str]=None) -> str:
        if False:
            i = 10
            return i + 15
        "Returns the response body as a string.\n\n        :param optional[str] encoding: The encoding you want to decode the text with. Can\n         also be set independently through our encoding property\n        :return: The response's content decoded as a string.\n        :rtype: str\n        "

    @abc.abstractmethod
    def json(self) -> Any:
        if False:
            return 10
        'Returns the whole body as a json object.\n\n        :return: The JSON deserialized response body\n        :rtype: any\n        :raises json.decoder.JSONDecodeError or ValueError (in python 2.7) if object is not JSON decodable:\n        '

    @abc.abstractmethod
    def raise_for_status(self) -> None:
        if False:
            i = 10
            return i + 15
        'Raises an HttpResponseError if the response has an error status code.\n\n        If response is good, does nothing.\n\n        :raises ~azure.core.HttpResponseError if the object has an error status code.:\n        '

class HttpResponse(_HttpResponseBase):
    """Abstract base class for HTTP responses.

    Use this abstract base class to create your own transport responses.

    Responses implementing this ABC are returned from your client's `send_request` method
    if you pass in an :class:`~azure.core.rest.HttpRequest`

    >>> from azure.core.rest import HttpRequest
    >>> request = HttpRequest('GET', 'http://www.example.com')
    <HttpRequest [GET], url: 'http://www.example.com'>
    >>> response = client.send_request(request)
    <HttpResponse: 200 OK>
    """

    @abc.abstractmethod
    def __enter__(self) -> 'HttpResponse':
        if False:
            return 10
        ...

    @abc.abstractmethod
    def __exit__(self, *args: Any) -> None:
        if False:
            i = 10
            return i + 15
        ...

    @abc.abstractmethod
    def close(self) -> None:
        if False:
            print('Hello World!')
        ...

    @abc.abstractmethod
    def read(self) -> bytes:
        if False:
            return 10
        "Read the response's bytes.\n\n        :return: The read in bytes\n        :rtype: bytes\n        "

    @abc.abstractmethod
    def iter_raw(self, **kwargs: Any) -> Iterator[bytes]:
        if False:
            i = 10
            return i + 15
        "Iterates over the response's bytes. Will not decompress in the process.\n\n        :return: An iterator of bytes from the response\n        :rtype: Iterator[str]\n        "

    @abc.abstractmethod
    def iter_bytes(self, **kwargs: Any) -> Iterator[bytes]:
        if False:
            print('Hello World!')
        "Iterates over the response's bytes. Will decompress in the process.\n\n        :return: An iterator of bytes from the response\n        :rtype: Iterator[str]\n        "

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        content_type_str = ', Content-Type: {}'.format(self.content_type) if self.content_type else ''
        return '<HttpResponse: {} {}{}>'.format(self.status_code, self.reason, content_type_str)

class AsyncHttpResponse(_HttpResponseBase, AsyncContextManager['AsyncHttpResponse']):
    """Abstract base class for Async HTTP responses.

    Use this abstract base class to create your own transport responses.

    Responses implementing this ABC are returned from your async client's `send_request`
    method if you pass in an :class:`~azure.core.rest.HttpRequest`

    >>> from azure.core.rest import HttpRequest
    >>> request = HttpRequest('GET', 'http://www.example.com')
    <HttpRequest [GET], url: 'http://www.example.com'>
    >>> response = await client.send_request(request)
    <AsyncHttpResponse: 200 OK>
    """

    @abc.abstractmethod
    async def read(self) -> bytes:
        """Read the response's bytes into memory.

        :return: The response's bytes
        :rtype: bytes
        """

    @abc.abstractmethod
    async def iter_raw(self, **kwargs: Any) -> AsyncIterator[bytes]:
        """Asynchronously iterates over the response's bytes. Will not decompress in the process.

        :return: An async iterator of bytes from the response
        :rtype: AsyncIterator[bytes]
        """
        raise NotImplementedError()
        yield

    @abc.abstractmethod
    async def iter_bytes(self, **kwargs: Any) -> AsyncIterator[bytes]:
        """Asynchronously iterates over the response's bytes. Will decompress in the process.

        :return: An async iterator of bytes from the response
        :rtype: AsyncIterator[bytes]
        """
        raise NotImplementedError()
        yield

    @abc.abstractmethod
    async def close(self) -> None:
        ...