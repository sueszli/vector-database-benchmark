from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Literal, TypedDict, cast
from litestar._parsers import parse_cookie_string
from litestar.connection.request import Request
from litestar.datastructures.upload_file import UploadFile
from litestar.enums import HttpMethod, RequestEncodingType
__all__ = ('ConnectionDataExtractor', 'ExtractedRequestData', 'ExtractedResponseData', 'ResponseDataExtractor', 'RequestExtractorField', 'ResponseExtractorField')
if TYPE_CHECKING:
    from litestar.connection import ASGIConnection
    from litestar.types import Method
    from litestar.types.asgi_types import HTTPResponseBodyEvent, HTTPResponseStartEvent

def _obfuscate(values: dict[str, Any], fields_to_obfuscate: set[str]) -> dict[str, Any]:
    if False:
        while True:
            i = 10
    'Obfuscate values in a dictionary, replacing values with `******`\n\n    Args:\n        values: A dictionary of strings\n        fields_to_obfuscate: keys to obfuscate\n\n    Returns:\n        A dictionary with obfuscated strings\n    '
    return {key: '*****' if key.lower() in fields_to_obfuscate else value for (key, value) in values.items()}
RequestExtractorField = Literal['path', 'method', 'content_type', 'headers', 'cookies', 'query', 'path_params', 'body', 'scheme', 'client']
ResponseExtractorField = Literal['status_code', 'headers', 'body', 'cookies']

class ExtractedRequestData(TypedDict, total=False):
    """Dictionary representing extracted request data."""
    body: Coroutine[Any, Any, Any]
    client: tuple[str, int]
    content_type: tuple[str, dict[str, str]]
    cookies: dict[str, str]
    headers: dict[str, str]
    method: Method
    path: str
    path_params: dict[str, Any]
    query: bytes | dict[str, Any]
    scheme: str

class ConnectionDataExtractor:
    """Utility class to extract data from an :class:`ASGIConnection <litestar.connection.ASGIConnection>`,
    :class:`Request <litestar.connection.Request>` or :class:`WebSocket <litestar.connection.WebSocket>` instance.
    """
    __slots__ = ('connection_extractors', 'request_extractors', 'parse_body', 'parse_query', 'obfuscate_headers', 'obfuscate_cookies')

    def __init__(self, extract_body: bool=True, extract_client: bool=True, extract_content_type: bool=True, extract_cookies: bool=True, extract_headers: bool=True, extract_method: bool=True, extract_path: bool=True, extract_path_params: bool=True, extract_query: bool=True, extract_scheme: bool=True, obfuscate_cookies: set[str] | None=None, obfuscate_headers: set[str] | None=None, parse_body: bool=False, parse_query: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Initialize ``ConnectionDataExtractor``\n\n        Args:\n            extract_body: Whether to extract body, (for requests only).\n            extract_client: Whether to extract the client (host, port) mapping.\n            extract_content_type: Whether to extract the content type and any options.\n            extract_cookies: Whether to extract cookies.\n            extract_headers: Whether to extract headers.\n            extract_method: Whether to extract the HTTP method, (for requests only).\n            extract_path: Whether to extract the path.\n            extract_path_params: Whether to extract path parameters.\n            extract_query: Whether to extract query parameters.\n            extract_scheme: Whether to extract the http scheme.\n            obfuscate_headers: headers keys to obfuscate. Obfuscated values are replaced with '*****'.\n            obfuscate_cookies: cookie keys to obfuscate. Obfuscated values are replaced with '*****'.\n            parse_body: Whether to parse the body value or return the raw byte string, (for requests only).\n            parse_query: Whether to parse query parameters or return the raw byte string.\n        "
        self.parse_body = parse_body
        self.parse_query = parse_query
        self.obfuscate_headers = {h.lower() for h in obfuscate_headers or set()}
        self.obfuscate_cookies = {c.lower() for c in obfuscate_cookies or set()}
        self.connection_extractors: dict[str, Callable[[ASGIConnection[Any, Any, Any, Any]], Any]] = {}
        self.request_extractors: dict[RequestExtractorField, Callable[[Request[Any, Any, Any]], Any]] = {}
        if extract_scheme:
            self.connection_extractors['scheme'] = self.extract_scheme
        if extract_client:
            self.connection_extractors['client'] = self.extract_client
        if extract_path:
            self.connection_extractors['path'] = self.extract_path
        if extract_headers:
            self.connection_extractors['headers'] = self.extract_headers
        if extract_cookies:
            self.connection_extractors['cookies'] = self.extract_cookies
        if extract_query:
            self.connection_extractors['query'] = self.extract_query
        if extract_path_params:
            self.connection_extractors['path_params'] = self.extract_path_params
        if extract_method:
            self.request_extractors['method'] = self.extract_method
        if extract_content_type:
            self.request_extractors['content_type'] = self.extract_content_type
        if extract_body:
            self.request_extractors['body'] = self.extract_body

    def __call__(self, connection: ASGIConnection[Any, Any, Any, Any]) -> ExtractedRequestData:
        if False:
            i = 10
            return i + 15
        'Extract data from the connection, returning a dictionary of values.\n\n        Notes:\n            - The value for ``body`` - if present - is an unresolved Coroutine and as such should be awaited by the receiver.\n\n        Args:\n            connection: An ASGI connection or its subclasses.\n\n        Returns:\n            A string keyed dictionary of extracted values.\n        '
        extractors = {**self.connection_extractors, **self.request_extractors} if isinstance(connection, Request) else self.connection_extractors
        return cast('ExtractedRequestData', {key: extractor(connection) for (key, extractor) in extractors.items()})

    @staticmethod
    def extract_scheme(connection: ASGIConnection[Any, Any, Any, Any]) -> str:
        if False:
            return 10
        'Extract the scheme from an ``ASGIConnection``\n\n        Args:\n            connection: An :class:`ASGIConnection <litestar.connection.ASGIConnection>` instance.\n\n        Returns:\n            The connection\'s scope["scheme"] value\n        '
        return connection.scope['scheme']

    @staticmethod
    def extract_client(connection: ASGIConnection[Any, Any, Any, Any]) -> tuple[str, int]:
        if False:
            while True:
                i = 10
        'Extract the client from an ``ASGIConnection``\n\n        Args:\n            connection: An :class:`ASGIConnection <litestar.connection.ASGIConnection>` instance.\n\n        Returns:\n            The connection\'s scope["client"] value or a default value.\n        '
        return connection.scope.get('client') or ('', 0)

    @staticmethod
    def extract_path(connection: ASGIConnection[Any, Any, Any, Any]) -> str:
        if False:
            while True:
                i = 10
        'Extract the path from an ``ASGIConnection``\n\n        Args:\n            connection: An :class:`ASGIConnection <litestar.connection.ASGIConnection>` instance.\n\n        Returns:\n            The connection\'s scope["path"] value\n        '
        return connection.scope['path']

    def extract_headers(self, connection: ASGIConnection[Any, Any, Any, Any]) -> dict[str, str]:
        if False:
            while True:
                i = 10
        "Extract headers from an ``ASGIConnection``\n\n        Args:\n            connection: An :class:`ASGIConnection <litestar.connection.ASGIConnection>` instance.\n\n        Returns:\n            A dictionary with the connection's headers.\n        "
        headers = {k.decode('latin-1'): v.decode('latin-1') for (k, v) in connection.scope['headers']}
        return _obfuscate(headers, self.obfuscate_headers) if self.obfuscate_headers else headers

    def extract_cookies(self, connection: ASGIConnection[Any, Any, Any, Any]) -> dict[str, str]:
        if False:
            print('Hello World!')
        "Extract cookies from an ``ASGIConnection``\n\n        Args:\n            connection: An :class:`ASGIConnection <litestar.connection.ASGIConnection>` instance.\n\n        Returns:\n            A dictionary with the connection's cookies.\n        "
        return _obfuscate(connection.cookies, self.obfuscate_cookies) if self.obfuscate_cookies else connection.cookies

    def extract_query(self, connection: ASGIConnection[Any, Any, Any, Any]) -> Any:
        if False:
            i = 10
            return i + 15
        "Extract query from an ``ASGIConnection``\n\n        Args:\n            connection: An :class:`ASGIConnection <litestar.connection.ASGIConnection>` instance.\n\n        Returns:\n            Either a dictionary with the connection's parsed query string or the raw query byte-string.\n        "
        return connection.query_params.dict() if self.parse_query else connection.scope.get('query_string', b'')

    @staticmethod
    def extract_path_params(connection: ASGIConnection[Any, Any, Any, Any]) -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        "Extract the path parameters from an ``ASGIConnection``\n\n        Args:\n            connection: An :class:`ASGIConnection <litestar.connection.ASGIConnection>` instance.\n\n        Returns:\n            A dictionary with the connection's path parameters.\n        "
        return connection.path_params

    @staticmethod
    def extract_method(request: Request[Any, Any, Any]) -> Method:
        if False:
            i = 10
            return i + 15
        'Extract the method from an ``ASGIConnection``\n\n        Args:\n            request: A :class:`Request <litestar.connection.Request>` instance.\n\n        Returns:\n            The request\'s scope["method"] value.\n        '
        return request.scope['method']

    @staticmethod
    def extract_content_type(request: Request[Any, Any, Any]) -> tuple[str, dict[str, str]]:
        if False:
            print('Hello World!')
        "Extract the content-type from an ``ASGIConnection``\n\n        Args:\n            request: A :class:`Request <litestar.connection.Request>` instance.\n\n        Returns:\n            A tuple containing the request's parsed 'Content-Type' header.\n        "
        return request.content_type

    async def extract_body(self, request: Request[Any, Any, Any]) -> Any:
        """Extract the body from an ``ASGIConnection``

        Args:
            request: A :class:`Request <litestar.connection.Request>` instance.

        Returns:
            Either the parsed request body or the raw byte-string.
        """
        if request.method == HttpMethod.GET:
            return None
        if not self.parse_body:
            return await request.body()
        request_encoding_type = request.content_type[0]
        if request_encoding_type == RequestEncodingType.JSON:
            return await request.json()
        form_data = await request.form()
        if request_encoding_type == RequestEncodingType.URL_ENCODED:
            return dict(form_data)
        return {key: repr(value) if isinstance(value, UploadFile) else value for (key, value) in form_data.multi_items()}

class ExtractedResponseData(TypedDict, total=False):
    """Dictionary representing extracted response data."""
    body: bytes
    status_code: int
    headers: dict[str, str]
    cookies: dict[str, str]

class ResponseDataExtractor:
    """Utility class to extract data from a ``Message``"""
    __slots__ = ('extractors', 'parse_headers', 'obfuscate_headers', 'obfuscate_cookies')

    def __init__(self, extract_body: bool=True, extract_cookies: bool=True, extract_headers: bool=True, extract_status_code: bool=True, obfuscate_cookies: set[str] | None=None, obfuscate_headers: set[str] | None=None) -> None:
        if False:
            while True:
                i = 10
        "Initialize ``ResponseDataExtractor`` with options.\n\n        Args:\n            extract_body: Whether to extract the body.\n            extract_cookies: Whether to extract the cookies.\n            extract_headers: Whether to extract the headers.\n            extract_status_code: Whether to extract the status code.\n            obfuscate_cookies: cookie keys to obfuscate. Obfuscated values are replaced with '*****'.\n            obfuscate_headers: headers keys to obfuscate. Obfuscated values are replaced with '*****'.\n        "
        self.obfuscate_headers = {h.lower() for h in obfuscate_headers or set()}
        self.obfuscate_cookies = {c.lower() for c in obfuscate_cookies or set()}
        self.extractors: dict[ResponseExtractorField, Callable[[tuple[HTTPResponseStartEvent, HTTPResponseBodyEvent]], Any]] = {}
        if extract_body:
            self.extractors['body'] = self.extract_response_body
        if extract_status_code:
            self.extractors['status_code'] = self.extract_status_code
        if extract_headers:
            self.extractors['headers'] = self.extract_headers
        if extract_cookies:
            self.extractors['cookies'] = self.extract_cookies

    def __call__(self, messages: tuple[HTTPResponseStartEvent, HTTPResponseBodyEvent]) -> ExtractedResponseData:
        if False:
            for i in range(10):
                print('nop')
        'Extract data from the response, returning a dictionary of values.\n\n        Args:\n            messages: A tuple containing\n                :class:`HTTPResponseStartEvent <litestar.types.asgi_types.HTTPResponseStartEvent>`\n                and :class:`HTTPResponseBodyEvent <litestar.types.asgi_types.HTTPResponseBodyEvent>`.\n\n        Returns:\n            A string keyed dictionary of extracted values.\n        '
        return cast('ExtractedResponseData', {key: extractor(messages) for (key, extractor) in self.extractors.items()})

    @staticmethod
    def extract_response_body(messages: tuple[HTTPResponseStartEvent, HTTPResponseBodyEvent]) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        "Extract the response body from a ``Message``\n\n        Args:\n            messages: A tuple containing\n                :class:`HTTPResponseStartEvent <litestar.types.asgi_types.HTTPResponseStartEvent>`\n                and :class:`HTTPResponseBodyEvent <litestar.types.asgi_types.HTTPResponseBodyEvent>`.\n\n        Returns:\n            The Response's body as a byte-string.\n        "
        return messages[1]['body']

    @staticmethod
    def extract_status_code(messages: tuple[HTTPResponseStartEvent, HTTPResponseBodyEvent]) -> int:
        if False:
            for i in range(10):
                print('nop')
        "Extract a status code from a ``Message``\n\n        Args:\n            messages: A tuple containing\n                :class:`HTTPResponseStartEvent <litestar.types.asgi_types.HTTPResponseStartEvent>`\n                and :class:`HTTPResponseBodyEvent <litestar.types.asgi_types.HTTPResponseBodyEvent>`.\n\n        Returns:\n            The Response's status-code.\n        "
        return messages[0]['status']

    def extract_headers(self, messages: tuple[HTTPResponseStartEvent, HTTPResponseBodyEvent]) -> dict[str, str]:
        if False:
            return 10
        "Extract headers from a ``Message``\n\n        Args:\n            messages: A tuple containing\n                :class:`HTTPResponseStartEvent <litestar.types.asgi_types.HTTPResponseStartEvent>`\n                and :class:`HTTPResponseBodyEvent <litestar.types.asgi_types.HTTPResponseBodyEvent>`.\n\n        Returns:\n            The Response's headers dict.\n        "
        headers = {key.decode('latin-1'): value.decode('latin-1') for (key, value) in filter(lambda x: x[0].lower() != b'set-cookie', messages[0]['headers'])}
        return _obfuscate(headers, self.obfuscate_headers) if self.obfuscate_headers else headers

    def extract_cookies(self, messages: tuple[HTTPResponseStartEvent, HTTPResponseBodyEvent]) -> dict[str, str]:
        if False:
            print('Hello World!')
        "Extract cookies from a ``Message``\n\n        Args:\n            messages: A tuple containing\n                :class:`HTTPResponseStartEvent <litestar.types.asgi_types.HTTPResponseStartEvent>`\n                and :class:`HTTPResponseBodyEvent <litestar.types.asgi_types.HTTPResponseBodyEvent>`.\n\n        Returns:\n            The Response's cookies dict.\n        "
        if (cookie_string := ';'.join([x[1].decode('latin-1') for x in filter(lambda x: x[0].lower() == b'set-cookie', messages[0]['headers'])])):
            parsed_cookies = parse_cookie_string(cookie_string)
            return _obfuscate(parsed_cookies, self.obfuscate_cookies) if self.obfuscate_cookies else parsed_cookies
        return {}