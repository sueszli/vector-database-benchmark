from __future__ import annotations
import inspect
import datetime
import functools
from typing import TYPE_CHECKING, Any, Union, Generic, TypeVar, Callable, cast
from typing_extensions import Awaitable, ParamSpec, get_args, override, get_origin
import httpx
import pydantic
from ._types import NoneType, UnknownResponse, BinaryResponseContent
from ._utils import is_given
from ._models import BaseModel
from ._constants import RAW_RESPONSE_HEADER
from ._exceptions import APIResponseValidationError
if TYPE_CHECKING:
    from ._models import FinalRequestOptions
    from ._base_client import Stream, BaseClient, AsyncStream
P = ParamSpec('P')
R = TypeVar('R')

class APIResponse(Generic[R]):
    _cast_to: type[R]
    _client: BaseClient[Any, Any]
    _parsed: R | None
    _stream: bool
    _stream_cls: type[Stream[Any]] | type[AsyncStream[Any]] | None
    _options: FinalRequestOptions
    http_response: httpx.Response

    def __init__(self, *, raw: httpx.Response, cast_to: type[R], client: BaseClient[Any, Any], stream: bool, stream_cls: type[Stream[Any]] | type[AsyncStream[Any]] | None, options: FinalRequestOptions) -> None:
        if False:
            return 10
        self._cast_to = cast_to
        self._client = client
        self._parsed = None
        self._stream = stream
        self._stream_cls = stream_cls
        self._options = options
        self.http_response = raw

    def parse(self) -> R:
        if False:
            print('Hello World!')
        if self._parsed is not None:
            return self._parsed
        parsed = self._parse()
        if is_given(self._options.post_parser):
            parsed = self._options.post_parser(parsed)
        self._parsed = parsed
        return parsed

    @property
    def headers(self) -> httpx.Headers:
        if False:
            print('Hello World!')
        return self.http_response.headers

    @property
    def http_request(self) -> httpx.Request:
        if False:
            while True:
                i = 10
        return self.http_response.request

    @property
    def status_code(self) -> int:
        if False:
            i = 10
            return i + 15
        return self.http_response.status_code

    @property
    def url(self) -> httpx.URL:
        if False:
            while True:
                i = 10
        return self.http_response.url

    @property
    def method(self) -> str:
        if False:
            i = 10
            return i + 15
        return self.http_request.method

    @property
    def content(self) -> bytes:
        if False:
            i = 10
            return i + 15
        return self.http_response.content

    @property
    def text(self) -> str:
        if False:
            return 10
        return self.http_response.text

    @property
    def http_version(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self.http_response.http_version

    @property
    def elapsed(self) -> datetime.timedelta:
        if False:
            print('Hello World!')
        'The time taken for the complete request/response cycle to complete.'
        return self.http_response.elapsed

    def _parse(self) -> R:
        if False:
            i = 10
            return i + 15
        if self._stream:
            if self._stream_cls:
                return cast(R, self._stream_cls(cast_to=_extract_stream_chunk_type(self._stream_cls), response=self.http_response, client=cast(Any, self._client)))
            stream_cls = cast('type[Stream[Any]] | type[AsyncStream[Any]] | None', self._client._default_stream_cls)
            if stream_cls is None:
                raise MissingStreamClassError()
            return cast(R, stream_cls(cast_to=self._cast_to, response=self.http_response, client=cast(Any, self._client)))
        cast_to = self._cast_to
        if cast_to is NoneType:
            return cast(R, None)
        response = self.http_response
        if cast_to == str:
            return cast(R, response.text)
        origin = get_origin(cast_to) or cast_to
        if inspect.isclass(origin) and issubclass(origin, BinaryResponseContent):
            return cast(R, cast_to(response))
        if origin == APIResponse:
            raise RuntimeError('Unexpected state - cast_to is `APIResponse`')
        if inspect.isclass(origin) and issubclass(origin, httpx.Response):
            if cast_to != httpx.Response:
                raise ValueError(f'Subclasses of httpx.Response cannot be passed to `cast_to`')
            return cast(R, response)
        if cast_to is not UnknownResponse and (not origin is list) and (not origin is dict) and (not origin is Union) and (not issubclass(origin, BaseModel)):
            raise RuntimeError(f'Invalid state, expected {cast_to} to be a subclass type of {BaseModel}, {dict}, {list} or {Union}.')
        (content_type, *_) = response.headers.get('content-type').split(';')
        if content_type != 'application/json':
            if self._client._strict_response_validation:
                raise APIResponseValidationError(response=response, message=f'Expected Content-Type response header to be `application/json` but received `{content_type}` instead.', body=response.text)
            return response.text
        data = response.json()
        try:
            return self._client._process_response_data(data=data, cast_to=cast_to, response=response)
        except pydantic.ValidationError as err:
            raise APIResponseValidationError(response=response, body=data) from err

    @override
    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'<APIResponse [{self.status_code} {self.http_response.reason_phrase}] type={self._cast_to}>'

class MissingStreamClassError(TypeError):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__('The `stream` argument was set to `True` but the `stream_cls` argument was not given. See `openai._streaming` for reference')

def _extract_stream_chunk_type(stream_cls: type) -> type:
    if False:
        print('Hello World!')
    args = get_args(stream_cls)
    if not args:
        raise TypeError(f'Expected stream_cls to have been given a generic type argument, e.g. Stream[Foo] but received {stream_cls}')
    return cast(type, args[0])

def to_raw_response_wrapper(func: Callable[P, R]) -> Callable[P, APIResponse[R]]:
    if False:
        return 10
    'Higher order function that takes one of our bound API methods and wraps it\n    to support returning the raw `APIResponse` object directly.\n    '

    @functools.wraps(func)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> APIResponse[R]:
        if False:
            while True:
                i = 10
        extra_headers = {**(cast(Any, kwargs.get('extra_headers')) or {})}
        extra_headers[RAW_RESPONSE_HEADER] = 'true'
        kwargs['extra_headers'] = extra_headers
        return cast(APIResponse[R], func(*args, **kwargs))
    return wrapped

def async_to_raw_response_wrapper(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[APIResponse[R]]]:
    if False:
        return 10
    'Higher order function that takes one of our bound API methods and wraps it\n    to support returning the raw `APIResponse` object directly.\n    '

    @functools.wraps(func)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> APIResponse[R]:
        extra_headers = {**(cast(Any, kwargs.get('extra_headers')) or {})}
        extra_headers[RAW_RESPONSE_HEADER] = 'true'
        kwargs['extra_headers'] = extra_headers
        return cast(APIResponse[R], await func(*args, **kwargs))
    return wrapped