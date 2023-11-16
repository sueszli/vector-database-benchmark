"""Protocol that defines what functions wrappers of tracing libraries should implement."""
from __future__ import annotations
from enum import Enum
from urllib.parse import urlparse
from typing import Any, Sequence, Optional, Union, Callable, Dict, Type, Generic, TypeVar
from types import TracebackType
from typing_extensions import Protocol, ContextManager, runtime_checkable
from azure.core.pipeline.transport import HttpRequest, HttpResponse, AsyncHttpResponse
from azure.core.rest import HttpResponse as RestHttpResponse, AsyncHttpResponse as AsyncRestHttpResponse, HttpRequest as RestHttpRequest
HttpResponseType = Union[HttpResponse, AsyncHttpResponse, RestHttpResponse, AsyncRestHttpResponse]
HttpRequestType = Union[HttpRequest, RestHttpRequest]
AttributeValue = Union[str, bool, int, float, Sequence[str], Sequence[bool], Sequence[int], Sequence[float]]
Attributes = Dict[str, AttributeValue]
SpanType = TypeVar('SpanType')

class SpanKind(Enum):
    UNSPECIFIED = 1
    SERVER = 2
    CLIENT = 3
    PRODUCER = 4
    CONSUMER = 5
    INTERNAL = 6

@runtime_checkable
class AbstractSpan(Protocol, Generic[SpanType]):
    """Wraps a span from a distributed tracing implementation.

    If a span is given wraps the span. Else a new span is created.
    The optional argument name is given to the new span.

    :param span: The span to wrap
    :type span: Any
    :param name: The name of the span
    :type name: str
    """

    def __init__(self, span: Optional[SpanType]=None, name: Optional[str]=None, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def span(self, name: str='child_span', **kwargs: Any) -> AbstractSpan[SpanType]:
        if False:
            return 10
        '\n        Create a child span for the current span and append it to the child spans list.\n        The child span must be wrapped by an implementation of AbstractSpan\n\n        :param name: The name of the child span\n        :type name: str\n        :return: The child span\n        :rtype: AbstractSpan\n        '
        ...

    @property
    def kind(self) -> Optional[SpanKind]:
        if False:
            while True:
                i = 10
        'Get the span kind of this span.\n\n        :rtype: SpanKind\n        :return: The span kind of this span\n        '
        ...

    @kind.setter
    def kind(self, value: SpanKind) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the span kind of this span.\n\n        :param value: The span kind of this span\n        :type value: SpanKind\n        '
        ...

    def __enter__(self) -> AbstractSpan[SpanType]:
        if False:
            while True:
                i = 10
        'Start a span.'
        ...

    def __exit__(self, exception_type: Optional[Type[BaseException]], exception_value: Optional[BaseException], traceback: TracebackType) -> None:
        if False:
            i = 10
            return i + 15
        'Finish a span.\n\n        :param exception_type: The type of the exception\n        :type exception_type: type\n        :param exception_value: The value of the exception\n        :type exception_value: Exception\n        :param traceback: The traceback of the exception\n        :type traceback: Traceback\n        '
        ...

    def start(self) -> None:
        if False:
            while True:
                i = 10
        'Set the start time for a span.'
        ...

    def finish(self) -> None:
        if False:
            print('Hello World!')
        'Set the end time for a span.'
        ...

    def to_header(self) -> Dict[str, str]:
        if False:
            print('Hello World!')
        'Returns a dictionary with the header labels and values.\n\n        :return: A dictionary with the header labels and values\n        :rtype: dict\n        '
        ...

    def add_attribute(self, key: str, value: Union[str, int]) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Add attribute (key value pair) to the current span.\n\n        :param key: The key of the key value pair\n        :type key: str\n        :param value: The value of the key value pair\n        :type value: Union[str, int]\n        '
        ...

    def set_http_attributes(self, request: HttpRequestType, response: Optional[HttpResponseType]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Add correct attributes for a http client span.\n\n        :param request: The request made\n        :type request: azure.core.rest.HttpRequest\n        :param response: The response received by the server. Is None if no response received.\n        :type response: ~azure.core.pipeline.transport.HttpResponse or ~azure.core.pipeline.transport.AsyncHttpResponse\n        '
        ...

    def get_trace_parent(self) -> str:
        if False:
            i = 10
            return i + 15
        'Return traceparent string.\n\n        :return: a traceparent string\n        :rtype: str\n        '
        ...

    @property
    def span_instance(self) -> SpanType:
        if False:
            return 10
        '\n        Returns the span the class is wrapping.\n        '
        ...

    @classmethod
    def link(cls, traceparent: str, attributes: Optional[Attributes]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Given a traceparent, extracts the context and links the context to the current tracer.\n\n        :param traceparent: A string representing a traceparent\n        :type traceparent: str\n        :param attributes: Any additional attributes that should be added to link\n        :type attributes: dict\n        '
        ...

    @classmethod
    def link_from_headers(cls, headers: Dict[str, str], attributes: Optional[Attributes]=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Given a dictionary, extracts the context and links the context to the current tracer.\n\n        :param headers: A dictionary of the request header as key value pairs.\n        :type headers: dict\n        :param attributes: Any additional attributes that should be added to link\n        :type attributes: dict\n        '
        ...

    @classmethod
    def get_current_span(cls) -> SpanType:
        if False:
            return 10
        '\n        Get the current span from the execution context. Return None otherwise.\n\n        :return: The current span\n        :rtype: AbstractSpan\n        '
        ...

    @classmethod
    def get_current_tracer(cls) -> Any:
        if False:
            return 10
        '\n        Get the current tracer from the execution context. Return None otherwise.\n\n        :return: The current tracer\n        :rtype: Any\n        '
        ...

    @classmethod
    def set_current_span(cls, span: SpanType) -> None:
        if False:
            print('Hello World!')
        'Set the given span as the current span in the execution context.\n\n        :param span: The span to set as the current span\n        :type span: Any\n        '
        ...

    @classmethod
    def set_current_tracer(cls, tracer: Any) -> None:
        if False:
            return 10
        'Set the given tracer as the current tracer in the execution context.\n\n        :param tracer: The tracer to set as the current tracer\n        :type tracer: Any\n        '
        ...

    @classmethod
    def change_context(cls, span: SpanType) -> ContextManager[SpanType]:
        if False:
            return 10
        'Change the context for the life of this context manager.\n\n        :param span: The span to run in the new context\n        :type span: Any\n        :rtype: contextmanager\n        :return: A context manager that will run the given span in the new context\n        '
        ...

    @classmethod
    def with_current_context(cls, func: Callable) -> Callable:
        if False:
            i = 10
            return i + 15
        'Passes the current spans to the new context the function will be run in.\n\n        :param func: The function that will be run in the new context\n        :type func: callable\n        :return: The target the pass in instead of the function\n        :rtype: callable\n        '
        ...

class HttpSpanMixin:
    """Can be used to get HTTP span attributes settings for free."""
    _SPAN_COMPONENT = 'component'
    _HTTP_USER_AGENT = 'http.user_agent'
    _HTTP_METHOD = 'http.method'
    _HTTP_URL = 'http.url'
    _HTTP_STATUS_CODE = 'http.status_code'
    _NET_PEER_NAME = 'net.peer.name'
    _NET_PEER_PORT = 'net.peer.port'

    def set_http_attributes(self: AbstractSpan, request: HttpRequestType, response: Optional[HttpResponseType]=None) -> None:
        if False:
            print('Hello World!')
        '\n        Add correct attributes for a http client span.\n\n        :param request: The request made\n        :type request: azure.core.rest.HttpRequest\n        :param response: The response received from the server. Is None if no response received.\n        :type response: ~azure.core.pipeline.transport.HttpResponse or ~azure.core.pipeline.transport.AsyncHttpResponse\n        '
        self.kind = SpanKind.CLIENT
        self.add_attribute(HttpSpanMixin._SPAN_COMPONENT, 'http')
        self.add_attribute(HttpSpanMixin._HTTP_METHOD, request.method)
        self.add_attribute(HttpSpanMixin._HTTP_URL, request.url)
        parsed_url = urlparse(request.url)
        if parsed_url.hostname:
            self.add_attribute(HttpSpanMixin._NET_PEER_NAME, parsed_url.hostname)
        if parsed_url.port and parsed_url.port not in [80, 443]:
            self.add_attribute(HttpSpanMixin._NET_PEER_PORT, parsed_url.port)
        user_agent = request.headers.get('User-Agent')
        if user_agent:
            self.add_attribute(HttpSpanMixin._HTTP_USER_AGENT, user_agent)
        if response and response.status_code:
            self.add_attribute(HttpSpanMixin._HTTP_STATUS_CODE, response.status_code)
        else:
            self.add_attribute(HttpSpanMixin._HTTP_STATUS_CODE, 504)

class Link:
    """
    This is a wrapper class to link the context to the current tracer.
    :param headers: A dictionary of the request header as key value pairs.
    :type headers: dict
    :param attributes: Any additional attributes that should be added to link
    :type attributes: dict
    """

    def __init__(self, headers: Dict[str, str], attributes: Optional[Attributes]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.headers = headers
        self.attributes = attributes