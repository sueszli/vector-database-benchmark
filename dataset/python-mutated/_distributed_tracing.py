"""Traces network calls using the implementation library from the settings."""
import logging
import sys
import urllib.parse
from typing import TYPE_CHECKING, Optional, Tuple, TypeVar, Union, Any, Type
from types import TracebackType
from azure.core.pipeline import PipelineRequest, PipelineResponse
from azure.core.pipeline.policies import SansIOHTTPPolicy
from azure.core.pipeline.transport import HttpResponse as LegacyHttpResponse, HttpRequest as LegacyHttpRequest
from azure.core.rest import HttpResponse, HttpRequest
from azure.core.settings import settings
from azure.core.tracing import SpanKind
if TYPE_CHECKING:
    from azure.core.tracing._abstract_span import AbstractSpan
HTTPResponseType = TypeVar('HTTPResponseType', HttpResponse, LegacyHttpResponse)
HTTPRequestType = TypeVar('HTTPRequestType', HttpRequest, LegacyHttpRequest)
ExcInfo = Tuple[Type[BaseException], BaseException, TracebackType]
OptExcInfo = Union[ExcInfo, Tuple[None, None, None]]
_LOGGER = logging.getLogger(__name__)

def _default_network_span_namer(http_request: HTTPRequestType) -> str:
    if False:
        i = 10
        return i + 15
    'Extract the path to be used as network span name.\n\n    :param http_request: The HTTP request\n    :type http_request: ~azure.core.pipeline.transport.HttpRequest\n    :returns: The string to use as network span name\n    :rtype: str\n    '
    path = urllib.parse.urlparse(http_request.url).path
    if not path:
        path = '/'
    return path

class DistributedTracingPolicy(SansIOHTTPPolicy[HTTPRequestType, HTTPResponseType]):
    """The policy to create spans for Azure calls.

    :keyword network_span_namer: A callable to customize the span name
    :type network_span_namer: callable[[~azure.core.pipeline.transport.HttpRequest], str]
    :keyword tracing_attributes: Attributes to set on all created spans
    :type tracing_attributes: dict[str, str]
    """
    TRACING_CONTEXT = 'TRACING_CONTEXT'
    _REQUEST_ID = 'x-ms-client-request-id'
    _RESPONSE_ID = 'x-ms-request-id'

    def __init__(self, **kwargs: Any):
        if False:
            print('Hello World!')
        self._network_span_namer = kwargs.get('network_span_namer', _default_network_span_namer)
        self._tracing_attributes = kwargs.get('tracing_attributes', {})

    def on_request(self, request: PipelineRequest[HTTPRequestType]) -> None:
        if False:
            while True:
                i = 10
        ctxt = request.context.options
        try:
            span_impl_type = settings.tracing_implementation()
            if span_impl_type is None:
                return
            namer = ctxt.pop('network_span_namer', self._network_span_namer)
            span_name = namer(request.http_request)
            span = span_impl_type(name=span_name, kind=SpanKind.CLIENT)
            for (attr, value) in self._tracing_attributes.items():
                span.add_attribute(attr, value)
            span.start()
            headers = span.to_header()
            request.http_request.headers.update(headers)
            request.context[self.TRACING_CONTEXT] = span
        except Exception as err:
            _LOGGER.warning('Unable to start network span: %s', err)

    def end_span(self, request: PipelineRequest[HTTPRequestType], response: Optional[HTTPResponseType]=None, exc_info: Optional[OptExcInfo]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Ends the span that is tracing the network and updates its status.\n\n        :param request: The PipelineRequest object\n        :type request: ~azure.core.pipeline.PipelineRequest\n        :param response: The HttpResponse object\n        :type response: ~azure.core.rest.HTTPResponse or ~azure.core.pipeline.transport.HttpResponse\n        :param exc_info: The exception information\n        :type exc_info: tuple\n        '
        if self.TRACING_CONTEXT not in request.context:
            return
        span: 'AbstractSpan' = request.context[self.TRACING_CONTEXT]
        http_request: Union[HttpRequest, LegacyHttpRequest] = request.http_request
        if span is not None:
            span.set_http_attributes(http_request, response=response)
            request_id = http_request.headers.get(self._REQUEST_ID)
            if request_id is not None:
                span.add_attribute(self._REQUEST_ID, request_id)
            if response and self._RESPONSE_ID in response.headers:
                span.add_attribute(self._RESPONSE_ID, response.headers[self._RESPONSE_ID])
            if exc_info:
                span.__exit__(*exc_info)
            else:
                span.finish()

    def on_response(self, request: PipelineRequest[HTTPRequestType], response: PipelineResponse[HTTPRequestType, HTTPResponseType]) -> None:
        if False:
            i = 10
            return i + 15
        self.end_span(request, response=response.http_response)

    def on_exception(self, request: PipelineRequest[HTTPRequestType]) -> None:
        if False:
            while True:
                i = 10
        self.end_span(request, exc_info=sys.exc_info())