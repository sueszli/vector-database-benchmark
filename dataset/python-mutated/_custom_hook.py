from typing import TypeVar, Any
from azure.core.pipeline import PipelineRequest, PipelineResponse
from azure.core.pipeline.transport import HttpResponse as LegacyHttpResponse, HttpRequest as LegacyHttpRequest
from azure.core.rest import HttpResponse, HttpRequest
from ._base import SansIOHTTPPolicy
HTTPResponseType = TypeVar('HTTPResponseType', HttpResponse, LegacyHttpResponse)
HTTPRequestType = TypeVar('HTTPRequestType', HttpRequest, LegacyHttpRequest)

class CustomHookPolicy(SansIOHTTPPolicy[HTTPRequestType, HTTPResponseType]):
    """A simple policy that enable the given callback
    with the response.

    :keyword callback raw_request_hook: Callback function. Will be invoked on request.
    :keyword callback raw_response_hook: Callback function. Will be invoked on response.
    """

    def __init__(self, **kwargs: Any):
        if False:
            for i in range(10):
                print('nop')
        self._request_callback = kwargs.get('raw_request_hook')
        self._response_callback = kwargs.get('raw_response_hook')

    def on_request(self, request: PipelineRequest[HTTPRequestType]) -> None:
        if False:
            print('Hello World!')
        'This is executed before sending the request to the next policy.\n\n        :param request: The PipelineRequest object.\n        :type request: ~azure.core.pipeline.PipelineRequest\n        '
        request_callback = request.context.options.pop('raw_request_hook', None)
        if request_callback:
            request.context['raw_request_hook'] = request_callback
            request_callback(request)
        elif self._request_callback:
            self._request_callback(request)
        response_callback = request.context.options.pop('raw_response_hook', None)
        if response_callback:
            request.context['raw_response_hook'] = response_callback

    def on_response(self, request: PipelineRequest[HTTPRequestType], response: PipelineResponse[HTTPRequestType, HTTPResponseType]) -> None:
        if False:
            return 10
        'This is executed after the request comes back from the policy.\n\n        :param request: The PipelineRequest object.\n        :type request: ~azure.core.pipeline.PipelineRequest\n        :param response: The PipelineResponse object.\n        :type response: ~azure.core.pipeline.PipelineResponse\n        '
        response_callback = response.context.get('raw_response_hook')
        if response_callback:
            response_callback(response)
        elif self._response_callback:
            self._response_callback(response)