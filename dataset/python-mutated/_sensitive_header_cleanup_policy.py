from typing import List, Optional, Any, TypeVar
from azure.core.pipeline import PipelineRequest
from azure.core.pipeline.transport import HttpRequest as LegacyHttpRequest, HttpResponse as LegacyHttpResponse
from azure.core.rest import HttpRequest, HttpResponse
from ._base import SansIOHTTPPolicy
HTTPResponseType = TypeVar('HTTPResponseType', HttpResponse, LegacyHttpResponse)
HTTPRequestType = TypeVar('HTTPRequestType', HttpRequest, LegacyHttpRequest)

class SensitiveHeaderCleanupPolicy(SansIOHTTPPolicy[HTTPRequestType, HTTPResponseType]):
    """A simple policy that cleans up sensitive headers

    :keyword list[str] blocked_redirect_headers: The headers to clean up when redirecting to another domain.
    :keyword bool disable_redirect_cleanup: Opt out cleaning up sensitive headers when redirecting to another domain.
    """
    DEFAULT_SENSITIVE_HEADERS = set(['Authorization', 'x-ms-authorization-auxiliary'])

    def __init__(self, *, blocked_redirect_headers: Optional[List[str]]=None, disable_redirect_cleanup: bool=False, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        self._disable_redirect_cleanup = disable_redirect_cleanup
        self._blocked_redirect_headers = SensitiveHeaderCleanupPolicy.DEFAULT_SENSITIVE_HEADERS if blocked_redirect_headers is None else blocked_redirect_headers

    def on_request(self, request: PipelineRequest[HTTPRequestType]) -> None:
        if False:
            while True:
                i = 10
        'This is executed before sending the request to the next policy.\n\n        :param request: The PipelineRequest object.\n        :type request: ~azure.core.pipeline.PipelineRequest\n        '
        insecure_domain_change = request.context.options.pop('insecure_domain_change', False)
        if not self._disable_redirect_cleanup and insecure_domain_change:
            for header in self._blocked_redirect_headers:
                request.http_request.headers.pop(header, None)