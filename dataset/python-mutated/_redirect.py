"""
This module is the requests implementation of Pipeline ABC
"""
import logging
from urllib.parse import urlparse
from typing import Optional, TypeVar, Dict, Any, Union, Type
from typing_extensions import Literal
from azure.core.exceptions import TooManyRedirectsError
from azure.core.pipeline import PipelineResponse, PipelineRequest
from azure.core.pipeline.transport import HttpResponse as LegacyHttpResponse, HttpRequest as LegacyHttpRequest, AsyncHttpResponse as LegacyAsyncHttpResponse
from azure.core.rest import HttpResponse, HttpRequest, AsyncHttpResponse
from ._base import HTTPPolicy, RequestHistory
from ._utils import get_domain
HTTPResponseType = TypeVar('HTTPResponseType', HttpResponse, LegacyHttpResponse)
AllHttpResponseType = TypeVar('AllHttpResponseType', HttpResponse, LegacyHttpResponse, AsyncHttpResponse, LegacyAsyncHttpResponse)
HTTPRequestType = TypeVar('HTTPRequestType', HttpRequest, LegacyHttpRequest)
ClsRedirectPolicy = TypeVar('ClsRedirectPolicy', bound='RedirectPolicyBase')
_LOGGER = logging.getLogger(__name__)

def domain_changed(original_domain: Optional[str], url: str) -> bool:
    if False:
        i = 10
        return i + 15
    'Checks if the domain has changed.\n    :param str original_domain: The original domain.\n    :param str url: The new url.\n    :rtype: bool\n    :return: Whether the domain has changed.\n    '
    domain = get_domain(url)
    if not original_domain:
        return False
    if original_domain == domain:
        return False
    return True

class RedirectPolicyBase:
    REDIRECT_STATUSES = frozenset([300, 301, 302, 303, 307, 308])
    REDIRECT_HEADERS_BLACKLIST = frozenset(['Authorization'])

    def __init__(self, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        self.allow: bool = kwargs.get('permit_redirects', True)
        self.max_redirects: int = kwargs.get('redirect_max', 30)
        remove_headers = set(kwargs.get('redirect_remove_headers', []))
        self._remove_headers_on_redirect = remove_headers.union(self.REDIRECT_HEADERS_BLACKLIST)
        redirect_status = set(kwargs.get('redirect_on_status_codes', []))
        self._redirect_on_status_codes = redirect_status.union(self.REDIRECT_STATUSES)
        super(RedirectPolicyBase, self).__init__()

    @classmethod
    def no_redirects(cls: Type[ClsRedirectPolicy]) -> ClsRedirectPolicy:
        if False:
            return 10
        'Disable redirects.\n\n        :return: A redirect policy with redirects disabled.\n        :rtype: ~azure.core.pipeline.policies.RedirectPolicy or ~azure.core.pipeline.policies.AsyncRedirectPolicy\n        '
        return cls(permit_redirects=False)

    def configure_redirects(self, options: Dict[str, Any]) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        'Configures the redirect settings.\n\n        :param options: Keyword arguments from context.\n        :type options: dict\n        :return: A dict containing redirect settings and a history of redirects.\n        :rtype: dict\n        '
        return {'allow': options.pop('permit_redirects', self.allow), 'redirects': options.pop('redirect_max', self.max_redirects), 'history': []}

    def get_redirect_location(self, response: PipelineResponse[Any, AllHttpResponseType]) -> Union[str, None, Literal[False]]:
        if False:
            for i in range(10):
                print('nop')
        'Checks for redirect status code and gets redirect location.\n\n        :param response: The PipelineResponse object\n        :type response: ~azure.core.pipeline.PipelineResponse\n        :return: Truthy redirect location string if we got a redirect status\n         code and valid location. ``None`` if redirect status and no\n         location. ``False`` if not a redirect status code.\n        :rtype: str or bool or None\n        '
        if response.http_response.status_code in [301, 302]:
            if response.http_request.method in ['GET', 'HEAD']:
                return response.http_response.headers.get('location')
            return False
        if response.http_response.status_code in self._redirect_on_status_codes:
            return response.http_response.headers.get('location')
        return False

    def increment(self, settings: Dict[str, Any], response: PipelineResponse[Any, AllHttpResponseType], redirect_location: str) -> bool:
        if False:
            while True:
                i = 10
        'Increment the redirect attempts for this request.\n\n        :param dict settings: The redirect settings\n        :param response: A pipeline response object.\n        :type response: ~azure.core.pipeline.PipelineResponse\n        :param str redirect_location: The redirected endpoint.\n        :return: Whether further redirect attempts are remaining.\n         False if exhausted; True if more redirect attempts available.\n        :rtype: bool\n        '
        settings['redirects'] -= 1
        settings['history'].append(RequestHistory(response.http_request, http_response=response.http_response))
        redirected = urlparse(redirect_location)
        if not redirected.netloc:
            base_url = urlparse(response.http_request.url)
            response.http_request.url = '{}://{}/{}'.format(base_url.scheme, base_url.netloc, redirect_location.lstrip('/'))
        else:
            response.http_request.url = redirect_location
        if response.http_response.status_code == 303:
            response.http_request.method = 'GET'
        for non_redirect_header in self._remove_headers_on_redirect:
            response.http_request.headers.pop(non_redirect_header, None)
        return settings['redirects'] >= 0

class RedirectPolicy(RedirectPolicyBase, HTTPPolicy[HTTPRequestType, HTTPResponseType]):
    """A redirect policy.

    A redirect policy in the pipeline can be configured directly or per operation.

    :keyword bool permit_redirects: Whether the client allows redirects. Defaults to True.
    :keyword int redirect_max: The maximum allowed redirects. Defaults to 30.

    .. admonition:: Example:

        .. literalinclude:: ../samples/test_example_sync.py
            :start-after: [START redirect_policy]
            :end-before: [END redirect_policy]
            :language: python
            :dedent: 4
            :caption: Configuring a redirect policy.
    """

    def send(self, request: PipelineRequest[HTTPRequestType]) -> PipelineResponse[HTTPRequestType, HTTPResponseType]:
        if False:
            i = 10
            return i + 15
        'Sends the PipelineRequest object to the next policy.\n        Uses redirect settings to send request to redirect endpoint if necessary.\n\n        :param request: The PipelineRequest object\n        :type request: ~azure.core.pipeline.PipelineRequest\n        :return: Returns the PipelineResponse or raises error if maximum redirects exceeded.\n        :rtype: ~azure.core.pipeline.PipelineResponse\n        :raises: ~azure.core.exceptions.TooManyRedirectsError if maximum redirects exceeded.\n        '
        retryable: bool = True
        redirect_settings = self.configure_redirects(request.context.options)
        original_domain = get_domain(request.http_request.url) if redirect_settings['allow'] else None
        while retryable:
            response = self.next.send(request)
            redirect_location = self.get_redirect_location(response)
            if redirect_location and redirect_settings['allow']:
                retryable = self.increment(redirect_settings, response, redirect_location)
                request.http_request = response.http_request
                if domain_changed(original_domain, request.http_request.url):
                    request.context.options['insecure_domain_change'] = True
                continue
            return response
        raise TooManyRedirectsError(redirect_settings['history'])