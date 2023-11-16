import time
from typing import TYPE_CHECKING, Optional, TypeVar, MutableMapping, Any
from azure.core.pipeline import PipelineRequest, PipelineResponse
from azure.core.pipeline.transport import HttpResponse as LegacyHttpResponse, HttpRequest as LegacyHttpRequest
from azure.core.rest import HttpResponse, HttpRequest
from . import HTTPPolicy, SansIOHTTPPolicy
from ...exceptions import ServiceRequestError
if TYPE_CHECKING:
    from azure.core.credentials import AccessToken, TokenCredential, AzureKeyCredential, AzureSasCredential
HTTPResponseType = TypeVar('HTTPResponseType', HttpResponse, LegacyHttpResponse)
HTTPRequestType = TypeVar('HTTPRequestType', HttpRequest, LegacyHttpRequest)

class _BearerTokenCredentialPolicyBase:
    """Base class for a Bearer Token Credential Policy.

    :param credential: The credential.
    :type credential: ~azure.core.credentials.TokenCredential
    :param str scopes: Lets you specify the type of access needed.
    :keyword bool enable_cae: Indicates whether to enable Continuous Access Evaluation (CAE) on all requested
        tokens. Defaults to False.
    """

    def __init__(self, credential: 'TokenCredential', *scopes: str, **kwargs: Any) -> None:
        if False:
            return 10
        super(_BearerTokenCredentialPolicyBase, self).__init__()
        self._scopes = scopes
        self._credential = credential
        self._token: Optional['AccessToken'] = None
        self._enable_cae: bool = kwargs.get('enable_cae', False)

    @staticmethod
    def _enforce_https(request: PipelineRequest[HTTPRequestType]) -> None:
        if False:
            for i in range(10):
                print('nop')
        option = request.context.options.pop('enforce_https', None)
        if option is False:
            request.context['enforce_https'] = option
        enforce_https = request.context.get('enforce_https', True)
        if enforce_https and (not request.http_request.url.lower().startswith('https')):
            raise ServiceRequestError('Bearer token authentication is not permitted for non-TLS protected (non-https) URLs.')

    @staticmethod
    def _update_headers(headers: MutableMapping[str, str], token: str) -> None:
        if False:
            while True:
                i = 10
        'Updates the Authorization header with the bearer token.\n\n        :param MutableMapping[str, str] headers: The HTTP Request headers\n        :param str token: The OAuth token.\n        '
        headers['Authorization'] = 'Bearer {}'.format(token)

    @property
    def _need_new_token(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return not self._token or self._token.expires_on - time.time() < 300

class BearerTokenCredentialPolicy(_BearerTokenCredentialPolicyBase, HTTPPolicy[HTTPRequestType, HTTPResponseType]):
    """Adds a bearer token Authorization header to requests.

    :param credential: The credential.
    :type credential: ~azure.core.TokenCredential
    :param str scopes: Lets you specify the type of access needed.
    :keyword bool enable_cae: Indicates whether to enable Continuous Access Evaluation (CAE) on all requested
        tokens. Defaults to False.
    :raises: :class:`~azure.core.exceptions.ServiceRequestError`
    """

    def on_request(self, request: PipelineRequest[HTTPRequestType]) -> None:
        if False:
            print('Hello World!')
        'Called before the policy sends a request.\n\n        The base implementation authorizes the request with a bearer token.\n\n        :param ~azure.core.pipeline.PipelineRequest request: the request\n        '
        self._enforce_https(request)
        if self._token is None or self._need_new_token:
            if self._enable_cae:
                self._token = self._credential.get_token(*self._scopes, enable_cae=self._enable_cae)
            else:
                self._token = self._credential.get_token(*self._scopes)
        self._update_headers(request.http_request.headers, self._token.token)

    def authorize_request(self, request: PipelineRequest[HTTPRequestType], *scopes: str, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        "Acquire a token from the credential and authorize the request with it.\n\n        Keyword arguments are passed to the credential's get_token method. The token will be cached and used to\n        authorize future requests.\n\n        :param ~azure.core.pipeline.PipelineRequest request: the request\n        :param str scopes: required scopes of authentication\n        "
        if self._enable_cae:
            kwargs.setdefault('enable_cae', self._enable_cae)
        self._token = self._credential.get_token(*scopes, **kwargs)
        self._update_headers(request.http_request.headers, self._token.token)

    def send(self, request: PipelineRequest[HTTPRequestType]) -> PipelineResponse[HTTPRequestType, HTTPResponseType]:
        if False:
            while True:
                i = 10
        'Authorize request with a bearer token and send it to the next policy\n\n        :param request: The pipeline request object\n        :type request: ~azure.core.pipeline.PipelineRequest\n        :return: The pipeline response object\n        :rtype: ~azure.core.pipeline.PipelineResponse\n        '
        self.on_request(request)
        try:
            response = self.next.send(request)
            self.on_response(request, response)
        except Exception:
            self.on_exception(request)
            raise
        else:
            if response.http_response.status_code == 401:
                self._token = None
                if 'WWW-Authenticate' in response.http_response.headers:
                    request_authorized = self.on_challenge(request, response)
                    if request_authorized:
                        request.context.options.pop('insecure_domain_change', False)
                        try:
                            response = self.next.send(request)
                            self.on_response(request, response)
                        except Exception:
                            self.on_exception(request)
                            raise
        return response

    def on_challenge(self, request: PipelineRequest[HTTPRequestType], response: PipelineResponse[HTTPRequestType, HTTPResponseType]) -> bool:
        if False:
            return 10
        "Authorize request according to an authentication challenge\n\n        This method is called when the resource provider responds 401 with a WWW-Authenticate header.\n\n        :param ~azure.core.pipeline.PipelineRequest request: the request which elicited an authentication challenge\n        :param ~azure.core.pipeline.PipelineResponse response: the resource provider's response\n        :returns: a bool indicating whether the policy should send the request\n        :rtype: bool\n        "
        return False

    def on_response(self, request: PipelineRequest[HTTPRequestType], response: PipelineResponse[HTTPRequestType, HTTPResponseType]) -> None:
        if False:
            while True:
                i = 10
        'Executed after the request comes back from the next policy.\n\n        :param request: Request to be modified after returning from the policy.\n        :type request: ~azure.core.pipeline.PipelineRequest\n        :param response: Pipeline response object\n        :type response: ~azure.core.pipeline.PipelineResponse\n        '

    def on_exception(self, request: PipelineRequest[HTTPRequestType]) -> None:
        if False:
            return 10
        'Executed when an exception is raised while executing the next policy.\n\n        This method is executed inside the exception handler.\n\n        :param request: The Pipeline request object\n        :type request: ~azure.core.pipeline.PipelineRequest\n        '
        return

class AzureKeyCredentialPolicy(SansIOHTTPPolicy[HTTPRequestType, HTTPResponseType]):
    """Adds a key header for the provided credential.

    :param credential: The credential used to authenticate requests.
    :type credential: ~azure.core.credentials.AzureKeyCredential
    :param str name: The name of the key header used for the credential.
    :keyword str prefix: The name of the prefix for the header value if any.
    :raises: ValueError or TypeError
    """

    def __init__(self, credential: 'AzureKeyCredential', name: str, *, prefix: Optional[str]=None, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        if not hasattr(credential, 'key'):
            raise TypeError('String is not a supported credential input type. Use an instance of AzureKeyCredential.')
        if not name:
            raise ValueError('name can not be None or empty')
        if not isinstance(name, str):
            raise TypeError('name must be a string.')
        self._credential = credential
        self._name = name
        self._prefix = prefix + ' ' if prefix else ''

    def on_request(self, request: PipelineRequest[HTTPRequestType]) -> None:
        if False:
            while True:
                i = 10
        request.http_request.headers[self._name] = f'{self._prefix}{self._credential.key}'

class AzureSasCredentialPolicy(SansIOHTTPPolicy[HTTPRequestType, HTTPResponseType]):
    """Adds a shared access signature to query for the provided credential.

    :param credential: The credential used to authenticate requests.
    :type credential: ~azure.core.credentials.AzureSasCredential
    :raises: ValueError or TypeError
    """

    def __init__(self, credential: 'AzureSasCredential', **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        super(AzureSasCredentialPolicy, self).__init__()
        if not credential:
            raise ValueError('credential can not be None')
        self._credential = credential

    def on_request(self, request: PipelineRequest[HTTPRequestType]) -> None:
        if False:
            return 10
        url = request.http_request.url
        query = request.http_request.query
        signature = self._credential.signature
        if signature.startswith('?'):
            signature = signature[1:]
        if query:
            if signature not in url:
                url = url + '&' + signature
        elif url.endswith('?'):
            url = url + signature
        else:
            url = url + '?' + signature
        request.http_request.url = url