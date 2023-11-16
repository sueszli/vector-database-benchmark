import asyncio
import time
from typing import Any, Awaitable, Optional, Dict, Union
from azure.core.pipeline.policies import AsyncHTTPPolicy
from azure.core.credentials import AccessToken
from azure.core.pipeline import PipelineRequest, PipelineResponse
from azure.cosmos import http_constants

async def await_result(func, *args, **kwargs):
    """If func returns an awaitable, await it.
    :param Callable func: The function to be awaited.
    :param List[Any] args: The explicit arguments for the function.
    :returns: The result from awaiting the function.
    :rtype: HttpResponse
    """
    result = func(*args, **kwargs)
    if hasattr(result, '__await__'):
        return await result
    return result

class _AsyncCosmosBearerTokenCredentialPolicyBase(object):
    """Base class for a Bearer Token Credential Policy.

    :param credential: The credential.
    :type credential: ~azure.core.credentials.TokenCredential
    :param str scopes: Lets you specify the type of access needed.
    """

    def __init__(self, credential, *scopes, **kwargs):
        if False:
            while True:
                i = 10
        super(_AsyncCosmosBearerTokenCredentialPolicyBase, self).__init__()
        self._scopes = scopes
        self._credential = credential
        self._token = None
        self._lock = asyncio.Lock()

    @staticmethod
    def _enforce_https(request):
        if False:
            return 10
        option = request.context.options.pop('enforce_https', None)
        if option is False:
            request.context['enforce_https'] = option
        enforce_https = request.context.get('enforce_https', True)
        if enforce_https and (not request.http_request.url.lower().startswith('https')):
            raise ValueError('Bearer token authentication is not permitted for non-TLS protected (non-https) URLs.')

    @staticmethod
    def _update_headers(headers, token):
        if False:
            for i in range(10):
                print('nop')
        "Updates the Authorization header with the cosmos signature and bearer token.\n        This is the main method that differentiates this policy from core's BearerTokenCredentialPolicy and works\n        to properly sign the authorization header for Cosmos' REST API. For more information:\n        https://docs.microsoft.com/rest/api/cosmos-db/access-control-on-cosmosdb-resources#authorization-header\n\n        :param dict headers: The HTTP Request headers\n        :param str token: The OAuth token.\n        "
        headers[http_constants.HttpHeaders.Authorization] = 'type=aad&ver=1.0&sig={}'.format(token)

    @property
    def _need_new_token(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return not self._token or self._token.expires_on - time.time() < 300

class AsyncCosmosBearerTokenCredentialPolicy(_AsyncCosmosBearerTokenCredentialPolicyBase, AsyncHTTPPolicy):
    """Adds a bearer token Authorization header to requests.

    :param credential: The credential.
    :type credential: ~azure.core.TokenCredential
    :param str scopes: Lets you specify the type of access needed.
    :raises ValueError: If https_enforce does not match with endpoint being used.
    """

    async def on_request(self, request: 'PipelineRequest') -> None:
        """Adds a bearer token Authorization header to request and sends request to next policy.

        :param request: The pipeline request object to be modified.
        :type request: ~azure.core.pipeline.PipelineRequest
        :raises: :class:`~azure.core.exceptions.ServiceRequestError`
        """
        self._enforce_https(request)
        if self._token is None or self._need_new_token:
            async with self._lock:
                if self._token is None or self._need_new_token:
                    self._token = await self._credential.get_token(*self._scopes)
        self._update_headers(request.http_request.headers, self._token.token)

    async def authorize_request(self, request: 'PipelineRequest', *scopes: str, **kwargs: 'Any') -> None:
        """Acquire a token from the credential and authorize the request with it.

        Keyword arguments are passed to the credential's get_token method. The token will be cached and used to
        authorize future requests.

        :param ~azure.core.pipeline.PipelineRequest request: the request
        :param str scopes: required scopes of authentication
        """
        async with self._lock:
            self._token = await self._credential.get_token(*scopes, **kwargs)
        self._update_headers(request.http_request.headers, self._token.token)

    async def send(self, request: 'PipelineRequest') -> 'PipelineResponse':
        """Authorize request with a bearer token and send it to the next policy

        :param request: The pipeline request object
        :type request: ~azure.core.pipeline.PipelineRequest
        :returns: The result of sending the request.
        :rtype: ~azure.core.pipeline.PipelineResponse
        """
        await await_result(self.on_request, request)
        try:
            response = await self.next.send(request)
            await await_result(self.on_response, request, response)
        except Exception:
            handled = await await_result(self.on_exception, request)
            if not handled:
                raise
        else:
            if response.http_response.status_code == 401:
                self._token = None
                if 'WWW-Authenticate' in response.http_response.headers:
                    request_authorized = await self.on_challenge(request, response)
                    if request_authorized:
                        try:
                            response = await self.next.send(request)
                            await await_result(self.on_response, request, response)
                        except Exception:
                            handled = await await_result(self.on_exception, request)
                            if not handled:
                                raise
        return response

    async def on_challenge(self, request: 'PipelineRequest', response: 'PipelineResponse') -> bool:
        """Authorize request according to an authentication challenge

        This method is called when the resource provider responds 401 with a WWW-Authenticate header.

        :param ~azure.core.pipeline.PipelineRequest request: the request which elicited an authentication challenge
        :param ~azure.core.pipeline.PipelineResponse response: the resource provider's response
        :returns: a boolean indicating whether the policy should send the request
        :rtype: bool
        """
        return False

    def on_response(self, request: PipelineRequest, response: PipelineResponse) -> Union[None, Awaitable[None]]:
        if False:
            print('Hello World!')
        'Executed after the request comes back from the next policy.\n\n        :param request: Request to be modified after returning from the policy.\n        :type request: ~azure.core.pipeline.PipelineRequest\n        :param response: Pipeline response object\n        :type response: ~azure.core.pipeline.PipelineResponse\n        '

    def on_exception(self, request: PipelineRequest) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Executed when an exception is raised while executing the next policy.\n\n        This method is executed inside the exception handler.\n\n        :param request: The Pipeline request object\n        :type request: ~azure.core.pipeline.PipelineRequest\n        '
        return