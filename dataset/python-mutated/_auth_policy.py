import time
from typing import Any, Dict, Optional
from azure.core.credentials import AccessToken
from azure.core.pipeline import PipelineRequest, PipelineResponse
from azure.core.pipeline.policies import HTTPPolicy
from azure.cosmos import http_constants

class _CosmosBearerTokenCredentialPolicyBase(object):
    """Base class for a Bearer Token Credential Policy.

    :param credential: The credential.
    :type credential: ~azure.core.credentials.TokenCredential
    :param str scopes: Lets you specify the type of access needed.
    """

    def __init__(self, credential, *scopes, **kwargs):
        if False:
            while True:
                i = 10
        super(_CosmosBearerTokenCredentialPolicyBase, self).__init__()
        self._scopes = scopes
        self._credential = credential
        self._token = None

    @staticmethod
    def _enforce_https(request):
        if False:
            i = 10
            return i + 15
        option = request.context.options.pop('enforce_https', None)
        if option is False:
            request.context['enforce_https'] = option
        enforce_https = request.context.get('enforce_https', True)
        if enforce_https and (not request.http_request.url.lower().startswith('https')):
            raise ValueError('Bearer token authentication is not permitted for non-TLS protected (non-https) URLs.')

    @staticmethod
    def _update_headers(headers, token):
        if False:
            i = 10
            return i + 15
        "Updates the Authorization header with the bearer token.\n        This is the main method that differentiates this policy from core's BearerTokenCredentialPolicy and works\n        to properly sign the authorization header for Cosmos' REST API. For more information:\n        https://docs.microsoft.com/rest/api/cosmos-db/access-control-on-cosmosdb-resources#authorization-header\n\n        :param dict headers: The HTTP Request headers\n        :param str token: The OAuth token.\n        "
        headers[http_constants.HttpHeaders.Authorization] = 'type=aad&ver=1.0&sig={}'.format(token)

    @property
    def _need_new_token(self):
        if False:
            for i in range(10):
                print('nop')
        return not self._token or self._token.expires_on - time.time() < 300

class CosmosBearerTokenCredentialPolicy(_CosmosBearerTokenCredentialPolicyBase, HTTPPolicy):
    """Adds a bearer token Authorization header to requests.

    :param credential: The credential.
    :type credential: ~azure.core.TokenCredential
    :param str scopes: Lets you specify the type of access needed.
    :raises ValueError: If https_enforce does not match with endpoint being used.
    """

    def on_request(self, request):
        if False:
            print('Hello World!')
        'Called before the policy sends a request.\n\n        The base implementation authorizes the request with a bearer token.\n\n        :param ~azure.core.pipeline.PipelineRequest request: the request\n        '
        self._enforce_https(request)
        if self._token is None or self._need_new_token:
            self._token = self._credential.get_token(*self._scopes)
        self._update_headers(request.http_request.headers, self._token.token)

    def authorize_request(self, request, *scopes, **kwargs):
        if False:
            while True:
                i = 10
        "Acquire a token from the credential and authorize the request with it.\n\n        Keyword arguments are passed to the credential's get_token method. The token will be cached and used to\n        authorize future requests.\n\n        :param ~azure.core.pipeline.PipelineRequest request: the request\n        :param str scopes: required scopes of authentication\n        "
        self._token = self._credential.get_token(*scopes, **kwargs)
        self._update_headers(request.http_request.headers, self._token.token)

    def send(self, request):
        if False:
            print('Hello World!')
        'Authorize request with a bearer token and send it to the next policy\n\n        :param request: The pipeline request object.\n        :type request: ~azure.core.pipeline.PipelineRequest\n        :returns: The pipeline response object.\n        :rtype: ~azure.core.pipeline.PipelineResponse\n        '
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
                        try:
                            response = self.next.send(request)
                            self.on_response(request, response)
                        except Exception:
                            self.on_exception(request)
                            raise
        return response

    def on_challenge(self, request, response):
        if False:
            print('Hello World!')
        "Authorize request according to an authentication challenge\n\n        This method is called when the resource provider responds 401 with a WWW-Authenticate header.\n\n        :param ~azure.core.pipeline.PipelineRequest request: the request which elicited an authentication challenge\n        :param ~azure.core.pipeline.PipelineResponse response: the resource provider's response\n        :returns: a boolean indicating whether the policy should send the request\n        :rtype: bool\n        "
        return False

    def on_response(self, request, response):
        if False:
            for i in range(10):
                print('nop')
        'Executed after the request comes back from the next policy.\n\n        :param request: Request to be modified after returning from the policy.\n        :type request: ~azure.core.pipeline.PipelineRequest\n        :param response: Pipeline response object\n        :type response: ~azure.core.pipeline.PipelineResponse\n        '

    def on_exception(self, request):
        if False:
            while True:
                i = 10
        'Executed when an exception is raised while executing the next policy.\n\n        This method is executed inside the exception handler.\n\n        :param request: The Pipeline request object\n        :type request: ~azure.core.pipeline.PipelineRequest\n        '
        return