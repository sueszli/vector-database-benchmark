from typing import Any, Union, Optional
from io import SEEK_SET, UnsupportedOperation
from azure.core.credentials import TokenCredential
from azure.core.pipeline import PipelineRequest, PipelineResponse
from azure.core.pipeline.policies import HTTPPolicy
from ._anonymous_exchange_client import AnonymousACRExchangeClient
from ._exchange_client import ACRExchangeClient
from ._helpers import _enforce_https

class ContainerRegistryChallengePolicy(HTTPPolicy):
    """Authentication policy for ACR which accepts a challenge"""

    def __init__(self, credential: Optional[TokenCredential], endpoint: str, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        super(ContainerRegistryChallengePolicy, self).__init__()
        self._credential = credential
        if self._credential is None:
            self._exchange_client: Union[AnonymousACRExchangeClient, ACRExchangeClient] = AnonymousACRExchangeClient(endpoint, **kwargs)
        else:
            self._exchange_client = ACRExchangeClient(endpoint, self._credential, **kwargs)

    def on_request(self, request: PipelineRequest) -> None:
        if False:
            i = 10
            return i + 15
        'Called before the policy sends a request.\n        The base implementation authorizes the request with a bearer token.\n\n        :param ~azure.core.pipeline.PipelineRequest request: the request\n        :return: None\n        :rtype: None\n        '
        pass

    def send(self, request: PipelineRequest) -> PipelineResponse:
        if False:
            while True:
                i = 10
        'Authorizes a request with a bearer token, possibly handling an authentication challenge\n\n        :param ~azure.core.pipeline.PipelineRequest request: The pipeline request object.\n        :return: The pipeline response object.\n        :rtype: ~azure.core.pipeline.PipelineResponse\n        '
        _enforce_https(request)
        self.on_request(request)
        response = self.next.send(request)
        if response.http_response.status_code == 401:
            challenge = response.http_response.headers.get('WWW-Authenticate')
            if challenge and self.on_challenge(request, response, challenge):
                if request.http_request.body and hasattr(request.http_request.body, 'read'):
                    try:
                        request.http_request.body.seek(0, SEEK_SET)
                    except (UnsupportedOperation, ValueError, AttributeError):
                        return response
                response = self.next.send(request)
        return response

    def on_challenge(self, request: PipelineRequest, response: PipelineResponse, challenge: str) -> bool:
        if False:
            i = 10
            return i + 15
        "Authorize request according to an authentication challenge\n        This method is called when the resource provider responds 401 with a WWW-Authenticate header.\n\n        :param ~azure.core.pipeline.PipelineRequest request: the request which elicited an authentication challenge\n        :param ~azure.core.pipeline.PipelineResponse response: the resource provider's response\n        :param str challenge: response's WWW-Authenticate header, unparsed. It may contain multiple challenges.\n        :returns: a bool indicating whether the policy should send the request\n        :rtype: bool\n        "
        access_token = self._exchange_client.get_acr_access_token(challenge)
        if access_token is not None:
            request.http_request.headers['Authorization'] = 'Bearer ' + access_token
        return access_token is not None

    def __enter__(self):
        if False:
            print('Hello World!')
        self._exchange_client.__enter__()
        return self

    def __exit__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self._exchange_client.__exit__(*args)