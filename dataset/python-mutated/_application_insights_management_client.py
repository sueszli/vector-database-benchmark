from copy import deepcopy
from typing import Any, TYPE_CHECKING
from azure.core.rest import HttpRequest, HttpResponse
from azure.mgmt.core import ARMPipelineClient
from . import models as _models
from .._serialization import Deserializer, Serializer
from ._configuration import ApplicationInsightsManagementClientConfiguration
from .operations import LiveTokenOperations, Operations
if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

class ApplicationInsightsManagementClient:
    """Composite Swagger for Application Insights Management Client.

    :ivar operations: Operations operations
    :vartype operations: azure.mgmt.applicationinsights.v2020_06_02_preview.operations.Operations
    :ivar live_token: LiveTokenOperations operations
    :vartype live_token:
     azure.mgmt.applicationinsights.v2020_06_02_preview.operations.LiveTokenOperations
    :param credential: Credential needed for the client to connect to Azure. Required.
    :type credential: ~azure.core.credentials.TokenCredential
    :param base_url: Service URL. Default value is "https://management.azure.com".
    :type base_url: str
    :keyword api_version: Api Version. Default value is "2020-06-02-preview". Note that overriding
     this default value may result in unsupported behavior.
    :paramtype api_version: str
    """

    def __init__(self, credential: 'TokenCredential', base_url: str='https://management.azure.com', **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        self._config = ApplicationInsightsManagementClientConfiguration(credential=credential, **kwargs)
        self._client: ARMPipelineClient = ARMPipelineClient(base_url=base_url, config=self._config, **kwargs)
        client_models = {k: v for (k, v) in _models.__dict__.items() if isinstance(v, type)}
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)
        self._serialize.client_side_validation = False
        self.operations = Operations(self._client, self._config, self._serialize, self._deserialize)
        self.live_token = LiveTokenOperations(self._client, self._config, self._serialize, self._deserialize)

    def _send_request(self, request: HttpRequest, **kwargs: Any) -> HttpResponse:
        if False:
            print('Hello World!')
        'Runs the network request through the client\'s chained policies.\n\n        >>> from azure.core.rest import HttpRequest\n        >>> request = HttpRequest("GET", "https://www.example.org/")\n        <HttpRequest [GET], url: \'https://www.example.org/\'>\n        >>> response = client._send_request(request)\n        <HttpResponse: 200 OK>\n\n        For more information on this code flow, see https://aka.ms/azsdk/dpcodegen/python/send_request\n\n        :param request: The network request you want to make. Required.\n        :type request: ~azure.core.rest.HttpRequest\n        :keyword bool stream: Whether the response payload will be streamed. Defaults to False.\n        :return: The response of your network call. Does not do error handling on your response.\n        :rtype: ~azure.core.rest.HttpResponse\n        '
        request_copy = deepcopy(request)
        request_copy.url = self._client.format_url(request_copy.url)
        return self._client.send_request(request_copy, **kwargs)

    def close(self) -> None:
        if False:
            while True:
                i = 10
        self._client.close()

    def __enter__(self) -> 'ApplicationInsightsManagementClient':
        if False:
            print('Hello World!')
        self._client.__enter__()
        return self

    def __exit__(self, *exc_details: Any) -> None:
        if False:
            i = 10
            return i + 15
        self._client.__exit__(*exc_details)