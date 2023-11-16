from copy import deepcopy
from typing import Any, Awaitable, TYPE_CHECKING
from azure.core.rest import AsyncHttpResponse, HttpRequest
from azure.mgmt.core import AsyncARMPipelineClient
from .. import models as _models
from ..._serialization import Deserializer, Serializer
from ._configuration import ContainerRegistryManagementClientConfiguration
from .operations import Operations, PrivateEndpointConnectionsOperations, RegistriesOperations, ReplicationsOperations, WebhooksOperations
if TYPE_CHECKING:
    from azure.core.credentials_async import AsyncTokenCredential

class ContainerRegistryManagementClient:
    """ContainerRegistryManagementClient.

    :ivar registries: RegistriesOperations operations
    :vartype registries:
     azure.mgmt.containerregistry.v2021_09_01.aio.operations.RegistriesOperations
    :ivar operations: Operations operations
    :vartype operations: azure.mgmt.containerregistry.v2021_09_01.aio.operations.Operations
    :ivar private_endpoint_connections: PrivateEndpointConnectionsOperations operations
    :vartype private_endpoint_connections:
     azure.mgmt.containerregistry.v2021_09_01.aio.operations.PrivateEndpointConnectionsOperations
    :ivar replications: ReplicationsOperations operations
    :vartype replications:
     azure.mgmt.containerregistry.v2021_09_01.aio.operations.ReplicationsOperations
    :ivar webhooks: WebhooksOperations operations
    :vartype webhooks: azure.mgmt.containerregistry.v2021_09_01.aio.operations.WebhooksOperations
    :param credential: Credential needed for the client to connect to Azure. Required.
    :type credential: ~azure.core.credentials_async.AsyncTokenCredential
    :param subscription_id: The Microsoft Azure subscription ID. Required.
    :type subscription_id: str
    :param base_url: Service URL. Default value is "https://management.azure.com".
    :type base_url: str
    :keyword api_version: Api Version. Default value is "2021-09-01". Note that overriding this
     default value may result in unsupported behavior.
    :paramtype api_version: str
    :keyword int polling_interval: Default waiting time between two polls for LRO operations if no
     Retry-After header is present.
    """

    def __init__(self, credential: 'AsyncTokenCredential', subscription_id: str, base_url: str='https://management.azure.com', **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._config = ContainerRegistryManagementClientConfiguration(credential=credential, subscription_id=subscription_id, **kwargs)
        self._client: AsyncARMPipelineClient = AsyncARMPipelineClient(base_url=base_url, config=self._config, **kwargs)
        client_models = {k: v for (k, v) in _models.__dict__.items() if isinstance(v, type)}
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)
        self._serialize.client_side_validation = False
        self.registries = RegistriesOperations(self._client, self._config, self._serialize, self._deserialize, '2021-09-01')
        self.operations = Operations(self._client, self._config, self._serialize, self._deserialize, '2021-09-01')
        self.private_endpoint_connections = PrivateEndpointConnectionsOperations(self._client, self._config, self._serialize, self._deserialize, '2021-09-01')
        self.replications = ReplicationsOperations(self._client, self._config, self._serialize, self._deserialize, '2021-09-01')
        self.webhooks = WebhooksOperations(self._client, self._config, self._serialize, self._deserialize, '2021-09-01')

    def _send_request(self, request: HttpRequest, **kwargs: Any) -> Awaitable[AsyncHttpResponse]:
        if False:
            return 10
        'Runs the network request through the client\'s chained policies.\n\n        >>> from azure.core.rest import HttpRequest\n        >>> request = HttpRequest("GET", "https://www.example.org/")\n        <HttpRequest [GET], url: \'https://www.example.org/\'>\n        >>> response = await client._send_request(request)\n        <AsyncHttpResponse: 200 OK>\n\n        For more information on this code flow, see https://aka.ms/azsdk/dpcodegen/python/send_request\n\n        :param request: The network request you want to make. Required.\n        :type request: ~azure.core.rest.HttpRequest\n        :keyword bool stream: Whether the response payload will be streamed. Defaults to False.\n        :return: The response of your network call. Does not do error handling on your response.\n        :rtype: ~azure.core.rest.AsyncHttpResponse\n        '
        request_copy = deepcopy(request)
        request_copy.url = self._client.format_url(request_copy.url)
        return self._client.send_request(request_copy, **kwargs)

    async def close(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> 'ContainerRegistryManagementClient':
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc_details: Any) -> None:
        await self._client.__aexit__(*exc_details)