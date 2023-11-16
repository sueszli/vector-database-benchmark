from copy import deepcopy
from typing import Any, Awaitable, TYPE_CHECKING
from azure.core.rest import AsyncHttpResponse, HttpRequest
from azure.mgmt.core import AsyncARMPipelineClient
from .. import models as _models
from .._serialization import Deserializer, Serializer
from ._configuration import ChaosManagementClientConfiguration
from .operations import CapabilitiesOperations, CapabilityTypesOperations, ExperimentsOperations, Operations, TargetTypesOperations, TargetsOperations
if TYPE_CHECKING:
    from azure.core.credentials_async import AsyncTokenCredential

class ChaosManagementClient:
    """Chaos Management Client.

    :ivar capabilities: CapabilitiesOperations operations
    :vartype capabilities: azure.mgmt.chaos.aio.operations.CapabilitiesOperations
    :ivar capability_types: CapabilityTypesOperations operations
    :vartype capability_types: azure.mgmt.chaos.aio.operations.CapabilityTypesOperations
    :ivar experiments: ExperimentsOperations operations
    :vartype experiments: azure.mgmt.chaos.aio.operations.ExperimentsOperations
    :ivar operations: Operations operations
    :vartype operations: azure.mgmt.chaos.aio.operations.Operations
    :ivar target_types: TargetTypesOperations operations
    :vartype target_types: azure.mgmt.chaos.aio.operations.TargetTypesOperations
    :ivar targets: TargetsOperations operations
    :vartype targets: azure.mgmt.chaos.aio.operations.TargetsOperations
    :param credential: Credential needed for the client to connect to Azure. Required.
    :type credential: ~azure.core.credentials_async.AsyncTokenCredential
    :param subscription_id: GUID that represents an Azure subscription ID. Required.
    :type subscription_id: str
    :param base_url: Service URL. Default value is "https://management.azure.com".
    :type base_url: str
    :keyword api_version: Api Version. Default value is "2023-04-15-preview". Note that overriding
     this default value may result in unsupported behavior.
    :paramtype api_version: str
    """

    def __init__(self, credential: 'AsyncTokenCredential', subscription_id: str, base_url: str='https://management.azure.com', **kwargs: Any) -> None:
        if False:
            return 10
        self._config = ChaosManagementClientConfiguration(credential=credential, subscription_id=subscription_id, **kwargs)
        self._client: AsyncARMPipelineClient = AsyncARMPipelineClient(base_url=base_url, config=self._config, **kwargs)
        client_models = {k: v for (k, v) in _models.__dict__.items() if isinstance(v, type)}
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)
        self._serialize.client_side_validation = False
        self.capabilities = CapabilitiesOperations(self._client, self._config, self._serialize, self._deserialize)
        self.capability_types = CapabilityTypesOperations(self._client, self._config, self._serialize, self._deserialize)
        self.experiments = ExperimentsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.operations = Operations(self._client, self._config, self._serialize, self._deserialize)
        self.target_types = TargetTypesOperations(self._client, self._config, self._serialize, self._deserialize)
        self.targets = TargetsOperations(self._client, self._config, self._serialize, self._deserialize)

    def _send_request(self, request: HttpRequest, **kwargs: Any) -> Awaitable[AsyncHttpResponse]:
        if False:
            while True:
                i = 10
        'Runs the network request through the client\'s chained policies.\n\n        >>> from azure.core.rest import HttpRequest\n        >>> request = HttpRequest("GET", "https://www.example.org/")\n        <HttpRequest [GET], url: \'https://www.example.org/\'>\n        >>> response = await client._send_request(request)\n        <AsyncHttpResponse: 200 OK>\n\n        For more information on this code flow, see https://aka.ms/azsdk/dpcodegen/python/send_request\n\n        :param request: The network request you want to make. Required.\n        :type request: ~azure.core.rest.HttpRequest\n        :keyword bool stream: Whether the response payload will be streamed. Defaults to False.\n        :return: The response of your network call. Does not do error handling on your response.\n        :rtype: ~azure.core.rest.AsyncHttpResponse\n        '
        request_copy = deepcopy(request)
        request_copy.url = self._client.format_url(request_copy.url)
        return self._client.send_request(request_copy, **kwargs)

    async def close(self) -> None:
        await self._client.close()

    async def __aenter__(self) -> 'ChaosManagementClient':
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc_details: Any) -> None:
        await self._client.__aexit__(*exc_details)