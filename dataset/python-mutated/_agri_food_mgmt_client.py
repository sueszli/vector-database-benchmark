from copy import deepcopy
from typing import Any, TYPE_CHECKING
from azure.core.rest import HttpRequest, HttpResponse
from azure.mgmt.core import ARMPipelineClient
from . import models as _models
from ._configuration import AgriFoodMgmtClientConfiguration
from ._serialization import Deserializer, Serializer
from .operations import ExtensionsOperations, FarmBeatsExtensionsOperations, FarmBeatsModelsOperations, LocationsOperations, Operations, PrivateEndpointConnectionsOperations, PrivateLinkResourcesOperations, SolutionsDiscoverabilityOperations, SolutionsOperations
if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

class AgriFoodMgmtClient:
    """APIs documentation for Azure AgFoodPlatform Resource Provider Service.

    :ivar extensions: ExtensionsOperations operations
    :vartype extensions: azure.mgmt.agrifood.operations.ExtensionsOperations
    :ivar farm_beats_extensions: FarmBeatsExtensionsOperations operations
    :vartype farm_beats_extensions: azure.mgmt.agrifood.operations.FarmBeatsExtensionsOperations
    :ivar farm_beats_models: FarmBeatsModelsOperations operations
    :vartype farm_beats_models: azure.mgmt.agrifood.operations.FarmBeatsModelsOperations
    :ivar locations: LocationsOperations operations
    :vartype locations: azure.mgmt.agrifood.operations.LocationsOperations
    :ivar operations: Operations operations
    :vartype operations: azure.mgmt.agrifood.operations.Operations
    :ivar private_endpoint_connections: PrivateEndpointConnectionsOperations operations
    :vartype private_endpoint_connections:
     azure.mgmt.agrifood.operations.PrivateEndpointConnectionsOperations
    :ivar private_link_resources: PrivateLinkResourcesOperations operations
    :vartype private_link_resources: azure.mgmt.agrifood.operations.PrivateLinkResourcesOperations
    :ivar solutions: SolutionsOperations operations
    :vartype solutions: azure.mgmt.agrifood.operations.SolutionsOperations
    :ivar solutions_discoverability: SolutionsDiscoverabilityOperations operations
    :vartype solutions_discoverability:
     azure.mgmt.agrifood.operations.SolutionsDiscoverabilityOperations
    :param credential: Credential needed for the client to connect to Azure. Required.
    :type credential: ~azure.core.credentials.TokenCredential
    :param solution_id: Solution Id of the solution. Required.
    :type solution_id: str
    :param subscription_id: The ID of the target subscription. The value must be an UUID. Required.
    :type subscription_id: str
    :param base_url: Service URL. Default value is "https://management.azure.com".
    :type base_url: str
    :keyword api_version: Api Version. Default value is "2021-09-01-preview". Note that overriding
     this default value may result in unsupported behavior.
    :paramtype api_version: str
    :keyword int polling_interval: Default waiting time between two polls for LRO operations if no
     Retry-After header is present.
    """

    def __init__(self, credential: 'TokenCredential', solution_id: str, subscription_id: str, base_url: str='https://management.azure.com', **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._config = AgriFoodMgmtClientConfiguration(credential=credential, solution_id=solution_id, subscription_id=subscription_id, **kwargs)
        self._client = ARMPipelineClient(base_url=base_url, config=self._config, **kwargs)
        client_models = {k: v for (k, v) in _models.__dict__.items() if isinstance(v, type)}
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)
        self._serialize.client_side_validation = False
        self.extensions = ExtensionsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.farm_beats_extensions = FarmBeatsExtensionsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.farm_beats_models = FarmBeatsModelsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.locations = LocationsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.operations = Operations(self._client, self._config, self._serialize, self._deserialize)
        self.private_endpoint_connections = PrivateEndpointConnectionsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.private_link_resources = PrivateLinkResourcesOperations(self._client, self._config, self._serialize, self._deserialize)
        self.solutions = SolutionsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.solutions_discoverability = SolutionsDiscoverabilityOperations(self._client, self._config, self._serialize, self._deserialize)

    def _send_request(self, request: HttpRequest, **kwargs: Any) -> HttpResponse:
        if False:
            return 10
        'Runs the network request through the client\'s chained policies.\n\n        >>> from azure.core.rest import HttpRequest\n        >>> request = HttpRequest("GET", "https://www.example.org/")\n        <HttpRequest [GET], url: \'https://www.example.org/\'>\n        >>> response = client._send_request(request)\n        <HttpResponse: 200 OK>\n\n        For more information on this code flow, see https://aka.ms/azsdk/dpcodegen/python/send_request\n\n        :param request: The network request you want to make. Required.\n        :type request: ~azure.core.rest.HttpRequest\n        :keyword bool stream: Whether the response payload will be streamed. Defaults to False.\n        :return: The response of your network call. Does not do error handling on your response.\n        :rtype: ~azure.core.rest.HttpResponse\n        '
        request_copy = deepcopy(request)
        request_copy.url = self._client.format_url(request_copy.url)
        return self._client.send_request(request_copy, **kwargs)

    def close(self) -> None:
        if False:
            i = 10
            return i + 15
        self._client.close()

    def __enter__(self) -> 'AgriFoodMgmtClient':
        if False:
            while True:
                i = 10
        self._client.__enter__()
        return self

    def __exit__(self, *exc_details) -> None:
        if False:
            print('Hello World!')
        self._client.__exit__(*exc_details)