from copy import deepcopy
from typing import Any, TYPE_CHECKING
from azure.core.rest import HttpRequest, HttpResponse
from azure.mgmt.core import ARMPipelineClient
from . import models as _models
from ._configuration import AzureDatabricksManagementClientConfiguration
from ._serialization import Deserializer, Serializer
from .operations import AccessConnectorsOperations, Operations, OutboundNetworkDependenciesEndpointsOperations, PrivateEndpointConnectionsOperations, PrivateLinkResourcesOperations, VNetPeeringOperations, WorkspacesOperations
if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

class AzureDatabricksManagementClient:
    """The Microsoft Azure management APIs allow end users to operate on Azure Databricks Workspace /
    Access Connector resources.

    :ivar workspaces: WorkspacesOperations operations
    :vartype workspaces: azure.mgmt.databricks.operations.WorkspacesOperations
    :ivar operations: Operations operations
    :vartype operations: azure.mgmt.databricks.operations.Operations
    :ivar private_link_resources: PrivateLinkResourcesOperations operations
    :vartype private_link_resources:
     azure.mgmt.databricks.operations.PrivateLinkResourcesOperations
    :ivar private_endpoint_connections: PrivateEndpointConnectionsOperations operations
    :vartype private_endpoint_connections:
     azure.mgmt.databricks.operations.PrivateEndpointConnectionsOperations
    :ivar outbound_network_dependencies_endpoints: OutboundNetworkDependenciesEndpointsOperations
     operations
    :vartype outbound_network_dependencies_endpoints:
     azure.mgmt.databricks.operations.OutboundNetworkDependenciesEndpointsOperations
    :ivar vnet_peering: VNetPeeringOperations operations
    :vartype vnet_peering: azure.mgmt.databricks.operations.VNetPeeringOperations
    :ivar access_connectors: AccessConnectorsOperations operations
    :vartype access_connectors: azure.mgmt.databricks.operations.AccessConnectorsOperations
    :param credential: Credential needed for the client to connect to Azure. Required.
    :type credential: ~azure.core.credentials.TokenCredential
    :param subscription_id: The ID of the target subscription. Required.
    :type subscription_id: str
    :param base_url: Service URL. Default value is "https://management.azure.com".
    :type base_url: str
    :keyword int polling_interval: Default waiting time between two polls for LRO operations if no
     Retry-After header is present.
    """

    def __init__(self, credential: 'TokenCredential', subscription_id: str, base_url: str='https://management.azure.com', **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._config = AzureDatabricksManagementClientConfiguration(credential=credential, subscription_id=subscription_id, **kwargs)
        self._client: ARMPipelineClient = ARMPipelineClient(base_url=base_url, config=self._config, **kwargs)
        client_models = {k: v for (k, v) in _models.__dict__.items() if isinstance(v, type)}
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)
        self._serialize.client_side_validation = False
        self.workspaces = WorkspacesOperations(self._client, self._config, self._serialize, self._deserialize)
        self.operations = Operations(self._client, self._config, self._serialize, self._deserialize)
        self.private_link_resources = PrivateLinkResourcesOperations(self._client, self._config, self._serialize, self._deserialize)
        self.private_endpoint_connections = PrivateEndpointConnectionsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.outbound_network_dependencies_endpoints = OutboundNetworkDependenciesEndpointsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.vnet_peering = VNetPeeringOperations(self._client, self._config, self._serialize, self._deserialize)
        self.access_connectors = AccessConnectorsOperations(self._client, self._config, self._serialize, self._deserialize)

    def _send_request(self, request: HttpRequest, **kwargs: Any) -> HttpResponse:
        if False:
            while True:
                i = 10
        'Runs the network request through the client\'s chained policies.\n\n        >>> from azure.core.rest import HttpRequest\n        >>> request = HttpRequest("GET", "https://www.example.org/")\n        <HttpRequest [GET], url: \'https://www.example.org/\'>\n        >>> response = client._send_request(request)\n        <HttpResponse: 200 OK>\n\n        For more information on this code flow, see https://aka.ms/azsdk/dpcodegen/python/send_request\n\n        :param request: The network request you want to make. Required.\n        :type request: ~azure.core.rest.HttpRequest\n        :keyword bool stream: Whether the response payload will be streamed. Defaults to False.\n        :return: The response of your network call. Does not do error handling on your response.\n        :rtype: ~azure.core.rest.HttpResponse\n        '
        request_copy = deepcopy(request)
        request_copy.url = self._client.format_url(request_copy.url)
        return self._client.send_request(request_copy, **kwargs)

    def close(self) -> None:
        if False:
            return 10
        self._client.close()

    def __enter__(self) -> 'AzureDatabricksManagementClient':
        if False:
            i = 10
            return i + 15
        self._client.__enter__()
        return self

    def __exit__(self, *exc_details: Any) -> None:
        if False:
            while True:
                i = 10
        self._client.__exit__(*exc_details)