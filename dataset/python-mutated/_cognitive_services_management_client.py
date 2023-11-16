from copy import deepcopy
from typing import Any, TYPE_CHECKING
from azure.core.rest import HttpRequest, HttpResponse
from azure.mgmt.core import ARMPipelineClient
from . import models as _models
from ._configuration import CognitiveServicesManagementClientConfiguration
from ._serialization import Deserializer, Serializer
from .operations import AccountsOperations, CognitiveServicesManagementClientOperationsMixin, CommitmentPlansOperations, CommitmentTiersOperations, DeletedAccountsOperations, DeploymentsOperations, ModelsOperations, Operations, PrivateEndpointConnectionsOperations, PrivateLinkResourcesOperations, ResourceSkusOperations, UsagesOperations
if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

class CognitiveServicesManagementClient(CognitiveServicesManagementClientOperationsMixin):
    """Cognitive Services Management Client.

    :ivar accounts: AccountsOperations operations
    :vartype accounts: azure.mgmt.cognitiveservices.operations.AccountsOperations
    :ivar deleted_accounts: DeletedAccountsOperations operations
    :vartype deleted_accounts: azure.mgmt.cognitiveservices.operations.DeletedAccountsOperations
    :ivar resource_skus: ResourceSkusOperations operations
    :vartype resource_skus: azure.mgmt.cognitiveservices.operations.ResourceSkusOperations
    :ivar usages: UsagesOperations operations
    :vartype usages: azure.mgmt.cognitiveservices.operations.UsagesOperations
    :ivar operations: Operations operations
    :vartype operations: azure.mgmt.cognitiveservices.operations.Operations
    :ivar commitment_tiers: CommitmentTiersOperations operations
    :vartype commitment_tiers: azure.mgmt.cognitiveservices.operations.CommitmentTiersOperations
    :ivar models: ModelsOperations operations
    :vartype models: azure.mgmt.cognitiveservices.operations.ModelsOperations
    :ivar private_endpoint_connections: PrivateEndpointConnectionsOperations operations
    :vartype private_endpoint_connections:
     azure.mgmt.cognitiveservices.operations.PrivateEndpointConnectionsOperations
    :ivar private_link_resources: PrivateLinkResourcesOperations operations
    :vartype private_link_resources:
     azure.mgmt.cognitiveservices.operations.PrivateLinkResourcesOperations
    :ivar deployments: DeploymentsOperations operations
    :vartype deployments: azure.mgmt.cognitiveservices.operations.DeploymentsOperations
    :ivar commitment_plans: CommitmentPlansOperations operations
    :vartype commitment_plans: azure.mgmt.cognitiveservices.operations.CommitmentPlansOperations
    :param credential: Credential needed for the client to connect to Azure. Required.
    :type credential: ~azure.core.credentials.TokenCredential
    :param subscription_id: The ID of the target subscription. Required.
    :type subscription_id: str
    :param base_url: Service URL. Default value is "https://management.azure.com".
    :type base_url: str
    :keyword api_version: Api Version. Default value is "2023-05-01". Note that overriding this
     default value may result in unsupported behavior.
    :paramtype api_version: str
    :keyword int polling_interval: Default waiting time between two polls for LRO operations if no
     Retry-After header is present.
    """

    def __init__(self, credential: 'TokenCredential', subscription_id: str, base_url: str='https://management.azure.com', **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        self._config = CognitiveServicesManagementClientConfiguration(credential=credential, subscription_id=subscription_id, **kwargs)
        self._client: ARMPipelineClient = ARMPipelineClient(base_url=base_url, config=self._config, **kwargs)
        client_models = {k: v for (k, v) in _models.__dict__.items() if isinstance(v, type)}
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)
        self._serialize.client_side_validation = False
        self.accounts = AccountsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.deleted_accounts = DeletedAccountsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.resource_skus = ResourceSkusOperations(self._client, self._config, self._serialize, self._deserialize)
        self.usages = UsagesOperations(self._client, self._config, self._serialize, self._deserialize)
        self.operations = Operations(self._client, self._config, self._serialize, self._deserialize)
        self.commitment_tiers = CommitmentTiersOperations(self._client, self._config, self._serialize, self._deserialize)
        self.models = ModelsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.private_endpoint_connections = PrivateEndpointConnectionsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.private_link_resources = PrivateLinkResourcesOperations(self._client, self._config, self._serialize, self._deserialize)
        self.deployments = DeploymentsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.commitment_plans = CommitmentPlansOperations(self._client, self._config, self._serialize, self._deserialize)

    def _send_request(self, request: HttpRequest, **kwargs: Any) -> HttpResponse:
        if False:
            i = 10
            return i + 15
        'Runs the network request through the client\'s chained policies.\n\n        >>> from azure.core.rest import HttpRequest\n        >>> request = HttpRequest("GET", "https://www.example.org/")\n        <HttpRequest [GET], url: \'https://www.example.org/\'>\n        >>> response = client._send_request(request)\n        <HttpResponse: 200 OK>\n\n        For more information on this code flow, see https://aka.ms/azsdk/dpcodegen/python/send_request\n\n        :param request: The network request you want to make. Required.\n        :type request: ~azure.core.rest.HttpRequest\n        :keyword bool stream: Whether the response payload will be streamed. Defaults to False.\n        :return: The response of your network call. Does not do error handling on your response.\n        :rtype: ~azure.core.rest.HttpResponse\n        '
        request_copy = deepcopy(request)
        request_copy.url = self._client.format_url(request_copy.url)
        return self._client.send_request(request_copy, **kwargs)

    def close(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._client.close()

    def __enter__(self) -> 'CognitiveServicesManagementClient':
        if False:
            i = 10
            return i + 15
        self._client.__enter__()
        return self

    def __exit__(self, *exc_details: Any) -> None:
        if False:
            i = 10
            return i + 15
        self._client.__exit__(*exc_details)