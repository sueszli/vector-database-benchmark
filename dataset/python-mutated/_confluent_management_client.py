from copy import deepcopy
from typing import Any, TYPE_CHECKING
from azure.core.rest import HttpRequest, HttpResponse
from azure.mgmt.core import ARMPipelineClient
from . import models
from ._configuration import ConfluentManagementClientConfiguration
from ._serialization import Deserializer, Serializer
from .operations import MarketplaceAgreementsOperations, OrganizationOperations, OrganizationOperationsOperations, ValidationsOperations
if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

class ConfluentManagementClient:
    """ConfluentManagementClient.

    :ivar marketplace_agreements: MarketplaceAgreementsOperations operations
    :vartype marketplace_agreements:
     azure.mgmt.confluent.operations.MarketplaceAgreementsOperations
    :ivar organization_operations: OrganizationOperationsOperations operations
    :vartype organization_operations:
     azure.mgmt.confluent.operations.OrganizationOperationsOperations
    :ivar organization: OrganizationOperations operations
    :vartype organization: azure.mgmt.confluent.operations.OrganizationOperations
    :ivar validations: ValidationsOperations operations
    :vartype validations: azure.mgmt.confluent.operations.ValidationsOperations
    :param credential: Credential needed for the client to connect to Azure. Required.
    :type credential: ~azure.core.credentials.TokenCredential
    :param subscription_id: Microsoft Azure subscription id. Required.
    :type subscription_id: str
    :param base_url: Service URL. Default value is "https://management.azure.com".
    :type base_url: str
    :keyword api_version: Api Version. Default value is "2021-12-01". Note that overriding this
     default value may result in unsupported behavior.
    :paramtype api_version: str
    :keyword int polling_interval: Default waiting time between two polls for LRO operations if no
     Retry-After header is present.
    """

    def __init__(self, credential: 'TokenCredential', subscription_id: str, base_url: str='https://management.azure.com', **kwargs: Any) -> None:
        if False:
            return 10
        self._config = ConfluentManagementClientConfiguration(credential=credential, subscription_id=subscription_id, **kwargs)
        self._client = ARMPipelineClient(base_url=base_url, config=self._config, **kwargs)
        client_models = {k: v for (k, v) in models.__dict__.items() if isinstance(v, type)}
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)
        self._serialize.client_side_validation = False
        self.marketplace_agreements = MarketplaceAgreementsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.organization_operations = OrganizationOperationsOperations(self._client, self._config, self._serialize, self._deserialize)
        self.organization = OrganizationOperations(self._client, self._config, self._serialize, self._deserialize)
        self.validations = ValidationsOperations(self._client, self._config, self._serialize, self._deserialize)

    def _send_request(self, request: HttpRequest, **kwargs: Any) -> HttpResponse:
        if False:
            print('Hello World!')
        'Runs the network request through the client\'s chained policies.\n\n        >>> from azure.core.rest import HttpRequest\n        >>> request = HttpRequest("GET", "https://www.example.org/")\n        <HttpRequest [GET], url: \'https://www.example.org/\'>\n        >>> response = client._send_request(request)\n        <HttpResponse: 200 OK>\n\n        For more information on this code flow, see https://aka.ms/azsdk/dpcodegen/python/send_request\n\n        :param request: The network request you want to make. Required.\n        :type request: ~azure.core.rest.HttpRequest\n        :keyword bool stream: Whether the response payload will be streamed. Defaults to False.\n        :return: The response of your network call. Does not do error handling on your response.\n        :rtype: ~azure.core.rest.HttpResponse\n        '
        request_copy = deepcopy(request)
        request_copy.url = self._client.format_url(request_copy.url)
        return self._client.send_request(request_copy, **kwargs)

    def close(self):
        if False:
            print('Hello World!')
        self._client.close()

    def __enter__(self):
        if False:
            print('Hello World!')
        self._client.__enter__()
        return self

    def __exit__(self, *exc_details):
        if False:
            i = 10
            return i + 15
        self._client.__exit__(*exc_details)