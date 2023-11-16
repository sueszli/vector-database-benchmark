from copy import deepcopy
from typing import Any, TYPE_CHECKING
from azure.core.rest import HttpRequest, HttpResponse
from azure.mgmt.core import ARMPipelineClient
from . import models as _models
from .._serialization import Deserializer, Serializer
from ._configuration import AuthorizationManagementClientConfiguration
from .operations import PermissionsOperations, RoleDefinitionsOperations
if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

class AuthorizationManagementClient:
    """Role based access control provides you a way to apply granular level policy administration down
    to individual resources or resource groups. These operations allow you to manage role
    definitions. A role definition describes the set of actions that can be performed on resources.

    :ivar permissions: PermissionsOperations operations
    :vartype permissions:
     azure.mgmt.authorization.v2022_05_01_preview.operations.PermissionsOperations
    :ivar role_definitions: RoleDefinitionsOperations operations
    :vartype role_definitions:
     azure.mgmt.authorization.v2022_05_01_preview.operations.RoleDefinitionsOperations
    :param credential: Credential needed for the client to connect to Azure. Required.
    :type credential: ~azure.core.credentials.TokenCredential
    :param subscription_id: The ID of the target subscription. Required.
    :type subscription_id: str
    :param base_url: Service URL. Default value is "https://management.azure.com".
    :type base_url: str
    :keyword api_version: Api Version. Default value is "2022-05-01-preview". Note that overriding
     this default value may result in unsupported behavior.
    :paramtype api_version: str
    """

    def __init__(self, credential: 'TokenCredential', subscription_id: str, base_url: str='https://management.azure.com', **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        self._config = AuthorizationManagementClientConfiguration(credential=credential, subscription_id=subscription_id, **kwargs)
        self._client: ARMPipelineClient = ARMPipelineClient(base_url=base_url, config=self._config, **kwargs)
        client_models = {k: v for (k, v) in _models.__dict__.items() if isinstance(v, type)}
        self._serialize = Serializer(client_models)
        self._deserialize = Deserializer(client_models)
        self._serialize.client_side_validation = False
        self.permissions = PermissionsOperations(self._client, self._config, self._serialize, self._deserialize, '2022-05-01-preview')
        self.role_definitions = RoleDefinitionsOperations(self._client, self._config, self._serialize, self._deserialize, '2022-05-01-preview')

    def _send_request(self, request: HttpRequest, **kwargs: Any) -> HttpResponse:
        if False:
            return 10
        'Runs the network request through the client\'s chained policies.\n\n        >>> from azure.core.rest import HttpRequest\n        >>> request = HttpRequest("GET", "https://www.example.org/")\n        <HttpRequest [GET], url: \'https://www.example.org/\'>\n        >>> response = client._send_request(request)\n        <HttpResponse: 200 OK>\n\n        For more information on this code flow, see https://aka.ms/azsdk/dpcodegen/python/send_request\n\n        :param request: The network request you want to make. Required.\n        :type request: ~azure.core.rest.HttpRequest\n        :keyword bool stream: Whether the response payload will be streamed. Defaults to False.\n        :return: The response of your network call. Does not do error handling on your response.\n        :rtype: ~azure.core.rest.HttpResponse\n        '
        request_copy = deepcopy(request)
        request_copy.url = self._client.format_url(request_copy.url)
        return self._client.send_request(request_copy, **kwargs)

    def close(self) -> None:
        if False:
            while True:
                i = 10
        self._client.close()

    def __enter__(self) -> 'AuthorizationManagementClient':
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