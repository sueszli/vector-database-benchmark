from typing import TYPE_CHECKING, Union
from azure.core.tracing.decorator import distributed_trace
from ._generated._communication_network_traversal_client import CommunicationNetworkTraversalClient as CommunicationNetworkTraversalClientGen
from ._shared.auth_policy_utils import get_authentication_policy
from ._shared.utils import parse_connection_str
from ._version import SDK_MONIKER
from ._generated.models import CommunicationRelayConfiguration
from ._api_versions import DEFAULT_VERSION
if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential, AzureKeyCredential
    from azure.communication.identity import CommunicationUserIdentifier
    from azure.communication.networktraversal import RouteType

class CommunicationRelayClient(object):
    """Azure Communication Services Relay client.

    :param str endpoint:
        The endpoint url for Azure Communication Service resource.
    :param Union[TokenCredential, AzureKeyCredential] credential:
        The credential we use to authenticate against the service.
    :keyword api_version: Azure Communication Network Traversal API version.
        Default value is "2022-03-01-preview".
        Note that overriding this default value may result in unsupported behavior.
    :paramtype api_version: str
    .. admonition:: Example:

        .. literalinclude:: ../samples/network_traversal_samples.py
            :language: python
            :dedent: 8
    """

    def __init__(self, endpoint, credential, **kwargs):
        if False:
            print('Hello World!')
        try:
            if not endpoint.lower().startswith('http'):
                endpoint = 'https://' + endpoint
        except AttributeError:
            raise ValueError('Account URL must be a string.')
        if not credential:
            raise ValueError('You need to provide account shared key to authenticate.')
        self._endpoint = endpoint
        self._api_version = kwargs.pop('api_version', DEFAULT_VERSION)
        self._network_traversal_service_client = CommunicationNetworkTraversalClientGen(self._endpoint, api_version=self._api_version, authentication_policy=get_authentication_policy(endpoint, credential), sdk_moniker=SDK_MONIKER, **kwargs)

    @classmethod
    def from_connection_string(cls, conn_str, **kwargs):
        if False:
            return 10
        'Create CommunicationRelayClient from a Connection String.\n\n        :param str conn_str: A connection string to an Azure Communication Service resource.\n        :returns: Instance of CommunicationRelayClient.\n        :rtype: ~azure.communication.networktraversal.CommunicationRelayClient\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/network_traversal_samples_async.py\n                :start-after: [START auth_from_connection_string]\n                :end-before: [END auth_from_connection_string]\n                :language: python\n                :dedent: 8\n                :caption: Creating the CommunicationRelayClient from a connection string.\n        '
        (endpoint, access_key) = parse_connection_str(conn_str)
        return cls(endpoint, access_key, **kwargs)

    @distributed_trace
    def get_relay_configuration(self, *, user=None, route_type=None, ttl=172800, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'get a Communication Relay configuration\n        :keyword user: Azure Communication User\n        :paramtype user: ~azure.communication.identity.CommunicationUserIdentifier\n        :keyword route_type: Azure Communication Route Type\n        :paramtype route_type: ~azure.communication.networktraversal.RouteType\n        :return: CommunicationRelayConfiguration\n        :rtype: ~azure.communication.networktraversal.models.CommunicationRelayConfiguration\n        '
        if user is None:
            return self._network_traversal_service_client.communication_network_traversal.issue_relay_configuration(route_type=route_type, ttl=ttl, **kwargs)
        return self._network_traversal_service_client.communication_network_traversal.issue_relay_configuration(id=user.properties['id'], route_type=route_type, ttl=ttl, **kwargs)