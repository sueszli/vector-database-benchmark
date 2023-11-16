from typing import TYPE_CHECKING, Any, Tuple, Union
from azure.core.tracing.decorator import distributed_trace
from azure.core.credentials import AccessToken
from ._generated._client import CommunicationIdentityClient as CommunicationIdentityClientGen
from ._shared.auth_policy_utils import get_authentication_policy
from ._shared.utils import parse_connection_str
from ._shared.models import CommunicationUserIdentifier
from ._version import SDK_MONIKER
from ._api_versions import DEFAULT_VERSION
from ._utils import convert_timedelta_to_mins
if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential, AzureKeyCredential

class CommunicationIdentityClient(object):
    """Azure Communication Services Identity client.

    :param str endpoint:
        The endpoint url for Azure Communication Service resource.
    :param Union[TokenCredential, AzureKeyCredential] credential:
        The credential we use to authenticate against the service.
    :keyword api_version: Azure Communication Identity API version.
        Default value is "2022-10-01". Note that overriding this default value may result in unsupported behavior.
    :paramtype api_version: str

    .. admonition:: Example:

        .. literalinclude:: ../samples/identity_samples.py
            :language: python
            :dedent: 8
    """

    def __init__(self, endpoint, credential, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        try:
            if not endpoint.lower().startswith('http'):
                endpoint = 'https://' + endpoint
        except AttributeError as err:
            raise ValueError('Account URL must be a string.') from err
        if not credential:
            raise ValueError('You need to provide account shared key to authenticate.')
        self._endpoint = endpoint
        self._api_version = kwargs.pop('api_version', DEFAULT_VERSION)
        self._identity_service_client = CommunicationIdentityClientGen(self._endpoint, api_version=self._api_version, authentication_policy=get_authentication_policy(endpoint, credential), sdk_moniker=SDK_MONIKER, **kwargs)

    @classmethod
    def from_connection_string(cls, conn_str, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Create CommunicationIdentityClient from a Connection String.\n\n        :param str conn_str: A connection string to an Azure Communication Service resource.\n        :returns: Instance of CommunicationIdentityClient.\n        :rtype: ~azure.communication.identity.CommunicationIdentityClient\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/identity_samples.py\n                :start-after: [START auth_from_connection_string]\n                :end-before: [END auth_from_connection_string]\n                :language: python\n                :dedent: 8\n                :caption: Creating the CommunicationIdentityClient from a connection string.\n        '
        (endpoint, access_key) = parse_connection_str(conn_str)
        return cls(endpoint, access_key, **kwargs)

    @distributed_trace
    def create_user(self, **kwargs):
        if False:
            print('Hello World!')
        'create a single Communication user\n\n        :return: CommunicationUserIdentifier\n        :rtype: ~azure.communication.identity.CommunicationUserIdentifier\n        '
        identity_access_token = self._identity_service_client.communication_identity.create(**kwargs)
        return CommunicationUserIdentifier(identity_access_token.identity.id, raw_id=identity_access_token.identity.id)

    @distributed_trace
    def create_user_and_token(self, scopes, **kwargs):
        if False:
            return 10
        'Create a single Communication user with an identity token.\n\n        :param scopes: List of scopes to be added to the token.\n        :type scopes: list[str or ~azure.communication.identity.CommunicationTokenScope]\n        :keyword token_expires_in: Custom validity period of the Communication Identity access token\n         within [1, 24] hours range. If not provided, the default value of 24 hours will be used.\n        :paramtype token_expires_in: ~datetime.timedelta\n        :return: A tuple of a CommunicationUserIdentifier and a AccessToken.\n        :rtype:\n            tuple of (~azure.communication.identity.CommunicationUserIdentifier, ~azure.core.credentials.AccessToken)\n        '
        token_expires_in = kwargs.pop('token_expires_in', None)
        request_body = {'createTokenWithScopes': scopes, 'expiresInMinutes': convert_timedelta_to_mins(token_expires_in)}
        identity_access_token = self._identity_service_client.communication_identity.create(body=request_body, **kwargs)
        user_identifier = CommunicationUserIdentifier(identity_access_token.identity.id, raw_id=identity_access_token.identity.id)
        access_token = AccessToken(identity_access_token.access_token.token, identity_access_token.access_token.expires_on)
        return (user_identifier, access_token)

    @distributed_trace
    def delete_user(self, user, **kwargs):
        if False:
            return 10
        'Triggers revocation event for user and deletes all its data.\n\n        :param user: Azure Communication User to delete\n        :type user: ~azure.communication.identity.CommunicationUserIdentifier\n        :return: None\n        :rtype: None\n        '
        self._identity_service_client.communication_identity.delete(user.properties['id'], **kwargs)

    @distributed_trace
    def get_token(self, user, scopes, **kwargs):
        if False:
            while True:
                i = 10
        'Generates a new token for an identity.\n\n        :param user: Azure Communication User\n        :type user: ~azure.communication.identity.CommunicationUserIdentifier\n        :param scopes: List of scopes to be added to the token.\n        :type scopes: list[str or ~azure.communication.identity.CommunicationTokenScope]\n        :keyword token_expires_in: Custom validity period of the Communication Identity access token\n         within [1, 24] hours range. If not provided, the default value of 24 hours will be used.\n        :paramtype token_expires_in: ~datetime.timedelta\n        :return: AccessToken\n        :rtype: ~azure.core.credentials.AccessToken\n        '
        token_expires_in = kwargs.pop('token_expires_in', None)
        request_body = {'scopes': scopes, 'expiresInMinutes': convert_timedelta_to_mins(token_expires_in)}
        access_token = self._identity_service_client.communication_identity.issue_access_token(user.properties['id'], body=request_body, **kwargs)
        return AccessToken(access_token.token, access_token.expires_on)

    @distributed_trace
    def revoke_tokens(self, user, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Schedule revocation of all tokens of an identity.\n\n        :param user: Azure Communication User.\n        :type user: ~azure.communication.identity.CommunicationUserIdentifier.\n        :return: None\n        :rtype: None\n        '
        return self._identity_service_client.communication_identity.revoke_access_tokens(user.properties['id'] if user else None, **kwargs)

    @distributed_trace
    def get_token_for_teams_user(self, aad_token, client_id, user_object_id, **kwargs):
        if False:
            print('Hello World!')
        'Exchanges an Azure AD access token of a Teams User for a new Communication Identity access token.\n\n        :param aad_token: an Azure AD access token of a Teams User.\n        :type aad_token: str\n        :param client_id: a Client ID of an Azure AD application to be verified against\n            the appId claim in the Azure AD access token.\n        :type client_id: str\n        :param user_object_id: an Object ID of an Azure AD user (Teams User) to be verified against\n            the OID claim in the Azure AD access token.\n        :type user_object_id: str\n        :return: AccessToken\n        :rtype: ~azure.core.credentials.AccessToken\n        '
        request_body = {'token': aad_token, 'appId': client_id, 'userId': user_object_id}
        access_token = self._identity_service_client.communication_identity.exchange_teams_user_access_token(body=request_body, **kwargs)
        return AccessToken(access_token.token, access_token.expires_on)