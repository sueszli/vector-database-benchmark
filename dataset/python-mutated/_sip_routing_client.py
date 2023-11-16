from typing import TYPE_CHECKING
from urllib.parse import urlparse
from azure.core.tracing.decorator import distributed_trace
from azure.core.paging import ItemPaged
from ._models import SipTrunk, SipTrunkRoute
from ._generated.models import SipConfiguration, SipTrunkInternal, SipTrunkRouteInternal
from ._generated._client import SIPRoutingService
from .._shared.auth_policy_utils import get_authentication_policy
from .._shared.utils import parse_connection_str
from .._version import SDK_MONIKER
if TYPE_CHECKING:
    from typing import Optional, Iterable, List, Any
    from azure.core.credentials import TokenCredential

class SipRoutingClient(object):
    """A client to interact with the AzureCommunicationService SIP routing gateway.
    This client provides operations to retrieve and manage SIP routing configuration.

    :param endpoint: The endpoint url for Azure Communication Service resource.
    :type endpoint: str
    :param credential: The credentials with which to authenticate.
    :type credential: TokenCredential
    :keyword api_version: Api Version. Default value is "2021-05-01-preview". Note that overriding
     this default value may result in unsupported behavior.
    :paramtype api_version: str
    """

    def __init__(self, endpoint, credential, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if not credential:
            raise ValueError('credential can not be None')
        try:
            if not endpoint.lower().startswith('http'):
                endpoint = 'https://' + endpoint
        except AttributeError:
            raise ValueError('Host URL must be a string')
        parsed_url = urlparse(endpoint.rstrip('/'))
        if not parsed_url.netloc:
            raise ValueError('Invalid URL: {}'.format(endpoint))
        self._endpoint = endpoint
        self._authentication_policy = get_authentication_policy(endpoint, credential)
        self._rest_service = SIPRoutingService(self._endpoint, authentication_policy=self._authentication_policy, sdk_moniker=SDK_MONIKER, **kwargs)

    @classmethod
    def from_connection_string(cls, conn_str, **kwargs):
        if False:
            i = 10
            return i + 15
        'Factory method for creating client from connection string.\n\n        :param str conn_str: Connection string containing endpoint and credentials.\n        :returns: The newly created client.\n        :rtype: ~azure.communication.siprouting.SipRoutingClient\n        '
        (endpoint, credential) = parse_connection_str(conn_str)
        return cls(endpoint, credential, **kwargs)

    @distributed_trace
    def get_trunk(self, trunk_fqdn, **kwargs):
        if False:
            while True:
                i = 10
        "Retrieve a single SIP trunk.\n\n        :param trunk_fqdn: FQDN of the desired SIP trunk.\n        :type trunk_fqdn: str\n        :returns: SIP trunk with specified trunk_fqdn. If it doesn't exist, throws KeyError.\n        :rtype: ~azure.communication.siprouting.models.SipTrunk\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError, KeyError\n        "
        if trunk_fqdn is None:
            raise ValueError("Parameter 'trunk_fqdn' must not be None.")
        config = self._rest_service.sip_routing.get(**kwargs)
        trunk = config.trunks[trunk_fqdn]
        return SipTrunk(fqdn=trunk_fqdn, sip_signaling_port=trunk.sip_signaling_port)

    @distributed_trace
    def set_trunk(self, trunk, **kwargs):
        if False:
            i = 10
            return i + 15
        "Modifies SIP trunk with the given FQDN. If it doesn't exist, adds a new trunk.\n\n        :param trunk: Trunk object to be set.\n        :type trunk: ~azure.communication.siprouting.models.SipTrunk\n        :returns: None\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n        "
        if trunk is None:
            raise ValueError("Parameter 'trunk' must not be None.")
        self._update_trunks_([trunk], **kwargs)

    @distributed_trace
    def delete_trunk(self, trunk_fqdn, **kwargs):
        if False:
            while True:
                i = 10
        'Deletes SIP trunk.\n\n        :param trunk_fqdn: FQDN of the trunk to be deleted.\n        :type trunk_fqdn: str\n        :returns: None\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n        '
        if trunk_fqdn is None:
            raise ValueError("Parameter 'trunk_fqdn' must not be None.")
        self._rest_service.sip_routing.update(body=SipConfiguration(trunks={trunk_fqdn: None}), **kwargs)

    @distributed_trace
    def list_trunks(self, **kwargs):
        if False:
            print('Hello World!')
        'Retrieves the currently configured SIP trunks.\n\n        :returns: Current SIP trunks configuration.\n        :rtype: ItemPaged[~azure.communication.siprouting.models.SipTrunk]\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '

        def extract_data(config):
            if False:
                return 10
            list_of_elem = [SipTrunk(fqdn=k, sip_signaling_port=v.sip_signaling_port) for (k, v) in config.trunks.items()]
            return (None, list_of_elem)

        def get_next(nextLink=None):
            if False:
                return 10
            return self._rest_service.sip_routing.get(**kwargs)
        return ItemPaged(get_next, extract_data)

    @distributed_trace
    def list_routes(self, **kwargs):
        if False:
            print('Hello World!')
        'Retrieves the currently configured SIP routes.\n\n        :returns: Current SIP routes configuration.\n        :rtype: ItemPaged[~azure.communication.siprouting.models.SipTrunkRoute]\n        :raises: ~azure.core.exceptions.HttpResponseError\n        '

        def extract_data(config):
            if False:
                for i in range(10):
                    print('nop')
            list_of_elem = [SipTrunkRoute(description=x.description, name=x.name, number_pattern=x.number_pattern, trunks=x.trunks) for x in config.routes]
            return (None, list_of_elem)

        def get_next(nextLink=None):
            if False:
                print('Hello World!')
            return self._rest_service.sip_routing.get(**kwargs)
        return ItemPaged(get_next, extract_data)

    @distributed_trace
    def set_trunks(self, trunks, **kwargs):
        if False:
            i = 10
            return i + 15
        'Overwrites the list of SIP trunks.\n\n        :param trunks: New list of trunks to be set.\n        :type trunks: List[SipTrunk]\n        :returns: None\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n        '
        if trunks is None:
            raise ValueError("Parameter 'trunks' must not be None.")
        trunks_dictionary = {x.fqdn: SipTrunkInternal(sip_signaling_port=x.sip_signaling_port) for x in trunks}
        config = SipConfiguration(trunks=trunks_dictionary)
        old_trunks = self._list_trunks_(**kwargs)
        for x in old_trunks:
            if x.fqdn not in [o.fqdn for o in trunks]:
                config.trunks[x.fqdn] = None
        if len(config.trunks) > 0:
            self._rest_service.sip_routing.update(body=config, **kwargs)

    @distributed_trace
    def set_routes(self, routes, **kwargs):
        if False:
            i = 10
            return i + 15
        'Overwrites the list of SIP routes.\n\n        :param routes: New list of routes to be set.\n        :type routes: List[SipTrunkRoute]\n        :returns: None\n        :rtype: None\n        :raises: ~azure.core.exceptions.HttpResponseError, ValueError\n        '
        if routes is None:
            raise ValueError("Parameter 'routes' must not be None.")
        routes_internal = [SipTrunkRouteInternal(description=x.description, name=x.name, number_pattern=x.number_pattern, trunks=x.trunks) for x in routes]
        self._rest_service.sip_routing.update(body=SipConfiguration(routes=routes_internal), **kwargs)

    def _list_trunks_(self, **kwargs):
        if False:
            while True:
                i = 10
        config = self._rest_service.sip_routing.get(**kwargs)
        return [SipTrunk(fqdn=k, sip_signaling_port=v.sip_signaling_port) for (k, v) in config.trunks.items()]

    def _update_trunks_(self, trunks, **kwargs):
        if False:
            while True:
                i = 10
        trunks_internal = {x.fqdn: SipTrunkInternal(sip_signaling_port=x.sip_signaling_port) for x in trunks}
        modified_config = SipConfiguration(trunks=trunks_internal)
        new_config = self._rest_service.sip_routing.update(body=modified_config, **kwargs)
        return [SipTrunk(fqdn=k, sip_signaling_port=v.sip_signaling_port) for (k, v) in new_config.trunks.items()]

    def close(self) -> None:
        if False:
            while True:
                i = 10
        self._rest_service.close()

    def __enter__(self) -> 'SipRoutingClient':
        if False:
            return 10
        self._rest_service.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._rest_service.__exit__(*args)