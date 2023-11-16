from typing import List
from unittest.mock import Mock, patch
from parameterized import parameterized
from twisted.test.proto_helpers import MemoryReactor
from synapse.app.generic_worker import GenericWorkerServer
from synapse.app.homeserver import SynapseHomeServer
from synapse.config.server import parse_listener_def
from synapse.server import HomeServer
from synapse.types import JsonDict
from synapse.util import Clock
from tests.server import make_request
from tests.unittest import HomeserverTestCase

class FederationReaderOpenIDListenerTests(HomeserverTestCase):

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            for i in range(10):
                print('nop')
        hs = self.setup_test_homeserver(homeserver_to_use=GenericWorkerServer)
        return hs

    def default_config(self) -> JsonDict:
        if False:
            i = 10
            return i + 15
        conf = super().default_config()
        conf['worker_app'] = 'yes'
        conf['instance_map'] = {'main': {'host': '127.0.0.1', 'port': 0}}
        return conf

    @parameterized.expand([(['federation'], 'auth_fail'), ([], 'no_resource'), (['openid', 'federation'], 'auth_fail'), (['openid'], 'auth_fail')])
    def test_openid_listener(self, names: List[str], expectation: str) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test different openid listener configurations.\n\n        401 is success here since it means we hit the handler and auth failed.\n        '
        config = {'port': 8080, 'type': 'http', 'bind_addresses': ['0.0.0.0'], 'resources': [{'names': names}]}
        hs = self.hs
        assert isinstance(hs, GenericWorkerServer)
        hs._listen_http(parse_listener_def(0, config))
        site = self.reactor.tcpServers[0][1]
        try:
            site.resource.children[b'_matrix'].children[b'federation']
        except KeyError:
            if expectation == 'no_resource':
                return
            raise
        channel = make_request(self.reactor, site, 'GET', '/_matrix/federation/v1/openid/userinfo')
        self.assertEqual(channel.code, 401)

@patch('synapse.app.homeserver.KeyResource', new=Mock())
class SynapseHomeserverOpenIDListenerTests(HomeserverTestCase):

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            return 10
        hs = self.setup_test_homeserver(homeserver_to_use=SynapseHomeServer)
        return hs

    @parameterized.expand([(['federation'], 'auth_fail'), ([], 'no_resource'), (['openid', 'federation'], 'auth_fail'), (['openid'], 'auth_fail')])
    def test_openid_listener(self, names: List[str], expectation: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test different openid listener configurations.\n\n        401 is success here since it means we hit the handler and auth failed.\n        '
        config = {'port': 8080, 'type': 'http', 'bind_addresses': ['0.0.0.0'], 'resources': [{'names': names}]}
        hs = self.hs
        assert isinstance(hs, SynapseHomeServer)
        hs._listener_http(self.hs.config, parse_listener_def(0, config))
        site = self.reactor.tcpServers[0][1]
        try:
            site.resource.children[b'_matrix'].children[b'federation']
        except KeyError:
            if expectation == 'no_resource':
                return
            raise
        channel = make_request(self.reactor, site, 'GET', '/_matrix/federation/v1/openid/userinfo')
        self.assertEqual(channel.code, 401)