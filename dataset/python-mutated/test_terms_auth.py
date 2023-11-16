from unittest.mock import Mock
from twisted.internet.interfaces import IReactorTime
from twisted.test.proto_helpers import MemoryReactor, MemoryReactorClock
from synapse.rest.client.register import register_servlets
from synapse.server import HomeServer
from synapse.types import JsonDict
from synapse.util import Clock
from tests import unittest

class TermsTestCase(unittest.HomeserverTestCase):
    servlets = [register_servlets]

    def default_config(self) -> JsonDict:
        if False:
            return 10
        config = super().default_config()
        config.update({'public_baseurl': 'https://example.org/', 'user_consent': {'version': '1.0', 'policy_name': 'My Cool Privacy Policy', 'template_dir': '/', 'require_at_registration': True}})
        return config

    def prepare(self, reactor: MemoryReactor, clock: Clock, homeserver: HomeServer) -> None:
        if False:
            print('Hello World!')
        self.clock: IReactorTime = MemoryReactorClock()
        self.hs_clock = Clock(self.clock)
        self.url = '/_matrix/client/r0/register'
        self.registration_handler = Mock()
        self.auth_handler = Mock()
        self.device_handler = Mock()

    def test_ui_auth(self) -> None:
        if False:
            print('Hello World!')
        request_data: JsonDict = {'username': 'kermit', 'password': 'monkey'}
        channel = self.make_request(b'POST', self.url, request_data)
        self.assertEqual(channel.code, 401, channel.result)
        self.assertTrue(channel.json_body is not None)
        self.assertIsInstance(channel.json_body['session'], str)
        self.assertIsInstance(channel.json_body['flows'], list)
        for flow in channel.json_body['flows']:
            self.assertIsInstance(flow['stages'], list)
            self.assertTrue(len(flow['stages']) > 0)
            self.assertTrue('m.login.terms' in flow['stages'])
        expected_params = {'m.login.terms': {'policies': {'privacy_policy': {'en': {'name': 'My Cool Privacy Policy', 'url': 'https://example.org/_matrix/consent?v=1.0'}, 'version': '1.0'}}}}
        self.assertIsInstance(channel.json_body['params'], dict)
        self.assertLessEqual(channel.json_body['params'].items(), expected_params.items())
        request_data = {'username': 'kermit', 'password': 'monkey', 'auth': {'session': channel.json_body['session'], 'type': 'm.login.dummy'}}
        self.registration_handler.check_username = Mock(return_value=True)
        channel = self.make_request(b'POST', self.url, request_data)
        self.assertEqual(channel.code, 401, channel.result)
        request_data = {'username': 'kermit', 'password': 'monkey', 'auth': {'session': channel.json_body['session'], 'type': 'm.login.terms'}}
        channel = self.make_request(b'POST', self.url, request_data)
        self.assertEqual(channel.code, 200, channel.result)
        self.assertTrue(channel.json_body is not None)
        self.assertIsInstance(channel.json_body['user_id'], str)
        self.assertIsInstance(channel.json_body['access_token'], str)
        self.assertIsInstance(channel.json_body['device_id'], str)