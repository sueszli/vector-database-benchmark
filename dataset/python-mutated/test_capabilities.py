from http import HTTPStatus
from twisted.test.proto_helpers import MemoryReactor
import synapse.rest.admin
from synapse.api.room_versions import KNOWN_ROOM_VERSIONS
from synapse.rest.client import capabilities, login
from synapse.server import HomeServer
from synapse.util import Clock
from tests import unittest
from tests.unittest import override_config

class CapabilitiesTestCase(unittest.HomeserverTestCase):
    servlets = [synapse.rest.admin.register_servlets_for_client_rest_resource, capabilities.register_servlets, login.register_servlets]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            while True:
                i = 10
        self.url = b'/capabilities'
        hs = self.setup_test_homeserver()
        self.config = hs.config
        self.auth_handler = hs.get_auth_handler()
        return hs

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            i = 10
            return i + 15
        self.localpart = 'user'
        self.password = 'pass'
        self.user = self.register_user(self.localpart, self.password)

    def test_check_auth_required(self) -> None:
        if False:
            while True:
                i = 10
        channel = self.make_request('GET', self.url)
        self.assertEqual(channel.code, 401)

    def test_get_room_version_capabilities(self) -> None:
        if False:
            print('Hello World!')
        access_token = self.login(self.localpart, self.password)
        channel = self.make_request('GET', self.url, access_token=access_token)
        capabilities = channel.json_body['capabilities']
        self.assertEqual(channel.code, 200)
        for room_version in capabilities['m.room_versions']['available'].keys():
            self.assertTrue(room_version in KNOWN_ROOM_VERSIONS, '' + room_version)
        self.assertEqual(self.config.server.default_room_version.identifier, capabilities['m.room_versions']['default'])

    def test_get_change_password_capabilities_password_login(self) -> None:
        if False:
            return 10
        access_token = self.login(self.localpart, self.password)
        channel = self.make_request('GET', self.url, access_token=access_token)
        capabilities = channel.json_body['capabilities']
        self.assertEqual(channel.code, 200)
        self.assertTrue(capabilities['m.change_password']['enabled'])

    @override_config({'password_config': {'localdb_enabled': False}})
    def test_get_change_password_capabilities_localdb_disabled(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        access_token = self.get_success(self.auth_handler.create_access_token_for_user_id(self.user, device_id=None, valid_until_ms=None))
        channel = self.make_request('GET', self.url, access_token=access_token)
        capabilities = channel.json_body['capabilities']
        self.assertEqual(channel.code, 200)
        self.assertFalse(capabilities['m.change_password']['enabled'])

    @override_config({'password_config': {'enabled': False}})
    def test_get_change_password_capabilities_password_disabled(self) -> None:
        if False:
            return 10
        access_token = self.get_success(self.auth_handler.create_access_token_for_user_id(self.user, device_id=None, valid_until_ms=None))
        channel = self.make_request('GET', self.url, access_token=access_token)
        capabilities = channel.json_body['capabilities']
        self.assertEqual(channel.code, 200)
        self.assertFalse(capabilities['m.change_password']['enabled'])

    def test_get_change_users_attributes_capabilities(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test that server returns capabilities by default.'
        access_token = self.login(self.localpart, self.password)
        channel = self.make_request('GET', self.url, access_token=access_token)
        capabilities = channel.json_body['capabilities']
        self.assertEqual(channel.code, HTTPStatus.OK)
        self.assertTrue(capabilities['m.change_password']['enabled'])
        self.assertTrue(capabilities['m.set_displayname']['enabled'])
        self.assertTrue(capabilities['m.set_avatar_url']['enabled'])
        self.assertTrue(capabilities['m.3pid_changes']['enabled'])

    @override_config({'enable_set_displayname': False})
    def test_get_set_displayname_capabilities_displayname_disabled(self) -> None:
        if False:
            return 10
        'Test if set displayname is disabled that the server responds it.'
        access_token = self.login(self.localpart, self.password)
        channel = self.make_request('GET', self.url, access_token=access_token)
        capabilities = channel.json_body['capabilities']
        self.assertEqual(channel.code, HTTPStatus.OK)
        self.assertFalse(capabilities['m.set_displayname']['enabled'])

    @override_config({'enable_set_avatar_url': False})
    def test_get_set_avatar_url_capabilities_avatar_url_disabled(self) -> None:
        if False:
            return 10
        'Test if set avatar_url is disabled that the server responds it.'
        access_token = self.login(self.localpart, self.password)
        channel = self.make_request('GET', self.url, access_token=access_token)
        capabilities = channel.json_body['capabilities']
        self.assertEqual(channel.code, HTTPStatus.OK)
        self.assertFalse(capabilities['m.set_avatar_url']['enabled'])

    @override_config({'enable_3pid_changes': False})
    def test_get_change_3pid_capabilities_3pid_disabled(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Test if change 3pid is disabled that the server responds it.'
        access_token = self.login(self.localpart, self.password)
        channel = self.make_request('GET', self.url, access_token=access_token)
        capabilities = channel.json_body['capabilities']
        self.assertEqual(channel.code, HTTPStatus.OK)
        self.assertFalse(capabilities['m.3pid_changes']['enabled'])

    @override_config({'experimental_features': {'msc3244_enabled': False}})
    def test_get_does_not_include_msc3244_fields_when_disabled(self) -> None:
        if False:
            while True:
                i = 10
        access_token = self.get_success(self.auth_handler.create_access_token_for_user_id(self.user, device_id=None, valid_until_ms=None))
        channel = self.make_request('GET', self.url, access_token=access_token)
        capabilities = channel.json_body['capabilities']
        self.assertEqual(channel.code, 200)
        self.assertNotIn('org.matrix.msc3244.room_capabilities', capabilities['m.room_versions'])

    def test_get_does_include_msc3244_fields_when_enabled(self) -> None:
        if False:
            return 10
        access_token = self.get_success(self.auth_handler.create_access_token_for_user_id(self.user, device_id=None, valid_until_ms=None))
        channel = self.make_request('GET', self.url, access_token=access_token)
        capabilities = channel.json_body['capabilities']
        self.assertEqual(channel.code, 200)
        for details in capabilities['m.room_versions']['org.matrix.msc3244.room_capabilities'].values():
            if details['preferred'] is not None:
                self.assertTrue(details['preferred'] in KNOWN_ROOM_VERSIONS, str(details['preferred']))
            self.assertGreater(len(details['support']), 0)
            for room_version in details['support']:
                self.assertTrue(room_version in KNOWN_ROOM_VERSIONS, str(room_version))

    def test_get_get_token_login_fields_when_disabled(self) -> None:
        if False:
            print('Hello World!')
        'By default login via an existing session is disabled.'
        access_token = self.get_success(self.auth_handler.create_access_token_for_user_id(self.user, device_id=None, valid_until_ms=None))
        channel = self.make_request('GET', self.url, access_token=access_token)
        capabilities = channel.json_body['capabilities']
        self.assertEqual(channel.code, HTTPStatus.OK)
        self.assertFalse(capabilities['m.get_login_token']['enabled'])

    @override_config({'login_via_existing_session': {'enabled': True}})
    def test_get_get_token_login_fields_when_enabled(self) -> None:
        if False:
            i = 10
            return i + 15
        access_token = self.get_success(self.auth_handler.create_access_token_for_user_id(self.user, device_id=None, valid_until_ms=None))
        channel = self.make_request('GET', self.url, access_token=access_token)
        capabilities = channel.json_body['capabilities']
        self.assertEqual(channel.code, HTTPStatus.OK)
        self.assertTrue(capabilities['m.get_login_token']['enabled'])