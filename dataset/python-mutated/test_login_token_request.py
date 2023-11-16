from twisted.test.proto_helpers import MemoryReactor
from synapse.rest import admin
from synapse.rest.client import login, login_token_request, versions
from synapse.server import HomeServer
from synapse.util import Clock
from tests import unittest
from tests.unittest import override_config
GET_TOKEN_ENDPOINT = '/_matrix/client/v1/login/get_token'

class LoginTokenRequestServletTestCase(unittest.HomeserverTestCase):
    servlets = [login.register_servlets, admin.register_servlets, login_token_request.register_servlets, versions.register_servlets]

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            print('Hello World!')
        self.hs = self.setup_test_homeserver()
        self.hs.config.registration.enable_registration = True
        self.hs.config.registration.registrations_require_3pid = []
        self.hs.config.registration.auto_join_rooms = []
        self.hs.config.captcha.enable_registration_captcha = False
        return self.hs

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            print('Hello World!')
        self.user = 'user123'
        self.password = 'password'

    def test_disabled(self) -> None:
        if False:
            i = 10
            return i + 15
        channel = self.make_request('POST', GET_TOKEN_ENDPOINT, {}, access_token=None)
        self.assertEqual(channel.code, 404)
        self.register_user(self.user, self.password)
        token = self.login(self.user, self.password)
        channel = self.make_request('POST', GET_TOKEN_ENDPOINT, {}, access_token=token)
        self.assertEqual(channel.code, 404)

    @override_config({'login_via_existing_session': {'enabled': True}})
    def test_require_auth(self) -> None:
        if False:
            i = 10
            return i + 15
        channel = self.make_request('POST', GET_TOKEN_ENDPOINT, {}, access_token=None)
        self.assertEqual(channel.code, 401)

    @override_config({'login_via_existing_session': {'enabled': True}})
    def test_uia_on(self) -> None:
        if False:
            i = 10
            return i + 15
        user_id = self.register_user(self.user, self.password)
        token = self.login(self.user, self.password)
        channel = self.make_request('POST', GET_TOKEN_ENDPOINT, {}, access_token=token)
        self.assertEqual(channel.code, 401)
        self.assertIn({'stages': ['m.login.password']}, channel.json_body['flows'])
        session = channel.json_body['session']
        uia = {'auth': {'type': 'm.login.password', 'identifier': {'type': 'm.id.user', 'user': self.user}, 'password': self.password, 'session': session}}
        channel = self.make_request('POST', GET_TOKEN_ENDPOINT, uia, access_token=token)
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body['expires_in_ms'], 300000)
        login_token = channel.json_body['login_token']
        channel = self.make_request('POST', '/login', content={'type': 'm.login.token', 'token': login_token})
        self.assertEqual(channel.code, 200, channel.result)
        self.assertEqual(channel.json_body['user_id'], user_id)

    @override_config({'login_via_existing_session': {'enabled': True, 'require_ui_auth': False}})
    def test_uia_off(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        user_id = self.register_user(self.user, self.password)
        token = self.login(self.user, self.password)
        channel = self.make_request('POST', GET_TOKEN_ENDPOINT, {}, access_token=token)
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body['expires_in_ms'], 300000)
        login_token = channel.json_body['login_token']
        channel = self.make_request('POST', '/login', content={'type': 'm.login.token', 'token': login_token})
        self.assertEqual(channel.code, 200, channel.result)
        self.assertEqual(channel.json_body['user_id'], user_id)

    @override_config({'login_via_existing_session': {'enabled': True, 'require_ui_auth': False, 'token_timeout': '15s'}})
    def test_expires_in(self) -> None:
        if False:
            print('Hello World!')
        self.register_user(self.user, self.password)
        token = self.login(self.user, self.password)
        channel = self.make_request('POST', GET_TOKEN_ENDPOINT, {}, access_token=token)
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body['expires_in_ms'], 15000)

    @override_config({'login_via_existing_session': {'enabled': True, 'require_ui_auth': False, 'token_timeout': '15s'}})
    def test_unstable_support(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        channel = self.make_request('GET', '/_matrix/client/versions', {}, access_token=None)
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body['unstable_features']['org.matrix.msc3882'], True)
        self.register_user(self.user, self.password)
        token = self.login(self.user, self.password)
        channel = self.make_request('POST', '/_matrix/client/unstable/org.matrix.msc3882/login/token', {}, access_token=token)
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.json_body['expires_in'], 15)