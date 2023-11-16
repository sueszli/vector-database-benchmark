"""Tests for the password_auth_provider interface"""
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Type, Union
from unittest.mock import AsyncMock, Mock
from twisted.test.proto_helpers import MemoryReactor
import synapse
from synapse.api.constants import LoginType
from synapse.api.errors import Codes
from synapse.handlers.account import AccountHandler
from synapse.module_api import ModuleApi
from synapse.rest.client import account, devices, login, logout, register
from synapse.server import HomeServer
from synapse.types import JsonDict, UserID
from synapse.util import Clock
from tests import unittest
from tests.server import FakeChannel
from tests.unittest import override_config
ADDITIONAL_LOGIN_FLOWS = [{'type': 'm.login.application_service'}]
mock_password_provider = Mock()

class LegacyPasswordOnlyAuthProvider:
    """A legacy password_provider which only implements `check_password`."""

    @staticmethod
    def parse_config(config: JsonDict) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def __init__(self, config: None, account_handler: AccountHandler):
        if False:
            i = 10
            return i + 15
        pass

    def check_password(self, *args: str) -> Mock:
        if False:
            i = 10
            return i + 15
        return mock_password_provider.check_password(*args)

class LegacyCustomAuthProvider:
    """A legacy password_provider which implements a custom login type."""

    @staticmethod
    def parse_config(config: JsonDict) -> None:
        if False:
            i = 10
            return i + 15
        pass

    def __init__(self, config: None, account_handler: AccountHandler):
        if False:
            i = 10
            return i + 15
        pass

    def get_supported_login_types(self) -> Dict[str, List[str]]:
        if False:
            print('Hello World!')
        return {'test.login_type': ['test_field']}

    def check_auth(self, *args: str) -> Mock:
        if False:
            for i in range(10):
                print('nop')
        return mock_password_provider.check_auth(*args)

class CustomAuthProvider:
    """A module which registers password_auth_provider callbacks for a custom login type."""

    @staticmethod
    def parse_config(config: JsonDict) -> None:
        if False:
            print('Hello World!')
        pass

    def __init__(self, config: None, api: ModuleApi):
        if False:
            while True:
                i = 10
        api.register_password_auth_provider_callbacks(auth_checkers={('test.login_type', ('test_field',)): self.check_auth})

    def check_auth(self, *args: Any) -> Mock:
        if False:
            print('Hello World!')
        return mock_password_provider.check_auth(*args)

class LegacyPasswordCustomAuthProvider:
    """A password_provider which implements password login via `check_auth`, as well
    as a custom type."""

    @staticmethod
    def parse_config(config: JsonDict) -> None:
        if False:
            return 10
        pass

    def __init__(self, config: None, account_handler: AccountHandler):
        if False:
            return 10
        pass

    def get_supported_login_types(self) -> Dict[str, List[str]]:
        if False:
            print('Hello World!')
        return {'m.login.password': ['password'], 'test.login_type': ['test_field']}

    def check_auth(self, *args: str) -> Mock:
        if False:
            return 10
        return mock_password_provider.check_auth(*args)

class PasswordCustomAuthProvider:
    """A module which registers password_auth_provider callbacks for a custom login type.
    as well as a password login"""

    @staticmethod
    def parse_config(config: JsonDict) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    def __init__(self, config: None, api: ModuleApi):
        if False:
            i = 10
            return i + 15
        api.register_password_auth_provider_callbacks(auth_checkers={('test.login_type', ('test_field',)): self.check_auth, ('m.login.password', ('password',)): self.check_auth})

    def check_auth(self, *args: Any) -> Mock:
        if False:
            for i in range(10):
                print('nop')
        return mock_password_provider.check_auth(*args)

    def check_pass(self, *args: str) -> Mock:
        if False:
            print('Hello World!')
        return mock_password_provider.check_password(*args)

def legacy_providers_config(*providers: Type[Any]) -> dict:
    if False:
        i = 10
        return i + 15
    'Returns a config dict that will enable the given legacy password auth providers'
    return {'password_providers': [{'module': '%s.%s' % (__name__, provider.__qualname__), 'config': {}} for provider in providers]}

def providers_config(*providers: Type[Any]) -> dict:
    if False:
        return 10
    'Returns a config dict that will enable the given modules'
    return {'modules': [{'module': '%s.%s' % (__name__, provider.__qualname__), 'config': {}} for provider in providers]}

class PasswordAuthProviderTests(unittest.HomeserverTestCase):
    servlets = [synapse.rest.admin.register_servlets, login.register_servlets, devices.register_servlets, logout.register_servlets, register.register_servlets, account.register_servlets]
    CALLBACK_USERNAME = 'get_username_for_registration'
    CALLBACK_DISPLAYNAME = 'get_displayname_for_registration'

    def prepare(self, reactor: MemoryReactor, clock: Clock, homeserver: HomeServer) -> None:
        if False:
            i = 10
            return i + 15
        mock_password_provider.reset_mock()
        self.register_user('u', 'not-the-tested-password')
        self.register_user('user', 'not-the-tested-password')

    @override_config(legacy_providers_config(LegacyPasswordOnlyAuthProvider))
    def test_password_only_auth_progiver_login_legacy(self) -> None:
        if False:
            print('Hello World!')
        self.password_only_auth_provider_login_test_body()

    def password_only_auth_provider_login_test_body(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        flows = self._get_login_flows()
        self.assertEqual(flows, [{'type': 'm.login.password'}] + ADDITIONAL_LOGIN_FLOWS)
        mock_password_provider.check_password = AsyncMock(return_value=True)
        channel = self._send_password_login('u', 'p')
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        self.assertEqual('@u:test', channel.json_body['user_id'])
        mock_password_provider.check_password.assert_called_once_with('@u:test', 'p')
        mock_password_provider.reset_mock()
        channel = self._send_password_login('@u:test', 'p')
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        self.assertEqual('@u:test', channel.json_body['user_id'])
        mock_password_provider.check_password.assert_called_once_with('@u:test', 'p')
        mock_password_provider.reset_mock()

    @override_config(legacy_providers_config(LegacyPasswordOnlyAuthProvider))
    def test_password_only_auth_provider_ui_auth_legacy(self) -> None:
        if False:
            return 10
        self.password_only_auth_provider_ui_auth_test_body()

    def password_only_auth_provider_ui_auth_test_body(self) -> None:
        if False:
            while True:
                i = 10
        'UI Auth should delegate correctly to the password provider'
        mock_password_provider.check_password = AsyncMock(return_value=True)
        tok1 = self.login('u', 'p')
        self.login('u', 'p', device_id='dev2')
        mock_password_provider.reset_mock()
        mock_password_provider.check_password = AsyncMock(return_value=False)
        session = self._start_delete_device_session(tok1, 'dev2')
        mock_password_provider.check_password.assert_not_called()
        channel = self._authed_delete_device(tok1, 'dev2', session, 'u', 'p')
        self.assertEqual(channel.code, 401)
        self.assertEqual(channel.json_body['errcode'], 'M_FORBIDDEN')
        mock_password_provider.check_password.assert_called_once_with('@u:test', 'p')
        mock_password_provider.reset_mock()
        mock_password_provider.check_password = AsyncMock(return_value=True)
        channel = self._authed_delete_device(tok1, 'dev2', session, 'u', 'p')
        self.assertEqual(channel.code, 200)
        mock_password_provider.check_password.assert_called_once_with('@u:test', 'p')

    @override_config(legacy_providers_config(LegacyPasswordOnlyAuthProvider))
    def test_local_user_fallback_login_legacy(self) -> None:
        if False:
            i = 10
            return i + 15
        self.local_user_fallback_login_test_body()

    def local_user_fallback_login_test_body(self) -> None:
        if False:
            while True:
                i = 10
        'rejected login should fall back to local db'
        self.register_user('localuser', 'localpass')
        mock_password_provider.check_password = AsyncMock(return_value=False)
        channel = self._send_password_login('u', 'p')
        self.assertEqual(channel.code, HTTPStatus.FORBIDDEN, channel.result)
        channel = self._send_password_login('localuser', 'localpass')
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        self.assertEqual('@localuser:test', channel.json_body['user_id'])

    @override_config(legacy_providers_config(LegacyPasswordOnlyAuthProvider))
    def test_local_user_fallback_ui_auth_legacy(self) -> None:
        if False:
            while True:
                i = 10
        self.local_user_fallback_ui_auth_test_body()

    def local_user_fallback_ui_auth_test_body(self) -> None:
        if False:
            return 10
        'rejected login should fall back to local db'
        self.register_user('localuser', 'localpass')
        mock_password_provider.check_password = AsyncMock(return_value=False)
        tok1 = self.login('localuser', 'localpass')
        self.login('localuser', 'localpass', device_id='dev2')
        mock_password_provider.check_password.reset_mock()
        session = self._start_delete_device_session(tok1, 'dev2')
        mock_password_provider.check_password.assert_not_called()
        channel = self._authed_delete_device(tok1, 'dev2', session, 'localuser', 'xxx')
        self.assertEqual(channel.code, 401)
        self.assertEqual(channel.json_body['errcode'], 'M_FORBIDDEN')
        mock_password_provider.check_password.assert_called_once_with('@localuser:test', 'xxx')
        mock_password_provider.reset_mock()
        channel = self._authed_delete_device(tok1, 'dev2', session, 'localuser', 'localpass')
        self.assertEqual(channel.code, 200)
        mock_password_provider.check_password.assert_called_once_with('@localuser:test', 'localpass')

    @override_config({**legacy_providers_config(LegacyPasswordOnlyAuthProvider), 'password_config': {'localdb_enabled': False}})
    def test_no_local_user_fallback_login_legacy(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.no_local_user_fallback_login_test_body()

    def no_local_user_fallback_login_test_body(self) -> None:
        if False:
            print('Hello World!')
        'localdb_enabled can block login with the local password'
        self.register_user('localuser', 'localpass')
        mock_password_provider.check_password = AsyncMock(return_value=False)
        channel = self._send_password_login('localuser', 'localpass')
        self.assertEqual(channel.code, 403)
        self.assertEqual(channel.json_body['errcode'], 'M_FORBIDDEN')
        mock_password_provider.check_password.assert_called_once_with('@localuser:test', 'localpass')

    @override_config({**legacy_providers_config(LegacyPasswordOnlyAuthProvider), 'password_config': {'localdb_enabled': False}})
    def test_no_local_user_fallback_ui_auth_legacy(self) -> None:
        if False:
            i = 10
            return i + 15
        self.no_local_user_fallback_ui_auth_test_body()

    def no_local_user_fallback_ui_auth_test_body(self) -> None:
        if False:
            i = 10
            return i + 15
        'localdb_enabled can block ui auth with the local password'
        self.register_user('localuser', 'localpass')
        mock_password_provider.check_password = AsyncMock(return_value=True)
        tok1 = self.login('localuser', 'p')
        self.login('localuser', 'p', device_id='dev2')
        mock_password_provider.check_password.reset_mock()
        channel = self._delete_device(tok1, 'dev2')
        self.assertEqual(channel.code, 401)
        self.assertEqual(channel.json_body['flows'], [{'stages': ['m.login.password']}])
        session = channel.json_body['session']
        mock_password_provider.check_password.assert_not_called()
        mock_password_provider.check_password = AsyncMock(return_value=False)
        channel = self._authed_delete_device(tok1, 'dev2', session, 'localuser', 'localpass')
        self.assertEqual(channel.code, 401)
        self.assertEqual(channel.json_body['errcode'], 'M_FORBIDDEN')
        mock_password_provider.check_password.assert_called_once_with('@localuser:test', 'localpass')

    @override_config({**legacy_providers_config(LegacyPasswordOnlyAuthProvider), 'password_config': {'enabled': False}})
    def test_password_auth_disabled_legacy(self) -> None:
        if False:
            return 10
        self.password_auth_disabled_test_body()

    def password_auth_disabled_test_body(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "password auth doesn't work if it's disabled across the board"
        flows = self._get_login_flows()
        self.assertEqual(flows, ADDITIONAL_LOGIN_FLOWS)
        channel = self._send_password_login('u', 'p')
        self.assertEqual(channel.code, HTTPStatus.BAD_REQUEST, channel.result)
        mock_password_provider.check_password.assert_not_called()

    @override_config(legacy_providers_config(LegacyCustomAuthProvider))
    def test_custom_auth_provider_login_legacy(self) -> None:
        if False:
            return 10
        self.custom_auth_provider_login_test_body()

    @override_config(providers_config(CustomAuthProvider))
    def test_custom_auth_provider_login(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.custom_auth_provider_login_test_body()

    def custom_auth_provider_login_test_body(self) -> None:
        if False:
            print('Hello World!')
        flows = self._get_login_flows()
        self.assertEqual(flows, [{'type': 'm.login.password'}, {'type': 'test.login_type'}] + ADDITIONAL_LOGIN_FLOWS)
        channel = self._send_login('test.login_type', 'u')
        self.assertEqual(channel.code, HTTPStatus.BAD_REQUEST, channel.result)
        mock_password_provider.check_auth.assert_not_called()
        mock_password_provider.check_auth = AsyncMock(return_value=('@user:test', None))
        channel = self._send_login('test.login_type', 'u', test_field='y')
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        self.assertEqual('@user:test', channel.json_body['user_id'])
        mock_password_provider.check_auth.assert_called_once_with('u', 'test.login_type', {'test_field': 'y'})
        mock_password_provider.reset_mock()

    @override_config(legacy_providers_config(LegacyCustomAuthProvider))
    def test_custom_auth_provider_ui_auth_legacy(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.custom_auth_provider_ui_auth_test_body()

    @override_config(providers_config(CustomAuthProvider))
    def test_custom_auth_provider_ui_auth(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.custom_auth_provider_ui_auth_test_body()

    def custom_auth_provider_ui_auth_test_body(self) -> None:
        if False:
            while True:
                i = 10
        self.register_user('localuser', 'localpass')
        tok1 = self.login('localuser', 'localpass')
        self.login('localuser', 'localpass', device_id='dev2')
        channel = self._delete_device(tok1, 'dev2')
        self.assertEqual(channel.code, 401)
        self.assertIn({'stages': ['m.login.password']}, channel.json_body['flows'])
        self.assertIn({'stages': ['test.login_type']}, channel.json_body['flows'])
        session = channel.json_body['session']
        body = {'auth': {'type': 'test.login_type', 'identifier': {'type': 'm.id.user', 'user': 'localuser'}, 'session': session}}
        channel = self._delete_device(tok1, 'dev2', body)
        self.assertEqual(channel.code, 400)
        self.assertIn('Missing parameters', channel.json_body['error'])
        mock_password_provider.check_auth.assert_not_called()
        mock_password_provider.reset_mock()
        mock_password_provider.check_auth = AsyncMock(return_value=('@user:test', None))
        body['auth']['test_field'] = 'foo'
        channel = self._delete_device(tok1, 'dev2', body)
        self.assertEqual(channel.code, 403)
        self.assertEqual(channel.json_body['errcode'], 'M_FORBIDDEN')
        mock_password_provider.check_auth.assert_called_once_with('localuser', 'test.login_type', {'test_field': 'foo'})
        mock_password_provider.reset_mock()
        mock_password_provider.check_auth = AsyncMock(return_value=('@localuser:test', None))
        channel = self._delete_device(tok1, 'dev2', body)
        self.assertEqual(channel.code, 200)
        mock_password_provider.check_auth.assert_called_once_with('localuser', 'test.login_type', {'test_field': 'foo'})

    @override_config(legacy_providers_config(LegacyCustomAuthProvider))
    def test_custom_auth_provider_callback_legacy(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.custom_auth_provider_callback_test_body()

    @override_config(providers_config(CustomAuthProvider))
    def test_custom_auth_provider_callback(self) -> None:
        if False:
            print('Hello World!')
        self.custom_auth_provider_callback_test_body()

    def custom_auth_provider_callback_test_body(self) -> None:
        if False:
            while True:
                i = 10
        callback = AsyncMock(return_value=None)
        mock_password_provider.check_auth = AsyncMock(return_value=('@user:test', callback))
        channel = self._send_login('test.login_type', 'u', test_field='y')
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        self.assertEqual('@user:test', channel.json_body['user_id'])
        mock_password_provider.check_auth.assert_called_once_with('u', 'test.login_type', {'test_field': 'y'})
        callback.assert_called_once()
        (call_args, call_kwargs) = callback.call_args
        self.assertEqual(len(call_args), 1)
        self.assertEqual(call_args[0]['user_id'], '@user:test')
        for p in ['user_id', 'access_token', 'device_id', 'home_server']:
            self.assertIn(p, call_args[0])

    @override_config({**legacy_providers_config(LegacyCustomAuthProvider), 'password_config': {'enabled': False}})
    def test_custom_auth_password_disabled_legacy(self) -> None:
        if False:
            while True:
                i = 10
        self.custom_auth_password_disabled_test_body()

    @override_config({**providers_config(CustomAuthProvider), 'password_config': {'enabled': False}})
    def test_custom_auth_password_disabled(self) -> None:
        if False:
            i = 10
            return i + 15
        self.custom_auth_password_disabled_test_body()

    def custom_auth_password_disabled_test_body(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test login with a custom auth provider where password login is disabled'
        self.register_user('localuser', 'localpass')
        flows = self._get_login_flows()
        self.assertEqual(flows, [{'type': 'test.login_type'}] + ADDITIONAL_LOGIN_FLOWS)
        channel = self._send_password_login('localuser', 'localpass')
        self.assertEqual(channel.code, HTTPStatus.BAD_REQUEST, channel.result)
        mock_password_provider.check_auth.assert_not_called()

    @override_config({**legacy_providers_config(LegacyCustomAuthProvider), 'password_config': {'enabled': False, 'localdb_enabled': False}})
    def test_custom_auth_password_disabled_localdb_enabled_legacy(self) -> None:
        if False:
            print('Hello World!')
        self.custom_auth_password_disabled_localdb_enabled_test_body()

    @override_config({**providers_config(CustomAuthProvider), 'password_config': {'enabled': False, 'localdb_enabled': False}})
    def test_custom_auth_password_disabled_localdb_enabled(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.custom_auth_password_disabled_localdb_enabled_test_body()

    def custom_auth_password_disabled_localdb_enabled_test_body(self) -> None:
        if False:
            i = 10
            return i + 15
        "Check the localdb_enabled == enabled == False\n\n        Regression test for https://github.com/matrix-org/synapse/issues/8914: check\n        that setting *both* `localdb_enabled` *and* `password: enabled` to False doesn't\n        cause an exception.\n        "
        self.register_user('localuser', 'localpass')
        flows = self._get_login_flows()
        self.assertEqual(flows, [{'type': 'test.login_type'}] + ADDITIONAL_LOGIN_FLOWS)
        channel = self._send_password_login('localuser', 'localpass')
        self.assertEqual(channel.code, HTTPStatus.BAD_REQUEST, channel.result)
        mock_password_provider.check_auth.assert_not_called()

    @override_config({**legacy_providers_config(LegacyPasswordCustomAuthProvider), 'password_config': {'enabled': False}})
    def test_password_custom_auth_password_disabled_login_legacy(self) -> None:
        if False:
            while True:
                i = 10
        self.password_custom_auth_password_disabled_login_test_body()

    @override_config({**providers_config(PasswordCustomAuthProvider), 'password_config': {'enabled': False}})
    def test_password_custom_auth_password_disabled_login(self) -> None:
        if False:
            print('Hello World!')
        self.password_custom_auth_password_disabled_login_test_body()

    def password_custom_auth_password_disabled_login_test_body(self) -> None:
        if False:
            while True:
                i = 10
        'log in with a custom auth provider which implements password, but password\n        login is disabled'
        self.register_user('localuser', 'localpass')
        flows = self._get_login_flows()
        self.assertEqual(flows, [{'type': 'test.login_type'}] + ADDITIONAL_LOGIN_FLOWS)
        channel = self._send_password_login('localuser', 'localpass')
        self.assertEqual(channel.code, HTTPStatus.BAD_REQUEST, channel.result)
        mock_password_provider.check_auth.assert_not_called()
        mock_password_provider.check_password.assert_not_called()

    @override_config({**legacy_providers_config(LegacyPasswordCustomAuthProvider), 'password_config': {'enabled': False}})
    def test_password_custom_auth_password_disabled_ui_auth_legacy(self) -> None:
        if False:
            print('Hello World!')
        self.password_custom_auth_password_disabled_ui_auth_test_body()

    @override_config({**providers_config(PasswordCustomAuthProvider), 'password_config': {'enabled': False}})
    def test_password_custom_auth_password_disabled_ui_auth(self) -> None:
        if False:
            print('Hello World!')
        self.password_custom_auth_password_disabled_ui_auth_test_body()

    def password_custom_auth_password_disabled_ui_auth_test_body(self) -> None:
        if False:
            return 10
        'UI Auth with a custom auth provider which implements password, but password\n        login is disabled'
        self.register_user('localuser', 'localpass')
        mock_password_provider.check_auth = AsyncMock(return_value=('@localuser:test', None))
        channel = self._send_login('test.login_type', 'localuser', test_field='')
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        tok1 = channel.json_body['access_token']
        channel = self._send_login('test.login_type', 'localuser', test_field='', device_id='dev2')
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        channel = self._delete_device(tok1, 'dev2')
        self.assertEqual(channel.code, 401)
        self.assertIn({'stages': ['test.login_type']}, channel.json_body['flows'])
        session = channel.json_body['session']
        mock_password_provider.reset_mock()
        body = {'auth': {'type': 'm.login.password', 'identifier': {'type': 'm.id.user', 'user': 'localuser'}, 'password': 'localpass', 'session': session}}
        channel = self._delete_device(tok1, 'dev2', body)
        self.assertEqual(channel.code, 400)
        self.assertEqual('Password login has been disabled.', channel.json_body['error'])
        mock_password_provider.check_auth.assert_not_called()
        mock_password_provider.check_password.assert_not_called()
        mock_password_provider.reset_mock()
        body['auth']['type'] = 'test.login_type'
        body['auth']['test_field'] = 'x'
        channel = self._delete_device(tok1, 'dev2', body)
        self.assertEqual(channel.code, 200)
        mock_password_provider.check_auth.assert_called_once_with('localuser', 'test.login_type', {'test_field': 'x'})
        mock_password_provider.check_password.assert_not_called()

    @override_config({**legacy_providers_config(LegacyCustomAuthProvider), 'password_config': {'localdb_enabled': False}})
    def test_custom_auth_no_local_user_fallback_legacy(self) -> None:
        if False:
            i = 10
            return i + 15
        self.custom_auth_no_local_user_fallback_test_body()

    @override_config({**providers_config(CustomAuthProvider), 'password_config': {'localdb_enabled': False}})
    def test_custom_auth_no_local_user_fallback(self) -> None:
        if False:
            print('Hello World!')
        self.custom_auth_no_local_user_fallback_test_body()

    def custom_auth_no_local_user_fallback_test_body(self) -> None:
        if False:
            return 10
        'Test login with a custom auth provider where the local db is disabled'
        self.register_user('localuser', 'localpass')
        flows = self._get_login_flows()
        self.assertEqual(flows, [{'type': 'test.login_type'}] + ADDITIONAL_LOGIN_FLOWS)
        channel = self._send_password_login('localuser', 'localpass')
        self.assertEqual(channel.code, HTTPStatus.BAD_REQUEST, channel.result)

    def test_on_logged_out(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that the on_logged_out callback is called when the user logs out.'
        self.register_user('rin', 'password')
        tok = self.login('rin', 'password')
        self.called = False

        async def on_logged_out(user_id: str, device_id: Optional[str], access_token: str) -> None:
            self.called = True
        on_logged_out = Mock(side_effect=on_logged_out)
        self.hs.get_password_auth_provider().on_logged_out_callbacks.append(on_logged_out)
        channel = self.make_request('POST', '/_matrix/client/v3/logout', {}, access_token=tok)
        self.assertEqual(channel.code, 200)
        on_logged_out.assert_called_once()
        self.assertTrue(self.called)

    def test_username(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that the get_username_for_registration callback can define the username\n        of a user when registering.\n        '
        self._setup_get_name_for_registration(callback_name=self.CALLBACK_USERNAME)
        username = 'rin'
        channel = self.make_request('POST', '/register', {'username': username, 'password': 'bar', 'auth': {'type': LoginType.DUMMY}})
        self.assertEqual(channel.code, 200)
        mxid = channel.json_body['user_id']
        self.assertEqual(UserID.from_string(mxid).localpart, username + '-foo')

    def test_username_uia(self) -> None:
        if False:
            print('Hello World!')
        'Tests that the get_username_for_registration callback is only called at the\n        end of the UIA flow.\n        '
        m = self._setup_get_name_for_registration(callback_name=self.CALLBACK_USERNAME)
        username = 'rin'
        res = self._do_uia_assert_mock_not_called(username, m)
        mxid = res['user_id']
        self.assertEqual(UserID.from_string(mxid).localpart, username + '-foo')
        m.assert_called_once()

    @override_config({'email': {'notif_from': 'noreply@test'}})
    def test_3pid_allowed(self) -> None:
        if False:
            i = 10
            return i + 15
        'Tests that an is_3pid_allowed_callbacks forbidding a 3PID makes Synapse refuse\n        to bind the new 3PID, and that one allowing a 3PID makes Synapse accept to bind\n        the 3PID. Also checks that the module is passed a boolean indicating whether the\n        user to bind this 3PID to is currently registering.\n        '
        self._test_3pid_allowed('rin', False)
        self._test_3pid_allowed('kitay', True)

    def test_displayname(self) -> None:
        if False:
            print('Hello World!')
        'Tests that the get_displayname_for_registration callback can define the\n        display name of a user when registering.\n        '
        self._setup_get_name_for_registration(callback_name=self.CALLBACK_DISPLAYNAME)
        username = 'rin'
        channel = self.make_request('POST', '/register', {'username': username, 'password': 'bar', 'auth': {'type': LoginType.DUMMY}})
        self.assertEqual(channel.code, 200)
        user_id = UserID.from_string(channel.json_body['user_id'])
        display_name = self.get_success(self.hs.get_profile_handler().get_displayname(user_id))
        self.assertEqual(display_name, username + '-foo')

    def test_displayname_uia(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Tests that the get_displayname_for_registration callback is only called at the\n        end of the UIA flow.\n        '
        m = self._setup_get_name_for_registration(callback_name=self.CALLBACK_DISPLAYNAME)
        username = 'rin'
        res = self._do_uia_assert_mock_not_called(username, m)
        user_id = UserID.from_string(res['user_id'])
        display_name = self.get_success(self.hs.get_profile_handler().get_displayname(user_id))
        self.assertEqual(display_name, username + '-foo')
        m.assert_called_once()

    def _test_3pid_allowed(self, username: str, registration: bool) -> None:
        if False:
            print('Hello World!')
        'Tests that the "is_3pid_allowed" module callback is called correctly, using\n        either /register or /account URLs depending on the arguments.\n\n        Args:\n            username: The username to use for the test.\n            registration: Whether to test with registration URLs.\n        '
        self.hs.get_identity_handler().send_threepid_validation = AsyncMock(return_value=0)
        m = AsyncMock(return_value=False)
        self.hs.get_password_auth_provider().is_3pid_allowed_callbacks = [m]
        self.register_user(username, 'password')
        tok = self.login(username, 'password')
        if registration:
            url = '/register/email/requestToken'
        else:
            url = '/account/3pid/email/requestToken'
        channel = self.make_request('POST', url, {'client_secret': 'foo', 'email': 'foo@test.com', 'send_attempt': 0}, access_token=tok)
        self.assertEqual(channel.code, HTTPStatus.FORBIDDEN, channel.result)
        self.assertEqual(channel.json_body['errcode'], Codes.THREEPID_DENIED, channel.json_body)
        m.assert_called_once_with('email', 'foo@test.com', registration)
        m = AsyncMock(return_value=True)
        self.hs.get_password_auth_provider().is_3pid_allowed_callbacks = [m]
        channel = self.make_request('POST', url, {'client_secret': 'foo', 'email': 'bar@test.com', 'send_attempt': 0}, access_token=tok)
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        self.assertIn('sid', channel.json_body)
        m.assert_called_once_with('email', 'bar@test.com', registration)

    def _setup_get_name_for_registration(self, callback_name: str) -> Mock:
        if False:
            print('Hello World!')
        'Registers either a get_username_for_registration callback or a\n        get_displayname_for_registration callback that appends "-foo" to the username the\n        client is trying to register.\n        '

        async def callback(uia_results: JsonDict, params: JsonDict) -> str:
            self.assertIn(LoginType.DUMMY, uia_results)
            username = params['username']
            return username + '-foo'
        m = Mock(side_effect=callback)
        password_auth_provider = self.hs.get_password_auth_provider()
        getattr(password_auth_provider, callback_name + '_callbacks').append(m)
        return m

    def _do_uia_assert_mock_not_called(self, username: str, m: Mock) -> JsonDict:
        if False:
            for i in range(10):
                print('nop')
        channel = self.make_request('POST', 'register', {'username': username, 'type': 'm.login.password', 'password': 'bar'})
        self.assertEqual(channel.code, 401)
        self.assertIn('session', channel.json_body)
        m.assert_not_called()
        session = channel.json_body['session']
        channel = self.make_request('POST', 'register', {'auth': {'session': session, 'type': LoginType.DUMMY}})
        self.assertEqual(channel.code, HTTPStatus.OK, channel.json_body)
        return channel.json_body

    def _get_login_flows(self) -> JsonDict:
        if False:
            while True:
                i = 10
        channel = self.make_request('GET', '/_matrix/client/r0/login')
        self.assertEqual(channel.code, HTTPStatus.OK, channel.result)
        return channel.json_body['flows']

    def _send_password_login(self, user: str, password: str) -> FakeChannel:
        if False:
            for i in range(10):
                print('nop')
        return self._send_login(type='m.login.password', user=user, password=password)

    def _send_login(self, type: str, user: str, **extra_params: str) -> FakeChannel:
        if False:
            i = 10
            return i + 15
        params = {'identifier': {'type': 'm.id.user', 'user': user}, 'type': type}
        params.update(extra_params)
        channel = self.make_request('POST', '/_matrix/client/r0/login', params)
        return channel

    def _start_delete_device_session(self, access_token: str, device_id: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Make an initial delete device request, and return the UI Auth session ID'
        channel = self._delete_device(access_token, device_id)
        self.assertEqual(channel.code, 401)
        self.assertIn({'stages': ['m.login.password']}, channel.json_body['flows'])
        return channel.json_body['session']

    def _authed_delete_device(self, access_token: str, device_id: str, session: str, user_id: str, password: str) -> FakeChannel:
        if False:
            i = 10
            return i + 15
        'Make a delete device request, authenticating with the given uid/password'
        return self._delete_device(access_token, device_id, {'auth': {'type': 'm.login.password', 'identifier': {'type': 'm.id.user', 'user': user_id}, 'password': password, 'session': session}})

    def _delete_device(self, access_token: str, device: str, body: Union[JsonDict, bytes]=b'') -> FakeChannel:
        if False:
            while True:
                i = 10
        'Delete an individual device.'
        channel = self.make_request('DELETE', 'devices/' + device, body, access_token=access_token)
        return channel