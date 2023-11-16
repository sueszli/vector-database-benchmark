from typing import Any, Dict
from unittest.mock import AsyncMock, Mock
from twisted.test.proto_helpers import MemoryReactor
from synapse.handlers.cas import CasResponse
from synapse.server import HomeServer
from synapse.util import Clock
from tests.unittest import HomeserverTestCase, override_config
BASE_URL = 'https://synapse/'
SERVER_URL = 'https://issuer/'

class CasHandlerTestCase(HomeserverTestCase):

    def default_config(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        config = super().default_config()
        config['public_baseurl'] = BASE_URL
        cas_config = {'enabled': True, 'server_url': SERVER_URL, 'service_url': BASE_URL}
        cas_config.update(config.get('cas_config', {}))
        config['cas_config'] = cas_config
        return config

    def make_homeserver(self, reactor: MemoryReactor, clock: Clock) -> HomeServer:
        if False:
            print('Hello World!')
        hs = self.setup_test_homeserver()
        self.handler = hs.get_cas_handler()
        sso_handler = hs.get_sso_handler()
        sso_handler._MAP_USERNAME_RETRIES = 3
        return hs

    def test_map_cas_user_to_user(self) -> None:
        if False:
            while True:
                i = 10
        'Ensure that mapping the CAS user returned from a provider to an MXID works properly.'
        auth_handler = self.hs.get_auth_handler()
        auth_handler.complete_sso_login = AsyncMock()
        cas_response = CasResponse('test_user', {})
        request = _mock_request()
        self.get_success(self.handler._handle_cas_response(request, cas_response, 'redirect_uri', ''))
        auth_handler.complete_sso_login.assert_called_once_with('@test_user:test', 'cas', request, 'redirect_uri', None, new_user=True, auth_provider_session_id=None)

    def test_map_cas_user_to_existing_user(self) -> None:
        if False:
            while True:
                i = 10
        'Existing users can log in with CAS account.'
        store = self.hs.get_datastores().main
        self.get_success(store.register_user(user_id='@test_user:test', password_hash=None))
        auth_handler = self.hs.get_auth_handler()
        auth_handler.complete_sso_login = AsyncMock()
        cas_response = CasResponse('test_user', {})
        request = _mock_request()
        self.get_success(self.handler._handle_cas_response(request, cas_response, 'redirect_uri', ''))
        auth_handler.complete_sso_login.assert_called_once_with('@test_user:test', 'cas', request, 'redirect_uri', None, new_user=False, auth_provider_session_id=None)
        auth_handler.complete_sso_login.reset_mock()
        self.get_success(self.handler._handle_cas_response(request, cas_response, 'redirect_uri', ''))
        auth_handler.complete_sso_login.assert_called_once_with('@test_user:test', 'cas', request, 'redirect_uri', None, new_user=False, auth_provider_session_id=None)

    def test_map_cas_user_to_invalid_localpart(self) -> None:
        if False:
            print('Hello World!')
        'CAS automaps invalid characters to base-64 encoding.'
        auth_handler = self.hs.get_auth_handler()
        auth_handler.complete_sso_login = AsyncMock()
        cas_response = CasResponse('föö', {})
        request = _mock_request()
        self.get_success(self.handler._handle_cas_response(request, cas_response, 'redirect_uri', ''))
        auth_handler.complete_sso_login.assert_called_once_with('@f=c3=b6=c3=b6:test', 'cas', request, 'redirect_uri', None, new_user=True, auth_provider_session_id=None)

    @override_config({'cas_config': {'required_attributes': {'userGroup': 'staff', 'department': None}}})
    def test_required_attributes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'The required attributes must be met from the CAS response.'
        auth_handler = self.hs.get_auth_handler()
        auth_handler.complete_sso_login = AsyncMock()
        cas_response = CasResponse('test_user', {})
        request = _mock_request()
        self.get_success(self.handler._handle_cas_response(request, cas_response, 'redirect_uri', ''))
        auth_handler.complete_sso_login.assert_not_called()
        cas_response = CasResponse('test_user', {'userGroup': ['staff']})
        request.reset_mock()
        self.get_success(self.handler._handle_cas_response(request, cas_response, 'redirect_uri', ''))
        auth_handler.complete_sso_login.assert_not_called()
        cas_response = CasResponse('test_user', {'userGroup': ['staff', 'admin'], 'department': ['sales']})
        request.reset_mock()
        self.get_success(self.handler._handle_cas_response(request, cas_response, 'redirect_uri', ''))
        auth_handler.complete_sso_login.assert_called_once_with('@test_user:test', 'cas', request, 'redirect_uri', None, new_user=True, auth_provider_session_id=None)

    @override_config({'cas_config': {'enable_registration': False}})
    def test_map_cas_user_does_not_register_new_user(self) -> None:
        if False:
            i = 10
            return i + 15
        'Ensures new users are not registered if the enabled registration flag is disabled.'
        auth_handler = self.hs.get_auth_handler()
        auth_handler.complete_sso_login = AsyncMock()
        cas_response = CasResponse('test_user', {})
        request = _mock_request()
        self.get_success(self.handler._handle_cas_response(request, cas_response, 'redirect_uri', ''))
        auth_handler.complete_sso_login.assert_not_called()

def _mock_request() -> Mock:
    if False:
        for i in range(10):
            print('nop')
    'Returns a mock which will stand in as a SynapseRequest'
    mock = Mock(spec=['finish', 'getClientAddress', 'getHeader', 'setHeader', 'setResponseCode', 'write'])
    mock._disconnected = False
    return mock