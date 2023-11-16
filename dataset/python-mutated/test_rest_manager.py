import shutil
from asyncio import CancelledError
from unittest.mock import Mock, patch
import pytest
from aiohttp import ClientOSError, ServerDisconnectedError
from aiohttp.web_protocol import RequestHandler
from pydantic import ValidationError
from tribler.core.components.restapi.rest.aiohttp_patch import get_transport_is_none_counter, patch_make_request
from tribler.core.components.restapi.rest.base_api_test import do_real_request
from tribler.core.components.restapi.rest.rest_endpoint import HTTP_UNAUTHORIZED
from tribler.core.components.restapi.rest.rest_manager import ApiKeyMiddleware, RESTManager, error_middleware
from tribler.core.components.restapi.rest.root_endpoint import RootEndpoint
from tribler.core.components.restapi.rest.settings_endpoint import SettingsEndpoint
from tribler.core.config.tribler_config import TriblerConfig
from tribler.core.tests.tools.common import TESTS_DIR

@pytest.fixture(name='tribler_config')
def tribler_config_fixture():
    if False:
        return 10
    return TriblerConfig()

@pytest.fixture(name='api_port')
def api_port_fixture(free_port):
    if False:
        while True:
            i = 10
    return free_port

@pytest.fixture(name='rest_manager')
async def rest_manager_fixture(request, tribler_config, api_port, tmp_path):
    config = tribler_config
    api_key_marker = request.node.get_closest_marker('api_key')
    if api_key_marker is not None:
        tribler_config.api.key = api_key_marker.args[0]
    enable_https_marker = request.node.get_closest_marker('enable_https')
    if enable_https_marker:
        tribler_config.api.https_enabled = True
        tribler_config.api.https_port = api_port
        shutil.copy(TESTS_DIR / 'data' / 'certfile.pem', tmp_path)
        config.api.put_path_as_relative('https_certfile', TESTS_DIR / 'data' / 'certfile.pem', tmp_path)
    else:
        tribler_config.api.http_enabled = True
        tribler_config.api.http_port = api_port
    root_endpoint = RootEndpoint(middlewares=[ApiKeyMiddleware(config.api.key), error_middleware])
    root_endpoint.add_endpoint('/settings', SettingsEndpoint(config))
    rest_manager = RESTManager(config=config.api, root_endpoint=root_endpoint, state_dir=tmp_path)
    await rest_manager.start()
    yield rest_manager
    await rest_manager.stop()

@pytest.mark.enable_https
async def test_https(tribler_config, rest_manager, api_port):
    await do_real_request(api_port, f'https://localhost:{api_port}/settings')

@pytest.mark.api_key('')
async def test_api_key_disabled(rest_manager, api_port):
    await do_real_request(api_port, 'settings')
    await do_real_request(api_port, 'settings?apikey=111')
    await do_real_request(api_port, 'settings', headers={'X-Api-Key': '111'})

@pytest.mark.api_key('0' * 32)
async def test_api_key_success(rest_manager, api_port):
    api_key = rest_manager.config.key
    await do_real_request(api_port, 'settings?apikey=' + api_key)
    await do_real_request(api_port, 'settings', headers={'X-Api-Key': api_key})

@pytest.mark.api_key('0' * 32)
async def test_api_key_fail(rest_manager, api_port):
    await do_real_request(api_port, 'settings', expected_code=HTTP_UNAUTHORIZED, expected_json={'error': 'Unauthorized access'})
    await do_real_request(api_port, 'settings?apikey=111', expected_code=HTTP_UNAUTHORIZED, expected_json={'error': 'Unauthorized access'})
    await do_real_request(api_port, 'settings', headers={'X-Api-Key': '111'}, expected_code=HTTP_UNAUTHORIZED, expected_json={'error': 'Unauthorized access'})

async def test_unhandled_exception(rest_manager, api_port):
    """
    Testing whether the API returns a formatted 500 error and
    calls exception handler if an unhandled Exception is raised
    """
    with patch('tribler.core.components.restapi.rest.rest_manager.default_core_exception_handler') as handler:
        response_dict = await do_real_request(api_port, 'settings', expected_code=500, post_data={'general': 'invalid schema'}, request_type='POST')
        handler.unhandled_error_observer.assert_called_once()
        exception_dict = handler.unhandled_error_observer.call_args.args[1]
        assert exception_dict['should_stop'] is False
        assert isinstance(exception_dict['exception'], ValidationError)
    assert response_dict
    assert not response_dict['error']['handled']
    assert response_dict['error']['code'] == 'ValidationError'

async def test_patch_make_request():
    app = Mock()
    app._make_request.patched = None
    assert patch_make_request(app)
    make_request = app._make_request
    assert make_request.patched
    assert not patch_make_request(app)
    assert app._make_request is make_request

async def test_aiohttp_assertion_patched(rest_manager, api_port):
    original_start = RequestHandler.start

    async def new_start(self: RequestHandler) -> None:
        original_request_factory = self._request_factory
        assert getattr(original_request_factory, 'patched', False)

        def new_request_factory(*args):
            if False:
                print('Hello World!')
            self.force_close()
            return original_request_factory(*args)
        with patch.object(self, '_request_factory', new=new_request_factory):
            with pytest.raises(CancelledError):
                await original_start(self)
            raise CancelledError
    counter_before = get_transport_is_none_counter()
    with pytest.raises((ServerDisconnectedError, ClientOSError)):
        with patch('aiohttp.web_protocol.RequestHandler.start', new=new_start):
            await do_real_request(api_port, 'settings')
    counter_after = get_transport_is_none_counter()
    assert counter_before + 1 == counter_after