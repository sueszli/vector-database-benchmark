"""The watch_app_state function enables us to trigger a callback function whenever the App state changes."""
import os
from unittest import mock
from lightning.app.core.constants import APP_SERVER_PORT
from lightning.app.frontend.panel.app_state_comm import _get_ws_url, _run_callbacks, _watch_app_state
FLOW_SUB = 'lit_flow'
FLOW = f'root.{FLOW_SUB}'

def do_nothing():
    if False:
        for i in range(10):
            print('nop')
    'Be lazy!'

def test_get_ws_url_when_local():
    if False:
        print('Hello World!')
    'The websocket uses port APP_SERVER_PORT when local.'
    assert _get_ws_url() == f'ws://localhost:{APP_SERVER_PORT}/api/v1/ws'

@mock.patch.dict(os.environ, {'LIGHTNING_APP_STATE_URL': 'some_url'})
def test_get_ws_url_when_cloud():
    if False:
        print('Hello World!')
    'The websocket uses port 8080 when LIGHTNING_APP_STATE_URL is set.'
    assert _get_ws_url() == 'ws://localhost:8080/api/v1/ws'

@mock.patch.dict(os.environ, {'LIGHTNING_FLOW_NAME': 'FLOW'})
def test_watch_app_state():
    if False:
        i = 10
        return i + 15
    'We can watch the App state and a callback function will be run when it changes.'
    callback = mock.MagicMock()
    _watch_app_state(callback)
    _run_callbacks()
    callback.assert_called_once()