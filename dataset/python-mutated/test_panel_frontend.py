"""The PanelFrontend wraps your Panel code in your LightningFlow."""
import os
import runpy
import sys
from unittest import mock
from unittest.mock import Mock
import pytest
from lightning.app import LightningFlow
from lightning.app.frontend.panel import PanelFrontend, panel_serve_render_fn
from lightning.app.frontend.panel.panel_frontend import _has_panel_autoreload
from lightning.app.utilities.state import AppState

@pytest.mark.skipif(True, reason='broken')
def test_stop_server_not_running():
    if False:
        i = 10
        return i + 15
    'If the server is not running but stopped an Exception should be raised.'
    frontend = PanelFrontend(entry_point=Mock())
    with pytest.raises(RuntimeError, match='Server is not running.'):
        frontend.stop_server()

def _noop_render_fn(_):
    if False:
        for i in range(10):
            print('nop')
    pass

class MockFlow(LightningFlow):
    """Test Flow."""

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        'Return name.'
        return 'root.my.flow'

    def run(self):
        if False:
            return 10
        'Be lazy!'

@mock.patch('lightning.app.frontend.panel.panel_frontend.subprocess')
@pytest.mark.skipif(True, reason='broken')
def test_panel_frontend_start_stop_server(subprocess_mock):
    if False:
        for i in range(10):
            print('nop')
    'Test that `PanelFrontend.start_server()` invokes subprocess.Popen with the right parameters.'
    frontend = PanelFrontend(entry_point=_noop_render_fn)
    frontend.flow = MockFlow()
    frontend.start_server(host='hostname', port=1111)
    subprocess_mock.Popen.assert_called_once()
    env_variables = subprocess_mock.method_calls[0].kwargs['env']
    call_args = subprocess_mock.method_calls[0].args[0]
    assert call_args == [sys.executable, '-m', 'panel', 'serve', panel_serve_render_fn.__file__, '--port', '1111', '--address', 'hostname', '--prefix', 'root.my.flow', '--allow-websocket-origin', '*']
    assert env_variables['LIGHTNING_FLOW_NAME'] == 'root.my.flow'
    assert env_variables['LIGHTNING_RENDER_ADDRESS'] == 'hostname'
    assert env_variables['LIGHTNING_RENDER_FUNCTION'] == '_noop_render_fn'
    assert env_variables['LIGHTNING_RENDER_MODULE_FILE'] == __file__
    assert env_variables['LIGHTNING_RENDER_PORT'] == '1111'
    assert 'LIGHTNING_FLOW_NAME' not in os.environ
    assert 'LIGHTNING_RENDER_FUNCTION' not in os.environ
    assert 'LIGHTNING_RENDER_MODULE_FILE' not in os.environ
    assert 'LIGHTNING_RENDER_MODULE_PORT' not in os.environ
    assert 'LIGHTNING_RENDER_MODULE_ADDRESS' not in os.environ
    frontend.stop_server()
    subprocess_mock.Popen().kill.assert_called_once()

def _call_me(state):
    if False:
        while True:
            i = 10
    assert isinstance(state, AppState)
    print(state)

@mock.patch.dict(os.environ, {'LIGHTNING_FLOW_NAME': 'root', 'LIGHTNING_RENDER_FUNCTION': '_call_me', 'LIGHTNING_RENDER_MODULE_FILE': __file__, 'LIGHTNING_RENDER_ADDRESS': '127.0.0.1', 'LIGHTNING_RENDER_PORT': '61896'})
def test_panel_wrapper_calls_entry_point(*_):
    if False:
        print('Hello World!')
    'Run the panel_serve_entry_point.'
    runpy.run_module('lightning.app.frontend.panel.panel_serve_render_fn')

@pytest.mark.skipif(True, reason='broken')
def test_method_exception():
    if False:
        while True:
            i = 10
    'The PanelFrontend does not support entry_point being a method and should raise an Exception.'

    class _DummyClass:

        def _render_fn(self):
            if False:
                i = 10
                return i + 15
            pass
    with pytest.raises(TypeError, match='being a method'):
        PanelFrontend(entry_point=_DummyClass()._render_fn)

@pytest.mark.skipif(True, reason='broken')
def test_open_close_log_files():
    if False:
        print('Hello World!')
    'We can open and close the log files.'
    frontend = PanelFrontend(_noop_render_fn)
    assert not frontend._log_files
    frontend._open_log_files()
    stdout = frontend._log_files['stdout']
    stderr = frontend._log_files['stderr']
    assert not stdout.closed
    assert not stderr.closed
    frontend._close_log_files()
    assert not frontend._log_files
    assert stdout.closed
    assert stderr.closed
    frontend._close_log_files()

@pytest.mark.parametrize(('value', 'expected'), [('Yes', True), ('yes', True), ('YES', True), ('Y', True), ('y', True), ('True', True), ('true', True), ('TRUE', True), ('No', False), ('no', False), ('NO', False), ('N', False), ('n', False), ('False', False), ('false', False), ('FALSE', False)])
def test_has_panel_autoreload(value, expected):
    if False:
        return 10
    'We can get and set autoreload using the environment variable PANEL_AUTORELOAD.'
    with mock.patch.dict(os.environ, {'PANEL_AUTORELOAD': value}):
        assert _has_panel_autoreload() == expected