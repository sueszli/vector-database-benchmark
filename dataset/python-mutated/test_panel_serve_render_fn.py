"""The panel_serve_render_fn_or_file file gets run by Python to launch a Panel Server with Lightning.

These tests are for serving a render_fn function.

"""
import inspect
import os
from unittest import mock
import pytest
from lightning.app.frontend.panel.app_state_watcher import AppStateWatcher
from lightning.app.frontend.panel.panel_serve_render_fn import _get_render_fn, _get_render_fn_from_environment
from lightning_utilities.core.imports import RequirementCache
_PARAM_AVAILABLE = RequirementCache('param')

@pytest.fixture(autouse=True)
def _mock_settings_env_vars():
    if False:
        return 10
    with mock.patch.dict(os.environ, {'LIGHTNING_FLOW_NAME': 'root.lit_flow', 'LIGHTNING_RENDER_ADDRESS': 'localhost', 'LIGHTNING_RENDER_MODULE_FILE': __file__, 'LIGHTNING_RENDER_PORT': '61896'}):
        yield

def render_fn(app):
    if False:
        i = 10
        return i + 15
    'Test render_fn function with app args.'
    return app

@pytest.mark.skipif(not _PARAM_AVAILABLE, reason='requires param')
@mock.patch.dict(os.environ, {'LIGHTNING_RENDER_FUNCTION': 'render_fn'})
def test_get_view_fn_args():
    if False:
        for i in range(10):
            print('nop')
    'We have a helper get_view_fn function that create a function for our view.\n\n    If the render_fn provides an argument an AppStateWatcher is provided as argument\n\n    '
    result = _get_render_fn()
    assert isinstance(result(), AppStateWatcher)

def render_fn_no_args():
    if False:
        while True:
            i = 10
    'Test function with no arguments.'
    return 'no_args'

@mock.patch.dict(os.environ, {'LIGHTNING_RENDER_FUNCTION': 'render_fn_no_args'})
def test_get_view_fn_no_args():
    if False:
        while True:
            i = 10
    'We have a helper get_view_fn function that create a function for our view.\n\n    If the render_fn provides an argument an AppStateWatcher is provided as argument\n\n    '
    result = _get_render_fn()
    assert result() == 'no_args'

def render_fn_2():
    if False:
        i = 10
        return i + 15
    'Do nothing.'

def test_get_render_fn_from_environment():
    if False:
        return 10
    'We have a method to get the render_fn from the environment.'
    result = _get_render_fn_from_environment('render_fn_2', __file__)
    assert result.__name__ == render_fn_2.__name__
    assert inspect.getmodule(result).__file__ == __file__