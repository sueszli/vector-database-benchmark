from __future__ import annotations
import pytest
pytest
from pathlib import Path
from unittest.mock import mock_open, patch
from pytest import raises
from bokeh.models import Slider
from bokeh.models import CustomJS

def test_js_callback() -> None:
    if False:
        while True:
            i = 10
    slider = Slider()
    cb = CustomJS(code='foo();', args=dict(x=slider))
    assert 'foo()' in cb.code
    assert cb.args['x'] is slider
    cb = CustomJS(code='foo();', args=dict(x=3))
    assert 'foo()' in cb.code
    assert cb.args['x'] == 3
    with raises(AttributeError):
        CustomJS(code='foo();', x=slider)

def test_CustomJS_from_code_mjs() -> None:
    if False:
        print('Hello World!')
    slider = Slider()
    with patch('builtins.open', mock_open(read_data="export default () => 'ESM'")):
        cb = CustomJS.from_file(Path('some/module.mjs'), some='something', slider=slider)
    assert cb.module is True
    assert cb.code == "export default () => 'ESM'"
    assert cb.args == dict(some='something', slider=slider)

def test_CustomJS_from_code_js() -> None:
    if False:
        return 10
    slider = Slider()
    with patch('builtins.open', mock_open(read_data="return 'function'")):
        cb = CustomJS.from_file(Path('some/module.js'), some='something', slider=slider)
    assert cb.module is False
    assert cb.code == "return 'function'"
    assert cb.args == dict(some='something', slider=slider)

def test_CustomJS_from_code_bad_file_type() -> None:
    if False:
        while True:
            i = 10
    with pytest.raises(RuntimeError):
        CustomJS.from_file(Path('some/module.css'))