from __future__ import annotations
import pytest
pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from bokeh.core.templates import FILE
from bokeh.io.state import curstate
from bokeh.models import Plot
from bokeh.resources import INLINE
import bokeh.io.saving as bis

def test__get_save_args_explicit_filename() -> None:
    if False:
        for i in range(10):
            print('nop')
    (filename, _, _) = bis._get_save_args(curstate(), 'filename', 'inline', 'title')
    assert filename == 'filename'
    (filename, _, _) = bis._get_save_args(curstate(), Path('some') / 'path' / 'filename', 'inline', 'title')
    assert filename == Path('some') / 'path' / 'filename'

def test__get_save_args_default_filename() -> None:
    if False:
        while True:
            i = 10
    curstate().reset()
    curstate().output_file('filename')
    (filename, _, _) = bis._get_save_args(curstate(), None, 'inline', 'title')
    assert filename == 'filename'

def test__get_save_args_explicit_resources() -> None:
    if False:
        print('Hello World!')
    (_, resources, _) = bis._get_save_args(curstate(), 'filename', 'inline', 'title')
    assert resources.mode == 'inline'
    (_, resources, _) = bis._get_save_args(curstate(), 'filename', INLINE, 'title')
    assert resources == INLINE

def test__get_save_args_default_resources() -> None:
    if False:
        print('Hello World!')
    state = curstate()
    state.reset()
    state.output_file('filename', mode='inline')
    assert state.file is not None
    assert state.file.resources.mode == 'inline'
    r = state.file.resources
    (_, resources, _) = bis._get_save_args(curstate(), 'filename', None, 'title')
    assert resources == r

@patch('bokeh.io.saving.warn')
def test__get_save_args_missing_resources(mock_warn: MagicMock) -> None:
    if False:
        i = 10
        return i + 15
    curstate().reset()
    (_, resources, _) = bis._get_save_args(curstate(), 'filename', None, 'title')
    assert resources.mode == 'cdn'
    assert mock_warn.call_count == 1
    assert mock_warn.call_args[0] == ('save() called but no resources were supplied and output_file(...) was never called, defaulting to resources.CDN',)
    assert mock_warn.call_args[1] == {}

def test__get_save_args_explicit_title() -> None:
    if False:
        print('Hello World!')
    (_, _, title) = bis._get_save_args(curstate(), 'filename', 'inline', 'title')
    assert title == 'title'

def test__get_save_args_default_title() -> None:
    if False:
        i = 10
        return i + 15
    state = curstate()
    state.reset()
    state.output_file('filename', title='title')
    assert state.file is not None
    assert state.file.title == 'title'
    (_, _, title) = bis._get_save_args(curstate(), 'filename', 'inline', None)
    assert title == 'title'

@patch('bokeh.io.saving.warn')
def test__get_save_args_missing_title(mock_warn: MagicMock) -> None:
    if False:
        i = 10
        return i + 15
    curstate().reset()
    (_, _, title) = bis._get_save_args(curstate(), 'filename', 'inline', None)
    assert title == 'Bokeh Plot'
    assert mock_warn.call_count == 1
    assert mock_warn.call_args[0] == ("save() called but no title was supplied and output_file(...) was never called, using default title 'Bokeh Plot'",)
    assert mock_warn.call_args[1] == {}

@patch('builtins.open')
@patch('bokeh.embed.file_html')
def test__save_helper(mock_file_html: MagicMock, mock_open: MagicMock) -> None:
    if False:
        while True:
            i = 10
    obj = Plot()
    (filename, resources, title) = bis._get_save_args(curstate(), 'filename', 'inline', 'title')
    mock_open.reset_mock()
    bis._save_helper(obj, filename, resources, title, None)
    assert mock_file_html.call_count == 1
    assert mock_file_html.call_args[0] == (obj,)
    assert mock_file_html.call_args[1] == dict(resources=resources, title='title', template=FILE, theme=None)
    assert mock_open.call_count == 1
    assert mock_open.call_args[0] == (filename,)
    assert mock_open.call_args[1] == dict(mode='w', encoding='utf-8')