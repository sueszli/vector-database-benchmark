from __future__ import annotations
import pytest
pytest
import os
from unittest.mock import MagicMock, patch
import bokeh.io.util as biu

def test_detect_current_filename() -> None:
    if False:
        print('Hello World!')
    filename = biu.detect_current_filename()
    assert filename and filename.endswith(('py.test', 'pytest', 'py.test-script.py', 'pytest-script.py'))

def test_temp_filename() -> None:
    if False:
        return 10
    with patch('bokeh.io.util.NamedTemporaryFile', **{'return_value.__enter__.return_value.name': 'Junk.test'}) as mock_tmp:
        r = biu.temp_filename('test')
        assert r == 'Junk.test'
        assert mock_tmp.called
        assert mock_tmp.call_args[0] == ()
        assert mock_tmp.call_args[1] == {'suffix': '.test'}

def test_default_filename() -> None:
    if False:
        i = 10
        return i + 15
    old_detect_current_filename = biu.detect_current_filename
    old__no_access = biu._no_access
    old__shares_exec_prefix = biu._shares_exec_prefix
    biu.detect_current_filename = lambda : '/a/b/foo.py'
    try:
        with pytest.raises(RuntimeError):
            biu.default_filename('py')

        def FALSE(_: str) -> bool:
            if False:
                print('Hello World!')
            return False

        def TRUE(_: str) -> bool:
            if False:
                i = 10
                return i + 15
            return True
        biu._no_access = FALSE
        r = biu.default_filename('test')
        assert os.path.normpath(r) == os.path.normpath('/a/b/foo.test')
        biu._no_access = TRUE
        r = biu.default_filename('test')
        assert os.path.normpath(r) != os.path.normpath('/a/b/foo.test')
        assert r.endswith('.test')
        biu._no_access = FALSE
        biu._shares_exec_prefix = TRUE
        r = biu.default_filename('test')
        assert os.path.normpath(r) != os.path.normpath('/a/b/foo.test')
        assert r.endswith('.test')
        biu.detect_current_filename = lambda : None
        biu._no_access = FALSE
        biu._shares_exec_prefix = FALSE
        r = biu.default_filename('test')
        assert os.path.normpath(r) != os.path.normpath('/a/b/foo.test')
        assert r.endswith('.test')
    finally:
        biu.detect_current_filename = old_detect_current_filename
        biu._no_access = old__no_access
        biu._shares_exec_prefix = old__shares_exec_prefix

@patch('os.access')
def test__no_access(mock_access: MagicMock) -> None:
    if False:
        for i in range(10):
            print('nop')
    biu._no_access('test')
    assert mock_access.called
    assert mock_access.call_args[0] == ('test', os.W_OK | os.X_OK)
    assert mock_access.call_args[1] == {}

def test__shares_exec_prefix() -> None:
    if False:
        while True:
            i = 10
    import sys
    old_ex = sys.exec_prefix
    try:
        sys.exec_prefix = '/foo/bar'
        assert biu._shares_exec_prefix('/foo/bar') is True
        sys.exec_prefix = '/baz/bar'
        assert biu._shares_exec_prefix('/foo/bar') is False
        sys.exec_prefix = None
        assert biu._shares_exec_prefix('/foo/bar') is False
    finally:
        sys.exec_prefix = old_ex