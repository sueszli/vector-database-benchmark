from __future__ import annotations
import pytest
pytest
import os
import tempfile
from unittest.mock import MagicMock, patch
from tests.support.util.types import Capture
import bokeh.command.util as util

def test_die(capsys: Capture) -> None:
    if False:
        while True:
            i = 10
    with pytest.raises(SystemExit):
        util.die('foo')
    (out, err) = capsys.readouterr()
    assert err == 'foo\n'
    assert out == ''

def test_build_single_handler_application_unknown_file() -> None:
    if False:
        print('Hello World!')
    with tempfile.NamedTemporaryFile(suffix='.bad') as f:
        with pytest.raises(ValueError) as e:
            util.build_single_handler_application(f.name)
    assert "Expected a '.py' script or '.ipynb' notebook, got: " in str(e.value)

def test_build_single_handler_application_nonexistent_file() -> None:
    if False:
        print('Hello World!')
    with pytest.raises(ValueError) as e:
        util.build_single_handler_application('junkjunkjunk')
    assert 'Path for Bokeh server application does not exist: ' in str(e.value)
DIRSTYLE_MAIN_WARNING_COPY = '\nIt looks like you might be running the main.py of a directory app directly.\nIf this is the case, to enable the features of directory style apps, you must\ncall "bokeh serve" on the directory instead. For example:\n\n    bokeh serve my_app_dir/\n\nIf this is not the case, renaming main.py will suppress this warning.\n'

@patch('warnings.warn')
def test_build_single_handler_application_main_py(mock_warn: MagicMock) -> None:
    if False:
        i = 10
        return i + 15
    f = tempfile.NamedTemporaryFile(suffix='main.py', delete=False)
    f.close()
    util.build_single_handler_application(f.name)
    assert mock_warn.called
    assert mock_warn.call_args[0] == (DIRSTYLE_MAIN_WARNING_COPY, None)
    assert mock_warn.call_args[1] == {'stacklevel': 3}
    os.remove(f.name)