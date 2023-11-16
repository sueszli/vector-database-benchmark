import os
import re
import tempfile
import pytest
from sacred.config.config_files import HANDLER_BY_EXT, load_config_file
data = {'foo': 42, 'baz': [1, 0.2, 'bar', True, {'some_number': -12, 'simon': 'hugo'}]}

@pytest.mark.parametrize('handler', HANDLER_BY_EXT.values())
def test_save_and_load(handler):
    if False:
        for i in range(10):
            print('nop')
    with tempfile.TemporaryFile('w+' + handler.mode) as f:
        handler.dump(data, f)
        f.seek(0)
        d = handler.load(f)
        assert d == data

@pytest.mark.parametrize('ext, handler', HANDLER_BY_EXT.items())
def test_load_config_file(ext, handler):
    if False:
        while True:
            i = 10
    (handle, f_name) = tempfile.mkstemp(suffix=ext)
    f = os.fdopen(handle, 'w' + handler.mode)
    handler.dump(data, f)
    f.close()
    d = load_config_file(f_name)
    assert d == data
    os.remove(f_name)

def test_load_config_file_exception_msg_invalid_ext():
    if False:
        i = 10
        return i + 15
    (handle, f_name) = tempfile.mkstemp(suffix='.invalid')
    f = os.fdopen(handle, 'w')
    f.close()
    try:
        exception_msg = re.compile('Configuration file ".*.invalid" has invalid or unsupported extension ".invalid".')
        with pytest.raises(ValueError) as excinfo:
            load_config_file(f_name)
        assert exception_msg.match(excinfo.value.args[0])
    finally:
        os.remove(f_name)