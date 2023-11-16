"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    TestCase for salt.modules.znc
"""
import pytest
import salt.modules.znc as znc
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {znc: {}}

def test_buildmod():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests build module using znc-buildmod\n    '
    with patch('os.path.exists', MagicMock(return_value=False)):
        assert znc.buildmod('modules.cpp') == 'Error: The file (modules.cpp) does not exist.'

def test_buildmod_module():
    if False:
        print('Hello World!')
    '\n    Tests build module using znc-buildmod\n    '
    mock = MagicMock(return_value='SALT')
    with patch.dict(znc.__salt__, {'cmd.run': mock}), patch('os.path.exists', MagicMock(return_value=True)):
        assert znc.buildmod('modules.cpp') == 'SALT'

def test_dumpconf():
    if False:
        i = 10
        return i + 15
    '\n    Tests write the active configuration state to config file\n    '
    mock = MagicMock(return_value='SALT')
    with patch.dict(znc.__salt__, {'ps.pkill': mock}), patch.object(znc, 'signal', MagicMock()):
        assert znc.dumpconf() == 'SALT'

def test_rehashconf():
    if False:
        i = 10
        return i + 15
    '\n    Tests rehash the active configuration state from config file\n    '
    mock = MagicMock(return_value='SALT')
    with patch.dict(znc.__salt__, {'ps.pkill': mock}), patch.object(znc, 'signal', MagicMock()):
        assert znc.rehashconf() == 'SALT'

def test_version():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests return server version from znc --version\n    '
    mock = MagicMock(return_value='ZNC 1.2 - http://znc.in')
    with patch.dict(znc.__salt__, {'cmd.run': mock}):
        assert znc.version() == 'ZNC 1.2'