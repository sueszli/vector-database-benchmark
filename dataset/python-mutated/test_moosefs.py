"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>
"""
import pytest
import salt.modules.moosefs as moosefs
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        for i in range(10):
            print('nop')
    return {moosefs: {}}

def test_dirinfo():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it return information on a directory located on the Moose\n    '
    mock = MagicMock(return_value={'stdout': 'Salt:salt'})
    with patch.dict(moosefs.__salt__, {'cmd.run_all': mock}):
        assert moosefs.dirinfo('/tmp/salt') == {'Salt': 'salt'}

def test_fileinfo():
    if False:
        return 10
    '\n    Test if it returns information on a file located on the Moose\n    '
    mock = MagicMock(return_value={'stdout': ''})
    with patch.dict(moosefs.__salt__, {'cmd.run_all': mock}):
        assert moosefs.fileinfo('/tmp/salt') == {}

def test_mounts():
    if False:
        i = 10
        return i + 15
    '\n    Test if it returns a list of current MooseFS mounts\n    '
    mock = MagicMock(return_value={'stdout': ''})
    with patch.dict(moosefs.__salt__, {'cmd.run_all': mock}):
        assert moosefs.mounts() == {}

def test_getgoal():
    if False:
        return 10
    '\n    Test if it returns goal(s) for a file or directory\n    '
    mock = MagicMock(return_value={'stdout': 'Salt: salt'})
    with patch.dict(moosefs.__salt__, {'cmd.run_all': mock}):
        assert moosefs.getgoal('/tmp/salt') == {'goal': 'salt'}