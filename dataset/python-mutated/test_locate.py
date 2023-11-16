"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.locate
"""
import pytest
import salt.modules.locate as locate
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {locate: {}}

def test_version():
    if False:
        print('Hello World!')
    '\n    Test if it returns the version of locate\n    '
    mock = MagicMock(return_value='mlocate 0.26')
    with patch.dict(locate.__salt__, {'cmd.run': mock}):
        assert locate.version() == ['mlocate 0.26']

def test_stats():
    if False:
        while True:
            i = 10
    '\n    Test if it returns statistics about the locate database\n    '
    ret = {'files': '75,253', 'directories': '49,252', 'bytes in file names': '93,214', 'bytes used to store database': '29,165', 'database': '/var/lib/mlocate/mlocate.db'}
    mock_ret = 'Database /var/lib/mlocate/mlocate.db:\n    49,252 directories\n    75,253 files\n    93,214 bytes in file names\n    29,165 bytes used to store database'
    with patch.dict(locate.__salt__, {'cmd.run': MagicMock(return_value=mock_ret)}):
        assert locate.stats() == ret

def test_updatedb():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it updates the locate database\n    '
    mock = MagicMock(return_value='')
    with patch.dict(locate.__salt__, {'cmd.run': mock}):
        assert locate.updatedb() == []

def test_locate():
    if False:
        while True:
            i = 10
    '\n    Test if it performs a file lookup.\n    '
    mock = MagicMock(return_value='')
    with patch.dict(locate.__salt__, {'cmd.run': mock}):
        assert locate.locate('wholename', database='myfile') == []