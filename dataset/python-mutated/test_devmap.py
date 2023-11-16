"""
    :codeauthor: Rupesh Tare <rupesht@saltstack.com>
"""
import os.path
import pytest
import salt.modules.devmap as devmap
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        for i in range(10):
            print('nop')
    return {devmap: {}}

def test_multipath_list():
    if False:
        return 10
    '\n    Test for Device-Mapper Multipath list\n    '
    mock = MagicMock(return_value='A')
    with patch.dict(devmap.__salt__, {'cmd.run': mock}):
        assert devmap.multipath_list() == ['A']

def test_multipath_flush():
    if False:
        i = 10
        return i + 15
    '\n    Test for Device-Mapper Multipath flush\n    '
    mock = MagicMock(return_value=False)
    with patch.object(os.path, 'exists', mock):
        assert devmap.multipath_flush('device') == 'device does not exist'
    mock = MagicMock(return_value=True)
    with patch.object(os.path, 'exists', mock):
        mock = MagicMock(return_value='A')
        with patch.dict(devmap.__salt__, {'cmd.run': mock}):
            assert devmap.multipath_flush('device') == ['A']