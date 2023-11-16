"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>
"""
import pytest
import salt.modules.extfs as extfs
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        for i in range(10):
            print('nop')
    return {extfs: {}}

def test_mkfs():
    if False:
        print('Hello World!')
    '\n    Tests if a file system created on the specified device\n    '
    mock = MagicMock()
    with patch.dict(extfs.__salt__, {'cmd.run': mock}):
        assert [] == extfs.mkfs('/dev/sda1', 'ext4')

def test_tune():
    if False:
        i = 10
        return i + 15
    '\n    Tests if specified group was added\n    '
    mock = MagicMock()
    with patch.dict(extfs.__salt__, {'cmd.run': mock}), patch('salt.modules.extfs.tune', MagicMock(return_value='')):
        assert '' == extfs.tune('/dev/sda1')

def test_dump():
    if False:
        i = 10
        return i + 15
    '\n    Tests if specified group was added\n    '
    mock = MagicMock()
    with patch.dict(extfs.__salt__, {'cmd.run': mock}):
        assert {'attributes': {}, 'blocks': {}} == extfs.dump('/dev/sda1')

def test_attributes():
    if False:
        print('Hello World!')
    '\n    Tests if specified group was added\n    '
    with patch('salt.modules.extfs.dump', MagicMock(return_value={'attributes': {}, 'blocks': {}})):
        assert {} == extfs.attributes('/dev/sda1')

def test_blocks():
    if False:
        print('Hello World!')
    '\n    Tests if specified group was added\n    '
    with patch('salt.modules.extfs.dump', MagicMock(return_value={'attributes': {}, 'blocks': {}})):
        assert {} == extfs.blocks('/dev/sda1')