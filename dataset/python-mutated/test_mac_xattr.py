import pytest
import salt.modules.mac_xattr as xattr
import salt.utils.mac_utils
from salt.exceptions import CommandExecutionError
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {xattr: {}}

def test_list():
    if False:
        i = 10
        return i + 15
    '\n    Test xattr.list\n    '
    expected = {'spongebob': 'squarepants', 'squidward': 'patrick'}
    with patch.object(xattr, 'read', MagicMock(side_effect=['squarepants', 'patrick'])), patch('salt.utils.mac_utils.execute_return_result', MagicMock(return_value='spongebob\nsquidward')):
        assert xattr.list_('path/to/file') == expected

def test_list_missing():
    if False:
        return 10
    '\n    Test listing attributes of a missing file\n    '
    with patch('salt.utils.mac_utils.execute_return_result', MagicMock(side_effect=CommandExecutionError('No such file'))):
        pytest.raises(CommandExecutionError, xattr.list_, '/path/to/file')

def test_read():
    if False:
        i = 10
        return i + 15
    '\n    Test reading a specific attribute from a file\n    '
    with patch('salt.utils.mac_utils.execute_return_result', MagicMock(return_value='expected results')):
        assert xattr.read('/path/to/file', 'com.attr') == 'expected results'

def test_read_hex():
    if False:
        print('Hello World!')
    '\n    Test reading a specific attribute from a file\n    '
    with patch.object(salt.utils.mac_utils, 'execute_return_result', MagicMock(return_value='expected results')) as mock:
        assert xattr.read('/path/to/file', 'com.attr', **{'hex': True}) == 'expected results'
        mock.assert_called_once_with(['xattr', '-p', '-x', 'com.attr', '/path/to/file'])

def test_read_missing():
    if False:
        while True:
            i = 10
    '\n    Test reading a specific attribute from a file\n    '
    with patch('salt.utils.mac_utils.execute_return_result', MagicMock(side_effect=CommandExecutionError('No such file'))):
        pytest.raises(CommandExecutionError, xattr.read, '/path/to/file', 'attribute')

def test_read_not_decodeable():
    if False:
        return 10
    '\n    Test reading an attribute which returns non-UTF-8 bytes\n    '
    with patch('salt.utils.mac_utils.execute_return_result', MagicMock(side_effect=UnicodeDecodeError('UTF-8', b'\xd1expected results', 0, 1, ''))):
        assert xattr.read('/path/to/file', 'com.attr') == '�expected results'

def test_write():
    if False:
        i = 10
        return i + 15
    '\n    Test writing a specific attribute to a file\n    '
    mock_cmd = MagicMock(return_value='squarepants')
    with patch.object(xattr, 'read', mock_cmd), patch('salt.utils.mac_utils.execute_return_success', MagicMock(return_value=True)):
        assert xattr.write('/path/to/file', 'spongebob', 'squarepants')

def test_write_missing():
    if False:
        i = 10
        return i + 15
    '\n    Test writing a specific attribute to a file\n    '
    with patch('salt.utils.mac_utils.execute_return_success', MagicMock(side_effect=CommandExecutionError('No such file'))):
        pytest.raises(CommandExecutionError, xattr.write, '/path/to/file', 'attribute', 'value')

def test_delete():
    if False:
        while True:
            i = 10
    '\n    Test deleting a specific attribute from a file\n    '
    mock_cmd = MagicMock(return_value={'spongebob': 'squarepants'})
    with patch.object(xattr, 'list_', mock_cmd), patch('salt.utils.mac_utils.execute_return_success', MagicMock(return_value=True)):
        assert xattr.delete('/path/to/file', 'attribute')

def test_delete_missing():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test deleting a specific attribute from a file\n    '
    with patch('salt.utils.mac_utils.execute_return_success', MagicMock(side_effect=CommandExecutionError('No such file'))):
        pytest.raises(CommandExecutionError, xattr.delete, '/path/to/file', 'attribute')

def test_clear():
    if False:
        print('Hello World!')
    '\n    Test clearing all attributes on a file\n    '
    mock_cmd = MagicMock(return_value={})
    with patch.object(xattr, 'list_', mock_cmd), patch('salt.utils.mac_utils.execute_return_success', MagicMock(return_value=True)):
        assert xattr.clear('/path/to/file')

def test_clear_missing():
    if False:
        return 10
    '\n    Test clearing all attributes on a file\n    '
    with patch('salt.utils.mac_utils.execute_return_success', MagicMock(side_effect=CommandExecutionError('No such file'))):
        pytest.raises(CommandExecutionError, xattr.clear, '/path/to/file')