import pytest
from saltfactories.utils import random_string
import salt.utils.stringutils
import salt.utils.win_reg as win_reg
from salt.exceptions import CommandExecutionError
from tests.support.mock import MagicMock, patch
try:
    import win32api
except ImportError:
    pass
pytestmark = [pytest.mark.windows_whitelisted, pytest.mark.skip_unless_on_windows]

@pytest.fixture
def fake_key():
    if False:
        return 10
    return 'SOFTWARE\\{}'.format(random_string('SaltTesting-', lowercase=False))

@pytest.fixture
def unicode_key():
    if False:
        while True:
            i = 10
    return 'Unicode Key ™'

@pytest.fixture
def unicode_value():
    if False:
        return 10
    return 'Unicode Value ©,™,®'

def test_broadcast_change_success():
    if False:
        i = 10
        return i + 15
    '\n    Tests the broadcast_change function\n    '
    with patch('win32gui.SendMessageTimeout', return_value=('', 0)):
        assert win_reg.broadcast_change()

def test_broadcast_change_fail():
    if False:
        i = 10
        return i + 15
    '\n    Tests the broadcast_change function failure\n    '
    with patch('win32gui.SendMessageTimeout', return_value=('', 1)):
        assert not win_reg.broadcast_change()

def test_key_exists_existing():
    if False:
        while True:
            i = 10
    '\n    Tests the key_exists function using a well known registry key\n    '
    assert win_reg.key_exists(hive='HKLM', key='SOFTWARE\\Microsoft')

def test_key_exists_non_existing():
    if False:
        i = 10
        return i + 15
    '\n    Tests the key_exists function using a non existing registry key\n    '
    assert not win_reg.key_exists(hive='HKLM', key='SOFTWARE\\FakeKey')

def test_key_exists_invalid_hive():
    if False:
        while True:
            i = 10
    '\n    Tests the key_exists function using an invalid hive\n    '
    pytest.raises(CommandExecutionError, win_reg.key_exists, hive='BADHIVE', key='SOFTWARE\\Microsoft')

def test_key_exists_unknown_key_error():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests the key_exists function with an unknown key error\n    '
    mock_error = MagicMock(side_effect=win32api.error(123, 'RegOpenKeyEx', 'Unknown error'))
    with patch('salt.utils.win_reg.win32api.RegOpenKeyEx', mock_error):
        pytest.raises(win32api.error, win_reg.key_exists, hive='HKLM', key='SOFTWARE\\Microsoft')

def test_value_exists_existing():
    if False:
        i = 10
        return i + 15
    '\n    Tests the value_exists function using a well known registry key\n    '
    assert win_reg.value_exists(hive='HKLM', key='SOFTWARE\\Microsoft\\Windows\\CurrentVersion', vname='CommonFilesDir')

def test_value_exists_non_existing():
    if False:
        return 10
    '\n    Tests the value_exists function using a non existing registry key\n    '
    assert not win_reg.value_exists(hive='HKLM', key='SOFTWARE\\Microsoft\\Windows\\CurrentVersion', vname='NonExistingValueName')

def test_value_exists_invalid_hive():
    if False:
        while True:
            i = 10
    '\n    Tests the value_exists function using an invalid hive\n    '
    pytest.raises(CommandExecutionError, win_reg.value_exists, hive='BADHIVE', key='SOFTWARE\\Microsoft\\Windows\\CurrentVersion', vname='CommonFilesDir')

def test_value_exists_key_not_exist():
    if False:
        while True:
            i = 10
    '\n    Tests the value_exists function when the key does not exist\n    '
    mock_error = MagicMock(side_effect=win32api.error(2, 'RegOpenKeyEx', 'Unknown error'))
    with patch('salt.utils.win_reg.win32api.RegOpenKeyEx', mock_error):
        assert not win_reg.value_exists(hive='HKLM', key='SOFTWARE\\Microsoft\\Windows\\CurrentVersion', vname='CommonFilesDir')

def test_value_exists_unknown_key_error():
    if False:
        while True:
            i = 10
    '\n    Tests the value_exists function with an unknown error when opening the\n    key\n    '
    mock_error = MagicMock(side_effect=win32api.error(123, 'RegOpenKeyEx', 'Unknown error'))
    with patch('salt.utils.win_reg.win32api.RegOpenKeyEx', mock_error):
        pytest.raises(win32api.error, win_reg.value_exists, hive='HKLM', key='SOFTWARE\\Microsoft\\Windows\\CurrentVersion', vname='CommonFilesDir')

def test_value_exists_empty_default_value():
    if False:
        print('Hello World!')
    '\n    Tests the value_exists function when querying the default value\n    '
    mock_error = MagicMock(side_effect=win32api.error(2, 'RegQueryValueEx', 'Empty Value'))
    with patch('salt.utils.win_reg.win32api.RegQueryValueEx', mock_error):
        assert win_reg.value_exists(hive='HKLM', key='SOFTWARE\\Microsoft\\Windows\\CurrentVersion', vname=None)

def test_value_exists_no_vname():
    if False:
        return 10
    '\n    Tests the value_exists function when the vname does not exist\n    '
    mock_error = MagicMock(side_effect=win32api.error(123, 'RegQueryValueEx', 'Empty Value'))
    with patch('salt.utils.win_reg.win32api.RegQueryValueEx', mock_error):
        assert not win_reg.value_exists(hive='HKLM', key='SOFTWARE\\Microsoft\\Windows\\CurrentVersion', vname='NonExistingValuePair')

def test_list_keys_existing():
    if False:
        i = 10
        return i + 15
    '\n    Test the list_keys function using a well known registry key\n    '
    assert 'Microsoft' in win_reg.list_keys(hive='HKLM', key='SOFTWARE')

def test_list_keys_non_existing(fake_key):
    if False:
        i = 10
        return i + 15
    '\n    Test the list_keys function using a non existing registry key\n    '
    expected = (False, 'Cannot find key: HKLM\\{}'.format(fake_key))
    assert win_reg.list_keys(hive='HKLM', key=fake_key) == expected

def test_list_keys_access_denied(fake_key):
    if False:
        return 10
    '\n    Test the list_keys function using a registry key when access is denied\n    '
    expected = (False, 'Access is denied: HKLM\\{}'.format(fake_key))
    mock_error = MagicMock(side_effect=win32api.error(5, 'RegOpenKeyEx', 'Access is denied'))
    with patch('salt.utils.win_reg.win32api.RegOpenKeyEx', mock_error):
        assert win_reg.list_keys(hive='HKLM', key=fake_key) == expected

def test_list_keys_invalid_hive():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the list_keys function when passing an invalid hive\n    '
    pytest.raises(CommandExecutionError, win_reg.list_keys, hive='BADHIVE', key='SOFTWARE\\Microsoft')

def test_list_keys_unknown_key_error():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests the list_keys function with an unknown key error\n    '
    mock_error = MagicMock(side_effect=win32api.error(123, 'RegOpenKeyEx', 'Unknown error'))
    with patch('salt.utils.win_reg.win32api.RegOpenKeyEx', mock_error):
        pytest.raises(win32api.error, win_reg.list_keys, hive='HKLM', key='SOFTWARE\\Microsoft')

def test_list_values_existing():
    if False:
        while True:
            i = 10
    '\n    Test the list_values function using a well known registry key\n    '
    values = win_reg.list_values(hive='HKLM', key='SOFTWARE\\Microsoft\\Windows\\CurrentVersion')
    keys = []
    for value in values:
        keys.append(value['vname'])
    assert 'ProgramFilesDir' in keys

def test_list_values_non_existing(fake_key):
    if False:
        print('Hello World!')
    '\n    Test the list_values function using a non existing registry key\n    '
    expected = (False, 'Cannot find key: HKLM\\{}'.format(fake_key))
    assert win_reg.list_values(hive='HKLM', key=fake_key) == expected

def test_list_values_access_denied(fake_key):
    if False:
        while True:
            i = 10
    '\n    Test the list_values function using a registry key when access is denied\n    '
    expected = (False, 'Access is denied: HKLM\\{}'.format(fake_key))
    mock_error = MagicMock(side_effect=win32api.error(5, 'RegOpenKeyEx', 'Access is denied'))
    with patch('salt.utils.win_reg.win32api.RegOpenKeyEx', mock_error):
        assert win_reg.list_values(hive='HKLM', key=fake_key) == expected

def test_list_values_invalid_hive():
    if False:
        print('Hello World!')
    '\n    Test the list_values function when passing an invalid hive\n    '
    pytest.raises(CommandExecutionError, win_reg.list_values, hive='BADHIVE', key='SOFTWARE\\Microsoft')

def test_list_values_unknown_key_error():
    if False:
        i = 10
        return i + 15
    '\n    Tests the list_values function with an unknown key error\n    '
    mock_error = MagicMock(side_effect=win32api.error(123, 'RegOpenKeyEx', 'Unknown error'))
    with patch('salt.utils.win_reg.win32api.RegOpenKeyEx', mock_error):
        pytest.raises(win32api.error, win_reg.list_values, hive='HKLM', key='SOFTWARE\\Microsoft')

def test_read_value_existing():
    if False:
        i = 10
        return i + 15
    '\n    Test the read_value function using a well known registry value\n    '
    ret = win_reg.read_value(hive='HKLM', key='SOFTWARE\\Microsoft\\Windows\\CurrentVersion', vname='ProgramFilesPath')
    assert ret['vdata'] == '%ProgramFiles%'

def test_read_value_default():
    if False:
        print('Hello World!')
    '\n    Test the read_value function reading the default value using a well\n    known registry key\n    '
    ret = win_reg.read_value(hive='HKLM', key='SOFTWARE\\Microsoft\\Windows\\CurrentVersion')
    assert ret['vdata'] == '(value not set)'

def test_read_value_non_existing():
    if False:
        while True:
            i = 10
    '\n    Test the read_value function using a non existing value pair\n    '
    expected = {'comment': 'Cannot find fake_name in HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion', 'vdata': None, 'vtype': None, 'vname': 'fake_name', 'success': False, 'hive': 'HKLM', 'key': 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion'}
    assert win_reg.read_value(hive='HKLM', key='SOFTWARE\\Microsoft\\Windows\\CurrentVersion', vname='fake_name') == expected

def test_read_value_non_existing_key(fake_key):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the read_value function using a non existing registry key\n    '
    expected = {'comment': 'Cannot find key: HKLM\\{}'.format(fake_key), 'vdata': None, 'vtype': None, 'vname': 'fake_name', 'success': False, 'hive': 'HKLM', 'key': fake_key}
    assert win_reg.read_value(hive='HKLM', key=fake_key, vname='fake_name') == expected

def test_read_value_access_denied(fake_key):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the read_value function using a registry key when access is denied\n    '
    expected = {'comment': 'Access is denied: HKLM\\{}'.format(fake_key), 'vdata': None, 'vtype': None, 'vname': 'fake_name', 'success': False, 'hive': 'HKLM', 'key': fake_key}
    mock_error = MagicMock(side_effect=win32api.error(5, 'RegOpenKeyEx', 'Access is denied'))
    with patch('salt.utils.win_reg.win32api.RegOpenKeyEx', mock_error):
        assert win_reg.read_value(hive='HKLM', key=fake_key, vname='fake_name') == expected

def test_read_value_invalid_hive():
    if False:
        i = 10
        return i + 15
    '\n    Test the read_value function when passing an invalid hive\n    '
    pytest.raises(CommandExecutionError, win_reg.read_value, hive='BADHIVE', key='SOFTWARE\\Microsoft', vname='ProgramFilesPath')

def test_read_value_unknown_key_error():
    if False:
        return 10
    '\n    Tests the read_value function with an unknown key error\n    '
    mock_error = MagicMock(side_effect=win32api.error(123, 'RegOpenKeyEx', 'Unknown error'))
    with patch('salt.utils.win_reg.win32api.RegOpenKeyEx', mock_error):
        pytest.raises(win32api.error, win_reg.read_value, hive='HKLM', key='SOFTWARE\\Microsoft\\Windows\\CurrentVersion', vname='ProgramFilesPath')

def test_read_value_unknown_value_error():
    if False:
        while True:
            i = 10
    '\n    Tests the read_value function with an unknown value error\n    '
    mock_error = MagicMock(side_effect=win32api.error(123, 'RegQueryValueEx', 'Unknown error'))
    with patch('salt.utils.win_reg.win32api.RegQueryValueEx', mock_error):
        pytest.raises(win32api.error, win_reg.read_value, hive='HKLM', key='SOFTWARE\\Microsoft\\Windows\\CurrentVersion', vname='ProgramFilesPath')

@pytest.mark.destructive_test
def test_read_value_multi_sz_empty_list(fake_key):
    if False:
        i = 10
        return i + 15
    '\n    An empty REG_MULTI_SZ value should return an empty list, not None\n    '
    try:
        assert win_reg.set_value(hive='HKLM', key=fake_key, vname='empty_list', vdata=[], vtype='REG_MULTI_SZ')
        expected = {'hive': 'HKLM', 'key': fake_key, 'success': True, 'vdata': [], 'vname': 'empty_list', 'vtype': 'REG_MULTI_SZ'}
        assert win_reg.read_value(hive='HKLM', key=fake_key, vname='empty_list') == expected
    finally:
        win_reg.delete_key_recursive(hive='HKLM', key=fake_key)

@pytest.mark.destructive_test
def test_set_value(fake_key):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the set_value function\n    '
    try:
        assert win_reg.set_value(hive='HKLM', key=fake_key, vname='fake_name', vdata='fake_data')
        expected = {'hive': 'HKLM', 'key': fake_key, 'success': True, 'vdata': 'fake_data', 'vname': 'fake_name', 'vtype': 'REG_SZ'}
        assert win_reg.read_value(hive='HKLM', key=fake_key, vname='fake_name') == expected
    finally:
        win_reg.delete_key_recursive(hive='HKLM', key=fake_key)

@pytest.mark.destructive_test
def test_set_value_default(fake_key):
    if False:
        return 10
    '\n    Test the set_value function on the default value\n    '
    try:
        assert win_reg.set_value(hive='HKLM', key=fake_key, vdata='fake_default_data')
        expected = {'hive': 'HKLM', 'key': fake_key, 'success': True, 'vdata': 'fake_default_data', 'vname': '(Default)', 'vtype': 'REG_SZ'}
        assert win_reg.read_value(hive='HKLM', key=fake_key) == expected
    finally:
        win_reg.delete_key_recursive(hive='HKLM', key=fake_key)

@pytest.mark.destructive_test
def test_set_value_unicode_key(fake_key, unicode_key):
    if False:
        print('Hello World!')
    '\n    Test the set_value function on a unicode key\n    '
    try:
        assert win_reg.set_value(hive='HKLM', key='\\'.join([fake_key, unicode_key]), vname='fake_name', vdata='fake_value')
        expected = {'hive': 'HKLM', 'key': '\\'.join([fake_key, unicode_key]), 'success': True, 'vdata': 'fake_value', 'vname': 'fake_name', 'vtype': 'REG_SZ'}
        assert win_reg.read_value(hive='HKLM', key='\\'.join([fake_key, unicode_key]), vname='fake_name') == expected
    finally:
        win_reg.delete_key_recursive(hive='HKLM', key=fake_key)

@pytest.mark.destructive_test
def test_set_value_unicode_value(fake_key, unicode_value):
    if False:
        print('Hello World!')
    '\n    Test the set_value function on a unicode value\n    '
    try:
        assert win_reg.set_value(hive='HKLM', key=fake_key, vname='fake_unicode', vdata=unicode_value)
        expected = {'hive': 'HKLM', 'key': fake_key, 'success': True, 'vdata': unicode_value, 'vname': 'fake_unicode', 'vtype': 'REG_SZ'}
        assert win_reg.read_value(hive='HKLM', key=fake_key, vname='fake_unicode') == expected
    finally:
        win_reg.delete_key_recursive(hive='HKLM', key=fake_key)

@pytest.mark.destructive_test
def test_set_value_reg_dword(fake_key):
    if False:
        print('Hello World!')
    '\n    Test the set_value function on a REG_DWORD value\n    '
    try:
        assert win_reg.set_value(hive='HKLM', key=fake_key, vname='dword_value', vdata=123, vtype='REG_DWORD')
        expected = {'hive': 'HKLM', 'key': fake_key, 'success': True, 'vdata': 123, 'vname': 'dword_value', 'vtype': 'REG_DWORD'}
        assert win_reg.read_value(hive='HKLM', key=fake_key, vname='dword_value') == expected
    finally:
        win_reg.delete_key_recursive(hive='HKLM', key=fake_key)

@pytest.mark.destructive_test
def test_set_value_reg_qword(fake_key):
    if False:
        print('Hello World!')
    '\n    Test the set_value function on a REG_QWORD value\n    '
    try:
        assert win_reg.set_value(hive='HKLM', key=fake_key, vname='qword_value', vdata=123, vtype='REG_QWORD')
        expected = {'hive': 'HKLM', 'key': fake_key, 'success': True, 'vdata': 123, 'vname': 'qword_value', 'vtype': 'REG_QWORD'}
        assert win_reg.read_value(hive='HKLM', key=fake_key, vname='qword_value') == expected
    finally:
        win_reg.delete_key_recursive(hive='HKLM', key=fake_key)

def test_set_value_invalid_hive(fake_key):
    if False:
        print('Hello World!')
    '\n    Test the set_value function when passing an invalid hive\n    '
    pytest.raises(CommandExecutionError, win_reg.set_value, hive='BADHIVE', key=fake_key, vname='fake_name', vdata='fake_data')

def test_set_value_open_create_failure(fake_key):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the set_value function when there is a problem opening/creating\n    the key\n    '
    mock_error = MagicMock(side_effect=win32api.error(123, 'RegCreateKeyEx', 'Unknown error'))
    with patch('salt.utils.win_reg.win32api.RegCreateKeyEx', mock_error):
        assert not win_reg.set_value(hive='HKLM', key=fake_key, vname='fake_name', vdata='fake_data')

def test_set_value_type_error(fake_key):
    if False:
        print('Hello World!')
    '\n    Test the set_value function when the wrong type of data is passed\n    '
    mock_error = MagicMock(side_effect=TypeError('Mocked TypeError'))
    with patch('salt.utils.win_reg.win32api.RegSetValueEx', mock_error):
        assert not win_reg.set_value(hive='HKLM', key=fake_key, vname='fake_name', vdata='fake_data')

def test_set_value_system_error(fake_key):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the set_value function when a SystemError occurs while setting the\n    value\n    '
    mock_error = MagicMock(side_effect=SystemError('Mocked SystemError'))
    with patch('salt.utils.win_reg.win32api.RegSetValueEx', mock_error):
        assert not win_reg.set_value(hive='HKLM', key=fake_key, vname='fake_name', vdata='fake_data')

def test_set_value_value_error(fake_key):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the set_value function when a ValueError occurs while setting the\n    value\n    '
    mock_error = MagicMock(side_effect=ValueError('Mocked ValueError'))
    with patch('salt.utils.win_reg.win32api.RegSetValueEx', mock_error):
        assert not win_reg.set_value(hive='HKLM', key=fake_key, vname='fake_name', vdata='fake_data')

def test_cast_vdata_reg_binary():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the cast_vdata function with REG_BINARY\n    Should always return binary data\n    '
    vdata = salt.utils.stringutils.to_bytes('test data')
    result = win_reg.cast_vdata(vdata=vdata, vtype='REG_BINARY')
    assert isinstance(result, bytes)
    vdata = salt.utils.stringutils.to_str('test data')
    result = win_reg.cast_vdata(vdata=vdata, vtype='REG_BINARY')
    assert isinstance(result, bytes)
    vdata = salt.utils.stringutils.to_unicode('test data')
    result = win_reg.cast_vdata(vdata=vdata, vtype='REG_BINARY')
    assert isinstance(result, bytes)

def test_cast_vdata_reg_dword():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the cast_vdata function with REG_DWORD\n    Should always return integer\n    '
    vdata = 1
    expected = 1
    result = win_reg.cast_vdata(vdata=vdata, vtype='REG_DWORD')
    assert result == expected
    vdata = '1'
    result = win_reg.cast_vdata(vdata=vdata, vtype='REG_DWORD')
    assert result == expected
    vdata = '0000001'
    result = win_reg.cast_vdata(vdata=vdata, vtype='REG_DWORD')
    assert result == expected

def test_cast_vdata_reg_expand_sz():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the cast_vdata function with REG_EXPAND_SZ\n    Should always return unicode\n    '
    vdata = salt.utils.stringutils.to_str('test data')
    result = win_reg.cast_vdata(vdata=vdata, vtype='REG_EXPAND_SZ')
    assert isinstance(result, str)
    vdata = salt.utils.stringutils.to_bytes('test data')
    result = win_reg.cast_vdata(vdata=vdata, vtype='REG_EXPAND_SZ')
    assert isinstance(result, str)

def test_cast_vdata_reg_multi_sz():
    if False:
        return 10
    '\n    Test the cast_vdata function with REG_MULTI_SZ\n    Should always return a list of unicode strings\n    '
    vdata = [salt.utils.stringutils.to_str('test string'), salt.utils.stringutils.to_bytes('test bytes')]
    result = win_reg.cast_vdata(vdata=vdata, vtype='REG_MULTI_SZ')
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, str)

def test_cast_vdata_reg_qword():
    if False:
        print('Hello World!')
    '\n    Test the cast_vdata function with REG_QWORD\n    Should always return a long integer\n    `int` is `long` is default on Py3\n    '
    vdata = 1
    result = win_reg.cast_vdata(vdata=vdata, vtype='REG_QWORD')
    assert isinstance(result, int)
    vdata = '1'
    result = win_reg.cast_vdata(vdata=vdata, vtype='REG_QWORD')
    assert isinstance(result, int)

def test_cast_vdata_reg_sz():
    if False:
        while True:
            i = 10
    '\n    Test the cast_vdata function with REG_SZ\n    Should always return unicode\n    '
    vdata = salt.utils.stringutils.to_str('test data')
    result = win_reg.cast_vdata(vdata=vdata, vtype='REG_SZ')
    assert isinstance(result, str)
    vdata = salt.utils.stringutils.to_bytes('test data')
    result = win_reg.cast_vdata(vdata=vdata, vtype='REG_SZ')
    assert isinstance(result, str)
    vdata = None
    result = win_reg.cast_vdata(vdata=vdata, vtype='REG_SZ')
    assert isinstance(result, str)
    assert result == ''

@pytest.mark.destructive_test
def test_delete_value(fake_key):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the delete_value function\n    '
    try:
        assert win_reg.set_value(hive='HKLM', key=fake_key, vname='fake_name', vdata='fake_data')
        assert win_reg.delete_value(hive='HKLM', key=fake_key, vname='fake_name')
        assert not win_reg.value_exists(hive='HKLM', key=fake_key, vname='fake_name')
    finally:
        win_reg.delete_key_recursive(hive='HKLM', key=fake_key)

def test_delete_value_non_existing(fake_key):
    if False:
        print('Hello World!')
    '\n    Test the delete_value function on non existing value\n    '
    mock_error = MagicMock(side_effect=win32api.error(2, 'RegOpenKeyEx', 'Unknown error'))
    with patch('salt.utils.win_reg.win32api.RegOpenKeyEx', mock_error):
        assert win_reg.delete_value(hive='HKLM', key=fake_key, vname='fake_name') is None

def test_delete_value_invalid_hive(fake_key):
    if False:
        i = 10
        return i + 15
    '\n    Test the delete_value function when passing an invalid hive\n    '
    pytest.raises(CommandExecutionError, win_reg.delete_value, hive='BADHIVE', key=fake_key, vname='fake_name')

def test_delete_value_unknown_error(fake_key):
    if False:
        i = 10
        return i + 15
    '\n    Test the delete_value function when there is a problem opening the key\n    '
    mock_error = MagicMock(side_effect=win32api.error(123, 'RegOpenKeyEx', 'Unknown error'))
    with patch('salt.utils.win_reg.win32api.RegOpenKeyEx', mock_error):
        pytest.raises(win32api.error, win_reg.delete_value, hive='HKLM', key=fake_key, vname='fake_name')

@pytest.mark.destructive_test
def test_delete_value_unicode(fake_key, unicode_value):
    if False:
        return 10
    '\n    Test the delete_value function on a unicode value\n    '
    try:
        assert win_reg.set_value(hive='HKLM', key=fake_key, vname='fake_unicode', vdata=unicode_value)
        assert win_reg.delete_value(hive='HKLM', key=fake_key, vname='fake_unicode')
        assert not win_reg.value_exists(hive='HKLM', key=fake_key, vname='fake_unicode')
    finally:
        win_reg.delete_key_recursive(hive='HKLM', key=fake_key)

@pytest.mark.destructive_test
def test_delete_value_unicode_vname(fake_key, unicode_key):
    if False:
        print('Hello World!')
    '\n    Test the delete_value function on a unicode vname\n    '
    try:
        assert win_reg.set_value(hive='HKLM', key=fake_key, vname=unicode_key, vdata='junk data')
        assert win_reg.delete_value(hive='HKLM', key=fake_key, vname=unicode_key)
        assert not win_reg.value_exists(hive='HKLM', key=fake_key, vname=unicode_key)
    finally:
        win_reg.delete_key_recursive(hive='HKLM', key=fake_key)

@pytest.mark.destructive_test
def test_delete_value_unicode_key(fake_key, unicode_key):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the delete_value function on a unicode key\n    '
    try:
        assert win_reg.set_value(hive='HKLM', key='\\'.join([fake_key, unicode_key]), vname='fake_name', vdata='junk data')
        assert win_reg.delete_value(hive='HKLM', key='\\'.join([fake_key, unicode_key]), vname='fake_name')
        assert not win_reg.value_exists(hive='HKLM', key='\\'.join([fake_key, unicode_key]), vname='fake_name')
    finally:
        win_reg.delete_key_recursive(hive='HKLM', key=fake_key)

def test_delete_key_recursive_invalid_hive(fake_key):
    if False:
        return 10
    '\n    Test the delete_key_recursive function when passing an invalid hive\n    '
    pytest.raises(CommandExecutionError, win_reg.delete_key_recursive, hive='BADHIVE', key=fake_key)

def test_delete_key_recursive_key_not_found(fake_key):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the delete_key_recursive function when the passed key to delete is\n    not found.\n    '
    assert not win_reg.key_exists(hive='HKLM', key=fake_key)
    assert not win_reg.delete_key_recursive(hive='HKLM', key=fake_key)

def test_delete_key_recursive_too_close():
    if False:
        while True:
            i = 10
    '\n    Test the delete_key_recursive function when the passed key to delete is\n    too close to root, such as\n    '
    mock_true = MagicMock(return_value=True)
    with patch('salt.utils.win_reg.key_exists', mock_true):
        assert not win_reg.delete_key_recursive(hive='HKLM', key='FAKE_KEY')

@pytest.mark.destructive_test
def test_delete_key_recursive(fake_key):
    if False:
        return 10
    '\n    Test the delete_key_recursive function\n    '
    try:
        assert win_reg.set_value(hive='HKLM', key=fake_key, vname='fake_name', vdata='fake_value')
        expected = {'Deleted': ['\\'.join(['HKLM', fake_key])], 'Failed': []}
        assert win_reg.delete_key_recursive(hive='HKLM', key=fake_key) == expected
        assert not win_reg.key_exists(hive='HKLM', key=fake_key)
    finally:
        win_reg.delete_key_recursive(hive='HKLM', key=fake_key)

@pytest.mark.destructive_test
def test_delete_key_recursive_failed_to_open_key(fake_key):
    if False:
        while True:
            i = 10
    '\n    Test the delete_key_recursive function on failure to open the key\n    '
    try:
        assert win_reg.set_value(hive='HKLM', key=fake_key, vname='fake_name', vdata='fake_value')
        expected = {'Deleted': [], 'Failed': ['\\'.join(['HKLM', fake_key]) + ' Failed to connect to key']}
        mock_true = MagicMock(return_value=True)
        mock_error = MagicMock(side_effect=[1, win32api.error(3, 'RegOpenKeyEx', 'Failed to connect to key')])
        with patch('salt.utils.win_reg.key_exists', mock_true), patch('salt.utils.win_reg.win32api.RegOpenKeyEx', mock_error):
            assert win_reg.delete_key_recursive(hive='HKLM', key=fake_key) == expected
    finally:
        win_reg.delete_key_recursive(hive='HKLM', key=fake_key)

@pytest.mark.destructive_test
def test_delete_key_recursive_failed_to_delete(fake_key):
    if False:
        print('Hello World!')
    '\n    Test the delete_key_recursive function on failure to delete a key\n    '
    try:
        assert win_reg.set_value(hive='HKLM', key=fake_key, vname='fake_name', vdata='fake_value')
        expected = {'Deleted': [], 'Failed': ['\\'.join(['HKLM', fake_key]) + ' Unknown error']}
        mock_error = MagicMock(side_effect=WindowsError('Unknown error'))
        with patch('salt.utils.win_reg.win32api.RegDeleteKey', mock_error):
            assert win_reg.delete_key_recursive(hive='HKLM', key=fake_key) == expected
    finally:
        win_reg.delete_key_recursive(hive='HKLM', key=fake_key)

@pytest.mark.destructive_test
def test_delete_key_recursive_unicode(fake_key, unicode_key):
    if False:
        while True:
            i = 10
    '\n    Test the delete_key_recursive function on value within a unicode key\n    '
    try:
        assert win_reg.set_value(hive='HKLM', key='\\'.join([fake_key, unicode_key]), vname='fake_name', vdata='fake_value')
        expected = {'Deleted': ['\\'.join(['HKLM', fake_key, unicode_key])], 'Failed': []}
        assert win_reg.delete_key_recursive(hive='HKLM', key='\\'.join([fake_key, unicode_key])) == expected
    finally:
        win_reg.delete_key_recursive(hive='HKLM', key=fake_key)

def test__to_unicode_int():
    if False:
        return 10
    '\n    Test the ``_to_unicode`` function when it receives an integer value.\n    Should return a unicode value, which is str in PY3.\n    '
    assert isinstance(win_reg._to_unicode(1), str)