import pathlib
import pytest
import salt.modules.win_file as win_file
import salt.modules.win_lgpo_reg as lgpo_reg
import salt.utils.files
import salt.utils.win_dacl
import salt.utils.win_lgpo_reg
import salt.utils.win_reg
from salt.exceptions import SaltInvocationError
pytestmark = [pytest.mark.windows_whitelisted, pytest.mark.skip_unless_on_windows, pytest.mark.destructive_test]

@pytest.fixture
def configure_loader_modules():
    if False:
        for i in range(10):
            print('nop')
    return {win_file: {'__utils__': {'dacl.set_perms': salt.utils.win_dacl.set_perms, 'dacl.set_permissions': salt.utils.win_dacl.set_permissions}}}

@pytest.fixture
def empty_reg_pol_mach():
    if False:
        print('Hello World!')
    class_info = salt.utils.win_lgpo_reg.CLASS_INFO
    reg_pol_file = pathlib.Path(class_info['Machine']['policy_path'])
    if not reg_pol_file.parent.exists():
        reg_pol_file.parent.mkdir(parents=True)
    with salt.utils.files.fopen(str(reg_pol_file), 'wb') as f:
        f.write(salt.utils.win_lgpo_reg.REG_POL_HEADER.encode('utf-16-le'))
    salt.utils.win_reg.delete_key_recursive(hive='HKLM', key='SOFTWARE\\MyKey1')
    salt.utils.win_reg.delete_key_recursive(hive='HKLM', key='SOFTWARE\\MyKey2')
    yield
    salt.utils.win_reg.delete_key_recursive(hive='HKLM', key='SOFTWARE\\MyKey1')
    salt.utils.win_reg.delete_key_recursive(hive='HKLM', key='SOFTWARE\\MyKey2')
    with salt.utils.files.fopen(str(reg_pol_file), 'wb') as f:
        f.write(salt.utils.win_lgpo_reg.REG_POL_HEADER.encode('utf-16-le'))

@pytest.fixture
def empty_reg_pol_user():
    if False:
        return 10
    class_info = salt.utils.win_lgpo_reg.CLASS_INFO
    reg_pol_file = pathlib.Path(class_info['User']['policy_path'])
    if not reg_pol_file.parent.exists():
        reg_pol_file.parent.mkdir(parents=True)
    with salt.utils.files.fopen(str(reg_pol_file), 'wb') as f:
        f.write(salt.utils.win_lgpo_reg.REG_POL_HEADER.encode('utf-16-le'))
    salt.utils.win_reg.delete_key_recursive(hive='HKCU', key='SOFTWARE\\MyKey1')
    salt.utils.win_reg.delete_key_recursive(hive='HKCU', key='SOFTWARE\\MyKey2')
    yield
    salt.utils.win_reg.delete_key_recursive(hive='HKCU', key='SOFTWARE\\MyKey1')
    salt.utils.win_reg.delete_key_recursive(hive='HKCU', key='SOFTWARE\\MyKey2')
    with salt.utils.files.fopen(str(reg_pol_file), 'wb') as f:
        f.write(salt.utils.win_lgpo_reg.REG_POL_HEADER.encode('utf-16-le'))

@pytest.fixture
def reg_pol_mach():
    if False:
        return 10
    data_to_write = {'SOFTWARE\\MyKey1': {'MyValue1': {'data': 'squidward', 'type': 'REG_SZ'}, '**del.MyValue2': {'data': ' ', 'type': 'REG_SZ'}}, 'SOFTWARE\\MyKey2': {'MyValue3': {'data': ['spongebob', 'squarepants'], 'type': 'REG_MULTI_SZ'}}}
    lgpo_reg.write_reg_pol(data_to_write)
    salt.utils.win_reg.set_value(hive='HKLM', key='SOFTWARE\\MyKey1', vname='MyValue1', vdata='squidward', vtype='REG_SZ')
    salt.utils.win_reg.set_value(hive='HKLM', key='SOFTWARE\\MyKey2', vname='MyValue3', vdata=['spongebob', 'squarepants'], vtype='REG_MULTI_SZ')
    yield
    salt.utils.win_reg.delete_key_recursive(hive='HKLM', key='SOFTWARE\\MyKey1')
    salt.utils.win_reg.delete_key_recursive(hive='HKLM', key='SOFTWARE\\MyKey2')
    class_info = salt.utils.win_lgpo_reg.CLASS_INFO
    reg_pol_file = class_info['Machine']['policy_path']
    with salt.utils.files.fopen(reg_pol_file, 'wb') as f:
        f.write(salt.utils.win_lgpo_reg.REG_POL_HEADER.encode('utf-16-le'))

@pytest.fixture
def reg_pol_user():
    if False:
        for i in range(10):
            print('nop')
    data_to_write = {'SOFTWARE\\MyKey1': {'MyValue1': {'data': 'squidward', 'type': 'REG_SZ'}, '**del.MyValue2': {'data': ' ', 'type': 'REG_SZ'}}, 'SOFTWARE\\MyKey2': {'MyValue3': {'data': ['spongebob', 'squarepants'], 'type': 'REG_MULTI_SZ'}}}
    lgpo_reg.write_reg_pol(data_to_write, policy_class='User')
    salt.utils.win_reg.set_value(hive='HKCU', key='SOFTWARE\\MyKey1', vname='MyValue1', vdata='squidward', vtype='REG_SZ')
    salt.utils.win_reg.set_value(hive='HKCU', key='SOFTWARE\\MyKey2', vname='MyValue3', vdata=['spongebob', 'squarepants'], vtype='REG_MULTI_SZ')
    yield
    salt.utils.win_reg.delete_key_recursive(hive='HKCU', key='SOFTWARE\\MyKey1')
    salt.utils.win_reg.delete_key_recursive(hive='HKCU', key='SOFTWARE\\MyKey2')
    class_info = salt.utils.win_lgpo_reg.CLASS_INFO
    reg_pol_file = class_info['User']['policy_path']
    with salt.utils.files.fopen(reg_pol_file, 'wb') as f:
        f.write(salt.utils.win_lgpo_reg.REG_POL_HEADER.encode('utf-16-le'))

def test_invalid_policy_class_delete_value():
    if False:
        print('Hello World!')
    pytest.raises(SaltInvocationError, lgpo_reg.delete_value, key='', v_name='', policy_class='Invalid')

def test_invalid_policy_class_disable_value():
    if False:
        i = 10
        return i + 15
    pytest.raises(SaltInvocationError, lgpo_reg.disable_value, key='', v_name='', policy_class='Invalid')

def test_invalid_policy_class_get_key():
    if False:
        return 10
    pytest.raises(SaltInvocationError, lgpo_reg.get_key, key='', policy_class='Invalid')

def test_invalid_policy_class_get_value():
    if False:
        for i in range(10):
            print('nop')
    pytest.raises(SaltInvocationError, lgpo_reg.get_value, key='', v_name='', policy_class='Invalid')

def test_invalid_policy_class_read_reg_pol():
    if False:
        for i in range(10):
            print('nop')
    pytest.raises(SaltInvocationError, lgpo_reg.read_reg_pol, policy_class='Invalid')

def test_invalid_policy_class_set_value():
    if False:
        return 10
    pytest.raises(SaltInvocationError, lgpo_reg.set_value, key='', v_name='', v_data='', policy_class='Invalid')

def test_invalid_policy_class_write_reg_pol():
    if False:
        for i in range(10):
            print('nop')
    pytest.raises(SaltInvocationError, lgpo_reg.write_reg_pol, data={}, policy_class='Invalid')

def test_set_value_invalid_reg_type():
    if False:
        print('Hello World!')
    pytest.raises(SaltInvocationError, lgpo_reg.set_value, key='', v_name='', v_data='', v_type='REG_INVALID')

def test_set_value_invalid_reg_sz():
    if False:
        print('Hello World!')
    pytest.raises(SaltInvocationError, lgpo_reg.set_value, key='', v_name='', v_data=[], v_type='REG_SZ')

def test_set_value_invalid_reg_multi_sz():
    if False:
        i = 10
        return i + 15
    pytest.raises(SaltInvocationError, lgpo_reg.set_value, key='', v_name='', v_data=1, v_type='REG_MULTI_SZ')

def test_set_value_invalid_reg_dword():
    if False:
        print('Hello World!')
    pytest.raises(SaltInvocationError, lgpo_reg.set_value, key='', v_name='', v_data='string')

def test_mach_read_reg_pol(empty_reg_pol_mach):
    if False:
        print('Hello World!')
    expected = {}
    result = lgpo_reg.read_reg_pol()
    assert result == expected

def test_mach_write_reg_pol(empty_reg_pol_mach):
    if False:
        return 10
    data_to_write = {'SOFTWARE\\MyKey': {'MyValue': {'data': 'string', 'type': 'REG_SZ'}}}
    lgpo_reg.write_reg_pol(data_to_write)
    result = lgpo_reg.read_reg_pol()
    assert result == data_to_write

def test_mach_get_value(reg_pol_mach):
    if False:
        while True:
            i = 10
    expected = {'data': 'squidward', 'type': 'REG_SZ'}
    result = lgpo_reg.get_value(key='SOFTWARE\\MyKey1', v_name='MyValue1')
    assert result == expected

def test_mach_get_key(reg_pol_mach):
    if False:
        i = 10
        return i + 15
    expected = {'MyValue3': {'data': ['spongebob', 'squarepants'], 'type': 'REG_MULTI_SZ'}}
    result = lgpo_reg.get_key(key='SOFTWARE\\MyKey2')
    assert result == expected

def test_mach_set_value(empty_reg_pol_mach):
    if False:
        return 10
    key = 'SOFTWARE\\MyKey'
    v_name = 'MyValue'
    result = lgpo_reg.set_value(key=key, v_name=v_name, v_data='1')
    assert result is True
    expected = {'data': 1, 'type': 'REG_DWORD'}
    result = lgpo_reg.get_value(key=key, v_name=v_name)
    assert result == expected
    expected = {'hive': 'HKLM', 'key': key, 'vname': v_name, 'vdata': 1, 'vtype': 'REG_DWORD', 'success': True}
    result = salt.utils.win_reg.read_value(hive='HKLM', key=key, vname=v_name)
    assert result == expected

def test_mach_set_value_existing_change(reg_pol_mach):
    if False:
        print('Hello World!')
    expected = {'data': 1, 'type': 'REG_DWORD'}
    key = 'SOFTWARE\\MyKey'
    v_name = 'MyValue1'
    lgpo_reg.set_value(key=key, v_name=v_name, v_data='1')
    result = lgpo_reg.get_value(key=key, v_name=v_name)
    assert result == expected
    expected = {'hive': 'HKLM', 'key': key, 'vname': v_name, 'vdata': 1, 'vtype': 'REG_DWORD', 'success': True}
    result = salt.utils.win_reg.read_value(hive='HKLM', key=key, vname=v_name)
    assert result == expected

def test_mach_set_value_existing_no_change(reg_pol_mach):
    if False:
        print('Hello World!')
    expected = {'data': 'squidward', 'type': 'REG_SZ'}
    key = 'SOFTWARE\\MyKey'
    v_name = 'MyValue1'
    lgpo_reg.set_value(key=key, v_name=v_name, v_data='squidward', v_type='REG_SZ')
    result = lgpo_reg.get_value(key=key, v_name=v_name)
    assert result == expected

def test_mach_disable_value(reg_pol_mach):
    if False:
        return 10
    key = 'SOFTWARE\\MyKey1'
    result = lgpo_reg.disable_value(key=key, v_name='MyValue1')
    assert result is True
    expected = {'**del.MyValue1': {'data': ' ', 'type': 'REG_SZ'}, '**del.MyValue2': {'data': ' ', 'type': 'REG_SZ'}}
    result = lgpo_reg.get_key(key=key)
    assert result == expected
    result = salt.utils.win_reg.value_exists(hive='HKLM', key=key, vname='MyValue1')
    assert result is False

def test_mach_disable_value_no_change(reg_pol_mach):
    if False:
        print('Hello World!')
    expected = {'MyValue1': {'data': 'squidward', 'type': 'REG_SZ'}, '**del.MyValue2': {'data': ' ', 'type': 'REG_SZ'}}
    key = 'SOFTWARE\\MyKey1'
    lgpo_reg.disable_value(key=key, v_name='MyValue2')
    result = lgpo_reg.get_key(key=key)
    assert result == expected

def test_mach_delete_value_existing(reg_pol_mach):
    if False:
        return 10
    key = 'SOFTWARE\\MyKey1'
    result = lgpo_reg.delete_value(key=key, v_name='MyValue1')
    assert result is True
    expected = {'**del.MyValue2': {'data': ' ', 'type': 'REG_SZ'}}
    result = lgpo_reg.get_key(key=key)
    assert result == expected
    result = salt.utils.win_reg.value_exists(hive='HKLM', key=key, vname='MyValue2')
    assert result is False

def test_mach_delete_value_no_change(empty_reg_pol_mach):
    if False:
        for i in range(10):
            print('nop')
    expected = {}
    key = 'SOFTWARE\\MyKey1'
    lgpo_reg.delete_value(key=key, v_name='MyValue2')
    result = lgpo_reg.get_key(key=key)
    assert result == expected

def test_user_read_reg_pol(empty_reg_pol_user):
    if False:
        while True:
            i = 10
    expected = {}
    result = lgpo_reg.read_reg_pol(policy_class='User')
    assert result == expected

def test_user_write_reg_pol(empty_reg_pol_user):
    if False:
        print('Hello World!')
    data_to_write = {'SOFTWARE\\MyKey': {'MyValue': {'data': 'string', 'type': 'REG_SZ'}}}
    lgpo_reg.write_reg_pol(data_to_write, policy_class='User')
    result = lgpo_reg.read_reg_pol(policy_class='User')
    assert result == data_to_write

def test_user_get_value(reg_pol_user):
    if False:
        print('Hello World!')
    expected = {'data': 'squidward', 'type': 'REG_SZ'}
    result = lgpo_reg.get_value(key='SOFTWARE\\MyKey1', v_name='MyValue1', policy_class='User')
    assert result == expected

def test_user_get_key(reg_pol_user):
    if False:
        for i in range(10):
            print('nop')
    expected = {'MyValue3': {'data': ['spongebob', 'squarepants'], 'type': 'REG_MULTI_SZ'}}
    result = lgpo_reg.get_key(key='SOFTWARE\\MyKey2', policy_class='User')
    assert result == expected

def test_user_set_value(empty_reg_pol_user):
    if False:
        i = 10
        return i + 15
    key = 'SOFTWARE\\MyKey'
    v_name = 'MyValue'
    result = lgpo_reg.set_value(key=key, v_name=v_name, v_data='1', policy_class='User')
    assert result is True
    expected = {'data': 1, 'type': 'REG_DWORD'}
    result = lgpo_reg.get_value(key=key, v_name=v_name, policy_class='User')
    assert result == expected
    expected = {'hive': 'HKCU', 'key': key, 'vname': v_name, 'vdata': 1, 'vtype': 'REG_DWORD', 'success': True}
    result = salt.utils.win_reg.read_value(hive='HKCU', key=key, vname=v_name)
    assert result == expected

def test_user_set_value_existing_change(reg_pol_user):
    if False:
        while True:
            i = 10
    expected = {'data': 1, 'type': 'REG_DWORD'}
    key = 'SOFTWARE\\MyKey'
    v_name = 'MyValue1'
    lgpo_reg.set_value(key=key, v_name=v_name, v_data='1', policy_class='User')
    result = lgpo_reg.get_value(key=key, v_name=v_name, policy_class='User')
    assert result == expected
    expected = {'hive': 'HKCU', 'key': key, 'vname': v_name, 'vdata': 1, 'vtype': 'REG_DWORD', 'success': True}
    result = salt.utils.win_reg.read_value(hive='HKCU', key=key, vname=v_name)
    assert result == expected

def test_user_set_value_existing_no_change(reg_pol_user):
    if False:
        for i in range(10):
            print('nop')
    expected = {'data': 'squidward', 'type': 'REG_SZ'}
    key = 'SOFTWARE\\MyKey'
    v_name = 'MyValue1'
    lgpo_reg.set_value(key=key, v_name=v_name, v_data='squidward', v_type='REG_SZ', policy_class='User')
    result = lgpo_reg.get_value(key=key, v_name=v_name, policy_class='User')
    assert result == expected

def test_user_disable_value(reg_pol_user):
    if False:
        while True:
            i = 10
    key = 'SOFTWARE\\MyKey1'
    result = lgpo_reg.disable_value(key=key, v_name='MyValue1', policy_class='User')
    assert result is True
    expected = {'**del.MyValue1': {'data': ' ', 'type': 'REG_SZ'}, '**del.MyValue2': {'data': ' ', 'type': 'REG_SZ'}}
    result = lgpo_reg.get_key(key=key, policy_class='User')
    assert result == expected
    result = salt.utils.win_reg.value_exists(hive='HKCU', key=key, vname='MyValue1')
    assert result is False

def test_user_disable_value_no_change(reg_pol_user):
    if False:
        i = 10
        return i + 15
    expected = {'MyValue1': {'data': 'squidward', 'type': 'REG_SZ'}, '**del.MyValue2': {'data': ' ', 'type': 'REG_SZ'}}
    key = 'SOFTWARE\\MyKey1'
    lgpo_reg.disable_value(key=key, v_name='MyValue2', policy_class='User')
    result = lgpo_reg.get_key(key=key, policy_class='User')
    assert result == expected

def test_user_delete_value_existing(reg_pol_user):
    if False:
        return 10
    key = 'SOFTWARE\\MyKey1'
    result = lgpo_reg.delete_value(key=key, v_name='MyValue1', policy_class='User')
    assert result is True
    expected = {'**del.MyValue2': {'data': ' ', 'type': 'REG_SZ'}}
    result = lgpo_reg.get_key(key=key, policy_class='User')
    assert result == expected
    result = salt.utils.win_reg.value_exists(hive='HKCU', key=key, vname='MyValue2')
    assert result is False

def test_user_delete_value_no_change(empty_reg_pol_user):
    if False:
        return 10
    expected = {}
    key = 'SOFTWARE\\MyKey1'
    lgpo_reg.delete_value(key=key, v_name='MyValue2', policy_class='User')
    result = lgpo_reg.get_key(key=key, policy_class='User')
    assert result == expected