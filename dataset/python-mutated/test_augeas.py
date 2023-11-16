"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>
    :codeauthor: Andrew Colin Kissa <andrew@topdog.za.net>
"""
import os
import pytest
import salt.states.augeas as augeas
from tests.support.mock import MagicMock, mock_open, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        print('Hello World!')
    return {augeas: {}}

@pytest.fixture
def setup_func():
    if False:
        return 10
    name = 'zabbix'
    context = '/files/etc/services'
    changes = ['ins service-name after service-name[last()]', 'set service-name[last()] zabbix-agent']
    fp_changes = ['ins service-name after /files/etc/services/service-name[last()]', 'set /files/etc/services/service-name[last()] zabbix-agent']
    ret = {'name': name, 'result': False, 'changes': {}, 'comment': ''}
    method_map = {'set': 'set', 'setm': 'setm', 'mv': 'move', 'move': 'move', 'ins': 'insert', 'insert': 'insert', 'rm': 'remove', 'remove': 'remove'}
    mock_method_map = MagicMock(return_value=method_map)
    return mock_method_map

def test_change_non_list_changes():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if none list changes handled correctly\n    '
    name = 'zabbix'
    ret = {'name': name, 'result': False, 'changes': {}, 'comment': ''}
    comt = "'changes' must be specified as a list"
    ret.update({'comment': comt})
    assert augeas.change(name) == ret

def test_change_non_list_load_path():
    if False:
        while True:
            i = 10
    '\n    Test if none list load_path is handled correctly\n    '
    name = 'zabbix'
    context = '/files/etc/services'
    changes = ['ins service-name after service-name[last()]', 'set service-name[last()] zabbix-agent']
    ret = {'name': name, 'result': False, 'changes': {}, 'comment': ''}
    comt = "'load_path' must be specified as a list"
    ret.update({'comment': comt})
    assert augeas.change(name, context, changes, load_path='x') == ret

def test_change_in_test_mode():
    if False:
        return 10
    '\n    Test test mode handling\n    '
    name = 'zabbix'
    context = '/files/etc/services'
    changes = ['ins service-name after service-name[last()]', 'set service-name[last()] zabbix-agent']
    ret = {'name': name, 'result': False, 'changes': {}, 'comment': ''}
    comt = 'Executing commands in file "/files/etc/services":\nins service-name after service-name[last()]\nset service-name[last()] zabbix-agent'
    ret.update({'comment': comt, 'result': True})
    with patch.dict(augeas.__opts__, {'test': True}):
        assert augeas.change(name, context, changes) == ret

def test_change_no_context_without_full_path(setup_func):
    if False:
        i = 10
        return i + 15
    '\n    Test handling of no context without full path\n    '
    name = 'zabbix'
    context = '/files/etc/services'
    changes = ['ins service-name after service-name[last()]', 'set service-name[last()] zabbix-agent']
    ret = {'name': name, 'result': False, 'changes': {}, 'comment': ''}
    comt = 'Error: Changes should be prefixed with /files if no context is provided, change: {}'.format(changes[0])
    ret.update({'comment': comt, 'result': False})
    with patch.dict(augeas.__opts__, {'test': False}):
        mock_dict_ = {'augeas.method_map': setup_func.mock_method_map}
        with patch.dict(augeas.__salt__, mock_dict_):
            assert augeas.change(name, changes=changes) == ret

def test_change_no_context_with_full_path_fail(setup_func):
    if False:
        print('Hello World!')
    '\n    Test handling of no context with full path with execute fail\n    '
    name = 'zabbix'
    fp_changes = ['ins service-name after /files/etc/services/service-name[last()]', 'set /files/etc/services/service-name[last()] zabbix-agent']
    ret = {'name': name, 'result': False, 'changes': {}, 'comment': ''}
    ret.update({'comment': 'Error: error', 'result': False})
    with patch.dict(augeas.__opts__, {'test': False}):
        mock_execute = MagicMock(return_value=dict(retval=False, error='error'))
        mock_dict_ = {'augeas.execute': mock_execute, 'augeas.method_map': setup_func.mock_method_map}
        with patch.dict(augeas.__salt__, mock_dict_):
            assert augeas.change(name, changes=fp_changes) == ret

def test_change_no_context_with_full_path_pass(setup_func):
    if False:
        print('Hello World!')
    '\n    Test handling of no context with full path with execute pass\n    '
    name = 'zabbix'
    fp_changes = ['ins service-name after /files/etc/services/service-name[last()]', 'set /files/etc/services/service-name[last()] zabbix-agent']
    ret = {'name': name, 'result': False, 'changes': {}, 'comment': ''}
    ret.update(dict(comment='Changes have been saved', result=True, changes={'diff': '+ zabbix-agent'}))
    with patch.dict(augeas.__opts__, {'test': False}):
        mock_execute = MagicMock(return_value=dict(retval=True))
        mock_dict_ = {'augeas.execute': mock_execute, 'augeas.method_map': setup_func.mock_method_map}
        with patch.dict(augeas.__salt__, mock_dict_):
            mock_filename = MagicMock(return_value='/etc/services')
            with patch.object(augeas, '_workout_filename', mock_filename), patch('os.path.isfile', MagicMock(return_value=True)):
                with patch('salt.utils.files.fopen', MagicMock(mock_open)):
                    mock_diff = MagicMock(return_value=['+ zabbix-agent'])
                    with patch('difflib.unified_diff', mock_diff):
                        assert augeas.change(name, changes=fp_changes) == ret

def test_change_no_context_without_full_path_invalid_cmd(setup_func):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test handling of invalid commands when no context supplied\n    '
    name = 'zabbix'
    ret = {'name': name, 'result': False, 'changes': {}, 'comment': ''}
    ret.update(dict(comment='Error: Command det is not supported (yet)', result=False))
    with patch.dict(augeas.__opts__, {'test': False}):
        mock_execute = MagicMock(return_value=dict(retval=True))
        mock_dict_ = {'augeas.execute': mock_execute, 'augeas.method_map': setup_func.mock_method_map}
        with patch.dict(augeas.__salt__, mock_dict_):
            changes = ['det service-name[last()] zabbix-agent']
            assert augeas.change(name, changes=changes) == ret

def test_change_no_context_without_full_path_invalid_change(setup_func):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test handling of invalid change when no context supplied\n    '
    name = 'zabbix'
    ret = {'name': name, 'result': False, 'changes': {}, 'comment': ''}
    comt = 'Error: Invalid formatted command, see debug log for details: require'
    ret.update(dict(comment=comt, result=False))
    changes = ['require']
    with patch.dict(augeas.__opts__, {'test': False}):
        mock_execute = MagicMock(return_value=dict(retval=True))
        mock_dict_ = {'augeas.execute': mock_execute, 'augeas.method_map': setup_func.mock_method_map}
        with patch.dict(augeas.__salt__, mock_dict_):
            assert augeas.change(name, changes=changes) == ret

def test_change_no_context_with_full_path_multiple_files(setup_func):
    if False:
        while True:
            i = 10
    '\n    Test handling of different paths with no context supplied\n    '
    name = 'zabbix'
    changes = ['set /files/etc/hosts/service-name test', 'set /files/etc/services/service-name test']
    filename = '/etc/hosts/service-name'
    filename_ = '/etc/services/service-name'
    comt = 'Error: Changes should be made to one file at a time, detected changes to {} and {}'.format(filename, filename_)
    ret = {'name': name, 'result': False, 'changes': {}, 'comment': ''}
    ret.update(dict(comment=comt, result=False))
    with patch.dict(augeas.__opts__, {'test': False}):
        mock_execute = MagicMock(return_value=dict(retval=True))
        mock_dict_ = {'augeas.execute': mock_execute, 'augeas.method_map': setup_func.mock_method_map}
        with patch.dict(augeas.__salt__, mock_dict_):
            assert augeas.change(name, changes=changes) == ret

def test_change_with_context_without_full_path_fail(setup_func):
    if False:
        i = 10
        return i + 15
    '\n    Test handling of context without full path fails\n    '
    name = 'zabbix'
    context = '/files/etc/services'
    changes = ['ins service-name after service-name[last()]', 'set service-name[last()] zabbix-agent']
    ret = {'name': name, 'result': False, 'changes': {}, 'comment': ''}
    ret.update(dict(comment='Error: error', result=False))
    with patch.dict(augeas.__opts__, {'test': False}):
        mock_execute = MagicMock(return_value=dict(retval=False, error='error'))
        mock_dict_ = {'augeas.execute': mock_execute, 'augeas.method_map': setup_func.mock_method_map}
        with patch.dict(augeas.__salt__, mock_dict_):
            with patch('salt.utils.files.fopen', MagicMock(mock_open)):
                assert augeas.change(name, context=context, changes=changes) == ret

def test_change_with_context_without_old_file(setup_func):
    if False:
        while True:
            i = 10
    '\n    Test handling of context without oldfile pass\n    '
    name = 'zabbix'
    context = '/files/etc/services'
    changes = ['ins service-name after service-name[last()]', 'set service-name[last()] zabbix-agent']
    ret = {'name': name, 'result': False, 'changes': {}, 'comment': ''}
    ret.update(dict(comment='Changes have been saved', result=True, changes={'updates': changes}))
    with patch.dict(augeas.__opts__, {'test': False}):
        mock_execute = MagicMock(return_value=dict(retval=True))
        mock_dict_ = {'augeas.execute': mock_execute, 'augeas.method_map': setup_func.mock_method_map}
        with patch.dict(augeas.__salt__, mock_dict_):
            mock_isfile = MagicMock(return_value=False)
            with patch.object(os.path, 'isfile', mock_isfile):
                assert augeas.change(name, context=context, changes=changes) == ret