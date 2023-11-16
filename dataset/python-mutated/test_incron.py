"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.incron
"""
import pytest
import salt.modules.incron as incron
from tests.support.mock import MagicMock, call, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {incron: {}}

def test_write_incron_file():
    if False:
        for i in range(10):
            print('nop')
    "\n    Test if it writes the contents of a file to a user's crontab\n    "
    mock = MagicMock(return_value=0)
    with patch.dict(incron.__salt__, {'cmd.retcode': mock}), patch('salt.modules.incron._get_incron_cmdstr', MagicMock(return_value='incrontab')):
        assert incron.write_incron_file('cybage', '/home/cybage/new_cron')

def test_write_cron_file_verbose():
    if False:
        i = 10
        return i + 15
    "\n    Test if it writes the contents of a file to a user's crontab and\n    return error message on error\n    "
    mock = MagicMock(return_value=True)
    with patch.dict(incron.__salt__, {'cmd.run_all': mock}), patch('salt.modules.incron._get_incron_cmdstr', MagicMock(return_value='incrontab')):
        assert incron.write_incron_file_verbose('cybage', '/home/cybage/new_cron')

def test_raw_system_incron():
    if False:
        while True:
            i = 10
    '\n    Test if it return the contents of the system wide incrontab\n    '
    with patch('salt.modules.incron._read_file', MagicMock(return_value='salt')):
        assert incron.raw_system_incron() == 'salt'

def test_raw_incron():
    if False:
        i = 10
        return i + 15
    "\n    Test if it return the contents of the user's incrontab\n    "
    mock = MagicMock(return_value='incrontab')
    expected_calls = [call('incrontab -l cybage', python_shell=False, rstrip=False, runas='cybage')]
    with patch.dict(incron.__grains__, {'os_family': mock}):
        cmd_run_mock = MagicMock(return_value='salt')
        with patch.dict(incron.__salt__, {'cmd.run_stdout': cmd_run_mock}):
            assert incron.raw_incron('cybage') == 'salt'
            cmd_run_mock.assert_has_calls(expected_calls)
            cmd = cmd_run_mock.call_args[0][0]
            assert 'incrontab -l cybage' == cmd
            assert '-u' not in cmd

def test_list_tab():
    if False:
        return 10
    "\n    Test if it return the contents of the specified user's incrontab\n    "
    mock = MagicMock(return_value='incrontab')
    with patch.dict(incron.__grains__, {'os_family': mock}):
        mock = MagicMock(return_value='salt')
        with patch.dict(incron.__salt__, {'cmd.run_stdout': mock}):
            assert incron.list_tab('cybage') == {'pre': ['salt'], 'crons': []}

def test_set_job():
    if False:
        i = 10
        return i + 15
    '\n    Test if it sets a cron job up for a specified user.\n    '
    assert incron.set_job('cybage', '/home/cybage', 'TO_MODIFY', 'echo "$$ $@ $# $% $&"') == 'Invalid mask type: TO_MODIFY'
    val = {'pre': [], 'crons': [{'path': '/home/cybage', 'mask': 'IN_MODIFY', 'cmd': 'echo "SALT"'}]}
    with patch.object(incron, 'list_tab', MagicMock(return_value=val)):
        assert incron.set_job('cybage', '/home/cybage', 'IN_MODIFY', 'echo "SALT"') == 'present'
    with patch.object(incron, 'list_tab', MagicMock(return_value={'pre': ['salt'], 'crons': []})):
        mock = MagicMock(return_value='incrontab')
        with patch.dict(incron.__grains__, {'os_family': mock}):
            with patch.object(incron, '_write_incron_lines', MagicMock(return_value={'retcode': True, 'stderr': 'error'})):
                assert incron.set_job('cybage', '/home/cybage', 'IN_MODIFY', 'echo "SALT"') == 'error'
    with patch.object(incron, 'list_tab', MagicMock(return_value={'pre': ['salt'], 'crons': []})):
        mock = MagicMock(return_value='incrontab')
        with patch.dict(incron.__grains__, {'os_family': mock}):
            with patch.object(incron, '_write_incron_lines', MagicMock(return_value={'retcode': False, 'stderr': 'error'})):
                assert incron.set_job('cybage', '/home/cybage', 'IN_MODIFY', 'echo "SALT"') == 'new'
    val = {'pre': [], 'crons': [{'path': '/home/cybage', 'mask': 'IN_MODIFY,IN_DELETE', 'cmd': 'echo "SALT"'}]}
    with patch.object(incron, 'list_tab', MagicMock(return_value=val)):
        mock = MagicMock(return_value='incrontab')
        with patch.dict(incron.__grains__, {'os_family': mock}):
            with patch.object(incron, '_write_incron_lines', MagicMock(return_value={'retcode': False, 'stderr': 'error'})):
                assert incron.set_job('cybage', '/home/cybage', 'IN_DELETE', 'echo "SALT"') == 'updated'

def test_rm_job():
    if False:
        print('Hello World!')
    '\n    Test if it remove a cron job for a specified user. If any of the\n    day/time params are specified, the job will only be removed if\n    the specified params match.\n    '
    assert incron.rm_job('cybage', '/home/cybage', 'TO_MODIFY', 'echo "$$ $@ $# $% $&"') == 'Invalid mask type: TO_MODIFY'
    with patch.object(incron, 'list_tab', MagicMock(return_value={'pre': ['salt'], 'crons': []})):
        mock = MagicMock(return_value='incrontab')
        with patch.dict(incron.__grains__, {'os_family': mock}):
            with patch.object(incron, '_write_incron_lines', MagicMock(return_value={'retcode': True, 'stderr': 'error'})):
                assert incron.rm_job('cybage', '/home/cybage', 'IN_MODIFY', 'echo "SALT"') == 'error'
    with patch.object(incron, 'list_tab', MagicMock(return_value={'pre': ['salt'], 'crons': []})):
        mock = MagicMock(return_value='incrontab')
        with patch.dict(incron.__grains__, {'os_family': mock}):
            with patch.object(incron, '_write_incron_lines', MagicMock(return_value={'retcode': False, 'stderr': 'error'})):
                assert incron.rm_job('cybage', '/home/cybage', 'IN_MODIFY', 'echo "SALT"') == 'absent'