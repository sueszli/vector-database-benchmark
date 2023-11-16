import pytest
import salt.modules.pf as pf
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        for i in range(10):
            print('nop')
    return {pf: {}}

def test_enable_when_disabled():
    if False:
        i = 10
        return i + 15
    "\n    Tests enabling pf when it's not enabled yet.\n    "
    ret = {}
    ret['stderr'] = 'pf enabled'
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        assert pf.enable()['changes']

def test_enable_when_enabled():
    if False:
        return 10
    '\n    Tests enabling pf when it already enabled.\n    '
    ret = {}
    ret['stderr'] = 'pfctl: pf already enabled'
    ret['retcode'] = 1
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        assert not pf.enable()['changes']

def test_disable_when_enabled():
    if False:
        i = 10
        return i + 15
    "\n    Tests disabling pf when it's enabled.\n    "
    ret = {}
    ret['stderr'] = 'pf disabled'
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        assert pf.disable()['changes']

def test_disable_when_disabled():
    if False:
        while True:
            i = 10
    '\n    Tests disabling pf when it already disabled.\n    '
    ret = {}
    ret['stderr'] = 'pfctl: pf not enabled'
    ret['retcode'] = 1
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        assert not pf.disable()['changes']

def test_loglevel_freebsd():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests setting a loglevel.\n    '
    ret = {}
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(pf.__grains__, {'os': 'FreeBSD'}):
        res = pf.loglevel('urgent')
        mock_cmd.assert_called_once_with('pfctl -x urgent', output_loglevel='trace', python_shell=False)
        assert res['changes']

def test_loglevel_openbsd():
    if False:
        return 10
    '\n    Tests setting a loglevel.\n    '
    ret = {}
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}), patch.dict(pf.__grains__, {'os': 'OpenBSD'}):
        res = pf.loglevel('crit')
        mock_cmd.assert_called_once_with('pfctl -x crit', output_loglevel='trace', python_shell=False)
        assert res['changes']

def test_load():
    if False:
        while True:
            i = 10
    '\n    Tests loading ruleset.\n    '
    ret = {}
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        res = pf.load()
        mock_cmd.assert_called_once_with(['pfctl', '-f', '/etc/pf.conf'], output_loglevel='trace', python_shell=False)
        assert res['changes']

def test_load_noop():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests evaluating but not actually loading ruleset.\n    '
    ret = {}
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        res = pf.load(noop=True)
        mock_cmd.assert_called_once_with(['pfctl', '-f', '/etc/pf.conf', '-n'], output_loglevel='trace', python_shell=False)
        assert not res['changes']

def test_flush():
    if False:
        return 10
    '\n    Tests a regular flush command.\n    '
    ret = {}
    ret['stderr'] = 'pf: statistics cleared'
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        res = pf.flush('info')
        mock_cmd.assert_called_once_with('pfctl -v -F info', output_loglevel='trace', python_shell=False)
        assert res['changes']

def test_flush_capital():
    if False:
        return 10
    '\n    Tests a flush command starting with a capital letter.\n    '
    ret = {}
    ret['stderr'] = '2 tables cleared'
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        res = pf.flush('tables')
        mock_cmd.assert_called_once_with('pfctl -v -F Tables', output_loglevel='trace', python_shell=False)
        assert res['changes']

def test_flush_without_changes():
    if False:
        i = 10
        return i + 15
    '\n    Tests a flush command that has no changes.\n    '
    ret = {}
    ret['stderr'] = '0 tables cleared'
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        assert not pf.flush('tables')['changes']

def test_table():
    if False:
        return 10
    '\n    Tests a regular table command.\n    '
    ret = {}
    ret['stderr'] = '42 addresses deleted'
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        assert pf.table('flush', table='bad_hosts')['changes']

def test_table_expire():
    if False:
        i = 10
        return i + 15
    '\n    Tests the table expire command.\n    '
    ret = {}
    ret['stderr'] = '1/1 addresses expired.'
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        assert pf.table('expire', table='bad_hosts', number=300)['changes']

def test_table_add_addresses():
    if False:
        return 10
    '\n    Tests adding addresses to a table.\n    '
    ret = {}
    ret['stderr'] = '2/2 addressess added.'
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        assert pf.table('add', table='bad_hosts', addresses=['1.2.3.4', '5.6.7.8'])['changes']

def test_table_delete_addresses():
    if False:
        print('Hello World!')
    '\n    Tests deleting addresses in a table.\n    '
    ret = {}
    ret['stderr'] = '2/2 addressess deleted.'
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        assert pf.table('delete', table='bad_hosts', addresses=['1.2.3.4', '5.6.7.8'])['changes']

def test_table_test_address():
    if False:
        i = 10
        return i + 15
    '\n    Tests testing addresses in a table.\n    '
    ret = {}
    ret['stderr'] = '1/2 addressess match.'
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        assert pf.table('test', table='bad_hosts', addresses=['1.2.3.4'])['matches']

def test_table_no_changes():
    if False:
        print('Hello World!')
    '\n    Tests a table command that has no changes.\n    '
    ret = {}
    ret['stderr'] = '0/1 addresses expired.'
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        assert not pf.table('expire', table='bad_hosts', number=300)['changes']

def test_table_show():
    if False:
        while True:
            i = 10
    '\n    Tests showing table contents.\n    '
    ret = {}
    ret['stdout'] = '1.2.3.4\n5.6.7.8'
    ret['retcode'] = 0
    expected = ['1.2.3.4', '5.6.7.8']
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        assert pf.table('show', table='bad_hosts')['comment'] == expected

def test_table_zero():
    if False:
        return 10
    '\n    Tests clearing all the statistics of a table.\n    '
    ret = {}
    ret['stderr'] = '42 addresses has been cleared'
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        assert pf.table('zero', table='bad_hosts')['changes']

def test_show_rules():
    if False:
        i = 10
        return i + 15
    '\n    Tests show rules command.\n    '
    ret = {}
    ret['stdout'] = 'block return\npass'
    ret['retcode'] = 0
    expected = ['block return', 'pass']
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        assert pf.show('rules')['comment'] == expected

def test_show_states():
    if False:
        while True:
            i = 10
    '\n    Tests show states command.\n    '
    ret = {}
    ret['stdout'] = 'all udp 192.168.1.1:3478\n'
    ret['retcode'] = 0
    expected = ['all udp 192.168.1.1:3478', '']
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        assert pf.show('states')['comment'] == expected

def test_show_tables():
    if False:
        i = 10
        return i + 15
    '\n    Tests show tables command.\n    '
    ret = {}
    ret['stdout'] = 'bad_hosts'
    ret['retcode'] = 0
    mock_cmd = MagicMock(return_value=ret)
    with patch.dict(pf.__salt__, {'cmd.run_all': mock_cmd}):
        res = pf.show('tables')
        mock_cmd.assert_called_once_with('pfctl -s Tables', output_loglevel='trace', python_shell=False)
        assert not res['changes']