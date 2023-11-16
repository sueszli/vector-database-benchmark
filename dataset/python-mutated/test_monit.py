"""
    :codeauthor: Rupesh Tare <rupesht@saltstack.com>
"""
import pytest
import salt.modules.monit as monit
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {monit: {}}

def test_start():
    if False:
        i = 10
        return i + 15
    '\n    Test for start\n    '
    with patch.dict(monit.__salt__, {'cmd.retcode': MagicMock(return_value=False)}):
        assert monit.start('name')

def test_stop():
    if False:
        print('Hello World!')
    '\n    Test for Stops service via monit\n    '
    with patch.dict(monit.__salt__, {'cmd.retcode': MagicMock(return_value=False)}):
        assert monit.stop('name')

def test_restart():
    if False:
        print('Hello World!')
    '\n    Test for Restart service via monit\n    '
    with patch.dict(monit.__salt__, {'cmd.retcode': MagicMock(return_value=False)}):
        assert monit.restart('name')

def test_unmonitor():
    if False:
        i = 10
        return i + 15
    '\n    Test for Unmonitor service via monit\n    '
    with patch.dict(monit.__salt__, {'cmd.retcode': MagicMock(return_value=False)}):
        assert monit.unmonitor('name')

def test_monitor():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for monitor service via monit\n    '
    with patch.dict(monit.__salt__, {'cmd.retcode': MagicMock(return_value=False)}):
        assert monit.monitor('name')

def test_summary():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Display a summary from monit\n    '
    mock = MagicMock(side_effect=['daemon is not running', 'A\nB\nC\nD\nE'])
    with patch.dict(monit.__salt__, {'cmd.run': mock}):
        assert monit.summary() == {'monit': 'daemon is not running', 'result': False}
        assert monit.summary() == {}

def test_status():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Display a process status from monit\n    '
    with patch.dict(monit.__salt__, {'cmd.run': MagicMock(return_value='Process')}):
        assert monit.status('service') == 'No such service'

def test_reload():
    if False:
        print('Hello World!')
    '\n    Test for Reload configuration\n    '
    mock = MagicMock(return_value=0)
    with patch.dict(monit.__salt__, {'cmd.retcode': mock}):
        assert monit.reload_()

def test_version():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Display version from monit -V\n    '
    mock = MagicMock(return_value='This is Monit version 5.14\nA\nB')
    with patch.dict(monit.__salt__, {'cmd.run': mock}):
        assert monit.version() == '5.14'

def test_id():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Display unique id\n    '
    mock = MagicMock(return_value='Monit ID: d3b1aba48527dd599db0e86f5ad97120')
    with patch.dict(monit.__salt__, {'cmd.run': mock}):
        assert monit.id_() == 'd3b1aba48527dd599db0e86f5ad97120'

def test_reset_id():
    if False:
        return 10
    '\n    Test for Regenerate a unique id\n    '
    expected = {'stdout': 'Monit id d3b1aba48527dd599db0e86f5ad97120 and ...'}
    mock = MagicMock(return_value=expected)
    with patch.dict(monit.__salt__, {'cmd.run_all': mock}):
        assert monit.id_(reset=True) == 'd3b1aba48527dd599db0e86f5ad97120'

def test_configtest():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Check configuration syntax\n    '
    excepted = {'stdout': 'Control file syntax OK', 'retcode': 0, 'stderr': ''}
    mock = MagicMock(return_value=excepted)
    with patch.dict(monit.__salt__, {'cmd.run_all': mock}):
        assert monit.configtest()['result']
        assert monit.configtest()['comment'] == 'Syntax OK'

def test_validate():
    if False:
        return 10
    '\n    Test for Check all services are monitored\n    '
    mock = MagicMock(return_value=0)
    with patch.dict(monit.__salt__, {'cmd.retcode': mock}):
        assert monit.validate()