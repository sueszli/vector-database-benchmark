"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.varnish
"""
import pytest
import salt.modules.varnish as varnish
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        for i in range(10):
            print('nop')
    return {varnish: {}}

def test_version():
    if False:
        i = 10
        return i + 15
    '\n    Test to return server version from varnishd -V\n    '
    with patch.dict(varnish.__salt__, {'cmd.run': MagicMock(return_value='(varnish-2.0)')}):
        assert varnish.version() == '2.0'

def test_ban():
    if False:
        while True:
            i = 10
    '\n    Test to add ban to the varnish cache\n    '
    with patch.object(varnish, '_run_varnishadm', return_value={'retcode': 0}):
        assert varnish.ban('ban_expression')

def test_ban_list():
    if False:
        print('Hello World!')
    '\n    Test to list varnish cache current bans\n    '
    with patch.object(varnish, '_run_varnishadm', return_value={'retcode': True}):
        assert not varnish.ban_list()
    with patch.object(varnish, '_run_varnishadm', return_value={'retcode': False, 'stdout': 'A\nB\nC'}):
        assert varnish.ban_list() == ['B', 'C']

def test_purge():
    if False:
        print('Hello World!')
    '\n    Test to purge the varnish cache\n    '
    with patch.object(varnish, 'ban', return_value=True):
        assert varnish.purge()

def test_param_set():
    if False:
        print('Hello World!')
    '\n    Test to set a param in varnish cache\n    '
    with patch.object(varnish, '_run_varnishadm', return_value={'retcode': 0}):
        assert varnish.param_set('param', 'value')

def test_param_show():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test to show params of varnish cache\n    '
    with patch.object(varnish, '_run_varnishadm', return_value={'retcode': True, 'stdout': 'A\nB\nC'}):
        assert not varnish.param_show('param')
    with patch.object(varnish, '_run_varnishadm', return_value={'retcode': False, 'stdout': 'A .1\nB .2\n'}):
        assert varnish.param_show('param') == {'A': '.1'}