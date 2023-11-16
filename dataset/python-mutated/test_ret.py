"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.ret
"""
import pytest
import salt.loader
import salt.modules.ret as ret
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {ret: {}}

def test_get_jid():
    if False:
        print('Hello World!')
    '\n    Test if it return the information for a specified job id\n    '
    mock_ret = MagicMock(return_value='DB')
    with patch.object(salt.loader, 'returners', MagicMock(return_value={'redis.get_jid': mock_ret})):
        assert ret.get_jid('redis', 'net') == 'DB'

def test_get_fun():
    if False:
        i = 10
        return i + 15
    '\n    Test if it return info about last time fun was called on each minion\n    '
    mock_ret = MagicMock(return_value='DB')
    with patch.object(salt.loader, 'returners', MagicMock(return_value={'mysql.get_fun': mock_ret})):
        assert ret.get_fun('mysql', 'net') == 'DB'

def test_get_jids():
    if False:
        while True:
            i = 10
    '\n    Test if it return a list of all job ids\n    '
    mock_ret = MagicMock(return_value='DB')
    with patch.object(salt.loader, 'returners', MagicMock(return_value={'mysql.get_jids': mock_ret})):
        assert ret.get_jids('mysql') == 'DB'

def test_get_minions():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it return a list of all minions\n    '
    mock_ret = MagicMock(return_value='DB')
    with patch.object(salt.loader, 'returners', MagicMock(return_value={'mysql.get_minions': mock_ret})):
        assert ret.get_minions('mysql') == 'DB'