"""
    :codeauthor: Rupesh Tare <rupesht@saltstack.com>

    Test cases for salt.modules.nagios
"""
import os
import pytest
import salt.modules.nagios as nagios
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {nagios: {}}

def test_run():
    if False:
        while True:
            i = 10
    '\n    Test for Run nagios plugin and return all\n     the data execution with cmd.run\n    '
    with patch.object(nagios, '_execute_cmd', return_value='A'):
        assert nagios.run('plugin') == 'A'

def test_retcode():
    if False:
        i = 10
        return i + 15
    '\n    Test for Run one nagios plugin and return retcode of the execution\n    '
    with patch.object(nagios, '_execute_cmd', return_value='A'):
        assert nagios.retcode('plugin', key_name='key') == {'key': {'status': 'A'}}

def test_run_all():
    if False:
        return 10
    '\n    Test for Run nagios plugin and return all\n     the data execution with cmd.run_all\n    '
    with patch.object(nagios, '_execute_cmd', return_value='A'):
        assert nagios.run_all('plugin') == 'A'

def test_retcode_pillar():
    if False:
        return 10
    '\n    Test for Run one or more nagios plugins from pillar data and\n     get the result of cmd.retcode\n    '
    with patch.dict(nagios.__salt__, {'pillar.get': MagicMock(return_value={})}):
        assert nagios.retcode_pillar('pillar_name') == {}

def test_run_pillar():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for Run one or more nagios plugins from pillar data\n     and get the result of cmd.run\n    '
    with patch.object(nagios, '_execute_pillar', return_value='A'):
        assert nagios.run_pillar('pillar') == 'A'

def test_run_all_pillar():
    if False:
        i = 10
        return i + 15
    '\n    Test for Run one or more nagios plugins from pillar data\n     and get the result of cmd.run\n    '
    with patch.object(nagios, '_execute_pillar', return_value='A'):
        assert nagios.run_all_pillar('pillar') == 'A'

def test_list_plugins():
    if False:
        return 10
    '\n    Test for List all the nagios plugins\n    '
    with patch.object(os, 'listdir', return_value=[]):
        assert nagios.list_plugins() == []