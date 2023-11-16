"""
    :codeauthor: Rupesh Tare <rupesht@saltstack.com>
"""
import os
import pytest
import salt.modules.daemontools as daemontools
from salt.exceptions import CommandExecutionError
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {daemontools: {}}

def test_start():
    if False:
        i = 10
        return i + 15
    '\n    Test for Starts service via daemontools\n    '
    mock = MagicMock(return_value=None)
    with patch.dict(daemontools.__salt__, {'file.remove': mock}):
        mock = MagicMock(return_value='')
        with patch.object(daemontools, '_service_path', mock):
            mock = MagicMock(return_value=False)
            with patch.dict(daemontools.__salt__, {'cmd.retcode': mock}):
                assert daemontools.start('name')

def test_stop():
    if False:
        while True:
            i = 10
    '\n    Test for Stops service via daemontools\n    '
    mock = MagicMock(return_value=None)
    with patch.dict(daemontools.__salt__, {'file.touch': mock}):
        mock = MagicMock(return_value='')
        with patch.object(daemontools, '_service_path', mock):
            mock = MagicMock(return_value=False)
            with patch.dict(daemontools.__salt__, {'cmd.retcode': mock}):
                assert daemontools.stop('name')

def test_term():
    if False:
        print('Hello World!')
    '\n    Test for Send a TERM to service via daemontools\n    '
    mock = MagicMock(return_value='')
    with patch.object(daemontools, '_service_path', mock):
        mock = MagicMock(return_value=False)
        with patch.dict(daemontools.__salt__, {'cmd.retcode': mock}):
            assert daemontools.term('name')

def test_reload_():
    if False:
        i = 10
        return i + 15
    '\n    Test for Wrapper for term()\n    '
    mock = MagicMock(return_value=None)
    with patch.object(daemontools, 'term', mock):
        assert daemontools.reload_('name') is None

def test_restart():
    if False:
        print('Hello World!')
    '\n    Test for Restart service via daemontools. This will stop/start service\n    '
    mock = MagicMock(return_value=False)
    with patch.object(daemontools, 'stop', mock):
        assert daemontools.restart('name') == 'restart False'

def test_full_restart():
    if False:
        while True:
            i = 10
    '\n    Test for Calls daemontools.restart() function\n    '
    mock = MagicMock(return_value=None)
    with patch.object(daemontools, 'restart', mock):
        assert daemontools.restart('name') is None

def test_status():
    if False:
        i = 10
        return i + 15
    '\n    Test for Return the status for a service via\n    daemontools, return pid if running\n    '
    with patch('re.search', MagicMock(return_value=1)):
        mock = MagicMock(return_value='')
        with patch.object(daemontools, '_service_path', mock):
            mock = MagicMock(return_value='name')
            with patch.dict(daemontools.__salt__, {'cmd.run_stdout': mock}):
                assert daemontools.status('name') == ''

def test_available():
    if False:
        return 10
    '\n    Test for Returns ``True`` if the specified service\n    is available, otherwise returns``False``.\n    '
    mock = MagicMock(return_value=[])
    with patch.object(daemontools, 'get_all', mock):
        assert not daemontools.available('name')

def test_missing():
    if False:
        return 10
    '\n    Test for The inverse of daemontools.available.\n    '
    mock = MagicMock(return_value=[])
    with patch.object(daemontools, 'get_all', mock):
        assert daemontools.missing('name')

def test_get_all():
    if False:
        i = 10
        return i + 15
    '\n    Test for Return a list of all available services\n    '
    pytest.raises(CommandExecutionError, daemontools.get_all)
    with patch.object(daemontools, 'SERVICE_DIR', 'A'):
        mock = MagicMock(return_value='A')
        with patch.object(os, 'listdir', mock):
            assert daemontools.get_all() == ['A']