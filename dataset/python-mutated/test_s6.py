"""
    :codeauthor: Marek Skrobacki <skrobul@skrobul.com>

    Test cases for salt.modules.s6
"""
import os
import pytest
import salt.modules.s6 as s6
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        print('Hello World!')
    return {s6: {'SERVICE_DIR': '/etc/service'}}

def test_start():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it starts service via s6-svc.\n    '
    mock_ret = MagicMock(return_value=False)
    with patch.dict(s6.__salt__, {'cmd.retcode': mock_ret}):
        assert s6.start('ssh')

def test_stop():
    if False:
        return 10
    '\n    Test if it stops service via s6.\n    '
    mock_ret = MagicMock(return_value=False)
    with patch.dict(s6.__salt__, {'cmd.retcode': mock_ret}):
        assert s6.stop('ssh')

def test_term():
    if False:
        while True:
            i = 10
    '\n    Test if it send a TERM to service via s6.\n    '
    mock_ret = MagicMock(return_value=False)
    with patch.dict(s6.__salt__, {'cmd.retcode': mock_ret}):
        assert s6.term('ssh')

def test_reload():
    if False:
        while True:
            i = 10
    '\n    Test if it send a HUP to service via s6.\n    '
    mock_ret = MagicMock(return_value=False)
    with patch.dict(s6.__salt__, {'cmd.retcode': mock_ret}):
        assert s6.reload_('ssh')

def test_restart():
    if False:
        while True:
            i = 10
    '\n    Test if it restart service via s6. This will stop/start service.\n    '
    mock_ret = MagicMock(return_value=False)
    with patch.dict(s6.__salt__, {'cmd.retcode': mock_ret}):
        assert s6.restart('ssh')

def test_full_restart():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it calls s6.restart() function.\n    '
    mock_ret = MagicMock(return_value=False)
    with patch.dict(s6.__salt__, {'cmd.retcode': mock_ret}):
        assert s6.full_restart('ssh') is None

def test_status():
    if False:
        i = 10
        return i + 15
    '\n    Test if it return the status for a service via s6,\n    return pid if running.\n    '
    mock_run = MagicMock(return_value='salt')
    with patch.dict(s6.__salt__, {'cmd.run_stdout': mock_run}):
        assert s6.status('ssh') == ''

def test_available():
    if False:
        return 10
    '\n    Test if it returns ``True`` if the specified service is available,\n    otherwise returns ``False``.\n    '
    with patch.object(os, 'listdir', MagicMock(return_value=['/etc/service'])):
        assert s6.available('/etc/service')

def test_missing():
    if False:
        while True:
            i = 10
    '\n    Test if it returns ``True`` if the specified service is not available,\n    otherwise returns ``False``.\n    '
    with patch.object(os, 'listdir', MagicMock(return_value=['/etc/service'])):
        assert s6.missing('foo')

def test_get_all():
    if False:
        while True:
            i = 10
    '\n    Test if it return a list of all available services.\n    '
    with patch.object(os, 'listdir', MagicMock(return_value=['/etc/service'])):
        assert s6.get_all() == ['/etc/service']