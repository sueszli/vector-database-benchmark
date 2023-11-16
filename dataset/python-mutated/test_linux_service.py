"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.linux_service
"""
import os
import pytest
import salt.modules.linux_service as service
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {service: {}}

def test_start():
    if False:
        print('Hello World!')
    '\n    Test to start the specified service\n    '
    with patch.object(os.path, 'join', return_value='A'):
        with patch.object(service, 'run', MagicMock(return_value=True)):
            assert service.start('name')

def test_stop():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test to stop the specified service\n    '
    with patch.object(os.path, 'join', return_value='A'):
        with patch.object(service, 'run', MagicMock(return_value=True)):
            assert service.stop('name')

def test_restart():
    if False:
        while True:
            i = 10
    '\n    Test to restart the specified service\n    '
    with patch.object(os.path, 'join', return_value='A'):
        with patch.object(service, 'run', MagicMock(return_value=True)):
            assert service.restart('name')

def test_status():
    if False:
        return 10
    '\n    Test to return the status for a service, returns the PID or an empty\n    string if the service is running or not, pass a signature to use to\n    find the service via ps\n    '
    with patch.dict(service.__salt__, {'status.pid': MagicMock(return_value=True)}):
        assert service.status('name')

def test_reload_():
    if False:
        print('Hello World!')
    '\n    Test to restart the specified service\n    '
    with patch.object(os.path, 'join', return_value='A'):
        with patch.object(service, 'run', MagicMock(return_value=True)):
            assert service.reload_('name')

def test_run():
    if False:
        i = 10
        return i + 15
    '\n    Test to run the specified service\n    '
    with patch.object(os.path, 'join', return_value='A'):
        with patch.object(service, 'run', MagicMock(return_value=True)):
            assert service.run('name', 'action')

def test_available():
    if False:
        return 10
    '\n    Test to returns ``True`` if the specified service is available,\n    otherwise returns ``False``.\n    '
    with patch.object(service, 'get_all', return_value=['name', 'A']):
        assert service.available('name')

def test_missing():
    if False:
        while True:
            i = 10
    '\n    Test to inverse of service.available.\n    '
    with patch.object(service, 'get_all', return_value=['name1', 'A']):
        assert service.missing('name')

def test_get_all():
    if False:
        print('Hello World!')
    '\n    Test to return a list of all available services\n    '
    with patch.object(os.path, 'isdir', side_effect=[False, True]):
        assert service.get_all() == []
        with patch.object(os, 'listdir', return_value=['A', 'B']):
            assert service.get_all() == ['A', 'B']