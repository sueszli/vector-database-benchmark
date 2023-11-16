"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.logrotate
"""
import pytest
import salt.modules.logrotate as logrotate
from salt.exceptions import SaltInvocationError
from tests.support.mock import MagicMock, patch

@pytest.fixture
def PARSE_CONF():
    if False:
        while True:
            i = 10
    return {'include files': {'rsyslog': ['/var/log/syslog']}, 'rotate': 1, '/var/log/wtmp': {'rotate': 1}}

@pytest.fixture
def configure_loader_modules():
    if False:
        print('Hello World!')
    return {logrotate: {}}

def test_show_conf():
    if False:
        i = 10
        return i + 15
    '\n    Test if it show parsed configuration\n    '
    with patch('salt.modules.logrotate._parse_conf', MagicMock(return_value=True)):
        assert logrotate.show_conf()

def test_set(PARSE_CONF):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it set a new value for a specific configuration line\n    '
    with patch('salt.modules.logrotate._parse_conf', MagicMock(return_value=PARSE_CONF)), patch.dict(logrotate.__salt__, {'file.replace': MagicMock(return_value=True)}):
        assert logrotate.set_('rotate', '2')

def test_set_failed(PARSE_CONF):
    if False:
        while True:
            i = 10
    '\n    Test if it fails to set a new value for a specific configuration line\n    '
    with patch('salt.modules.logrotate._parse_conf', MagicMock(return_value=PARSE_CONF)):
        kwargs = {'key': '/var/log/wtmp', 'value': 2}
        pytest.raises(SaltInvocationError, logrotate.set_, **kwargs)

def test_set_setting(PARSE_CONF):
    if False:
        while True:
            i = 10
    '\n    Test if it set a new value for a specific configuration line\n    '
    with patch.dict(logrotate.__salt__, {'file.replace': MagicMock(return_value=True)}), patch('salt.modules.logrotate._parse_conf', MagicMock(return_value=PARSE_CONF)):
        assert logrotate.set_('/var/log/wtmp', 'rotate', '2')

def test_set_setting_failed(PARSE_CONF):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it fails to set a new value for a specific configuration line\n    '
    with patch('salt.modules.logrotate._parse_conf', MagicMock(return_value=PARSE_CONF)):
        kwargs = {'key': 'rotate', 'value': '/var/log/wtmp', 'setting': '2'}
        pytest.raises(SaltInvocationError, logrotate.set_, **kwargs)

def test_get(PARSE_CONF):
    if False:
        i = 10
        return i + 15
    '\n    Test if get a value for a specific configuration line\n    '
    with patch('salt.modules.logrotate._parse_conf', MagicMock(return_value=PARSE_CONF)):
        assert logrotate.get('rotate') == 1
        assert logrotate.get('rotate') != 2
        assert logrotate.get('/var/log/wtmp', 'rotate') == 1
        assert logrotate.get('/var/log/wtmp', 'rotate') != 2
        with patch.object(logrotate, '_LOG') as log_mock:
            res = logrotate.get('/var/log/utmp', 'rotate')
            assert log_mock.debug.called
            assert not log_mock.warn.called