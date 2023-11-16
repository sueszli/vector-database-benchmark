"""
    tests.pytests.unit.beacons.test_glxinfo
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    glxinfo beacon test cases
"""
import pytest
import salt.beacons.glxinfo as glxinfo
from tests.support.mock import Mock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {glxinfo: {'last_state': {}}}

def test_no_glxinfo_command():
    if False:
        print('Hello World!')
    with patch('salt.utils.path.which') as mock:
        mock.return_value = None
        ret = glxinfo.__virtual__()
        mock.assert_called_once_with('glxinfo')
        assert ret == (False, 'glxinfo is missing.')

def test_with_glxinfo_command():
    if False:
        for i in range(10):
            print('nop')
    with patch('salt.utils.path.which') as mock:
        mock.return_value = '/usr/bin/glxinfo'
        ret = glxinfo.__virtual__()
        mock.assert_called_once_with('glxinfo')
        assert ret == 'glxinfo'

def test_non_list_config():
    if False:
        for i in range(10):
            print('nop')
    config = {}
    ret = glxinfo.validate(config)
    assert ret == (False, 'Configuration for glxinfo beacon must be a list.')

def test_no_user():
    if False:
        i = 10
        return i + 15
    config = [{'screen_event': True}]
    _expected = (False, 'Configuration for glxinfo beacon must include a user as glxinfo is not available to root.')
    ret = glxinfo.validate(config)
    assert ret == _expected

def test_screen_state():
    if False:
        print('Hello World!')
    config = [{'screen_event': True, 'user': 'frank'}]
    mock = Mock(return_value=0)
    with patch.dict(glxinfo.__salt__, {'cmd.retcode': mock}):
        ret = glxinfo.validate(config)
        assert ret == (True, 'Valid beacon configuration')
        ret = glxinfo.beacon(config)
        assert ret == [{'tag': 'screen_event', 'screen_available': True}]
        mock.assert_called_once_with('DISPLAY=:0 glxinfo', runas='frank', python_shell=True)

def test_screen_state_missing():
    if False:
        print('Hello World!')
    config = [{'screen_event': True, 'user': 'frank'}]
    mock = Mock(return_value=255)
    with patch.dict(glxinfo.__salt__, {'cmd.retcode': mock}):
        ret = glxinfo.validate(config)
        assert ret == (True, 'Valid beacon configuration')
        ret = glxinfo.beacon(config)
        assert ret == [{'tag': 'screen_event', 'screen_available': False}]

def test_screen_state_no_repeat():
    if False:
        i = 10
        return i + 15
    config = [{'screen_event': True, 'user': 'frank'}]
    mock = Mock(return_value=255)
    with patch.dict(glxinfo.__salt__, {'cmd.retcode': mock}):
        ret = glxinfo.validate(config)
        assert ret == (True, 'Valid beacon configuration')
        ret = glxinfo.beacon(config)
        assert ret == [{'tag': 'screen_event', 'screen_available': False}]
        ret = glxinfo.beacon(config)
        assert ret == []

def test_screen_state_change():
    if False:
        for i in range(10):
            print('nop')
    config = [{'screen_event': True, 'user': 'frank'}]
    mock = Mock(side_effect=[255, 0])
    with patch.dict(glxinfo.__salt__, {'cmd.retcode': mock}):
        ret = glxinfo.validate(config)
        assert ret == (True, 'Valid beacon configuration')
        ret = glxinfo.beacon(config)
        assert ret == [{'tag': 'screen_event', 'screen_available': False}]
        ret = glxinfo.beacon(config)
        assert ret == [{'tag': 'screen_event', 'screen_available': True}]