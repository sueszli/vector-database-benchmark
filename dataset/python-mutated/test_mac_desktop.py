"""
Unit Tests for the mac_desktop execution module.
"""
import pytest
import salt.modules.mac_desktop as mac_desktop
from salt.exceptions import CommandExecutionError
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {mac_desktop: {}}

def test_get_output_volume():
    if False:
        print('Hello World!')
    '\n    Test if it get the output volume (range 0 to 100)\n    '
    mock = MagicMock(return_value={'retcode': 0, 'stdout': '25'})
    with patch.dict(mac_desktop.__salt__, {'cmd.run_all': mock}):
        assert mac_desktop.get_output_volume() == '25'

def test_get_output_volume_error():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests that an error is raised when cmd.run_all errors\n    '
    mock = MagicMock(return_value={'retcode': 1})
    with patch.dict(mac_desktop.__salt__, {'cmd.run_all': mock}):
        pytest.raises(CommandExecutionError, mac_desktop.get_output_volume)

def test_set_output_volume():
    if False:
        i = 10
        return i + 15
    '\n    Test if it set the volume of sound (range 0 to 100)\n    '
    mock = MagicMock(return_value={'retcode': 0})
    with patch.dict(mac_desktop.__salt__, {'cmd.run_all': mock}), patch('salt.modules.mac_desktop.get_output_volume', MagicMock(return_value='25')):
        assert mac_desktop.set_output_volume('25')

def test_set_output_volume_error():
    if False:
        i = 10
        return i + 15
    '\n    Tests that an error is raised when cmd.run_all errors\n    '
    mock = MagicMock(return_value={'retcode': 1})
    with patch.dict(mac_desktop.__salt__, {'cmd.run_all': mock}):
        pytest.raises(CommandExecutionError, mac_desktop.set_output_volume, '25')

def test_screensaver():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it launch the screensaver\n    '
    mock = MagicMock(return_value={'retcode': 0})
    with patch.dict(mac_desktop.__salt__, {'cmd.run_all': mock}):
        assert mac_desktop.screensaver()

def test_screensaver_error():
    if False:
        i = 10
        return i + 15
    '\n    Tests that an error is raised when cmd.run_all errors\n    '
    mock = MagicMock(return_value={'retcode': 1})
    with patch.dict(mac_desktop.__salt__, {'cmd.run_all': mock}):
        pytest.raises(CommandExecutionError, mac_desktop.screensaver)

def test_lock():
    if False:
        return 10
    '\n    Test if it lock the desktop session\n    '
    mock = MagicMock(return_value={'retcode': 0})
    with patch.dict(mac_desktop.__salt__, {'cmd.run_all': mock}):
        assert mac_desktop.lock()

def test_lock_error():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests that an error is raised when cmd.run_all errors\n    '
    mock = MagicMock(return_value={'retcode': 1})
    with patch.dict(mac_desktop.__salt__, {'cmd.run_all': mock}):
        pytest.raises(CommandExecutionError, mac_desktop.lock)

def test_say():
    if False:
        while True:
            i = 10
    '\n    Test if it says some words.\n    '
    mock = MagicMock(return_value={'retcode': 0})
    with patch.dict(mac_desktop.__salt__, {'cmd.run_all': mock}):
        assert mac_desktop.say()

def test_say_error():
    if False:
        i = 10
        return i + 15
    '\n    Tests that an error is raised when cmd.run_all errors\n    '
    mock = MagicMock(return_value={'retcode': 1})
    with patch.dict(mac_desktop.__salt__, {'cmd.run_all': mock}):
        pytest.raises(CommandExecutionError, mac_desktop.say)