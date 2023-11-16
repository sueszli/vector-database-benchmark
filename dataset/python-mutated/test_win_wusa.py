"""
Test the win_wusa execution module
"""
import pytest
import salt.modules.win_wusa as win_wusa
from salt.exceptions import CommandExecutionError
from tests.support.mock import MagicMock, patch
pytestmark = [pytest.mark.windows_whitelisted, pytest.mark.skip_unless_on_windows]

@pytest.fixture
def configure_loader_modules():
    if False:
        print('Hello World!')
    return {win_wusa: {}}

def test_is_installed_false():
    if False:
        i = 10
        return i + 15
    '\n    test is_installed function when the KB is not installed\n    '
    mock_retcode = MagicMock(return_value=1)
    with patch.dict(win_wusa.__salt__, {'cmd.retcode': mock_retcode}):
        assert win_wusa.is_installed('KB123456') is False

def test_is_installed_true():
    if False:
        print('Hello World!')
    '\n    test is_installed function when the KB is installed\n    '
    mock_retcode = MagicMock(return_value=0)
    with patch.dict(win_wusa.__salt__, {'cmd.retcode': mock_retcode}):
        assert win_wusa.is_installed('KB123456') is True

def test_list():
    if False:
        for i in range(10):
            print('nop')
    '\n    test list function\n    '
    ret = [{'HotFixID': 'KB123456'}, {'HotFixID': 'KB123457'}]
    mock_all = MagicMock(return_value=ret)
    with patch('salt.utils.win_pwsh.run_dict', mock_all):
        expected = ['KB123456', 'KB123457']
        returned = win_wusa.list_()
        assert returned == expected

def test_install():
    if False:
        print('Hello World!')
    '\n    test install function\n    '
    mock_retcode = MagicMock(return_value=0)
    path = 'C:\\KB123456.msu'
    with patch.dict(win_wusa.__salt__, {'cmd.retcode': mock_retcode}):
        assert win_wusa.install(path) is True
    mock_retcode.assert_called_once_with(['wusa.exe', path, '/quiet', '/norestart'], ignore_retcode=True)

def test_install_restart():
    if False:
        i = 10
        return i + 15
    '\n    test install function with restart=True\n    '
    mock_retcode = MagicMock(return_value=0)
    path = 'C:\\KB123456.msu'
    with patch.dict(win_wusa.__salt__, {'cmd.retcode': mock_retcode}):
        assert win_wusa.install(path, restart=True) is True
    mock_retcode.assert_called_once_with(['wusa.exe', path, '/quiet', '/forcerestart'], ignore_retcode=True)

def test_install_already_installed():
    if False:
        print('Hello World!')
    '\n    test install function when KB already installed\n    '
    retcode = 2359302
    mock_retcode = MagicMock(return_value=retcode)
    path = 'C:\\KB123456.msu'
    name = 'KB123456.msu'
    with patch.dict(win_wusa.__salt__, {'cmd.retcode': mock_retcode}):
        with pytest.raises(CommandExecutionError) as excinfo:
            win_wusa.install(path)
    mock_retcode.assert_called_once_with(['wusa.exe', path, '/quiet', '/norestart'], ignore_retcode=True)
    assert f'{name} is already installed. Additional info follows:\n\n{retcode}' == excinfo.value.message

def test_install_reboot_needed():
    if False:
        i = 10
        return i + 15
    '\n    test install function when KB need a reboot\n    '
    retcode = 3010
    mock_retcode = MagicMock(return_value=retcode)
    path = 'C:\\KB123456.msu'
    name = 'KB123456.msu'
    with patch.dict(win_wusa.__salt__, {'cmd.retcode': mock_retcode}):
        with pytest.raises(CommandExecutionError) as excinfo:
            win_wusa.install(path)
    mock_retcode.assert_called_once_with(['wusa.exe', path, '/quiet', '/norestart'], ignore_retcode=True)
    assert f'{name} correctly installed but server reboot is needed to complete installation. Additional info follows:\n\n{retcode}' == excinfo.value.message

def test_install_error_87():
    if False:
        i = 10
        return i + 15
    '\n    test install function when error 87 returned\n    '
    retcode = 87
    mock_retcode = MagicMock(return_value=retcode)
    path = 'C:\\KB123456.msu'
    with patch.dict(win_wusa.__salt__, {'cmd.retcode': mock_retcode}):
        with pytest.raises(CommandExecutionError) as excinfo:
            win_wusa.install(path)
    mock_retcode.assert_called_once_with(['wusa.exe', path, '/quiet', '/norestart'], ignore_retcode=True)
    assert f'Unknown error. Additional info follows:\n\n{retcode}' == excinfo.value.message

def test_install_error_other():
    if False:
        print('Hello World!')
    '\n    test install function on other unknown error\n    '
    mock_retcode = MagicMock(return_value=1234)
    path = 'C:\\KB123456.msu'
    with patch.dict(win_wusa.__salt__, {'cmd.retcode': mock_retcode}):
        with pytest.raises(CommandExecutionError) as excinfo:
            win_wusa.install(path)
    mock_retcode.assert_called_once_with(['wusa.exe', path, '/quiet', '/norestart'], ignore_retcode=True)
    assert 'Unknown error: 1234' == excinfo.value.message

def test_uninstall_kb():
    if False:
        return 10
    '\n    test uninstall function passing kb name\n    '
    mock_retcode = MagicMock(return_value=0)
    kb = 'KB123456'
    with patch.dict(win_wusa.__salt__, {'cmd.retcode': mock_retcode}), patch('os.path.exists', MagicMock(return_value=False)):
        assert win_wusa.uninstall(kb) is True
    mock_retcode.assert_called_once_with(['wusa.exe', '/uninstall', '/quiet', f'/kb:{kb[2:]}', '/norestart'], ignore_retcode=True)

def test_uninstall_path():
    if False:
        i = 10
        return i + 15
    '\n    test uninstall function passing full path to .msu file\n    '
    mock_retcode = MagicMock(return_value=0)
    path = 'C:\\KB123456.msu'
    with patch.dict(win_wusa.__salt__, {'cmd.retcode': mock_retcode}), patch('os.path.exists', MagicMock(return_value=True)):
        assert win_wusa.uninstall(path) is True
    mock_retcode.assert_called_once_with(['wusa.exe', '/uninstall', '/quiet', path, '/norestart'], ignore_retcode=True)

def test_uninstall_path_restart():
    if False:
        print('Hello World!')
    '\n    test uninstall function with full path and restart=True\n    '
    mock_retcode = MagicMock(return_value=0)
    path = 'C:\\KB123456.msu'
    with patch.dict(win_wusa.__salt__, {'cmd.retcode': mock_retcode}), patch('os.path.exists', MagicMock(return_value=True)):
        assert win_wusa.uninstall(path, restart=True) is True
    mock_retcode.assert_called_once_with(['wusa.exe', '/uninstall', '/quiet', path, '/forcerestart'], ignore_retcode=True)

def test_uninstall_already_uninstalled():
    if False:
        i = 10
        return i + 15
    '\n    test uninstall function when KB already uninstalled\n    '
    retcode = 2359303
    mock_retcode = MagicMock(return_value=retcode)
    kb = 'KB123456'
    with patch.dict(win_wusa.__salt__, {'cmd.retcode': mock_retcode}):
        with pytest.raises(CommandExecutionError) as excinfo:
            win_wusa.uninstall(kb)
    mock_retcode.assert_called_once_with(['wusa.exe', '/uninstall', '/quiet', f'/kb:{kb[2:]}', '/norestart'], ignore_retcode=True)
    assert f'{kb} not installed. Additional info follows:\n\n{retcode}' == excinfo.value.message

def test_uninstall_path_error_other():
    if False:
        return 10
    '\n    test uninstall function with unknown error\n    '
    mock_retcode = MagicMock(return_value=1234)
    path = 'C:\\KB123456.msu'
    with patch.dict(win_wusa.__salt__, {'cmd.retcode': mock_retcode}), patch('os.path.exists', MagicMock(return_value=True)), pytest.raises(CommandExecutionError) as excinfo:
        win_wusa.uninstall(path)
    mock_retcode.assert_called_once_with(['wusa.exe', '/uninstall', '/quiet', path, '/norestart'], ignore_retcode=True)
    assert 'Unknown error: 1234' == excinfo.value.message