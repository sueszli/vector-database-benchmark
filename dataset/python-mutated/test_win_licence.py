"""
    Test cases for salt.modules.win_licence
"""
import pytest
import salt.modules.win_license as win_license
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {win_license: {}}

def test_installed():
    if False:
        print('Hello World!')
    '\n    Test to see if the given license key is installed\n    '
    mock = MagicMock(return_value='Partial Product Key: ABCDE')
    with patch.dict(win_license.__salt__, {'cmd.run': mock}):
        out = win_license.installed('AAAAA-AAAAA-AAAAA-AAAA-AAAAA-ABCDE')
        mock.assert_called_once_with('cscript C:\\Windows\\System32\\slmgr.vbs /dli')
        assert out

def test_installed_diff():
    if False:
        print('Hello World!')
    '\n    Test to see if the given license key is installed when the key is different\n    '
    mock = MagicMock(return_value='Partial Product Key: 12345')
    with patch.dict(win_license.__salt__, {'cmd.run': mock}):
        out = win_license.installed('AAAAA-AAAAA-AAAAA-AAAA-AAAAA-ABCDE')
        mock.assert_called_once_with('cscript C:\\Windows\\System32\\slmgr.vbs /dli')
        assert not out

def test_install():
    if False:
        while True:
            i = 10
    '\n    Test installing the given product key\n    '
    mock = MagicMock()
    with patch.dict(win_license.__salt__, {'cmd.run': mock}):
        win_license.install('AAAAA-AAAAA-AAAAA-AAAA-AAAAA-ABCDE')
        mock.assert_called_once_with('cscript C:\\Windows\\System32\\slmgr.vbs /ipk AAAAA-AAAAA-AAAAA-AAAA-AAAAA-ABCDE')

def test_uninstall():
    if False:
        print('Hello World!')
    '\n    Test uninstalling the given product key\n    '
    mock = MagicMock()
    with patch.dict(win_license.__salt__, {'cmd.run': mock}):
        win_license.uninstall()
        mock.assert_called_once_with('cscript C:\\Windows\\System32\\slmgr.vbs /upk')

def test_activate():
    if False:
        return 10
    '\n    Test activating the current product key\n    '
    mock = MagicMock()
    with patch.dict(win_license.__salt__, {'cmd.run': mock}):
        win_license.activate()
        mock.assert_called_once_with('cscript C:\\Windows\\System32\\slmgr.vbs /ato')

def test_licensed():
    if False:
        print('Hello World!')
    '\n    Test checking if the minion is licensed\n    '
    mock = MagicMock(return_value='License Status: Licensed')
    with patch.dict(win_license.__salt__, {'cmd.run': mock}):
        win_license.licensed()
        mock.assert_called_once_with('cscript C:\\Windows\\System32\\slmgr.vbs /dli')

def test_info():
    if False:
        i = 10
        return i + 15
    '\n    Test getting the info about the current license key\n    '
    expected = {'description': 'Prof', 'licensed': True, 'name': 'Win7', 'partial_key': '12345'}
    mock = MagicMock(return_value='Name: Win7\r\nDescription: Prof\r\nPartial Product Key: 12345\r\nLicense Status: Licensed')
    with patch.dict(win_license.__salt__, {'cmd.run': mock}):
        out = win_license.info()
        mock.assert_called_once_with('cscript C:\\Windows\\System32\\slmgr.vbs /dli')
        assert out == expected