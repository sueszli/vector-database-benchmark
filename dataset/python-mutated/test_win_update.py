import pytest
import salt.utils.win_update as win_update
from tests.support.mock import MagicMock, patch
pytestmark = [pytest.mark.windows_whitelisted, pytest.mark.skip_unless_on_windows]

def test_installed_no_updates():
    if False:
        while True:
            i = 10
    '\n    Test installed when there are no updates on the system\n    '
    with patch('salt.utils.winapi.Com', autospec=True), patch('win32com.client.Dispatch', autospec=True), patch.object(win_update.WindowsUpdateAgent, 'refresh', autospec=True):
        wua = win_update.WindowsUpdateAgent(online=False)
        wua._updates = []
        installed_updates = wua.installed()
        assert installed_updates.updates.Add.call_count == 0

def test_installed_no_updates_installed():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test installed when there are no Installed updates on the system\n    '
    with patch('salt.utils.winapi.Com', autospec=True), patch('win32com.client.Dispatch', autospec=True), patch.object(win_update.WindowsUpdateAgent, 'refresh', autospec=True):
        wua = win_update.WindowsUpdateAgent(online=False)
        wua._updates = [MagicMock(IsInstalled=False), MagicMock(IsInstalled=False), MagicMock(IsInstalled=False)]
        installed_updates = wua.installed()
        assert installed_updates.updates.Add.call_count == 0

def test_installed_updates_all_installed():
    if False:
        return 10
    '\n    Test installed when all updates on the system are Installed\n    '
    with patch('salt.utils.winapi.Com', autospec=True), patch('win32com.client.Dispatch', autospec=True), patch.object(win_update.WindowsUpdateAgent, 'refresh', autospec=True):
        wua = win_update.WindowsUpdateAgent(online=False)
        wua._updates = [MagicMock(IsInstalled=True), MagicMock(IsInstalled=True), MagicMock(IsInstalled=True)]
        installed_updates = wua.installed()
        assert installed_updates.updates.Add.call_count == 3

def test_installed_updates_some_installed():
    if False:
        return 10
    '\n    Test installed when some updates are installed on the system\n    '
    with patch('salt.utils.winapi.Com', autospec=True), patch('win32com.client.Dispatch', autospec=True), patch.object(win_update.WindowsUpdateAgent, 'refresh', autospec=True):
        wua = win_update.WindowsUpdateAgent(online=False)
        wua._updates = [MagicMock(IsInstalled=True), MagicMock(IsInstalled=False), MagicMock(IsInstalled=True), MagicMock(IsInstalled=False), MagicMock(IsInstalled=True)]
        installed_updates = wua.installed()
        assert installed_updates.updates.Add.call_count == 3