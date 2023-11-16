"""
    :codeauthor: Gareth J. Greenaway <gareth@saltstack.com>
"""
import pytest
import salt.modules.cmdmod as cmdmod
import salt.modules.win_system as win_system
from tests.support.mock import MagicMock, patch
pytestmark = [pytest.mark.windows_whitelisted, pytest.mark.skip_unless_on_windows]

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {win_system: {'__salt__': {'cmd.run': cmdmod.run}}}

def test_lock():
    if False:
        return 10
    '\n    Test locking workstation\n    '
    with patch('ctypes.windll.user32.LockWorkStation', MagicMock()):
        ret = win_system.lock()
        assert ret

def test_get_computer_name():
    if False:
        i = 10
        return i + 15
    '\n    Test getting the computer name\n    '
    ret = win_system.get_computer_name()
    assert ret

def test_set_computer_name():
    if False:
        return 10
    '\n    Test getting the computer name\n    '
    with patch('ctypes.windll.kernel32.SetComputerNameExW', MagicMock()):
        ret = win_system.set_computer_name('test_hostname')
        assert ret

def test_get_hostname():
    if False:
        while True:
            i = 10
    '\n    Test getting the computer name\n    '
    ret = win_system.get_hostname()
    assert ret

def test_set_system_time():
    if False:
        print('Hello World!')
    '\n    Test setting the system time\n    '
    with patch.object(win_system, 'set_system_date_time', MagicMock()):
        ret = win_system.set_system_time('12:05 AM')
        assert ret

def test_get_system_time():
    if False:
        print('Hello World!')
    '\n    Test setting the system time\n    '
    ret = win_system.get_system_time()
    assert ret

def test_set_system_date_time():
    if False:
        while True:
            i = 10
    '\n    Test setting the system time\n    '
    get_local_time = (2022, 6, 3, 11, 0, 0, 0)
    with patch('win32api.GetLocalTime', MagicMock(return_value=get_local_time)), patch('ctypes.windll.kernel32.SetLocalTime', MagicMock()) as set_local_time:
        ret = win_system.set_system_time('12:05 AM')
        assert ret

def test_get_system_date():
    if False:
        return 10
    '\n    Test setting the system date\n    '
    ret = win_system.get_system_date()
    assert ret

def test_set_system_date():
    if False:
        while True:
            i = 10
    '\n    Test setting the system time\n    '
    with patch.object(win_system, 'set_system_date_time', MagicMock()):
        ret = win_system.set_system_date('03-28-13')
        assert ret