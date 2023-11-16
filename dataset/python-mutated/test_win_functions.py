import pytest
import salt.utils.win_functions as win_functions
from tests.support.mock import MagicMock, patch
HAS_WIN32 = False
HAS_PYWIN = False
try:
    import win32net
    HAS_WIN32 = True

    class WinError(win32net.error):
        winerror = 0
except ImportError:
    HAS_WIN32 = False
try:
    import pywintypes
    HAS_PYWIN = True

    class PyWinError(pywintypes.error):
        pywinerror = 0
except ImportError:
    HAS_PYWIN = False

@pytest.mark.skip_unless_on_windows(reason='Test is only applicable to Windows.')
def test_escape_argument_simple():
    if False:
        return 10
    '\n    Test to make sure we encode simple arguments correctly\n    '
    encoded = win_functions.escape_argument('simple')
    assert encoded == 'simple'

@pytest.mark.skip_unless_on_windows(reason='Test is only applicable to Windows.')
def test_escape_argument_with_space():
    if False:
        while True:
            i = 10
    '\n    Test to make sure we encode arguments containing spaces correctly\n    '
    encoded = win_functions.escape_argument('with space')
    assert encoded == '^"with space^"'

@pytest.mark.skip_unless_on_windows(reason='Test is only applicable to Windows.')
def test_escape_argument_simple_path():
    if False:
        return 10
    '\n    Test to make sure we encode simple path arguments correctly\n    '
    encoded = win_functions.escape_argument('C:\\some\\path')
    assert encoded == 'C:\\some\\path'

@pytest.mark.skip_unless_on_windows(reason='Test is only applicable to Windows.')
def test_escape_argument_path_with_space():
    if False:
        return 10
    '\n    Test to make sure we encode path arguments containing spaces correctly\n    '
    encoded = win_functions.escape_argument('C:\\Some Path\\With Spaces')
    assert encoded == '^"C:\\Some Path\\With Spaces^"'

@pytest.mark.skip_unless_on_windows(reason='Test is only applicable to Windows.')
def test_broadcast_setting_change():
    if False:
        while True:
            i = 10
    '\n    Test to rehash the Environment variables\n    '
    assert win_functions.broadcast_setting_change()

@pytest.mark.skip_unless_on_windows(reason='Test is only applicable to Windows.')
def test_get_user_groups():
    if False:
        for i in range(10):
            print('nop')
    groups = ['Administrators', 'Users']
    with patch('win32net.NetUserGetLocalGroups', return_value=groups):
        assert win_functions.get_user_groups('Administrator') == groups

@pytest.mark.skip_unless_on_windows(reason='Test is only applicable to Windows.')
def test_get_user_groups_sid():
    if False:
        for i in range(10):
            print('nop')
    groups = ['Administrators', 'Users']
    expected = ['S-1-5-32-544', 'S-1-5-32-545']
    with patch('win32net.NetUserGetLocalGroups', return_value=groups):
        assert win_functions.get_user_groups('Administrator', sid=True) == expected

@pytest.mark.skip_unless_on_windows(reason='Test is only applicable to Windows.')
def test_get_user_groups_system():
    if False:
        return 10
    groups = ['SYSTEM']
    with patch('win32net.NetUserGetLocalGroups', return_value=groups):
        assert win_functions.get_user_groups('SYSTEM') == groups

@pytest.mark.skip_unless_on_windows(reason='Test is only applicable to Windows.')
@pytest.mark.skipif(not HAS_WIN32, reason='Requires Win32 libraries')
def test_get_user_groups_unavailable_dc():
    if False:
        i = 10
        return i + 15
    groups = ['Administrators', 'Users']
    win_error = WinError()
    win_error.winerror = 1722
    effect = [win_error, groups]
    with patch('win32net.NetUserGetLocalGroups', side_effect=effect):
        assert win_functions.get_user_groups('Administrator') == groups

@pytest.mark.skip_unless_on_windows(reason='Test is only applicable to Windows.')
@pytest.mark.skipif(not HAS_WIN32, reason='Requires Win32 libraries')
def test_get_user_groups_unknown_dc():
    if False:
        return 10
    groups = ['Administrators', 'Users']
    win_error = WinError()
    win_error.winerror = 2453
    effect = [win_error, groups]
    with patch('win32net.NetUserGetLocalGroups', side_effect=effect):
        assert win_functions.get_user_groups('Administrator') == groups

@pytest.mark.skip_unless_on_windows(reason='Test is only applicable to Windows.')
@pytest.mark.skipif(not HAS_WIN32, reason='Requires Win32 libraries')
def test_get_user_groups_missing_permission():
    if False:
        for i in range(10):
            print('nop')
    groups = ['Administrators', 'Users']
    win_error = WinError()
    win_error.winerror = 5
    effect = [win_error, groups]
    with patch('win32net.NetUserGetLocalGroups', side_effect=effect):
        assert win_functions.get_user_groups('Administrator') == groups

@pytest.mark.skip_unless_on_windows(reason='Test is only applicable to Windows.')
@pytest.mark.skipif(not HAS_WIN32, reason='Requires Win32 libraries')
def test_get_user_groups_error():
    if False:
        return 10
    win_error = WinError()
    win_error.winerror = 1927
    mock_error = MagicMock(side_effect=win_error)
    with patch('win32net.NetUserGetLocalGroups', side_effect=mock_error):
        with pytest.raises(WinError):
            win_functions.get_user_groups('Administrator')

@pytest.mark.skip_unless_on_windows(reason='Test is only applicable to Windows.')
@pytest.mark.skipif(not HAS_PYWIN, reason='Requires pywintypes libraries')
def test_get_user_groups_local_pywin_error():
    if False:
        i = 10
        return i + 15
    win_error = PyWinError()
    win_error.winerror = 1355
    mock_error = MagicMock(side_effect=win_error)
    with patch('win32net.NetUserGetLocalGroups', side_effect=mock_error):
        with pytest.raises(PyWinError):
            win_functions.get_user_groups('Administrator')

@pytest.mark.skip_unless_on_windows(reason='Test is only applicable to Windows.')
@pytest.mark.skipif(not HAS_PYWIN, reason='Requires pywintypes libraries')
def test_get_user_groups_pywin_error():
    if False:
        while True:
            i = 10
    win_error = PyWinError()
    win_error.winerror = 1355
    mock_error = MagicMock(side_effect=win_error)
    with patch('win32net.NetUserGetLocalGroups', side_effect=mock_error):
        with patch('win32net.NetUserGetGroups', side_effect=mock_error):
            with pytest.raises(PyWinError):
                win_functions.get_user_groups('Administrator')