"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>
"""
import pytest
import salt.modules.win_timezone as win_timezone
from salt.exceptions import CommandExecutionError
from tests.support.mock import MagicMock, patch
pytestmark = [pytest.mark.skipif(not win_timezone.HAS_PYTZ, reason='This test requires pytz')]

@pytest.fixture
def configure_loader_modules():
    if False:
        while True:
            i = 10
    return {win_timezone: {'__opts__': {}, '__salt__': {}, '__utils__': {}}}

def test_get_zone_normal():
    if False:
        while True:
            i = 10
    '\n    Test if it get current timezone (i.e. Asia/Calcutta)\n    '
    mock_read_ok = MagicMock(return_value={'pid': 78, 'retcode': 0, 'stderr': '', 'stdout': 'India Standard Time'})
    with patch.dict(win_timezone.__salt__, {'cmd.run_all': mock_read_ok}):
        assert win_timezone.get_zone() == 'Asia/Calcutta'

def test_get_zone_normal_dstoff():
    if False:
        print('Hello World!')
    '\n    Test if it gets current timezone with dst off (i.e. America/Denver)\n    '
    mock_read_ok = MagicMock(return_value={'pid': 78, 'retcode': 0, 'stderr': '', 'stdout': 'Mountain Standard Time_dstoff'})
    with patch.dict(win_timezone.__salt__, {'cmd.run_all': mock_read_ok}):
        assert win_timezone.get_zone() == 'America/Denver'

def test_get_zone_normal_dstoff_issue():
    if False:
        while True:
            i = 10
    '\n    Test regression with dstoff fix stripping unwanted characters\n    '
    mock_read_ok = MagicMock(return_value={'pid': 78, 'retcode': 0, 'stderr': '', 'stdout': 'FLE Standard Time'})
    with patch.dict(win_timezone.__salt__, {'cmd.run_all': mock_read_ok}):
        assert win_timezone.get_zone() == 'Europe/Kiev'

@pytest.mark.parametrize('timezone', win_timezone.mapper.list_win())
def test_get_zone_all(timezone):
    if False:
        i = 10
        return i + 15
    '\n    Test all Win zones are properly resolved and none returns Unknown\n    '
    mock_read_ok = MagicMock(return_value={'pid': 78, 'retcode': 0, 'stderr': '', 'stdout': timezone})
    with patch.dict(win_timezone.__salt__, {'cmd.run_all': mock_read_ok}):
        assert win_timezone.get_zone() != 'Unknown'

def test_get_zone_unknown():
    if False:
        while True:
            i = 10
    '\n    Test get_zone with unknown timezone (i.e. Indian Standard Time)\n    '
    mock_read_error = MagicMock(return_value={'pid': 78, 'retcode': 0, 'stderr': '', 'stdout': 'Indian Standard Time'})
    with patch.dict(win_timezone.__salt__, {'cmd.run_all': mock_read_error}):
        assert win_timezone.get_zone() == 'Unknown'

def test_get_zone_error():
    if False:
        print('Hello World!')
    '\n    Test get_zone when it encounters an error\n    '
    mock_read_fatal = MagicMock(return_value={'pid': 78, 'retcode': 1, 'stderr': '', 'stdout': ''})
    with patch.dict(win_timezone.__salt__, {'cmd.run_all': mock_read_fatal}):
        with pytest.raises(CommandExecutionError):
            win_timezone.get_zone()

def test_get_offset():
    if False:
        i = 10
        return i + 15
    '\n    Test if it get current numeric timezone offset from UCT (i.e. +0530)\n    '
    mock_read = MagicMock(return_value={'pid': 78, 'retcode': 0, 'stderr': '', 'stdout': 'India Standard Time'})
    with patch.dict(win_timezone.__salt__, {'cmd.run_all': mock_read}):
        assert win_timezone.get_offset() == '+0530'

def test_get_zonecode():
    if False:
        print('Hello World!')
    '\n    Test if it get current timezone (i.e. PST, MDT, etc)\n    '
    mock_read = MagicMock(return_value={'pid': 78, 'retcode': 0, 'stderr': '', 'stdout': 'India Standard Time'})
    with patch.dict(win_timezone.__salt__, {'cmd.run_all': mock_read}):
        assert win_timezone.get_zonecode() == 'IST'

def test_set_zone():
    if False:
        i = 10
        return i + 15
    '\n    Test if it unlinks, then symlinks /etc/localtime to the set timezone.\n    '
    mock_write = MagicMock(return_value={'pid': 78, 'retcode': 0, 'stderr': '', 'stdout': ''})
    mock_read = MagicMock(return_value={'pid': 78, 'retcode': 0, 'stderr': '', 'stdout': 'India Standard Time'})
    with patch.dict(win_timezone.__salt__, {'cmd.run_all': mock_write}), patch.dict(win_timezone.__salt__, {'cmd.run_all': mock_read}):
        assert win_timezone.set_zone('Asia/Calcutta')

def test_zone_compare():
    if False:
        return 10
    '\n    Test if it checks the md5sum between the given timezone, and\n    the one set in /etc/localtime. Returns True if they match,\n    and False if not. Mostly useful for running state checks.\n    '
    mock_read = MagicMock(return_value={'pid': 78, 'retcode': 0, 'stderr': '', 'stdout': 'India Standard Time'})
    with patch.dict(win_timezone.__salt__, {'cmd.run_all': mock_read}):
        assert win_timezone.zone_compare('Asia/Calcutta')

def test_get_hwclock():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it get current hardware clock setting (UTC or localtime)\n    '
    assert win_timezone.get_hwclock() == 'localtime'

def test_set_hwclock():
    if False:
        while True:
            i = 10
    '\n    Test if it sets the hardware clock to be either UTC or localtime\n    '
    assert not win_timezone.set_hwclock('UTC')