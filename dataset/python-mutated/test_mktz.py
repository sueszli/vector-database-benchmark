from datetime import datetime as dt
import tzlocal
from mock import patch
from pytest import raises
from arctic.date import mktz, TimezoneError
DEFAULT_TIME_ZONE_NAME = tzlocal.get_localzone().zone

def test_mktz():
    if False:
        for i in range(10):
            print('nop')
    tz = mktz('Europe/London')
    d = dt(2012, 2, 2, tzinfo=tz)
    assert d.tzname() == 'GMT'
    d = dt(2012, 7, 2, tzinfo=tz)
    assert d.tzname() == 'BST'
    tz = mktz('UTC')
    d = dt(2012, 2, 2, tzinfo=tz)
    assert d.tzname() == 'UTC'
    d = dt(2012, 7, 2, tzinfo=tz)
    assert d.tzname() == 'UTC'

def test_mktz_noarg():
    if False:
        return 10
    tz = mktz()
    assert DEFAULT_TIME_ZONE_NAME in str(tz)

def test_mktz_zone():
    if False:
        while True:
            i = 10
    tz = mktz('UTC')
    assert tz.zone == 'UTC'
    tz = mktz('/usr/share/zoneinfo/UTC')
    assert tz.zone == 'UTC'

def test_mktz_fails_if_invalid_timezone():
    if False:
        i = 10
        return i + 15
    with patch('os.path.exists') as file_exists:
        file_exists.return_value = False
        with raises(TimezoneError):
            mktz('junk')