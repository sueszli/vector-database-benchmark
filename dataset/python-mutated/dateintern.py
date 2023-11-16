from __future__ import absolute_import, division, print_function, unicode_literals
import datetime
import math
import time as _time
from .py3 import string_types
ZERO = datetime.timedelta(0)
STDOFFSET = datetime.timedelta(seconds=-_time.timezone)
if _time.daylight:
    DSTOFFSET = datetime.timedelta(seconds=-_time.altzone)
else:
    DSTOFFSET = STDOFFSET
DSTDIFF = DSTOFFSET - STDOFFSET
TIME_MAX = datetime.time(23, 59, 59, 999990)
TIME_MIN = datetime.time.min

def tzparse(tz):
    if False:
        i = 10
        return i + 15
    tzstr = isinstance(tz, string_types)
    if tz is None or not tzstr:
        return Localizer(tz)
    try:
        import pytz
    except ImportError:
        return Localizer(tz)
    tzs = tz
    if tzs == 'CST':
        tzs = 'CST6CDT'
    try:
        tz = pytz.timezone(tzs)
    except pytz.UnknownTimeZoneError:
        return Localizer(tz)
    return tz

def Localizer(tz):
    if False:
        return 10
    import types

    def localize(self, dt):
        if False:
            for i in range(10):
                print('nop')
        return dt.replace(tzinfo=self)
    if tz is not None and (not hasattr(tz, 'localize')):
        tz.localize = types.MethodType(localize, tz)
    return tz

class _UTC(datetime.tzinfo):
    """UTC"""

    def utcoffset(self, dt):
        if False:
            while True:
                i = 10
        return ZERO

    def tzname(self, dt):
        if False:
            return 10
        return 'UTC'

    def dst(self, dt):
        if False:
            while True:
                i = 10
        return ZERO

    def localize(self, dt):
        if False:
            for i in range(10):
                print('nop')
        return dt.replace(tzinfo=self)

class _LocalTimezone(datetime.tzinfo):

    def utcoffset(self, dt):
        if False:
            print('Hello World!')
        if self._isdst(dt):
            return DSTOFFSET
        else:
            return STDOFFSET

    def dst(self, dt):
        if False:
            while True:
                i = 10
        if self._isdst(dt):
            return DSTDIFF
        else:
            return ZERO

    def tzname(self, dt):
        if False:
            while True:
                i = 10
        return _time.tzname[self._isdst(dt)]

    def _isdst(self, dt):
        if False:
            i = 10
            return i + 15
        tt = (dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, dt.weekday(), 0, 0)
        try:
            stamp = _time.mktime(tt)
        except (ValueError, OverflowError):
            return False
        tt = _time.localtime(stamp)
        return tt.tm_isdst > 0

    def localize(self, dt):
        if False:
            i = 10
            return i + 15
        return dt.replace(tzinfo=self)
UTC = _UTC()
TZLocal = _LocalTimezone()
HOURS_PER_DAY = 24.0
MINUTES_PER_HOUR = 60.0
SECONDS_PER_MINUTE = 60.0
MUSECONDS_PER_SECOND = 1000000.0
MINUTES_PER_DAY = MINUTES_PER_HOUR * HOURS_PER_DAY
SECONDS_PER_DAY = SECONDS_PER_MINUTE * MINUTES_PER_DAY
MUSECONDS_PER_DAY = MUSECONDS_PER_SECOND * SECONDS_PER_DAY

def num2date(x, tz=None, naive=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    *x* is a float value which gives the number of days\n    (fraction part represents hours, minutes, seconds) since\n    0001-01-01 00:00:00 UTC *plus* *one*.\n    The addition of one here is a historical artifact.  Also, note\n    that the Gregorian calendar is assumed; this is not universal\n    practice.  For details, see the module docstring.\n    Return value is a :class:`datetime` instance in timezone *tz* (default to\n    rcparams TZ value).\n    If *x* is a sequence, a sequence of :class:`datetime` objects will\n    be returned.\n    '
    ix = int(x)
    dt = datetime.datetime.fromordinal(ix)
    remainder = float(x) - ix
    (hour, remainder) = divmod(HOURS_PER_DAY * remainder, 1)
    (minute, remainder) = divmod(MINUTES_PER_HOUR * remainder, 1)
    (second, remainder) = divmod(SECONDS_PER_MINUTE * remainder, 1)
    microsecond = int(MUSECONDS_PER_SECOND * remainder)
    if microsecond < 10:
        microsecond = 0
    if True and tz is not None:
        dt = datetime.datetime(dt.year, dt.month, dt.day, int(hour), int(minute), int(second), microsecond, tzinfo=UTC)
        dt = dt.astimezone(tz)
        if naive:
            dt = dt.replace(tzinfo=None)
    else:
        dt = datetime.datetime(dt.year, dt.month, dt.day, int(hour), int(minute), int(second), microsecond)
    if microsecond > 999990:
        dt += datetime.timedelta(microseconds=1000000.0 - microsecond)
    return dt

def num2dt(num, tz=None, naive=True):
    if False:
        while True:
            i = 10
    return num2date(num, tz=tz, naive=naive).date()

def num2time(num, tz=None, naive=True):
    if False:
        for i in range(10):
            print('nop')
    return num2date(num, tz=tz, naive=naive).time()

def date2num(dt, tz=None):
    if False:
        return 10
    '\n    Convert :mod:`datetime` to the Gregorian date as UTC float days,\n    preserving hours, minutes, seconds and microseconds.  Return value\n    is a :func:`float`.\n    '
    if tz is not None:
        dt = tz.localize(dt)
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        delta = dt.tzinfo.utcoffset(dt)
        if delta is not None:
            dt -= delta
    base = float(dt.toordinal())
    if hasattr(dt, 'hour'):
        base = math.fsum((base, dt.hour / HOURS_PER_DAY, dt.minute / MINUTES_PER_DAY, dt.second / SECONDS_PER_DAY, dt.microsecond / MUSECONDS_PER_DAY))
    return base

def time2num(tm):
    if False:
        for i in range(10):
            print('nop')
    '\n    Converts the hour/minute/second/microsecond part of tm (datetime.datetime\n    or time) to a num\n    '
    num = tm.hour / HOURS_PER_DAY + tm.minute / MINUTES_PER_DAY + tm.second / SECONDS_PER_DAY + tm.microsecond / MUSECONDS_PER_DAY
    return num