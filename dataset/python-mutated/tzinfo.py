"""Base classes and helpers for building zone specific tzinfo classes"""
from datetime import datetime, timedelta, tzinfo
from bisect import bisect_right
try:
    set
except NameError:
    from sets import Set as set
import pytz
from pytz.exceptions import AmbiguousTimeError, NonExistentTimeError
__all__ = []
_timedelta_cache = {}

def memorized_timedelta(seconds):
    if False:
        return 10
    'Create only one instance of each distinct timedelta'
    try:
        return _timedelta_cache[seconds]
    except KeyError:
        delta = timedelta(seconds=seconds)
        _timedelta_cache[seconds] = delta
        return delta
_epoch = datetime.utcfromtimestamp(0)
_datetime_cache = {0: _epoch}

def memorized_datetime(seconds):
    if False:
        i = 10
        return i + 15
    'Create only one instance of each distinct datetime'
    try:
        return _datetime_cache[seconds]
    except KeyError:
        dt = _epoch + timedelta(seconds=seconds)
        _datetime_cache[seconds] = dt
        return dt
_ttinfo_cache = {}

def memorized_ttinfo(*args):
    if False:
        print('Hello World!')
    'Create only one instance of each distinct tuple'
    try:
        return _ttinfo_cache[args]
    except KeyError:
        ttinfo = (memorized_timedelta(args[0]), memorized_timedelta(args[1]), args[2])
        _ttinfo_cache[args] = ttinfo
        return ttinfo
_notime = memorized_timedelta(0)

def _to_seconds(td):
    if False:
        i = 10
        return i + 15
    'Convert a timedelta to seconds'
    return td.seconds + td.days * 24 * 60 * 60

class BaseTzInfo(tzinfo):
    _utcoffset = None
    _tzname = None
    zone = None

    def __str__(self):
        if False:
            print('Hello World!')
        return self.zone

class StaticTzInfo(BaseTzInfo):
    """A timezone that has a constant offset from UTC

    These timezones are rare, as most locations have changed their
    offset at some point in their history
    """

    def fromutc(self, dt):
        if False:
            print('Hello World!')
        'See datetime.tzinfo.fromutc'
        if dt.tzinfo is not None and dt.tzinfo is not self:
            raise ValueError('fromutc: dt.tzinfo is not self')
        return (dt + self._utcoffset).replace(tzinfo=self)

    def utcoffset(self, dt, is_dst=None):
        if False:
            i = 10
            return i + 15
        'See datetime.tzinfo.utcoffset\n\n        is_dst is ignored for StaticTzInfo, and exists only to\n        retain compatibility with DstTzInfo.\n        '
        return self._utcoffset

    def dst(self, dt, is_dst=None):
        if False:
            i = 10
            return i + 15
        'See datetime.tzinfo.dst\n\n        is_dst is ignored for StaticTzInfo, and exists only to\n        retain compatibility with DstTzInfo.\n        '
        return _notime

    def tzname(self, dt, is_dst=None):
        if False:
            while True:
                i = 10
        'See datetime.tzinfo.tzname\n\n        is_dst is ignored for StaticTzInfo, and exists only to\n        retain compatibility with DstTzInfo.\n        '
        return self._tzname

    def localize(self, dt, is_dst=False):
        if False:
            print('Hello World!')
        'Convert naive time to local time'
        if dt.tzinfo is not None:
            raise ValueError('Not naive datetime (tzinfo is already set)')
        return dt.replace(tzinfo=self)

    def normalize(self, dt, is_dst=False):
        if False:
            print('Hello World!')
        "Correct the timezone information on the given datetime.\n\n        This is normally a no-op, as StaticTzInfo timezones never have\n        ambiguous cases to correct:\n\n        >>> from pytz import timezone\n        >>> gmt = timezone('GMT')\n        >>> isinstance(gmt, StaticTzInfo)\n        True\n        >>> dt = datetime(2011, 5, 8, 1, 2, 3, tzinfo=gmt)\n        >>> gmt.normalize(dt) is dt\n        True\n\n        The supported method of converting between timezones is to use\n        datetime.astimezone(). Currently normalize() also works:\n\n        >>> la = timezone('America/Los_Angeles')\n        >>> dt = la.localize(datetime(2011, 5, 7, 1, 2, 3))\n        >>> fmt = '%Y-%m-%d %H:%M:%S %Z (%z)'\n        >>> gmt.normalize(dt).strftime(fmt)\n        '2011-05-07 08:02:03 GMT (+0000)'\n        "
        if dt.tzinfo is self:
            return dt
        if dt.tzinfo is None:
            raise ValueError('Naive time - no tzinfo set')
        return dt.astimezone(self)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return '<StaticTzInfo %r>' % (self.zone,)

    def __reduce__(self):
        if False:
            while True:
                i = 10
        return (pytz._p, (self.zone,))

class DstTzInfo(BaseTzInfo):
    """A timezone that has a variable offset from UTC

    The offset might change if daylight saving time comes into effect,
    or at a point in history when the region decides to change their
    timezone definition.
    """
    _utc_transition_times = None
    _transition_info = None
    zone = None
    _tzinfos = None
    _dst = None

    def __init__(self, _inf=None, _tzinfos=None):
        if False:
            print('Hello World!')
        if _inf:
            self._tzinfos = _tzinfos
            (self._utcoffset, self._dst, self._tzname) = _inf
        else:
            _tzinfos = {}
            self._tzinfos = _tzinfos
            (self._utcoffset, self._dst, self._tzname) = self._transition_info[0]
            _tzinfos[self._transition_info[0]] = self
            for inf in self._transition_info[1:]:
                if inf not in _tzinfos:
                    _tzinfos[inf] = self.__class__(inf, _tzinfos)

    def fromutc(self, dt):
        if False:
            print('Hello World!')
        'See datetime.tzinfo.fromutc'
        if dt.tzinfo is not None and getattr(dt.tzinfo, '_tzinfos', None) is not self._tzinfos:
            raise ValueError('fromutc: dt.tzinfo is not self')
        dt = dt.replace(tzinfo=None)
        idx = max(0, bisect_right(self._utc_transition_times, dt) - 1)
        inf = self._transition_info[idx]
        return (dt + inf[0]).replace(tzinfo=self._tzinfos[inf])

    def normalize(self, dt):
        if False:
            i = 10
            return i + 15
        "Correct the timezone information on the given datetime\n\n        If date arithmetic crosses DST boundaries, the tzinfo\n        is not magically adjusted. This method normalizes the\n        tzinfo to the correct one.\n\n        To test, first we need to do some setup\n\n        >>> from pytz import timezone\n        >>> utc = timezone('UTC')\n        >>> eastern = timezone('US/Eastern')\n        >>> fmt = '%Y-%m-%d %H:%M:%S %Z (%z)'\n\n        We next create a datetime right on an end-of-DST transition point,\n        the instant when the wallclocks are wound back one hour.\n\n        >>> utc_dt = datetime(2002, 10, 27, 6, 0, 0, tzinfo=utc)\n        >>> loc_dt = utc_dt.astimezone(eastern)\n        >>> loc_dt.strftime(fmt)\n        '2002-10-27 01:00:00 EST (-0500)'\n\n        Now, if we subtract a few minutes from it, note that the timezone\n        information has not changed.\n\n        >>> before = loc_dt - timedelta(minutes=10)\n        >>> before.strftime(fmt)\n        '2002-10-27 00:50:00 EST (-0500)'\n\n        But we can fix that by calling the normalize method\n\n        >>> before = eastern.normalize(before)\n        >>> before.strftime(fmt)\n        '2002-10-27 01:50:00 EDT (-0400)'\n\n        The supported method of converting between timezones is to use\n        datetime.astimezone(). Currently, normalize() also works:\n\n        >>> th = timezone('Asia/Bangkok')\n        >>> am = timezone('Europe/Amsterdam')\n        >>> dt = th.localize(datetime(2011, 5, 7, 1, 2, 3))\n        >>> fmt = '%Y-%m-%d %H:%M:%S %Z (%z)'\n        >>> am.normalize(dt).strftime(fmt)\n        '2011-05-06 20:02:03 CEST (+0200)'\n        "
        if dt.tzinfo is None:
            raise ValueError('Naive time - no tzinfo set')
        offset = dt.tzinfo._utcoffset
        dt = dt.replace(tzinfo=None)
        dt = dt - offset
        return self.fromutc(dt)

    def localize(self, dt, is_dst=False):
        if False:
            i = 10
            return i + 15
        "Convert naive time to local time.\n\n        This method should be used to construct localtimes, rather\n        than passing a tzinfo argument to a datetime constructor.\n\n        is_dst is used to determine the correct timezone in the ambigous\n        period at the end of daylight saving time.\n\n        >>> from pytz import timezone\n        >>> fmt = '%Y-%m-%d %H:%M:%S %Z (%z)'\n        >>> amdam = timezone('Europe/Amsterdam')\n        >>> dt  = datetime(2004, 10, 31, 2, 0, 0)\n        >>> loc_dt1 = amdam.localize(dt, is_dst=True)\n        >>> loc_dt2 = amdam.localize(dt, is_dst=False)\n        >>> loc_dt1.strftime(fmt)\n        '2004-10-31 02:00:00 CEST (+0200)'\n        >>> loc_dt2.strftime(fmt)\n        '2004-10-31 02:00:00 CET (+0100)'\n        >>> str(loc_dt2 - loc_dt1)\n        '1:00:00'\n\n        Use is_dst=None to raise an AmbiguousTimeError for ambiguous\n        times at the end of daylight saving time\n\n        >>> try:\n        ...     loc_dt1 = amdam.localize(dt, is_dst=None)\n        ... except AmbiguousTimeError:\n        ...     print('Ambiguous')\n        Ambiguous\n\n        is_dst defaults to False\n\n        >>> amdam.localize(dt) == amdam.localize(dt, False)\n        True\n\n        is_dst is also used to determine the correct timezone in the\n        wallclock times jumped over at the start of daylight saving time.\n\n        >>> pacific = timezone('US/Pacific')\n        >>> dt = datetime(2008, 3, 9, 2, 0, 0)\n        >>> ploc_dt1 = pacific.localize(dt, is_dst=True)\n        >>> ploc_dt2 = pacific.localize(dt, is_dst=False)\n        >>> ploc_dt1.strftime(fmt)\n        '2008-03-09 02:00:00 PDT (-0700)'\n        >>> ploc_dt2.strftime(fmt)\n        '2008-03-09 02:00:00 PST (-0800)'\n        >>> str(ploc_dt2 - ploc_dt1)\n        '1:00:00'\n\n        Use is_dst=None to raise a NonExistentTimeError for these skipped\n        times.\n\n        >>> try:\n        ...     loc_dt1 = pacific.localize(dt, is_dst=None)\n        ... except NonExistentTimeError:\n        ...     print('Non-existent')\n        Non-existent\n        "
        if dt.tzinfo is not None:
            raise ValueError('Not naive datetime (tzinfo is already set)')
        possible_loc_dt = set()
        for delta in [timedelta(days=-1), timedelta(days=1)]:
            loc_dt = dt + delta
            idx = max(0, bisect_right(self._utc_transition_times, loc_dt) - 1)
            inf = self._transition_info[idx]
            tzinfo = self._tzinfos[inf]
            loc_dt = tzinfo.normalize(dt.replace(tzinfo=tzinfo))
            if loc_dt.replace(tzinfo=None) == dt:
                possible_loc_dt.add(loc_dt)
        if len(possible_loc_dt) == 1:
            return possible_loc_dt.pop()
        if len(possible_loc_dt) == 0:
            if is_dst is None:
                raise NonExistentTimeError(dt)
            elif is_dst:
                return self.localize(dt + timedelta(hours=6), is_dst=True) - timedelta(hours=6)
            else:
                return self.localize(dt - timedelta(hours=6), is_dst=False) + timedelta(hours=6)
        if is_dst is None:
            raise AmbiguousTimeError(dt)
        filtered_possible_loc_dt = [p for p in possible_loc_dt if bool(p.tzinfo._dst) == is_dst]
        if len(filtered_possible_loc_dt) == 1:
            return filtered_possible_loc_dt[0]
        if len(filtered_possible_loc_dt) == 0:
            filtered_possible_loc_dt = list(possible_loc_dt)
        dates = {}
        for local_dt in filtered_possible_loc_dt:
            utc_time = local_dt.replace(tzinfo=None) - local_dt.tzinfo._utcoffset
            assert utc_time not in dates
            dates[utc_time] = local_dt
        return dates[[min, max][not is_dst](dates)]

    def utcoffset(self, dt, is_dst=None):
        if False:
            return 10
        "See datetime.tzinfo.utcoffset\n\n        The is_dst parameter may be used to remove ambiguity during DST\n        transitions.\n\n        >>> from pytz import timezone\n        >>> tz = timezone('America/St_Johns')\n        >>> ambiguous = datetime(2009, 10, 31, 23, 30)\n\n        >>> str(tz.utcoffset(ambiguous, is_dst=False))\n        '-1 day, 20:30:00'\n\n        >>> str(tz.utcoffset(ambiguous, is_dst=True))\n        '-1 day, 21:30:00'\n\n        >>> try:\n        ...     tz.utcoffset(ambiguous)\n        ... except AmbiguousTimeError:\n        ...     print('Ambiguous')\n        Ambiguous\n\n        "
        if dt is None:
            return None
        elif dt.tzinfo is not self:
            dt = self.localize(dt, is_dst)
            return dt.tzinfo._utcoffset
        else:
            return self._utcoffset

    def dst(self, dt, is_dst=None):
        if False:
            print('Hello World!')
        "See datetime.tzinfo.dst\n\n        The is_dst parameter may be used to remove ambiguity during DST\n        transitions.\n\n        >>> from pytz import timezone\n        >>> tz = timezone('America/St_Johns')\n\n        >>> normal = datetime(2009, 9, 1)\n\n        >>> str(tz.dst(normal))\n        '1:00:00'\n        >>> str(tz.dst(normal, is_dst=False))\n        '1:00:00'\n        >>> str(tz.dst(normal, is_dst=True))\n        '1:00:00'\n\n        >>> ambiguous = datetime(2009, 10, 31, 23, 30)\n\n        >>> str(tz.dst(ambiguous, is_dst=False))\n        '0:00:00'\n        >>> str(tz.dst(ambiguous, is_dst=True))\n        '1:00:00'\n        >>> try:\n        ...     tz.dst(ambiguous)\n        ... except AmbiguousTimeError:\n        ...     print('Ambiguous')\n        Ambiguous\n\n        "
        if dt is None:
            return None
        elif dt.tzinfo is not self:
            dt = self.localize(dt, is_dst)
            return dt.tzinfo._dst
        else:
            return self._dst

    def tzname(self, dt, is_dst=None):
        if False:
            for i in range(10):
                print('nop')
        "See datetime.tzinfo.tzname\n\n        The is_dst parameter may be used to remove ambiguity during DST\n        transitions.\n\n        >>> from pytz import timezone\n        >>> tz = timezone('America/St_Johns')\n\n        >>> normal = datetime(2009, 9, 1)\n\n        >>> tz.tzname(normal)\n        'NDT'\n        >>> tz.tzname(normal, is_dst=False)\n        'NDT'\n        >>> tz.tzname(normal, is_dst=True)\n        'NDT'\n\n        >>> ambiguous = datetime(2009, 10, 31, 23, 30)\n\n        >>> tz.tzname(ambiguous, is_dst=False)\n        'NST'\n        >>> tz.tzname(ambiguous, is_dst=True)\n        'NDT'\n        >>> try:\n        ...     tz.tzname(ambiguous)\n        ... except AmbiguousTimeError:\n        ...     print('Ambiguous')\n        Ambiguous\n        "
        if dt is None:
            return self.zone
        elif dt.tzinfo is not self:
            dt = self.localize(dt, is_dst)
            return dt.tzinfo._tzname
        else:
            return self._tzname

    def __repr__(self):
        if False:
            print('Hello World!')
        if self._dst:
            dst = 'DST'
        else:
            dst = 'STD'
        if self._utcoffset > _notime:
            return '<DstTzInfo %r %s+%s %s>' % (self.zone, self._tzname, self._utcoffset, dst)
        else:
            return '<DstTzInfo %r %s%s %s>' % (self.zone, self._tzname, self._utcoffset, dst)

    def __reduce__(self):
        if False:
            return 10
        return (pytz._p, (self.zone, _to_seconds(self._utcoffset), _to_seconds(self._dst), self._tzname))

def unpickler(zone, utcoffset=None, dstoffset=None, tzname=None):
    if False:
        for i in range(10):
            print('nop')
    "Factory function for unpickling pytz tzinfo instances.\n\n    This is shared for both StaticTzInfo and DstTzInfo instances, because\n    database changes could cause a zones implementation to switch between\n    these two base classes and we can't break pickles on a pytz version\n    upgrade.\n    "
    tz = pytz.timezone(zone)
    if utcoffset is None:
        return tz
    utcoffset = memorized_timedelta(utcoffset)
    dstoffset = memorized_timedelta(dstoffset)
    try:
        return tz._tzinfos[utcoffset, dstoffset, tzname]
    except KeyError:
        pass
    for localized_tz in tz._tzinfos.values():
        if localized_tz._utcoffset == utcoffset and localized_tz._dst == dstoffset:
            return localized_tz
    inf = (utcoffset, dstoffset, tzname)
    tz._tzinfos[inf] = tz.__class__(inf, tz._tzinfos)
    return tz._tzinfos[inf]