import datetime as dt
from calendar import monthrange
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Optional
from hypothesis.errors import InvalidArgument
from hypothesis.internal.conjecture import utils
from hypothesis.internal.validation import check_type, check_valid_interval
from hypothesis.strategies._internal.core import sampled_from
from hypothesis.strategies._internal.misc import just, none
from hypothesis.strategies._internal.strategies import SearchStrategy
from hypothesis.strategies._internal.utils import defines_strategy
try:
    import zoneinfo
except ImportError:
    try:
        from backports import zoneinfo
    except ImportError:
        zoneinfo = None
DATENAMES = ('year', 'month', 'day')
TIMENAMES = ('hour', 'minute', 'second', 'microsecond')

def is_pytz_timezone(tz):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(tz, dt.tzinfo):
        return False
    module = type(tz).__module__
    return module == 'pytz' or module.startswith('pytz.')

def replace_tzinfo(value, timezone):
    if False:
        return 10
    if is_pytz_timezone(timezone):
        return timezone.localize(value, is_dst=not value.fold)
    return value.replace(tzinfo=timezone)

def datetime_does_not_exist(value):
    if False:
        for i in range(10):
            print('nop')
    'This function tests whether the given datetime can be round-tripped to and\n    from UTC.  It is an exact inverse of (and very similar to) the dateutil method\n    https://dateutil.readthedocs.io/en/stable/tz.html#dateutil.tz.datetime_exists\n    '
    if value.tzinfo is None:
        return False
    try:
        roundtrip = value.astimezone(dt.timezone.utc).astimezone(value.tzinfo)
    except OverflowError:
        return True
    if value.tzinfo is not roundtrip.tzinfo and value.utcoffset() != roundtrip.utcoffset():
        return True
    assert value.tzinfo is roundtrip.tzinfo, 'so only the naive portions are compared'
    return value != roundtrip

def draw_capped_multipart(data, min_value, max_value, duration_names=DATENAMES + TIMENAMES):
    if False:
        print('Hello World!')
    assert isinstance(min_value, (dt.date, dt.time, dt.datetime))
    assert type(min_value) == type(max_value)
    assert min_value <= max_value
    result = {}
    (cap_low, cap_high) = (True, True)
    for name in duration_names:
        low = getattr(min_value if cap_low else dt.datetime.min, name)
        high = getattr(max_value if cap_high else dt.datetime.max, name)
        if name == 'day' and (not cap_high):
            (_, high) = monthrange(**result)
        if name == 'year':
            val = utils.integer_range(data, low, high, 2000)
        else:
            val = utils.integer_range(data, low, high)
        result[name] = val
        cap_low = cap_low and val == low
        cap_high = cap_high and val == high
    if hasattr(min_value, 'fold'):
        result['fold'] = utils.integer_range(data, 0, 1)
    return result

class DatetimeStrategy(SearchStrategy):

    def __init__(self, min_value, max_value, timezones_strat, allow_imaginary):
        if False:
            print('Hello World!')
        assert isinstance(min_value, dt.datetime)
        assert isinstance(max_value, dt.datetime)
        assert min_value.tzinfo is None
        assert max_value.tzinfo is None
        assert min_value <= max_value
        assert isinstance(timezones_strat, SearchStrategy)
        assert isinstance(allow_imaginary, bool)
        self.min_value = min_value
        self.max_value = max_value
        self.tz_strat = timezones_strat
        self.allow_imaginary = allow_imaginary

    def do_draw(self, data):
        if False:
            for i in range(10):
                print('nop')
        tz = data.draw(self.tz_strat)
        result = self.draw_naive_datetime_and_combine(data, tz)
        if not self.allow_imaginary and datetime_does_not_exist(result):
            data.mark_invalid('nonexistent datetime')
        return result

    def draw_naive_datetime_and_combine(self, data, tz):
        if False:
            print('Hello World!')
        result = draw_capped_multipart(data, self.min_value, self.max_value)
        try:
            return replace_tzinfo(dt.datetime(**result), timezone=tz)
        except (ValueError, OverflowError):
            msg = 'Failed to draw a datetime between %r and %r with timezone from %r.'
            data.mark_invalid(msg % (self.min_value, self.max_value, self.tz_strat))

@defines_strategy(force_reusable_values=True)
def datetimes(min_value: dt.datetime=dt.datetime.min, max_value: dt.datetime=dt.datetime.max, *, timezones: SearchStrategy[Optional[dt.tzinfo]]=none(), allow_imaginary: bool=True) -> SearchStrategy[dt.datetime]:
    if False:
        return 10
    'datetimes(min_value=datetime.datetime.min, max_value=datetime.datetime.max, *, timezones=none(), allow_imaginary=True)\n\n    A strategy for generating datetimes, which may be timezone-aware.\n\n    This strategy works by drawing a naive datetime between ``min_value``\n    and ``max_value``, which must both be naive (have no timezone).\n\n    ``timezones`` must be a strategy that generates either ``None``, for naive\n    datetimes, or :class:`~python:datetime.tzinfo` objects for \'aware\' datetimes.\n    You can construct your own, though we recommend using one of these built-in\n    strategies:\n\n    * with Python 3.9 or newer or :pypi:`backports.zoneinfo`:\n      :func:`hypothesis.strategies.timezones`;\n    * with :pypi:`dateutil <python-dateutil>`:\n      :func:`hypothesis.extra.dateutil.timezones`; or\n    * with :pypi:`pytz`: :func:`hypothesis.extra.pytz.timezones`.\n\n    You may pass ``allow_imaginary=False`` to filter out "imaginary" datetimes\n    which did not (or will not) occur due to daylight savings, leap seconds,\n    timezone and calendar adjustments, etc.  Imaginary datetimes are allowed\n    by default, because malformed timestamps are a common source of bugs.\n\n    Examples from this strategy shrink towards midnight on January 1st 2000,\n    local time.\n    '
    check_type(bool, allow_imaginary, 'allow_imaginary')
    check_type(dt.datetime, min_value, 'min_value')
    check_type(dt.datetime, max_value, 'max_value')
    if min_value.tzinfo is not None:
        raise InvalidArgument(f'min_value={min_value!r} must not have tzinfo')
    if max_value.tzinfo is not None:
        raise InvalidArgument(f'max_value={max_value!r} must not have tzinfo')
    check_valid_interval(min_value, max_value, 'min_value', 'max_value')
    if not isinstance(timezones, SearchStrategy):
        raise InvalidArgument(f'timezones={timezones!r} must be a SearchStrategy that can provide tzinfo for datetimes (either None or dt.tzinfo objects)')
    return DatetimeStrategy(min_value, max_value, timezones, allow_imaginary)

class TimeStrategy(SearchStrategy):

    def __init__(self, min_value, max_value, timezones_strat):
        if False:
            return 10
        self.min_value = min_value
        self.max_value = max_value
        self.tz_strat = timezones_strat

    def do_draw(self, data):
        if False:
            i = 10
            return i + 15
        result = draw_capped_multipart(data, self.min_value, self.max_value, TIMENAMES)
        tz = data.draw(self.tz_strat)
        return dt.time(**result, tzinfo=tz)

@defines_strategy(force_reusable_values=True)
def times(min_value: dt.time=dt.time.min, max_value: dt.time=dt.time.max, *, timezones: SearchStrategy[Optional[dt.tzinfo]]=none()) -> SearchStrategy[dt.time]:
    if False:
        return 10
    'times(min_value=datetime.time.min, max_value=datetime.time.max, *, timezones=none())\n\n    A strategy for times between ``min_value`` and ``max_value``.\n\n    The ``timezones`` argument is handled as for :py:func:`datetimes`.\n\n    Examples from this strategy shrink towards midnight, with the timezone\n    component shrinking as for the strategy that provided it.\n    '
    check_type(dt.time, min_value, 'min_value')
    check_type(dt.time, max_value, 'max_value')
    if min_value.tzinfo is not None:
        raise InvalidArgument(f'min_value={min_value!r} must not have tzinfo')
    if max_value.tzinfo is not None:
        raise InvalidArgument(f'max_value={max_value!r} must not have tzinfo')
    check_valid_interval(min_value, max_value, 'min_value', 'max_value')
    return TimeStrategy(min_value, max_value, timezones)

class DateStrategy(SearchStrategy):

    def __init__(self, min_value, max_value):
        if False:
            return 10
        assert isinstance(min_value, dt.date)
        assert isinstance(max_value, dt.date)
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value

    def do_draw(self, data):
        if False:
            i = 10
            return i + 15
        return dt.date(**draw_capped_multipart(data, self.min_value, self.max_value, DATENAMES))

@defines_strategy(force_reusable_values=True)
def dates(min_value: dt.date=dt.date.min, max_value: dt.date=dt.date.max) -> SearchStrategy[dt.date]:
    if False:
        return 10
    'dates(min_value=datetime.date.min, max_value=datetime.date.max)\n\n    A strategy for dates between ``min_value`` and ``max_value``.\n\n    Examples from this strategy shrink towards January 1st 2000.\n    '
    check_type(dt.date, min_value, 'min_value')
    check_type(dt.date, max_value, 'max_value')
    check_valid_interval(min_value, max_value, 'min_value', 'max_value')
    if min_value == max_value:
        return just(min_value)
    return DateStrategy(min_value, max_value)

class TimedeltaStrategy(SearchStrategy):

    def __init__(self, min_value, max_value):
        if False:
            print('Hello World!')
        assert isinstance(min_value, dt.timedelta)
        assert isinstance(max_value, dt.timedelta)
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value

    def do_draw(self, data):
        if False:
            print('Hello World!')
        result = {}
        low_bound = True
        high_bound = True
        for name in ('days', 'seconds', 'microseconds'):
            low = getattr(self.min_value if low_bound else dt.timedelta.min, name)
            high = getattr(self.max_value if high_bound else dt.timedelta.max, name)
            val = utils.integer_range(data, low, high, 0)
            result[name] = val
            low_bound = low_bound and val == low
            high_bound = high_bound and val == high
        return dt.timedelta(**result)

@defines_strategy(force_reusable_values=True)
def timedeltas(min_value: dt.timedelta=dt.timedelta.min, max_value: dt.timedelta=dt.timedelta.max) -> SearchStrategy[dt.timedelta]:
    if False:
        for i in range(10):
            print('nop')
    'timedeltas(min_value=datetime.timedelta.min, max_value=datetime.timedelta.max)\n\n    A strategy for timedeltas between ``min_value`` and ``max_value``.\n\n    Examples from this strategy shrink towards zero.\n    '
    check_type(dt.timedelta, min_value, 'min_value')
    check_type(dt.timedelta, max_value, 'max_value')
    check_valid_interval(min_value, max_value, 'min_value', 'max_value')
    if min_value == max_value:
        return just(min_value)
    return TimedeltaStrategy(min_value=min_value, max_value=max_value)

@lru_cache(maxsize=None)
def _valid_key_cacheable(tzpath, key):
    if False:
        print('Hello World!')
    assert isinstance(tzpath, tuple)
    for root in tzpath:
        if Path(root).joinpath(key).exists():
            return True
    else:
        (*package_loc, resource_name) = key.split('/')
        package = 'tzdata.zoneinfo.' + '.'.join(package_loc)
        try:
            try:
                traversable = resources.files(package) / resource_name
                return traversable.exists()
            except (AttributeError, ValueError):
                return resources.is_resource(package, resource_name)
        except ModuleNotFoundError:
            return False

@defines_strategy(force_reusable_values=True)
def timezone_keys(*, allow_prefix: bool=True) -> SearchStrategy[str]:
    if False:
        return 10
    'A strategy for :wikipedia:`IANA timezone names <List_of_tz_database_time_zones>`.\n\n    As well as timezone names like ``"UTC"``, ``"Australia/Sydney"``, or\n    ``"America/New_York"``, this strategy can generate:\n\n    - Aliases such as ``"Antarctica/McMurdo"``, which links to ``"Pacific/Auckland"``.\n    - Deprecated names such as ``"Antarctica/South_Pole"``, which *also* links to\n      ``"Pacific/Auckland"``.  Note that most but\n      not all deprecated timezone names are also aliases.\n    - Timezone names with the ``"posix/"`` or ``"right/"`` prefixes, unless\n      ``allow_prefix=False``.\n\n    These strings are provided separately from Tzinfo objects - such as ZoneInfo\n    instances from the timezones() strategy - to facilitate testing of timezone\n    logic without needing workarounds to access non-canonical names.\n\n    .. note::\n\n        The :mod:`python:zoneinfo` module is new in Python 3.9, so you will need\n        to install the :pypi:`backports.zoneinfo` module on earlier versions.\n\n        `On Windows, you will also need to install the tzdata package\n        <https://docs.python.org/3/library/zoneinfo.html#data-sources>`__.\n\n        ``pip install hypothesis[zoneinfo]`` will install these conditional\n        dependencies if and only if they are needed.\n\n    On Windows, you may need to access IANA timezone data via the :pypi:`tzdata`\n    package.  For non-IANA timezones, such as Windows-native names or GNU TZ\n    strings, we recommend using :func:`~hypothesis.strategies.sampled_from` with\n    the :pypi:`dateutil <python-dateutil>` package, e.g.\n    :meth:`dateutil:dateutil.tz.tzwin.list`.\n    '
    check_type(bool, allow_prefix, 'allow_prefix')
    if zoneinfo is None:
        raise ModuleNotFoundError('The zoneinfo module is required, but could not be imported.  Run `pip install hypothesis[zoneinfo]` and try again.')
    available_timezones = ('UTC', *sorted(zoneinfo.available_timezones()))

    def valid_key(key):
        if False:
            print('Hello World!')
        return key == 'UTC' or _valid_key_cacheable(zoneinfo.TZPATH, key)
    strategy = sampled_from([key for key in available_timezones if valid_key(key)])
    if not allow_prefix:
        return strategy

    def sample_with_prefixes(zone):
        if False:
            while True:
                i = 10
        keys_with_prefixes = (zone, f'posix/{zone}', f'right/{zone}')
        return sampled_from([key for key in keys_with_prefixes if valid_key(key)])
    return strategy.flatmap(sample_with_prefixes)

@defines_strategy(force_reusable_values=True)
def timezones(*, no_cache: bool=False) -> SearchStrategy['zoneinfo.ZoneInfo']:
    if False:
        for i in range(10):
            print('nop')
    'A strategy for :class:`python:zoneinfo.ZoneInfo` objects.\n\n    If ``no_cache=True``, the generated instances are constructed using\n    :meth:`ZoneInfo.no_cache <python:zoneinfo.ZoneInfo.no_cache>` instead\n    of the usual constructor.  This may change the semantics of your datetimes\n    in surprising ways, so only use it if you know that you need to!\n\n    .. note::\n\n        The :mod:`python:zoneinfo` module is new in Python 3.9, so you will need\n        to install the :pypi:`backports.zoneinfo` module on earlier versions.\n\n        `On Windows, you will also need to install the tzdata package\n        <https://docs.python.org/3/library/zoneinfo.html#data-sources>`__.\n\n        ``pip install hypothesis[zoneinfo]`` will install these conditional\n        dependencies if and only if they are needed.\n    '
    check_type(bool, no_cache, 'no_cache')
    if zoneinfo is None:
        raise ModuleNotFoundError('The zoneinfo module is required, but could not be imported.  Run `pip install hypothesis[zoneinfo]` and try again.')
    return timezone_keys().map(zoneinfo.ZoneInfo.no_cache if no_cache else zoneinfo.ZoneInfo)