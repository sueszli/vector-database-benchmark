from __future__ import annotations
import datetime as dt
from typing import overload
import pendulum
from dateutil.relativedelta import relativedelta
from pendulum.datetime import DateTime
utc = pendulum.tz.timezone('UTC')

def is_localized(value):
    if False:
        return 10
    'Determine if a given datetime.datetime is aware.\n\n    The concept is defined in Python documentation. Assuming the tzinfo is\n    either None or a proper ``datetime.tzinfo`` instance, ``value.utcoffset()``\n    implements the appropriate logic.\n\n    .. seealso:: http://docs.python.org/library/datetime.html#datetime.tzinfo\n    '
    return value.utcoffset() is not None

def is_naive(value):
    if False:
        i = 10
        return i + 15
    'Determine if a given datetime.datetime is naive.\n\n    The concept is defined in Python documentation. Assuming the tzinfo is\n    either None or a proper ``datetime.tzinfo`` instance, ``value.utcoffset()``\n    implements the appropriate logic.\n\n    .. seealso:: http://docs.python.org/library/datetime.html#datetime.tzinfo\n    '
    return value.utcoffset() is None

def utcnow() -> dt.datetime:
    if False:
        for i in range(10):
            print('nop')
    'Get the current date and time in UTC.'
    result = dt.datetime.utcnow()
    result = result.replace(tzinfo=utc)
    return result

def utc_epoch() -> dt.datetime:
    if False:
        return 10
    "Get the epoch in the user's timezone."
    result = dt.datetime(1970, 1, 1)
    result = result.replace(tzinfo=utc)
    return result

@overload
def convert_to_utc(value: None) -> None:
    if False:
        while True:
            i = 10
    ...

@overload
def convert_to_utc(value: dt.datetime) -> DateTime:
    if False:
        for i in range(10):
            print('nop')
    ...

def convert_to_utc(value: dt.datetime | None) -> DateTime | None:
    if False:
        print('Hello World!')
    'Create a datetime with the default timezone added if none is associated.\n\n    :param value: datetime\n    :return: datetime with tzinfo\n    '
    if value is None:
        return value
    if not is_localized(value):
        from airflow.settings import TIMEZONE
        value = pendulum.instance(value, TIMEZONE)
    return pendulum.instance(value.astimezone(utc))

@overload
def make_aware(value: None, timezone: dt.tzinfo | None=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def make_aware(value: DateTime, timezone: dt.tzinfo | None=None) -> DateTime:
    if False:
        print('Hello World!')
    ...

@overload
def make_aware(value: dt.datetime, timezone: dt.tzinfo | None=None) -> dt.datetime:
    if False:
        return 10
    ...

def make_aware(value: dt.datetime | None, timezone: dt.tzinfo | None=None) -> dt.datetime | None:
    if False:
        return 10
    '\n    Make a naive datetime.datetime in a given time zone aware.\n\n    :param value: datetime\n    :param timezone: timezone\n    :return: localized datetime in settings.TIMEZONE or timezone\n    '
    if timezone is None:
        from airflow.settings import TIMEZONE
        timezone = TIMEZONE
    if not value:
        return None
    if is_localized(value):
        raise ValueError(f'make_aware expects a naive datetime, got {value}')
    if hasattr(value, 'fold'):
        value = value.replace(fold=1)
    localized = getattr(timezone, 'localize', None)
    if localized is not None:
        return localized(value)
    convert = getattr(timezone, 'convert', None)
    if convert is not None:
        return convert(value)
    return value.replace(tzinfo=timezone)

def make_naive(value, timezone=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Make an aware datetime.datetime naive in a given time zone.\n\n    :param value: datetime\n    :param timezone: timezone\n    :return: naive datetime\n    '
    if timezone is None:
        from airflow.settings import TIMEZONE
        timezone = TIMEZONE
    if is_naive(value):
        raise ValueError('make_naive() cannot be applied to a naive datetime')
    date = value.astimezone(timezone)
    naive = dt.datetime(date.year, date.month, date.day, date.hour, date.minute, date.second, date.microsecond)
    return naive

def datetime(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Wrap around datetime.datetime to add settings.TIMEZONE if tzinfo not specified.\n\n    :return: datetime.datetime\n    '
    if 'tzinfo' not in kwargs:
        from airflow.settings import TIMEZONE
        kwargs['tzinfo'] = TIMEZONE
    return dt.datetime(*args, **kwargs)

def parse(string: str, timezone=None, *, strict=False) -> DateTime:
    if False:
        print('Hello World!')
    '\n    Parse a time string and return an aware datetime.\n\n    :param string: time string\n    :param timezone: the timezone\n    :param strict: if False, it will fall back on the dateutil parser if unable to parse with pendulum\n    '
    from airflow.settings import TIMEZONE
    return pendulum.parse(string, tz=timezone or TIMEZONE, strict=strict)

@overload
def coerce_datetime(v: None, tz: dt.tzinfo | None=None) -> None:
    if False:
        print('Hello World!')
    ...

@overload
def coerce_datetime(v: DateTime, tz: dt.tzinfo | None=None) -> DateTime:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def coerce_datetime(v: dt.datetime, tz: dt.tzinfo | None=None) -> DateTime:
    if False:
        print('Hello World!')
    ...

def coerce_datetime(v: dt.datetime | None, tz: dt.tzinfo | None=None) -> DateTime | None:
    if False:
        while True:
            i = 10
    'Convert ``v`` into a timezone-aware ``pendulum.DateTime``.\n\n    * If ``v`` is *None*, *None* is returned.\n    * If ``v`` is a naive datetime, it is converted to an aware Pendulum DateTime.\n    * If ``v`` is an aware datetime, it is converted to a Pendulum DateTime.\n      Note that ``tz`` is **not** taken into account in this case; the datetime\n      will maintain its original tzinfo!\n    '
    if v is None:
        return None
    if isinstance(v, DateTime):
        return v if v.tzinfo else make_aware(v, tz)
    return pendulum.instance(v if v.tzinfo else make_aware(v, tz))

def td_format(td_object: None | dt.timedelta | float | int) -> str | None:
    if False:
        return 10
    '\n    Format a timedelta object or float/int into a readable string for time duration.\n\n    For example timedelta(seconds=3752) would become `1h:2M:32s`.\n    If the time is less than a second, the return will be `<1s`.\n    '
    if not td_object:
        return None
    if isinstance(td_object, dt.timedelta):
        delta = relativedelta() + td_object
    else:
        delta = relativedelta(seconds=int(td_object))
    (months, delta.days) = divmod(delta.days, 30)
    delta = delta.normalized() + relativedelta(months=months)

    def _format_part(key: str) -> str:
        if False:
            print('Hello World!')
        value = int(getattr(delta, key))
        if value < 1:
            return ''
        if key == 'minutes':
            key = key.upper()
        key = key[0]
        return f'{value}{key}'
    parts = map(_format_part, ('years', 'months', 'days', 'hours', 'minutes', 'seconds'))
    joined = ':'.join((part for part in parts if part))
    if not joined:
        return '<1s'
    return joined