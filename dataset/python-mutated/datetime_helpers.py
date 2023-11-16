import re
from datetime import datetime, timezone
from typing import Optional
import arrow
from freqtrade.constants import DATETIME_PRINT_FORMAT

def dt_now() -> datetime:
    if False:
        for i in range(10):
            print('nop')
    'Return the current datetime in UTC.'
    return datetime.now(timezone.utc)

def dt_utc(year: int, month: int, day: int, hour: int=0, minute: int=0, second: int=0, microsecond: int=0) -> datetime:
    if False:
        print('Hello World!')
    'Return a datetime in UTC.'
    return datetime(year, month, day, hour, minute, second, microsecond, tzinfo=timezone.utc)

def dt_ts(dt: Optional[datetime]=None) -> int:
    if False:
        print('Hello World!')
    '\n    Return dt in ms as a timestamp in UTC.\n    If dt is None, return the current datetime in UTC.\n    '
    if dt:
        return int(dt.timestamp() * 1000)
    return int(dt_now().timestamp() * 1000)

def dt_ts_def(dt: Optional[datetime], default: int=0) -> int:
    if False:
        for i in range(10):
            print('nop')
    '\n    Return dt in ms as a timestamp in UTC.\n    If dt is None, return the current datetime in UTC.\n    '
    if dt:
        return int(dt.timestamp() * 1000)
    return default

def dt_floor_day(dt: datetime) -> datetime:
    if False:
        return 10
    'Return the floor of the day for the given datetime.'
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)

def dt_from_ts(timestamp: float) -> datetime:
    if False:
        i = 10
        return i + 15
    '\n    Return a datetime from a timestamp.\n    :param timestamp: timestamp in seconds or milliseconds\n    '
    if timestamp > 10000000000.0:
        timestamp /= 1000
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)

def shorten_date(_date: str) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Trim the date so it fits on small screens\n    '
    new_date = re.sub('seconds?', 'sec', _date)
    new_date = re.sub('minutes?', 'min', new_date)
    new_date = re.sub('hours?', 'h', new_date)
    new_date = re.sub('days?', 'd', new_date)
    new_date = re.sub('^an?', '1', new_date)
    return new_date

def dt_humanize(dt: datetime, **kwargs) -> str:
    if False:
        print('Hello World!')
    "\n    Return a humanized string for the given datetime.\n    :param dt: datetime to humanize\n    :param kwargs: kwargs to pass to arrow's humanize()\n    "
    return arrow.get(dt).humanize(**kwargs)

def format_date(date: Optional[datetime]) -> str:
    if False:
        while True:
            i = 10
    '\n    Return a formatted date string.\n    Returns an empty string if date is None.\n    :param date: datetime to format\n    '
    if date:
        return date.strftime(DATETIME_PRINT_FORMAT)
    return ''

def format_ms_time(date: int) -> str:
    if False:
        print('Hello World!')
    '\n    convert MS date to readable format.\n    : epoch-string in ms\n    '
    return datetime.fromtimestamp(date / 1000.0).strftime('%Y-%m-%dT%H:%M:%S')