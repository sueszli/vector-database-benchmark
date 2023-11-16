import datetime
from typing import Union
from collections import namedtuple
import six
from dateutil.parser import parse
from rqalpha.utils.functools import lru_cache
from rqalpha.utils.exception import RQInvalidArgument
TimeRange = namedtuple('TimeRange', ['start', 'end'])

def convert_date_to_date_int(dt):
    if False:
        for i in range(10):
            print('nop')
    t = dt.year * 10000 + dt.month * 100 + dt.day
    return t

def convert_date_to_int(dt):
    if False:
        print('Hello World!')
    t = dt.year * 10000000000 + dt.month * 100000000 + dt.day * 1000000
    return t

def convert_dt_to_int(dt):
    if False:
        for i in range(10):
            print('nop')
    t = convert_date_to_int(dt)
    t += dt.hour * 10000 + dt.minute * 100 + dt.second
    return t

def convert_int_to_date(dt_int):
    if False:
        print('Hello World!')
    dt_int = int(dt_int)
    if dt_int > 100000000:
        dt_int //= 1000000
    return _convert_int_to_date(dt_int)

@lru_cache(None)
def _convert_int_to_date(dt_int):
    if False:
        i = 10
        return i + 15
    (year, r) = divmod(dt_int, 10000)
    (month, day) = divmod(r, 100)
    return datetime.datetime(year, month, day)

@lru_cache(20480)
def convert_int_to_datetime(dt_int):
    if False:
        print('Hello World!')
    dt_int = int(dt_int)
    (year, r) = divmod(dt_int, 10000000000)
    (month, r) = divmod(r, 100000000)
    (day, r) = divmod(r, 1000000)
    (hour, r) = divmod(r, 10000)
    (minute, second) = divmod(r, 100)
    return datetime.datetime(year, month, day, hour, minute, second)

def convert_ms_int_to_datetime(ms_dt_int):
    if False:
        return 10
    (dt_int, ms_int) = divmod(ms_dt_int, 1000)
    dt = convert_int_to_datetime(dt_int).replace(microsecond=ms_int * 1000)
    return dt

def convert_date_time_ms_int_to_datetime(date_int, time_int):
    if False:
        return 10
    (date_int, time_int) = (int(date_int), int(time_int))
    dt = _convert_int_to_date(date_int)
    (hours, r) = divmod(time_int, 10000000)
    (minutes, r) = divmod(r, 100000)
    (seconds, millisecond) = divmod(r, 1000)
    return dt.replace(hour=hours, minute=minutes, second=seconds, microsecond=millisecond * 1000)

def to_date(date):
    if False:
        i = 10
        return i + 15
    if isinstance(date, six.string_types):
        return parse(date).date()
    elif isinstance(date, datetime.date):
        return date
    elif isinstance(date, datetime.datetime):
        return date.date()
    else:
        raise RQInvalidArgument('unknown date value: {}'.format(date))