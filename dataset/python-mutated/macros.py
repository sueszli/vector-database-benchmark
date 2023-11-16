import builtins
import datetime
import numbers
from typing import Union
from dateutil import parser
from isodate import parse_duration
'\nThis file contains macros that can be evaluated by a `JinjaInterpolation` object\n'

def now_utc():
    if False:
        while True:
            i = 10
    '\n    Current local date and time in UTC timezone\n\n    Usage:\n    `"{{ now_utc() }}"`\n    '
    return datetime.datetime.now(datetime.timezone.utc)

def today_utc():
    if False:
        return 10
    '\n    Current date in UTC timezone\n\n    Usage:\n    `"{{ today_utc() }}"`\n    '
    return datetime.datetime.now(datetime.timezone.utc).date()

def timestamp(dt: Union[numbers.Number, str]):
    if False:
        print('Hello World!')
    '\n    Converts a number or a string to a timestamp\n\n    If dt is a number, then convert to an int\n    If dt is a string, then parse it using dateutil.parser\n\n    Usage:\n    `"{{ timestamp(1658505815.223235) }}"\n\n    :param dt: datetime to convert to timestamp\n    :return: unix timestamp\n    '
    if isinstance(dt, numbers.Number):
        return int(dt)
    else:
        return _str_to_datetime(dt).astimezone(datetime.timezone.utc).timestamp()

def _str_to_datetime(s: str) -> datetime.datetime:
    if False:
        while True:
            i = 10
    parsed_date = parser.isoparse(s)
    if not parsed_date.tzinfo:
        parsed_date = parsed_date.replace(tzinfo=datetime.timezone.utc)
    return parsed_date.astimezone(datetime.timezone.utc)

def max(*args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns biggest object of an iterable, or two or more arguments.\n\n    max(iterable, *[, default=obj, key=func]) -> value\n    max(arg1, arg2, *args, *[, key=func]) -> value\n\n    Usage:\n    `"{{ max(2,3) }}"\n\n    With a single iterable argument, return its biggest item. The\n    default keyword-only argument specifies an object to return if\n    the provided iterable is empty.\n    With two or more arguments, return the largest argument.\n    :param args: args to compare\n    :return: largest argument\n    '
    return builtins.max(*args)

def day_delta(num_days: int, format: str='%Y-%m-%dT%H:%M:%S.%f%z') -> str:
    if False:
        print('Hello World!')
    '\n    Returns datetime of now() + num_days\n\n    Usage:\n    `"{{ day_delta(25) }}"`\n\n    :param num_days: number of days to add to current date time\n    :return: datetime formatted as RFC3339\n    '
    return (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=num_days)).strftime(format)

def duration(datestring: str) -> datetime.timedelta:
    if False:
        i = 10
        return i + 15
    '\n    Converts ISO8601 duration to datetime.timedelta\n\n    Usage:\n    `"{{ now_utc() - duration(\'P1D\') }}"`\n    '
    return parse_duration(datestring)

def format_datetime(dt: Union[str, datetime.datetime], format: str) -> str:
    if False:
        return 10
    '\n    Converts datetime to another format\n\n    Usage:\n    `"{{ format_datetime(config.start_date, \'%Y-%m-%d\') }}"`\n    '
    if isinstance(dt, datetime.datetime):
        return dt.strftime(format)
    return _str_to_datetime(dt).strftime(format)
_macros_list = [now_utc, today_utc, timestamp, max, day_delta, duration, format_datetime]
macros = {f.__name__: f for f in _macros_list}