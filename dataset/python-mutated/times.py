import re
from datetime import datetime, timezone, tzinfo
from typing import Callable, Literal, Union, overload
from isodate import LOCAL, parse_datetime
UTC = timezone.utc

def now(tz: tzinfo=UTC) -> datetime:
    if False:
        while True:
            i = 10
    return datetime.now(tz=tz)

def localnow() -> datetime:
    if False:
        for i in range(10):
            print('nop')
    return datetime.now(tz=LOCAL)

def fromtimestamp(timestamp: float, tz: tzinfo=UTC) -> datetime:
    if False:
        while True:
            i = 10
    return datetime.fromtimestamp(timestamp, tz=tz)

def fromlocaltimestamp(timestamp: float) -> datetime:
    if False:
        for i in range(10):
            print('nop')
    return datetime.fromtimestamp(timestamp, tz=LOCAL)
_re_hms_float = re.compile('^-?\\d+(?:\\.\\d+)?$')
_re_hms_s = re.compile('\n        ^\n        -?\n        (?P<seconds>\\d+(?:\\.\\d+)?)\n        s\n        $\n    ', re.VERBOSE | re.IGNORECASE)
_re_hms_ms = re.compile('\n        ^\n        -?\n        (?P<minutes>\\d+)\n        (?:(?P<sep>m)|:(?=.))\n        (?:\n            (?P<seconds>[0-5]?[0-9](?:\\.\\d+)?)\n            (?(sep)s|)\n        )?\n        $\n    ', re.VERBOSE | re.IGNORECASE)
_re_hms_hms = re.compile('\n        ^\n        -?\n        (?P<hours>\\d+)\n        (?:(?P<sep>h)|:(?=.))\n        (?:\n            (?P<minutes>[0-5]?[0-9])\n            (?(sep)m|:(?=.))\n        )?\n        (?:\n            (?P<seconds>[0-5]?[0-9](?:\\.\\d+)?)\n            (?(sep)s|)\n        )?\n        $\n    ', re.VERBOSE | re.IGNORECASE)

@overload
def _hours_minutes_seconds(as_float: Literal[False]) -> Callable[[str], int]:
    if False:
        while True:
            i = 10
    ...

@overload
def _hours_minutes_seconds(as_float: Literal[True]) -> Callable[[str], float]:
    if False:
        i = 10
        return i + 15
    ...

def _hours_minutes_seconds(as_float: bool=True) -> Callable[[str], Union[float, int]]:
    if False:
        print('Hello World!')
    '\n    Convert an optionally negative HMS-timestamp string to seconds, as float or int\n\n    Accepted formats:\n\n    - seconds\n    - minutes":"seconds\n    - hours":"minutes":"seconds\n    - seconds"s"\n    - minutes"m"\n    - hours"h"\n    - minutes"m"seconds"s"\n    - hours"h"seconds"s"\n    - hours"h"minutes"m"\n    - hours"h"minutes"m"seconds"s"\n    '

    def inner(value: str) -> Union[int, float]:
        if False:
            i = 10
            return i + 15
        if _re_hms_float.match(value):
            return float(value) if as_float else int(float(value))
        match = _re_hms_s.match(value) or _re_hms_ms.match(value) or _re_hms_hms.match(value)
        if not match:
            raise ValueError
        data = match.groupdict()
        seconds = 0.0
        seconds += float(data.get('hours') or 0.0) * 3600.0
        seconds += float(data.get('minutes') or 0.0) * 60.0
        seconds += float(data.get('seconds') or 0.0)
        res = -seconds if value[0] == '-' else seconds
        return res if as_float else int(res)
    inner.__name__ = 'hours_minutes_seconds'
    return inner
hours_minutes_seconds = _hours_minutes_seconds(as_float=False)
hours_minutes_seconds_float = _hours_minutes_seconds(as_float=True)

def seconds_to_hhmmss(seconds):
    if False:
        while True:
            i = 10
    (hours, seconds) = divmod(seconds, 3600)
    (minutes, seconds) = divmod(seconds, 60)
    return '{0:02d}:{1:02d}:{2}'.format(int(hours), int(minutes), '{0:02.1f}'.format(seconds) if seconds % 1 else '{0:02d}'.format(int(seconds)))
__all__ = ['UTC', 'LOCAL', 'parse_datetime', 'now', 'localnow', 'fromtimestamp', 'fromlocaltimestamp', 'hours_minutes_seconds', 'hours_minutes_seconds_float', 'seconds_to_hhmmss']