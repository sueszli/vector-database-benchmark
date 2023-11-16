import datetime
import time
import collections.abc
from _sqlite3 import *
paramstyle = 'qmark'
threadsafety = 1
apilevel = '2.0'
Date = datetime.date
Time = datetime.time
Timestamp = datetime.datetime

def DateFromTicks(ticks):
    if False:
        return 10
    return Date(*time.localtime(ticks)[:3])

def TimeFromTicks(ticks):
    if False:
        for i in range(10):
            print('nop')
    return Time(*time.localtime(ticks)[3:6])

def TimestampFromTicks(ticks):
    if False:
        i = 10
        return i + 15
    return Timestamp(*time.localtime(ticks)[:6])
version_info = tuple([int(x) for x in version.split('.')])
sqlite_version_info = tuple([int(x) for x in sqlite_version.split('.')])
Binary = memoryview
collections.abc.Sequence.register(Row)

def register_adapters_and_converters():
    if False:
        print('Hello World!')

    def adapt_date(val):
        if False:
            while True:
                i = 10
        return val.isoformat()

    def adapt_datetime(val):
        if False:
            while True:
                i = 10
        return val.isoformat(' ')

    def convert_date(val):
        if False:
            for i in range(10):
                print('nop')
        return datetime.date(*map(int, val.split(b'-')))

    def convert_timestamp(val):
        if False:
            return 10
        (datepart, timepart) = val.split(b' ')
        (year, month, day) = map(int, datepart.split(b'-'))
        timepart_full = timepart.split(b'.')
        (hours, minutes, seconds) = map(int, timepart_full[0].split(b':'))
        if len(timepart_full) == 2:
            microseconds = int('{:0<6.6}'.format(timepart_full[1].decode()))
        else:
            microseconds = 0
        val = datetime.datetime(year, month, day, hours, minutes, seconds, microseconds)
        return val
    register_adapter(datetime.date, adapt_date)
    register_adapter(datetime.datetime, adapt_datetime)
    register_converter('date', convert_date)
    register_converter('timestamp', convert_timestamp)
register_adapters_and_converters()

def enable_shared_cache(enable):
    if False:
        print('Hello World!')
    from _sqlite3 import enable_shared_cache as _old_enable_shared_cache
    import warnings
    msg = 'enable_shared_cache is deprecated and will be removed in Python 3.12. Shared cache is strongly discouraged by the SQLite 3 documentation. If shared cache must be used, open the database in URI mode usingthe cache=shared query parameter.'
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return _old_enable_shared_cache(enable)
del register_adapters_and_converters