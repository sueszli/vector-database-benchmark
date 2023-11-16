from datetime import timedelta
from dateutil import tz
UTC = tz.UTC

def get_timezone(key):
    if False:
        print('Hello World!')
    if not key:
        raise KeyError('Unknown time zone: %s' % key)
    try:
        rv = tz.gettz(key)
    except Exception:
        rv = None
    if rv is None or not isinstance(rv, (tz.tzutc, tz.tzfile)):
        raise KeyError('Unknown time zone: %s' % key)
    return rv

def get_timezone_file(f, key=None):
    if False:
        print('Hello World!')
    return tz.tzfile(f)

def get_fixed_offset_zone(offset):
    if False:
        return 10
    return tz.tzoffset(None, timedelta(minutes=offset))

def is_ambiguous(dt):
    if False:
        for i in range(10):
            print('nop')
    return tz.datetime_ambiguous(dt)

def is_imaginary(dt):
    if False:
        return 10
    return not tz.datetime_exists(dt)
enfold = tz.enfold

def get_fold(dt):
    if False:
        print('Hello World!')
    return getattr(dt, 'fold', 0)