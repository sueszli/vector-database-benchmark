"""Time utils for getting and converting datetime objects for the Mycroft
system. This time is based on the setting in the Mycroft config and may or
may not match the system locale.
"""
from datetime import datetime
from dateutil.tz import gettz, tzlocal

def default_timezone():
    if False:
        while True:
            i = 10
    'Get the default timezone\n\n    Based on user location settings location.timezone.code or\n    the default system value if no setting exists.\n\n    Returns:\n        (datetime.tzinfo): Definition of the default timezone\n    '
    try:
        from mycroft.configuration import Configuration
        config = Configuration.get()
        code = config['location']['timezone']['code']
        return gettz(code)
    except Exception:
        return tzlocal()

def now_utc():
    if False:
        print('Hello World!')
    'Retrieve the current time in UTC\n\n    Returns:\n        (datetime): The current time in Universal Time, aka GMT\n    '
    return to_utc(datetime.utcnow())

def now_local(tz=None):
    if False:
        for i in range(10):
            print('nop')
    "Retrieve the current time\n\n    Args:\n        tz (datetime.tzinfo, optional): Timezone, default to user's settings\n\n    Returns:\n        (datetime): The current time\n    "
    if not tz:
        tz = default_timezone()
    return datetime.now(tz)

def to_utc(dt):
    if False:
        i = 10
        return i + 15
    'Convert a datetime with timezone info to a UTC datetime\n\n    Args:\n        dt (datetime): A datetime (presumably in some local zone)\n    Returns:\n        (datetime): time converted to UTC\n    '
    tzUTC = gettz('UTC')
    if dt.tzinfo:
        return dt.astimezone(tzUTC)
    else:
        return dt.replace(tzinfo=gettz('UTC')).astimezone(tzUTC)

def to_local(dt):
    if False:
        i = 10
        return i + 15
    "Convert a datetime to the user's local timezone\n\n    Args:\n        dt (datetime): A datetime (if no timezone, defaults to UTC)\n    Returns:\n        (datetime): time converted to the local timezone\n    "
    tz = default_timezone()
    if dt.tzinfo:
        return dt.astimezone(tz)
    else:
        return dt.replace(tzinfo=gettz('UTC')).astimezone(tz)

def to_system(dt):
    if False:
        while True:
            i = 10
    "Convert a datetime to the system's local timezone\n\n    Args:\n        dt (datetime): A datetime (if no timezone, assumed to be UTC)\n    Returns:\n        (datetime): time converted to the operation system's timezone\n    "
    tz = tzlocal()
    if dt.tzinfo:
        return dt.astimezone(tz)
    else:
        return dt.replace(tzinfo=gettz('UTC')).astimezone(tz)