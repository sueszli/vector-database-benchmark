"""
Convenience functions for dealing with datetime classes
"""
import datetime
import salt.utils.stringutils
from salt.utils.decorators.jinja import jinja_filter
try:
    import timelib
    HAS_TIMELIB = True
except ImportError:
    HAS_TIMELIB = False

def date_cast(date):
    if False:
        for i in range(10):
            print('nop')
    '\n    Casts any object into a datetime.datetime object\n\n    date\n      any datetime, time string representation...\n    '
    if date is None:
        return datetime.datetime.now()
    elif isinstance(date, datetime.datetime):
        return date
    try:
        if isinstance(date, str):
            try:
                if HAS_TIMELIB:
                    return timelib.strtodatetime(salt.utils.stringutils.to_bytes(date))
            except ValueError:
                pass
            if date.isdigit():
                date = int(date)
            else:
                date = float(date)
        return datetime.datetime.fromtimestamp(date)
    except Exception:
        if HAS_TIMELIB:
            raise ValueError('Unable to parse {}'.format(date))
        raise RuntimeError('Unable to parse {}. Consider installing timelib'.format(date))

@jinja_filter('date_format')
@jinja_filter('strftime')
def strftime(date=None, format='%Y-%m-%d'):
    if False:
        i = 10
        return i + 15
    "\n    Converts date into a time-based string\n\n    date\n      any datetime, time string representation...\n\n    format\n       :ref:`strftime<http://docs.python.org/2/library/datetime.html#datetime.datetime.strftime>` format\n\n    >>> import datetime\n    >>> src = datetime.datetime(2002, 12, 25, 12, 00, 00, 00)\n    >>> strftime(src)\n    '2002-12-25'\n    >>> src = '2002/12/25'\n    >>> strftime(src)\n    '2002-12-25'\n    >>> src = 1040814000\n    >>> strftime(src)\n    '2002-12-25'\n    >>> src = '1040814000'\n    >>> strftime(src)\n    '2002-12-25'\n    "
    return date_cast(date).strftime(format)

def total_seconds(td):
    if False:
        i = 10
        return i + 15
    '\n    Takes a timedelta and returns the total number of seconds\n    represented by the object. Wrapper for the total_seconds()\n    method which does not exist in versions of Python < 2.7.\n    '
    return (td.microseconds + (td.seconds + td.days * 24 * 3600) * 10 ** 6) / 10 ** 6