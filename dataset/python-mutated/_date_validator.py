import numpy
from calendar import timegm
from datetime import date, datetime
from dateutil.parser import parse
from pytz import UTC
from pandas import Period
from re import search
from time import mktime
from .libpsppy import t_dtype

def _normalize_timestamp(obj):
    if False:
        i = 10
        return i + 15
    'Convert a timestamp in seconds to milliseconds.\n\n    If the input overflows, it is treated as milliseconds - otherwise it is\n    treated as seconds and converted.\n    '
    try:
        datetime.fromtimestamp(obj)
        return int(obj * 1000)
    except (ValueError, OverflowError, OSError):
        return int(obj)

class _PerspectiveDateValidator(object):
    """Validate and parse dates using the `dateutil` package."""

    def parse(self, datestring):
        if False:
            while True:
                i = 10
        'Return a datetime.datetime object containing the parsed date, or\n        None if the date is invalid.\n\n        If a ISO date string with a timezone is provided, there is no guarantee\n        that timezones will be properly handled by the parser. Perspective\n        stores and serializes times in UTC as a milliseconds\n        since epoch timestamp. For more definitive timezone support, use\n        `datetime.datetime` objects or `pandas.Timestamp` objects with the\n        `timezone` property set.\n\n        Args:\n            datestring (:obj:`str`): the datestring to parse\n\n        Returns:\n            (:class:`datetime.date`/`datetime.datetime`/`None`): if parse is\n                successful.\n        '
        try:
            return parse(datestring)
        except (ValueError, OverflowError):
            return None

    def to_date_components(self, obj):
        if False:
            for i in range(10):
                print('nop')
        'Return a dictionary of string keys and integer values for `year`,\n        `month` (from 0 - 11), and `day`.\n\n        This method converts both datetime.date and numpy.datetime64 objects\n        that contain datetime.date.\n        '
        if obj is None:
            return obj
        if isinstance(obj, (int, float)):
            obj = datetime.fromtimestamp(_normalize_timestamp(obj) / 1000)
        if isinstance(obj, numpy.datetime64):
            if str(obj) == 'NaT':
                return None
            obj = obj.astype(datetime)
            if isinstance(obj, int):
                obj = datetime.fromtimestamp(obj / 1000000000)
        return {'year': obj.year, 'month': obj.month - 1, 'day': obj.day}

    def to_timestamp(self, obj):
        if False:
            i = 10
            return i + 15
        'Returns an integer corresponding to the number of milliseconds since\n        epoch in the local timezone.\n\n        If the `datetime.datetime` object has a `timezone` property set, this\n        method will convert the object into UTC before returning a timestamp.\n        '
        if obj is None:
            return obj
        if obj.__class__.__name__ == 'date':
            obj = datetime(obj.year, obj.month, obj.day)
        if isinstance(obj, Period):
            obj = obj.to_timestamp()
        converter = mktime
        to_timetuple = 'timetuple'
        if hasattr(obj, 'tzinfo') and obj.tzinfo is not None:
            obj = obj.astimezone(UTC)
            converter = timegm
            to_timetuple = 'utctimetuple'
        if isinstance(obj, numpy.datetime64):
            if str(obj) == 'NaT':
                return None
            obj = obj.astype(datetime)
            if isinstance(obj, date) and (not isinstance(obj, datetime)):
                return int(converter(getattr(obj, to_timetuple)()) * 1000)
            if isinstance(obj, int):
                return round(obj / 1000000)
        if isinstance(obj, (int, float, numpy.integer, numpy.float64, numpy.float32)):
            return _normalize_timestamp(obj)
        timetuple = getattr(obj, to_timetuple)()
        is_datetime_min = timetuple.tm_year == 1 and timetuple.tm_mon == 1 and (timetuple.tm_mday == 1) and (timetuple.tm_hour == 0) and (timetuple.tm_min == 0) and (timetuple.tm_sec == 0)
        if is_datetime_min:
            return 0
        if timetuple.tm_year < 1900:
            converter = timegm
        seconds_timestamp = converter(timetuple) + obj.microsecond / 1000000.0
        ms_timestamp = int(seconds_timestamp * 1000)
        return ms_timestamp

    def format(self, datestring):
        if False:
            print('Hello World!')
        "Return either t_dtype.DTYPE_DATE or t_dtype.DTYPE_TIME depending on\n        the format of the parsed date.\n\n        If the parsed date is invalid, return t_dtype.DTYPE_STR to prevent\n        further attempts at conversion.  Attempt to use heuristics about dates\n        to minimize false positives, i.e. do not parse dates without separators.\n\n        Args:\n            datestring (:obj:'str'): the datestring to parse.\n        "
        if isinstance(datestring, (bytes, bytearray)):
            datestring = datestring.decode('utf-8')
        has_separators = bool(search('[/. -]', datestring))
        dtype = t_dtype.DTYPE_STR
        if has_separators:
            try:
                parsed = parse(datestring)
                if (parsed.hour, parsed.minute, parsed.second, parsed.microsecond) == (0, 0, 0, 0):
                    dtype = t_dtype.DTYPE_DATE
                else:
                    dtype = t_dtype.DTYPE_TIME
            except (ValueError, OverflowError, TypeError):
                dtype = t_dtype.DTYPE_STR
        return dtype