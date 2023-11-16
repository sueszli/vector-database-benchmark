import datetime
from typing import Union

class DatetimeParser:
    """
    Parses and formats datetime objects according to a specified format.

    This class mainly acts as a wrapper to properly handling timestamp formatting through the "%s" directive.

    %s is part of the list of format codes required by  the 1989 C standard, but it is unreliable because it always return a datetime in the system's timezone.
    Instead of using the directive directly, we can use datetime.fromtimestamp and dt.timestamp()
    """
    _UNIX_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)

    def parse(self, date: Union[str, int], format: str) -> datetime.datetime:
        if False:
            i = 10
            return i + 15
        if format == '%s':
            return datetime.datetime.fromtimestamp(int(date), tz=datetime.timezone.utc)
        elif format == '%ms':
            return self._UNIX_EPOCH + datetime.timedelta(milliseconds=int(date))
        parsed_datetime = datetime.datetime.strptime(str(date), format)
        if self._is_naive(parsed_datetime):
            return parsed_datetime.replace(tzinfo=datetime.timezone.utc)
        return parsed_datetime

    def format(self, dt: datetime.datetime, format: str) -> str:
        if False:
            while True:
                i = 10
        if format == '%s':
            return str(int(dt.timestamp()))
        if format == '%ms':
            return str(int(dt.timestamp() * 1000))
        else:
            return dt.strftime(format)

    def _is_naive(self, dt: datetime.datetime) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None