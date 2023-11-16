"""
Time zone utilities.
"""
from datetime import datetime as DateTime, timedelta as TimeDelta, timezone, tzinfo as TZInfo
from typing import Optional
__all__ = ['FixedOffsetTimeZone', 'UTC']

class FixedOffsetTimeZone(TZInfo):
    """
    Represents a fixed timezone offset (without daylight saving time).

    @ivar name: A L{str} giving the name of this timezone; the name just
        includes how much time this offset represents.

    @ivar offset: A L{TimeDelta} giving the amount of time this timezone is
        offset.
    """

    def __init__(self, offset: TimeDelta, name: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a L{FixedOffsetTimeZone} with a fixed offset.\n\n        @param offset: a delta representing the offset from UTC.\n        @param name: A name to be given for this timezone.\n        '
        self.offset = offset
        self.name = name

    @classmethod
    def fromSignHoursMinutes(cls, sign: str, hours: int, minutes: int) -> 'FixedOffsetTimeZone':
        if False:
            while True:
                i = 10
        "\n        Construct a L{FixedOffsetTimeZone} from an offset described by sign\n        ('+' or '-'), hours, and minutes.\n\n        @note: For protocol compatibility with AMP, this method never uses 'Z'\n\n        @param sign: A string describing the positive or negative-ness of the\n            offset.\n        @param hours: The number of hours in the offset.\n        @param minutes: The number of minutes in the offset\n\n        @return: A time zone with the given offset, and a name describing the\n            offset.\n        "
        name = '%s%02i:%02i' % (sign, hours, minutes)
        if sign == '-':
            hours = -hours
            minutes = -minutes
        elif sign != '+':
            raise ValueError(f'Invalid sign for timezone {sign!r}')
        return cls(TimeDelta(hours=hours, minutes=minutes), name)

    @classmethod
    def fromLocalTimeStamp(cls, timeStamp: float) -> 'FixedOffsetTimeZone':
        if False:
            while True:
                i = 10
        "\n        Create a time zone with a fixed offset corresponding to a time stamp in\n        the system's locally configured time zone.\n        "
        offset = DateTime.fromtimestamp(timeStamp) - DateTime.fromtimestamp(timeStamp, timezone.utc).replace(tzinfo=None)
        return cls(offset)

    def utcoffset(self, dt: Optional[DateTime]) -> TimeDelta:
        if False:
            i = 10
            return i + 15
        "\n        Return the given timezone's offset from UTC.\n        "
        return self.offset

    def dst(self, dt: Optional[DateTime]) -> TimeDelta:
        if False:
            i = 10
            return i + 15
        '\n        Return a zero L{TimeDelta} for the daylight saving time\n        offset, since there is never one.\n        '
        return TimeDelta(0)

    def tzname(self, dt: Optional[DateTime]) -> str:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a string describing this timezone.\n        '
        if self.name is not None:
            return self.name
        dt = DateTime.fromtimestamp(0, self)
        return dt.strftime('UTC%z')
UTC = FixedOffsetTimeZone.fromSignHoursMinutes('+', 0, 0)