"""Timezone related utilities for BSON."""
from __future__ import annotations
from datetime import datetime, timedelta, tzinfo
from typing import Optional, Tuple, Union
ZERO: timedelta = timedelta(0)

class FixedOffset(tzinfo):
    """Fixed offset timezone, in minutes east from UTC.

    Implementation based from the Python `standard library documentation
    <http://docs.python.org/library/datetime.html#tzinfo-objects>`_.
    Defining __getinitargs__ enables pickling / copying.
    """

    def __init__(self, offset: Union[float, timedelta], name: str) -> None:
        if False:
            i = 10
            return i + 15
        if isinstance(offset, timedelta):
            self.__offset = offset
        else:
            self.__offset = timedelta(minutes=offset)
        self.__name = name

    def __getinitargs__(self) -> Tuple[timedelta, str]:
        if False:
            while True:
                i = 10
        return (self.__offset, self.__name)

    def utcoffset(self, dt: Optional[datetime]) -> timedelta:
        if False:
            return 10
        return self.__offset

    def tzname(self, dt: Optional[datetime]) -> str:
        if False:
            print('Hello World!')
        return self.__name

    def dst(self, dt: Optional[datetime]) -> timedelta:
        if False:
            i = 10
            return i + 15
        return ZERO
utc: FixedOffset = FixedOffset(0, 'UTC')
'Fixed offset timezone representing UTC.'