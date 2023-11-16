from __future__ import annotations
import sys
from typing import TYPE_CHECKING
if sys.version_info < (3, 9):
    from backports.zoneinfo import TZPATH
    from backports.zoneinfo import InvalidTZPathWarning
    from backports.zoneinfo import ZoneInfoNotFoundError
    from backports.zoneinfo import available_timezones
    from backports.zoneinfo import reset_tzpath
    if TYPE_CHECKING:
        from collections.abc import Iterable
        from datetime import datetime
        from datetime import timedelta
        from datetime import tzinfo
        from typing import Any
        from typing import Protocol
        from typing_extensions import Self

        class _IOBytes(Protocol):

            def read(self, __size: int) -> bytes:
                if False:
                    return 10
                ...

            def seek(self, __size: int, __whence: int=...) -> Any:
                if False:
                    i = 10
                    return i + 15
                ...

        class ZoneInfo(tzinfo):

            @property
            def key(self) -> str:
                if False:
                    print('Hello World!')
                ...

            def __init__(self, key: str) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                ...

            @classmethod
            def no_cache(cls, key: str) -> Self:
                if False:
                    return 10
                ...

            @classmethod
            def from_file(cls, __fobj: _IOBytes, key: str | None=...) -> Self:
                if False:
                    for i in range(10):
                        print('nop')
                ...

            @classmethod
            def clear_cache(cls, *, only_keys: Iterable[str] | None=...) -> None:
                if False:
                    for i in range(10):
                        print('nop')
                ...

            def tzname(self, __dt: datetime | None) -> str | None:
                if False:
                    while True:
                        i = 10
                ...

            def utcoffset(self, __dt: datetime | None) -> timedelta | None:
                if False:
                    while True:
                        i = 10
                ...

            def dst(self, __dt: datetime | None) -> timedelta | None:
                if False:
                    for i in range(10):
                        print('nop')
                ...
    else:
        from backports.zoneinfo import ZoneInfo
else:
    from zoneinfo import TZPATH
    from zoneinfo import InvalidTZPathWarning
    from zoneinfo import ZoneInfo
    from zoneinfo import ZoneInfoNotFoundError
    from zoneinfo import available_timezones
    from zoneinfo import reset_tzpath
__all__ = ['ZoneInfo', 'reset_tzpath', 'available_timezones', 'TZPATH', 'ZoneInfoNotFoundError', 'InvalidTZPathWarning']