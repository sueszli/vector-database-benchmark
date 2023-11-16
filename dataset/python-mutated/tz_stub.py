from __future__ import annotations
import datetime
import os
import time
import typing
from contextlib import contextmanager
import pytest
try:
    import zoneinfo
except ImportError:
    from backports import zoneinfo

@contextmanager
def stub_timezone_ctx(tzname: str | None) -> typing.Generator[None, None, None]:
    if False:
        while True:
            i = 10
    '\n    Switch to a locally-known timezone specified by `tzname`.\n    On exit, restore the previous timezone.\n    If `tzname` is `None`, do nothing.\n    '
    if tzname is None:
        yield
        return
    if not hasattr(time, 'tzset'):
        pytest.skip('Timezone patching is not supported')
    try:
        zoneinfo.ZoneInfo(tzname)
    except zoneinfo.ZoneInfoNotFoundError:
        raise ValueError(f'Invalid timezone specified: {tzname!r}')
    old_tzname = datetime.datetime.now().astimezone().tzname()
    if old_tzname is None:
        raise OSError('Cannot determine current timezone')
    os.environ['TZ'] = tzname
    time.tzset()
    yield
    os.environ['TZ'] = old_tzname
    time.tzset()