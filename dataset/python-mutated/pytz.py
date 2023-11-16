"""
----------------
hypothesis[pytz]
----------------

This module provides :pypi:`pytz` timezones.

You can use this strategy to make
:py:func:`hypothesis.strategies.datetimes` and
:py:func:`hypothesis.strategies.times` produce timezone-aware values.
"""
import datetime as dt
import pytz
from pytz.tzfile import StaticTzInfo
from hypothesis import strategies as st
from hypothesis.strategies._internal.utils import cacheable, defines_strategy
__all__ = ['timezones']

@cacheable
@defines_strategy()
def timezones() -> st.SearchStrategy[dt.tzinfo]:
    if False:
        print('Hello World!')
    'Any timezone in the Olsen database, as a pytz tzinfo object.\n\n    This strategy minimises to UTC, or the smallest possible fixed\n    offset, and is designed for use with\n    :py:func:`hypothesis.strategies.datetimes`.\n    '
    all_timezones = [pytz.timezone(tz) for tz in pytz.all_timezones]
    static: list = [pytz.UTC]
    static += sorted((t for t in all_timezones if isinstance(t, StaticTzInfo)), key=lambda tz: abs(tz.utcoffset(dt.datetime(2000, 1, 1))))
    dynamic = [tz for tz in all_timezones if tz not in static]
    return st.sampled_from(static + dynamic)