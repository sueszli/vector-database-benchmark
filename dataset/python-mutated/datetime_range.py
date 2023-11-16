from __future__ import annotations
import contextlib
from typing import TYPE_CHECKING, overload
from polars import functions as F
from polars.functions.range._utils import parse_interval_argument
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr
from polars.utils.deprecation import deprecate_saturating
with contextlib.suppress(ImportError):
    import polars.polars as plr
if TYPE_CHECKING:
    from datetime import date, datetime, timedelta
    from typing import Literal
    from polars import Expr, Series
    from polars.type_aliases import ClosedInterval, IntoExprColumn, TimeUnit

@overload
def datetime_range(start: datetime | date | IntoExprColumn, end: datetime | date | IntoExprColumn, interval: str | timedelta=..., *, closed: ClosedInterval=..., time_unit: TimeUnit | None=..., time_zone: str | None=..., eager: Literal[False]=...) -> Expr:
    if False:
        return 10
    ...

@overload
def datetime_range(start: datetime | date | IntoExprColumn, end: datetime | date | IntoExprColumn, interval: str | timedelta=..., *, closed: ClosedInterval=..., time_unit: TimeUnit | None=..., time_zone: str | None=..., eager: Literal[True]) -> Series:
    if False:
        while True:
            i = 10
    ...

@overload
def datetime_range(start: datetime | date | IntoExprColumn, end: datetime | date | IntoExprColumn, interval: str | timedelta=..., *, closed: ClosedInterval=..., time_unit: TimeUnit | None=..., time_zone: str | None=..., eager: bool) -> Series | Expr:
    if False:
        return 10
    ...

def datetime_range(start: datetime | date | IntoExprColumn, end: datetime | date | IntoExprColumn, interval: str | timedelta='1d', *, closed: ClosedInterval='both', time_unit: TimeUnit | None=None, time_zone: str | None=None, eager: bool=False) -> Series | Expr:
    if False:
        print('Hello World!')
    '\n    Generate a datetime range.\n\n    Parameters\n    ----------\n    start\n        Lower bound of the datetime range.\n    end\n        Upper bound of the datetime range.\n    interval\n        Interval of the range periods, specified as a Python `timedelta` object\n        or using the Polars duration string language (see "Notes" section below).\n    closed : {\'both\', \'left\', \'right\', \'none\'}\n        Define which sides of the range are closed (inclusive).\n    time_unit : {None, \'ns\', \'us\', \'ms\'}\n        Time unit of the resulting `Datetime` data type.\n    time_zone\n        Time zone of the resulting `Datetime` data type.\n    eager\n        Evaluate immediately and return a `Series`.\n        If set to `False` (default), return an expression instead.\n\n    Returns\n    -------\n    Expr or Series\n        Column of data type :class:`Datetime`.\n\n    Notes\n    -----\n    `interval` is created according to the following string language:\n\n    - 1ns   (1 nanosecond)\n    - 1us   (1 microsecond)\n    - 1ms   (1 millisecond)\n    - 1s    (1 second)\n    - 1m    (1 minute)\n    - 1h    (1 hour)\n    - 1d    (1 calendar day)\n    - 1w    (1 calendar week)\n    - 1mo   (1 calendar month)\n    - 1q    (1 calendar quarter)\n    - 1y    (1 calendar year)\n\n    Or combine them:\n    "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds\n\n    By "calendar day", we mean the corresponding time on the next day (which may\n    not be 24 hours, due to daylight savings). Similarly for "calendar week",\n    "calendar month", "calendar quarter", and "calendar year".\n\n    Examples\n    --------\n    Using Polars duration string to specify the interval:\n\n    >>> from datetime import datetime\n    >>> pl.datetime_range(datetime(2022, 1, 1), datetime(2022, 3, 1), "1mo", eager=True)\n    shape: (3,)\n    Series: \'datetime\' [datetime[μs]]\n    [\n        2022-01-01 00:00:00\n        2022-02-01 00:00:00\n        2022-03-01 00:00:00\n    ]\n\n    Using `timedelta` object to specify the interval:\n\n    >>> from datetime import date, timedelta\n    >>> pl.datetime_range(\n    ...     date(1985, 1, 1),\n    ...     date(1985, 1, 10),\n    ...     timedelta(days=1, hours=12),\n    ...     time_unit="ms",\n    ...     eager=True,\n    ... )\n    shape: (7,)\n    Series: \'datetime\' [datetime[ms]]\n    [\n        1985-01-01 00:00:00\n        1985-01-02 12:00:00\n        1985-01-04 00:00:00\n        1985-01-05 12:00:00\n        1985-01-07 00:00:00\n        1985-01-08 12:00:00\n        1985-01-10 00:00:00\n    ]\n\n    Specifying a time zone:\n\n    >>> pl.datetime_range(\n    ...     datetime(2022, 1, 1),\n    ...     datetime(2022, 3, 1),\n    ...     "1mo",\n    ...     time_zone="America/New_York",\n    ...     eager=True,\n    ... )\n    shape: (3,)\n    Series: \'datetime\' [datetime[μs, America/New_York]]\n    [\n        2022-01-01 00:00:00 EST\n        2022-02-01 00:00:00 EST\n        2022-03-01 00:00:00 EST\n    ]\n\n    '
    interval = deprecate_saturating(interval)
    interval = parse_interval_argument(interval)
    if time_unit is None and 'ns' in interval:
        time_unit = 'ns'
    start_pyexpr = parse_as_expression(start)
    end_pyexpr = parse_as_expression(end)
    result = wrap_expr(plr.datetime_range(start_pyexpr, end_pyexpr, interval, closed, time_unit, time_zone))
    if eager:
        return F.select(result).to_series()
    return result

@overload
def datetime_ranges(start: datetime | date | IntoExprColumn, end: datetime | date | IntoExprColumn, interval: str | timedelta=..., *, closed: ClosedInterval=..., time_unit: TimeUnit | None=..., time_zone: str | None=..., eager: Literal[False]=...) -> Expr:
    if False:
        while True:
            i = 10
    ...

@overload
def datetime_ranges(start: datetime | date | IntoExprColumn, end: datetime | date | IntoExprColumn, interval: str | timedelta=..., *, closed: ClosedInterval=..., time_unit: TimeUnit | None=..., time_zone: str | None=..., eager: Literal[True]) -> Series:
    if False:
        print('Hello World!')
    ...

@overload
def datetime_ranges(start: datetime | date | IntoExprColumn, end: datetime | date | IntoExprColumn, interval: str | timedelta=..., *, closed: ClosedInterval=..., time_unit: TimeUnit | None=..., time_zone: str | None=..., eager: bool) -> Series | Expr:
    if False:
        for i in range(10):
            print('nop')
    ...

def datetime_ranges(start: datetime | date | IntoExprColumn, end: datetime | date | IntoExprColumn, interval: str | timedelta='1d', *, closed: ClosedInterval='both', time_unit: TimeUnit | None=None, time_zone: str | None=None, eager: bool=False) -> Series | Expr:
    if False:
        i = 10
        return i + 15
    '\n    Create a column of datetime ranges.\n\n    Parameters\n    ----------\n    start\n        Lower bound of the datetime range.\n    end\n        Upper bound of the datetime range.\n    interval\n        Interval of the range periods, specified as a Python `timedelta` object\n        or using the Polars duration string language (see "Notes" section below).\n    closed : {\'both\', \'left\', \'right\', \'none\'}\n        Define which sides of the range are closed (inclusive).\n    time_unit : {None, \'ns\', \'us\', \'ms\'}\n        Time unit of the resulting `Datetime` data type.\n    time_zone\n        Time zone of the resulting `Datetime` data type.\n    eager\n        Evaluate immediately and return a `Series`.\n        If set to `False` (default), return an expression instead.\n\n    Notes\n    -----\n    `interval` is created according to the following string language:\n\n    - 1ns   (1 nanosecond)\n    - 1us   (1 microsecond)\n    - 1ms   (1 millisecond)\n    - 1s    (1 second)\n    - 1m    (1 minute)\n    - 1h    (1 hour)\n    - 1d    (1 calendar day)\n    - 1w    (1 calendar week)\n    - 1mo   (1 calendar month)\n    - 1q    (1 calendar quarter)\n    - 1y    (1 calendar year)\n\n    Or combine them:\n    "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds\n\n    By "calendar day", we mean the corresponding time on the next day (which may\n    not be 24 hours, due to daylight savings). Similarly for "calendar week",\n    "calendar month", "calendar quarter", and "calendar year".\n\n    Returns\n    -------\n    Expr or Series\n        Column of data type `List(Datetime)`.\n\n    '
    interval = deprecate_saturating(interval)
    interval = parse_interval_argument(interval)
    if time_unit is None and 'ns' in interval:
        time_unit = 'ns'
    start_pyexpr = parse_as_expression(start)
    end_pyexpr = parse_as_expression(end)
    result = wrap_expr(plr.datetime_ranges(start_pyexpr, end_pyexpr, interval, closed, time_unit, time_zone))
    if eager:
        return F.select(result).to_series()
    return result