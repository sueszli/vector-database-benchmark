from __future__ import annotations
import contextlib
from datetime import datetime
from typing import TYPE_CHECKING, overload
from polars import functions as F
from polars.functions.range._utils import parse_interval_argument
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr
from polars.utils.deprecation import deprecate_renamed_parameter, deprecate_saturating, issue_deprecation_warning
with contextlib.suppress(ImportError):
    import polars.polars as plr
if TYPE_CHECKING:
    from datetime import date, timedelta
    from typing import Literal
    from polars import Expr, Series
    from polars.type_aliases import ClosedInterval, IntoExprColumn, TimeUnit

@overload
def date_range(start: date | datetime | IntoExprColumn, end: date | datetime | IntoExprColumn, interval: str | timedelta=..., *, closed: ClosedInterval=..., time_unit: TimeUnit | None=..., time_zone: str | None=..., eager: Literal[False]=..., name: str | None=...) -> Expr:
    if False:
        print('Hello World!')
    ...

@overload
def date_range(start: date | datetime | IntoExprColumn, end: date | datetime | IntoExprColumn, interval: str | timedelta=..., *, closed: ClosedInterval=..., time_unit: TimeUnit | None=..., time_zone: str | None=..., eager: Literal[True], name: str | None=...) -> Series:
    if False:
        print('Hello World!')
    ...

@overload
def date_range(start: date | datetime | IntoExprColumn, end: date | datetime | IntoExprColumn, interval: str | timedelta=..., *, closed: ClosedInterval=..., time_unit: TimeUnit | None=..., time_zone: str | None=..., eager: bool, name: str | None=...) -> Series | Expr:
    if False:
        i = 10
        return i + 15
    ...

@deprecate_renamed_parameter('low', 'start', version='0.18.0')
@deprecate_renamed_parameter('high', 'end', version='0.18.0')
def date_range(start: date | datetime | IntoExprColumn, end: date | datetime | IntoExprColumn, interval: str | timedelta='1d', *, closed: ClosedInterval='both', time_unit: TimeUnit | None=None, time_zone: str | None=None, eager: bool=False, name: str | None=None) -> Series | Expr:
    if False:
        while True:
            i = 10
    '\n    Generate a date range.\n\n    Parameters\n    ----------\n    start\n        Lower bound of the date range.\n    end\n        Upper bound of the date range.\n    interval\n        Interval of the range periods, specified as a Python `timedelta` object\n        or using the Polars duration string language (see "Notes" section below).\n    closed : {\'both\', \'left\', \'right\', \'none\'}\n        Define which sides of the range are closed (inclusive).\n    time_unit : {None, \'ns\', \'us\', \'ms\'}\n        Time unit of the resulting `Datetime` data type.\n        Only takes effect if the output column is of type `Datetime`.\n    time_zone\n        Time zone of the resulting `Datetime` data type.\n        Only takes effect if the output column is of type `Datetime`.\n    eager\n        Evaluate immediately and return a `Series`.\n        If set to `False` (default), return an expression instead.\n    name\n        Name of the output column.\n\n        .. deprecated:: 0.18.0\n            This argument is deprecated. Use the `alias` method instead.\n\n    Returns\n    -------\n    Expr or Series\n        Column of data type :class:`Date` or :class:`Datetime`.\n\n    Notes\n    -----\n    1) If both `start` and `end` are passed as date types (not datetime), and the\n       interval granularity is no finer than 1d, the returned range is also of\n       type date. All other permutations return a datetime Series.\n\n       .. deprecated:: 0.19.3\n           In a future version of Polars, `date_range` will always return a `Date`.\n           Please use :func:`datetime_range` if you want a `Datetime` instead.\n\n    2) `interval` is created according to the following string language:\n\n       - 1ns   (1 nanosecond)\n       - 1us   (1 microsecond)\n       - 1ms   (1 millisecond)\n       - 1s    (1 second)\n       - 1m    (1 minute)\n       - 1h    (1 hour)\n       - 1d    (1 calendar day)\n       - 1w    (1 calendar week)\n       - 1mo   (1 calendar month)\n       - 1q    (1 calendar quarter)\n       - 1y    (1 calendar year)\n\n       Or combine them:\n       "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds\n\n       By "calendar day", we mean the corresponding time on the next day (which may\n       not be 24 hours, due to daylight savings). Similarly for "calendar week",\n       "calendar month", "calendar quarter", and "calendar year".\n\n    Examples\n    --------\n    Using Polars duration string to specify the interval:\n\n    >>> from datetime import date\n    >>> pl.date_range(date(2022, 1, 1), date(2022, 3, 1), "1mo", eager=True)\n    shape: (3,)\n    Series: \'date\' [date]\n    [\n        2022-01-01\n        2022-02-01\n        2022-03-01\n    ]\n\n    Using `timedelta` object to specify the interval:\n\n    >>> from datetime import timedelta\n    >>> pl.date_range(\n    ...     date(1985, 1, 1),\n    ...     date(1985, 1, 10),\n    ...     timedelta(days=2),\n    ...     eager=True,\n    ... )\n    shape: (5,)\n    Series: \'date\' [date]\n    [\n        1985-01-01\n        1985-01-03\n        1985-01-05\n        1985-01-07\n        1985-01-09\n    ]\n\n    '
    interval = deprecate_saturating(interval)
    if name is not None:
        issue_deprecation_warning('the `name` argument is deprecated. Use the `alias` method instead.', version='0.18.0')
    interval = parse_interval_argument(interval)
    if time_unit is None and 'ns' in interval:
        time_unit = 'ns'
    _warn_for_deprecated_date_range_use(start, end, interval, time_unit, time_zone)
    start_pyexpr = parse_as_expression(start)
    end_pyexpr = parse_as_expression(end)
    result = wrap_expr(plr.date_range(start_pyexpr, end_pyexpr, interval, closed, time_unit, time_zone))
    if name is not None:
        result = result.alias(name)
    if eager:
        return F.select(result).to_series()
    return result

@overload
def date_ranges(start: date | datetime | IntoExprColumn, end: date | datetime | IntoExprColumn, interval: str | timedelta=..., *, closed: ClosedInterval=..., time_unit: TimeUnit | None=..., time_zone: str | None=..., eager: Literal[False]=...) -> Expr:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def date_ranges(start: date | datetime | IntoExprColumn, end: date | datetime | IntoExprColumn, interval: str | timedelta=..., *, closed: ClosedInterval=..., time_unit: TimeUnit | None=..., time_zone: str | None=..., eager: Literal[True]) -> Series:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def date_ranges(start: date | datetime | IntoExprColumn, end: date | datetime | IntoExprColumn, interval: str | timedelta=..., *, closed: ClosedInterval=..., time_unit: TimeUnit | None=..., time_zone: str | None=..., eager: bool) -> Series | Expr:
    if False:
        i = 10
        return i + 15
    ...

def date_ranges(start: date | datetime | IntoExprColumn, end: date | datetime | IntoExprColumn, interval: str | timedelta='1d', *, closed: ClosedInterval='both', time_unit: TimeUnit | None=None, time_zone: str | None=None, eager: bool=False) -> Series | Expr:
    if False:
        while True:
            i = 10
    '\n    Create a column of date ranges.\n\n    Parameters\n    ----------\n    start\n        Lower bound of the date range.\n    end\n        Upper bound of the date range.\n    interval\n        Interval of the range periods, specified as a Python `timedelta` object\n        or using the Polars duration string language (see "Notes" section below).\n    closed : {\'both\', \'left\', \'right\', \'none\'}\n        Define which sides of the range are closed (inclusive).\n    time_unit : {None, \'ns\', \'us\', \'ms\'}\n        Time unit of the resulting `Datetime` data type.\n        Only takes effect if the output column is of type `Datetime`.\n    time_zone\n        Time zone of the resulting `Datetime` data type.\n        Only takes effect if the output column is of type `Datetime`.\n    eager\n        Evaluate immediately and return a `Series`.\n        If set to `False` (default), return an expression instead.\n\n    Returns\n    -------\n    Expr or Series\n        Column of data type `List(Date)` or `List(Datetime)`.\n\n    Notes\n    -----\n    `interval` is created according to the following string language:\n\n    - 1ns   (1 nanosecond)\n    - 1us   (1 microsecond)\n    - 1ms   (1 millisecond)\n    - 1s    (1 second)\n    - 1m    (1 minute)\n    - 1h    (1 hour)\n    - 1d    (1 calendar day)\n    - 1w    (1 calendar week)\n    - 1mo   (1 calendar month)\n    - 1q    (1 calendar quarter)\n    - 1y    (1 calendar year)\n\n    Or combine them:\n    "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds\n\n    By "calendar day", we mean the corresponding time on the next day (which may\n    not be 24 hours, due to daylight savings). Similarly for "calendar week",\n    "calendar month", "calendar quarter", and "calendar year".\n\n    Examples\n    --------\n    >>> from datetime import date\n    >>> df = pl.DataFrame(\n    ...     {\n    ...         "start": [date(2022, 1, 1), date(2022, 1, 2)],\n    ...         "end": date(2022, 1, 3),\n    ...     }\n    ... )\n    >>> df.with_columns(pl.date_ranges("start", "end"))\n    shape: (2, 3)\n    ┌────────────┬────────────┬───────────────────────────────────┐\n    │ start      ┆ end        ┆ date_range                        │\n    │ ---        ┆ ---        ┆ ---                               │\n    │ date       ┆ date       ┆ list[date]                        │\n    ╞════════════╪════════════╪═══════════════════════════════════╡\n    │ 2022-01-01 ┆ 2022-01-03 ┆ [2022-01-01, 2022-01-02, 2022-01… │\n    │ 2022-01-02 ┆ 2022-01-03 ┆ [2022-01-02, 2022-01-03]          │\n    └────────────┴────────────┴───────────────────────────────────┘\n\n    '
    interval = deprecate_saturating(interval)
    interval = parse_interval_argument(interval)
    if time_unit is None and 'ns' in interval:
        time_unit = 'ns'
    _warn_for_deprecated_date_range_use(start, end, interval, time_unit, time_zone)
    start_pyexpr = parse_as_expression(start)
    end_pyexpr = parse_as_expression(end)
    result = wrap_expr(plr.date_ranges(start_pyexpr, end_pyexpr, interval, closed, time_unit, time_zone))
    if eager:
        return F.select(result).to_series()
    return result

def _warn_for_deprecated_date_range_use(start: date | datetime | IntoExprColumn, end: date | datetime | IntoExprColumn, interval: str, time_unit: TimeUnit | None, time_zone: str | None) -> None:
    if False:
        while True:
            i = 10
    if isinstance(start, datetime) or isinstance(end, datetime) or time_unit is not None or (time_zone is not None) or ('h' in interval) or ('m' in interval.replace('mo', '')) or ('s' in interval.replace('saturating', '')):
        issue_deprecation_warning('Creating Datetime ranges using `date_range(s)` is deprecated. Use `datetime_range(s)` instead.', version='0.19.3')