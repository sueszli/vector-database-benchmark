from __future__ import annotations
import contextlib
from datetime import time
from typing import TYPE_CHECKING, overload
from polars import functions as F
from polars.functions.range._utils import parse_interval_argument
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr
from polars.utils.deprecation import deprecate_saturating, issue_deprecation_warning
with contextlib.suppress(ImportError):
    import polars.polars as plr
if TYPE_CHECKING:
    from datetime import timedelta
    from typing import Literal
    from polars import Expr, Series
    from polars.type_aliases import ClosedInterval, IntoExprColumn

@overload
def time_range(start: time | IntoExprColumn | None=..., end: time | IntoExprColumn | None=..., interval: str | timedelta=..., *, closed: ClosedInterval=..., eager: Literal[False]=..., name: str | None=...) -> Expr:
    if False:
        while True:
            i = 10
    ...

@overload
def time_range(start: time | IntoExprColumn | None=..., end: time | IntoExprColumn | None=..., interval: str | timedelta=..., *, closed: ClosedInterval=..., eager: Literal[True], name: str | None=...) -> Series:
    if False:
        print('Hello World!')
    ...

@overload
def time_range(start: time | IntoExprColumn | None=..., end: time | IntoExprColumn | None=..., interval: str | timedelta=..., *, closed: ClosedInterval=..., eager: bool, name: str | None=...) -> Series | Expr:
    if False:
        print('Hello World!')
    ...

def time_range(start: time | IntoExprColumn | None=None, end: time | IntoExprColumn | None=None, interval: str | timedelta='1h', *, closed: ClosedInterval='both', eager: bool=False, name: str | None=None) -> Series | Expr:
    if False:
        for i in range(10):
            print('nop')
    '\n    Generate a time range.\n\n    Parameters\n    ----------\n    start\n        Lower bound of the time range.\n        If omitted, defaults to `time(0,0,0,0)`.\n    end\n        Upper bound of the time range.\n        If omitted, defaults to `time(23,59,59,999999)`.\n    interval\n        Interval of the range periods, specified as a Python `timedelta` object\n        or using the Polars duration string language (see "Notes" section below).\n    closed : {\'both\', \'left\', \'right\', \'none\'}\n        Define which sides of the range are closed (inclusive).\n    eager\n        Evaluate immediately and return a `Series`.\n        If set to `False` (default), return an expression instead.\n    name\n        Name of the output column.\n\n        .. deprecated:: 0.18.0\n            This argument is deprecated. Use the `alias` method instead.\n\n    Returns\n    -------\n    Expr or Series\n        Column of data type `:class:Time`.\n\n    Notes\n    -----\n    `interval` is created according to the following string language:\n\n    - 1ns   (1 nanosecond)\n    - 1us   (1 microsecond)\n    - 1ms   (1 millisecond)\n    - 1s    (1 second)\n    - 1m    (1 minute)\n    - 1h    (1 hour)\n    - 1d    (1 calendar day)\n    - 1w    (1 calendar week)\n    - 1mo   (1 calendar month)\n    - 1q    (1 calendar quarter)\n    - 1y    (1 calendar year)\n\n    Or combine them:\n    "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds\n\n    By "calendar day", we mean the corresponding time on the next day (which may\n    not be 24 hours, due to daylight savings). Similarly for "calendar week",\n    "calendar month", "calendar quarter", and "calendar year".\n\n    See Also\n    --------\n    time_ranges : Create a column of time ranges.\n\n    Examples\n    --------\n    >>> from datetime import time, timedelta\n    >>> pl.time_range(\n    ...     start=time(14, 0),\n    ...     interval=timedelta(hours=3, minutes=15),\n    ...     eager=True,\n    ... )\n    shape: (4,)\n    Series: \'time\' [time]\n    [\n        14:00:00\n        17:15:00\n        20:30:00\n        23:45:00\n    ]\n\n    '
    interval = deprecate_saturating(interval)
    if name is not None:
        issue_deprecation_warning('the `name` argument is deprecated. Use the `alias` method instead.', version='0.18.0')
    interval = parse_interval_argument(interval)
    for unit in ('y', 'mo', 'w', 'd'):
        if unit in interval:
            raise ValueError(f'invalid interval unit for time_range: found {unit!r}')
    if start is None:
        start = time(0, 0, 0)
    if end is None:
        end = time(23, 59, 59, 999999)
    start_pyexpr = parse_as_expression(start)
    end_pyexpr = parse_as_expression(end)
    result = wrap_expr(plr.time_range(start_pyexpr, end_pyexpr, interval, closed))
    if name is not None:
        result = result.alias(name)
    if eager:
        return F.select(result).to_series()
    return result

@overload
def time_ranges(start: time | IntoExprColumn | None=..., end: time | IntoExprColumn | None=..., interval: str | timedelta=..., *, closed: ClosedInterval=..., eager: Literal[False]=...) -> Expr:
    if False:
        i = 10
        return i + 15
    ...

@overload
def time_ranges(start: time | IntoExprColumn | None=..., end: time | IntoExprColumn | None=..., interval: str | timedelta=..., *, closed: ClosedInterval=..., eager: Literal[True]) -> Series:
    if False:
        return 10
    ...

@overload
def time_ranges(start: time | IntoExprColumn | None=..., end: time | IntoExprColumn | None=..., interval: str | timedelta=..., *, closed: ClosedInterval=..., eager: bool) -> Series | Expr:
    if False:
        print('Hello World!')
    ...

def time_ranges(start: time | IntoExprColumn | None=None, end: time | IntoExprColumn | None=None, interval: str | timedelta='1h', *, closed: ClosedInterval='both', eager: bool=False) -> Series | Expr:
    if False:
        while True:
            i = 10
    '\n    Create a column of time ranges.\n\n    Parameters\n    ----------\n    start\n        Lower bound of the time range.\n        If omitted, defaults to `time(0, 0, 0, 0)`.\n    end\n        Upper bound of the time range.\n        If omitted, defaults to `time(23, 59, 59, 999999)`.\n    interval\n        Interval of the range periods, specified as a Python `timedelta` object\n        or using the Polars duration string language (see "Notes" section below).\n    closed : {\'both\', \'left\', \'right\', \'none\'}\n        Define which sides of the range are closed (inclusive).\n    eager\n        Evaluate immediately and return a `Series`.\n        If set to `False` (default), return an expression instead.\n\n    Returns\n    -------\n    Expr or Series\n        Column of data type `List(Time)`.\n\n    Notes\n    -----\n    `interval` is created according to the following string language:\n\n    - 1ns   (1 nanosecond)\n    - 1us   (1 microsecond)\n    - 1ms   (1 millisecond)\n    - 1s    (1 second)\n    - 1m    (1 minute)\n    - 1h    (1 hour)\n    - 1d    (1 calendar day)\n    - 1w    (1 calendar week)\n    - 1mo   (1 calendar month)\n    - 1q    (1 calendar quarter)\n    - 1y    (1 calendar year)\n\n    Or combine them:\n    "3d12h4m25s" # 3 days, 12 hours, 4 minutes, and 25 seconds\n\n    By "calendar day", we mean the corresponding time on the next day (which may\n    not be 24 hours, due to daylight savings). Similarly for "calendar week",\n    "calendar month", "calendar quarter", and "calendar year".\n\n    See Also\n    --------\n    time_range : Generate a single time range.\n\n    Examples\n    --------\n    >>> from datetime import time\n    >>> df = pl.DataFrame(\n    ...     {\n    ...         "start": [time(9, 0), time(10, 0)],\n    ...         "end": time(11, 0),\n    ...     }\n    ... )\n    >>> df.with_columns(pl.time_ranges("start", "end"))\n    shape: (2, 3)\n    ┌──────────┬──────────┬────────────────────────────────┐\n    │ start    ┆ end      ┆ time_range                     │\n    │ ---      ┆ ---      ┆ ---                            │\n    │ time     ┆ time     ┆ list[time]                     │\n    ╞══════════╪══════════╪════════════════════════════════╡\n    │ 09:00:00 ┆ 11:00:00 ┆ [09:00:00, 10:00:00, 11:00:00] │\n    │ 10:00:00 ┆ 11:00:00 ┆ [10:00:00, 11:00:00]           │\n    └──────────┴──────────┴────────────────────────────────┘\n\n    '
    interval = deprecate_saturating(interval)
    interval = parse_interval_argument(interval)
    for unit in ('y', 'mo', 'w', 'd'):
        if unit in interval:
            raise ValueError(f'invalid interval unit for time_range: found {unit!r}')
    if start is None:
        start = time(0, 0, 0)
    if end is None:
        end = time(23, 59, 59, 999999)
    start_pyexpr = parse_as_expression(start)
    end_pyexpr = parse_as_expression(end)
    result = wrap_expr(plr.time_ranges(start_pyexpr, end_pyexpr, interval, closed))
    if eager:
        return F.select(result).to_series()
    return result