from __future__ import annotations
import contextlib
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Any
import polars._reexport as pl
from polars.datatypes import Date, Datetime, Duration, Time
from polars.dependencies import _check_for_numpy
from polars.dependencies import numpy as np
from polars.utils._wrap import wrap_expr
from polars.utils.convert import _datetime_to_pl_timestamp, _time_to_pl_time, _timedelta_to_pl_timedelta
from polars.utils.deprecation import issue_deprecation_warning
with contextlib.suppress(ImportError):
    import polars.polars as plr
if TYPE_CHECKING:
    from polars import Expr
    from polars.type_aliases import PolarsDataType, TimeUnit

def lit(value: Any, dtype: PolarsDataType | None=None, *, allow_object: bool=False) -> Expr:
    if False:
        i = 10
        return i + 15
    '\n    Return an expression representing a literal value.\n\n    Parameters\n    ----------\n    value\n        Value that should be used as a `literal`.\n    dtype\n        Optionally define a dtype.\n    allow_object\n        If type is unknown use an \'object\' type.\n        By default, we will raise a `ValueException`\n        if the type is unknown.\n\n    Notes\n    -----\n    Expected datatypes\n\n    - `pl.lit([])` -> empty  Series Float32\n    - `pl.lit([1, 2, 3])` -> Series Int64\n    - `pl.lit([[]])`-> empty  Series List<Null>\n    - `pl.lit([[1, 2, 3]])` -> Series List<i64>\n    - `pl.lit(None)` -> Series Null\n\n    Examples\n    --------\n    Literal scalar values:\n\n    >>> pl.lit(1)  # doctest: +IGNORE_RESULT\n    >>> pl.lit(5.5)  # doctest: +IGNORE_RESULT\n    >>> pl.lit(None)  # doctest: +IGNORE_RESULT\n    >>> pl.lit("foo_bar")  # doctest: +IGNORE_RESULT\n    >>> pl.lit(date(2021, 1, 20))  # doctest: +IGNORE_RESULT\n    >>> pl.lit(datetime(2023, 3, 31, 10, 30, 45))  # doctest: +IGNORE_RESULT\n\n    Literal list/Series data (1D):\n\n    >>> pl.lit([1, 2, 3])  # doctest: +SKIP\n    >>> pl.lit(pl.Series("x", [1, 2, 3]))  # doctest: +IGNORE_RESULT\n\n    Literal list/Series data (2D):\n\n    >>> pl.lit([[1, 2], [3, 4]])  # doctest: +SKIP\n    >>> pl.lit(pl.Series("y", [[1, 2], [3, 4]]))  # doctest: +IGNORE_RESULT\n\n    '
    time_unit: TimeUnit
    if isinstance(value, datetime):
        time_unit = 'us' if dtype is None else getattr(dtype, 'time_unit', 'us')
        time_zone = value.tzinfo if getattr(dtype, 'time_zone', None) is None else getattr(dtype, 'time_zone', None)
        if value.tzinfo is not None and getattr(dtype, 'time_zone', None) is not None and (dtype.time_zone != str(value.tzinfo)):
            raise TypeError(f'time zone of dtype ({dtype.time_zone!r}) differs from time zone of value ({value.tzinfo!r})')
        e = lit(_datetime_to_pl_timestamp(value.replace(tzinfo=timezone.utc), time_unit)).cast(Datetime(time_unit))
        if time_zone is not None:
            return e.dt.replace_time_zone(str(time_zone), ambiguous='earliest' if value.fold == 0 else 'latest')
        else:
            return e
    elif isinstance(value, timedelta):
        time_unit = 'us' if dtype is None else getattr(dtype, 'time_unit', 'us')
        return lit(_timedelta_to_pl_timedelta(value, time_unit)).cast(Duration(time_unit))
    elif isinstance(value, time):
        return lit(_time_to_pl_time(value)).cast(Time)
    elif isinstance(value, date):
        return lit(datetime(value.year, value.month, value.day)).cast(Date)
    elif isinstance(value, pl.Series):
        name = value.name
        value = value._s
        e = wrap_expr(plr.lit(value, allow_object))
        if name == '':
            return e
        return e.alias(name)
    elif _check_for_numpy(value) and isinstance(value, np.ndarray):
        return lit(pl.Series('', value))
    elif isinstance(value, (list, tuple)):
        issue_deprecation_warning('Behavior for `lit` will change for sequence inputs. The result will change to be a literal of type List. To retain the old behavior, pass a Series instead, e.g. `Series(sequence)`.', version='0.18.14')
        return lit(pl.Series('', value))
    if dtype:
        return wrap_expr(plr.lit(value, allow_object)).cast(dtype)
    try:
        item = value.item()
        if isinstance(item, (datetime, timedelta)):
            return lit(item)
        if isinstance(item, int) and hasattr(value, 'dtype'):
            dtype_name = value.dtype.name
            if dtype_name.startswith('datetime64['):
                time_unit = dtype_name[len('datetime64['):-1]
                return lit(item).cast(Datetime(time_unit))
            if dtype_name.startswith('timedelta64['):
                time_unit = dtype_name[len('timedelta64['):-1]
                return lit(item).cast(Duration(time_unit))
    except AttributeError:
        item = value
    return wrap_expr(plr.lit(item, allow_object))