"""Implements an object to describe the context of a window aggregation.

For any particular aggregation such as ``sum``, ``mean``, etc we need to decide
based on the presence or absence of other expressions like ``group_by`` and
``order_by`` whether we should call a different method of aggregation.

Here are the different aggregation contexts and the conditions under which they
are used.

Note that in the pandas backend, only trailing and cumulative windows are
supported right now.

No ``group_by`` or ``order_by``: ``context.Summarize()``
--------------------------------------------------------
This is an aggregation on a column, repeated for every row in the table.

SQL

::

    SELECT SUM(value) OVER () AS sum_value FROM t

Pandas

::
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame(
    ...     {
    ...         "key": list("aabc"),
    ...         "value": np.random.randn(4),
    ...         "time": pd.date_range(periods=4, start="now"),
    ...     }
    ... )
    >>> s = pd.Series(df.value.sum(), index=df.index, name="sum_value")
    >>> s  # quartodoc: +SKIP # doctest: +SKIP

Ibis

::

    >>> import ibis
    >>> schema = dict(time="timestamp", key="string", value="double")
    >>> t = ibis.table(schema, name="t")
    >>> t[
    ...     t, t.value.sum().name("sum_value")
    ... ].sum_value  # quartodoc: +SKIP # doctest: +SKIP


``group_by``, no ``order_by``: ``context.Transform()``
------------------------------------------------------

This performs an aggregation per group and repeats it across every row in the
group.

SQL

::

    SELECT SUM(value) OVER (PARTITION BY key) AS sum_value
    FROM t

Pandas

::

    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame(
    ...     {
    ...         "key": list("aabc"),
    ...         "value": np.random.randn(4),
    ...         "time": pd.date_range(periods=4, start="now"),
    ...     }
    ... )
    >>> df.groupby("key").value.transform("sum")  # quartodoc: +SKIP # doctest: +SKIP

Ibis

::

    >>> import ibis
    >>> schema = dict(time="timestamp", key="string", value="double")
    >>> t = ibis.table(schema, name="t")
    >>> t.value.sum().over(
    ...     ibis.window(group_by=t.key)
    ... )  # quartodoc: +SKIP # doctest: +SKIP

``order_by``, no ``group_by``: ``context.Cumulative()``/``context.Rolling()``
-----------------------------------------------------------------------------

Cumulative and trailing window operations.

Cumulative
~~~~~~~~~~

Also called expanding.

SQL

::

    SELECT SUM(value) OVER (
        ORDER BY time ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS sum_value
    FROM t


Pandas

::

    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame(
    ...     {
    ...         "key": list("aabc"),
    ...         "value": np.random.randn(4),
    ...         "time": pd.date_range(periods=4, start="now"),
    ...     }
    ... )
    >>> df.sort_values("time").value.cumsum()  # quartodoc: +SKIP # doctest: +SKIP

Ibis

::

    >>> import ibis
    >>> schema = dict(time="timestamp", key="string", value="double")
    >>> t = ibis.table(schema, name="t")
    >>> window = ibis.cumulative_window(order_by=t.time)
    >>> t.value.sum().over(window)  # quartodoc: +SKIP # doctest: +SKIP

Moving
~~~~~~

Also called referred to as "rolling" in other libraries such as pandas.

SQL

::

    SELECT SUM(value) OVER (
        ORDER BY time ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
    ) AS sum_value
    FROM t


Pandas

::

    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame(
    ...     {
    ...         "key": list("aabc"),
    ...         "value": np.random.randn(4),
    ...         "time": pd.date_range(periods=4, start="now"),
    ...     }
    ... )
    >>> df.sort_values("time").value.rolling(
    ...     3
    ... ).sum()  # quartodoc: +SKIP # doctest: +SKIP

Ibis

::

    >>> import ibis
    >>> schema = dict(time="timestamp", key="string", value="double")
    >>> t = ibis.table(schema, name="t")
    >>> window = ibis.trailing_window(3, order_by=t.time)
    >>> t.value.sum().over(window)  # quartodoc: +SKIP # doctest: +SKIP


``group_by`` and ``order_by``: ``context.Cumulative()``/``context.Rolling()``
-----------------------------------------------------------------------------

This performs a cumulative or rolling operation within a group.

SQL

::

    SELECT SUM(value) OVER (
        PARTITION BY key ORDER BY time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
    ) AS sum_value
    FROM t


Pandas

::

    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame(
    ...     {
    ...         "key": list("aabc"),
    ...         "value": np.random.randn(4),
    ...         "time": pd.date_range(periods=4, start="now"),
    ...     }
    ... )
    >>> sorter = lambda df: df.sort_values("time")
    >>> gb = (
    ...     df.groupby("key", group_keys=False)
    ...     .apply(sorter)
    ...     .reset_index(drop=True)
    ...     .groupby("key")
    ... )
    >>> rolling = gb.value.rolling(2)
    >>> rolling.sum()  # quartodoc: +SKIP # doctest: +SKIP

Ibis

::

    >>> import ibis
    >>> schema = dict(time="timestamp", key="string", value="double")
    >>> t = ibis.table(schema, name="t")
    >>> window = ibis.trailing_window(2, order_by=t.time, group_by=t.key)
    >>> t.value.sum().over(window)  # quartodoc: +SKIP # doctest: +SKIP
"""
from __future__ import annotations
import abc
import functools
import itertools
import operator
from typing import TYPE_CHECKING, Any, Callable
import pandas as pd
from pandas.core.groupby import SeriesGroupBy
import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.util
from ibis.backends.base.df.timecontext import construct_time_context_aware_series, get_time_col
if TYPE_CHECKING:
    from collections.abc import Iterator
    import numpy as np

class AggregationContext(abc.ABC):
    __slots__ = ('parent', 'group_by', 'order_by', 'dtype', 'max_lookback', 'output_type')

    def __init__(self, parent=None, group_by=None, order_by=None, max_lookback=None, output_type=None):
        if False:
            for i in range(10):
                print('nop')
        self.parent = parent
        self.group_by = group_by
        self.order_by = order_by
        self.dtype = None if output_type is None else output_type.to_pandas()
        self.output_type = output_type
        self.max_lookback = max_lookback

    @abc.abstractmethod
    def agg(self, grouped_data, function, *args, **kwargs):
        if False:
            while True:
                i = 10
        pass

def wrap_for_apply(function: Callable, args: tuple[Any, ...] | None=None, kwargs: dict[str, Any] | None=None) -> Callable:
    if False:
        while True:
            i = 10
    'Wrap a function for use with Pandas `apply`.\n\n    Parameters\n    ----------\n    function : Callable\n        A function to be used with Pandas `apply`.\n    args : Optional[Tuple[Any, ...]]\n        args to be passed to function when it is called by Pandas `apply`\n    kwargs : Optional[Dict[str, Any]]\n        kwargs to be passed to function when it is called by Pandas `apply`\n    '
    assert callable(function), f'function {function} is not callable'
    new_args: tuple[Any, ...] = ()
    if args is not None:
        new_args = args
    new_kwargs: dict[str, Any] = {}
    if kwargs is not None:
        new_kwargs = kwargs

    @functools.wraps(function)
    def wrapped_func(data: Any, function: Callable=function, args: tuple[Any, ...]=new_args, kwargs: dict[str, Any]=new_kwargs) -> Callable:
        if False:
            i = 10
            return i + 15
        return function(data, *args, **kwargs)
    return wrapped_func

def wrap_for_agg(function: Callable, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Callable:
    if False:
        i = 10
        return i + 15
    'Wrap a function for use with Pandas `agg`.\n\n    This includes special logic that will force Pandas `agg` to always treat\n    the function as an aggregation function. Details:\n\n    When passed a function, Pandas `agg` will either:\n    1) Behave like Pandas `apply` and treat the function as a N->N mapping\n      function (i.e. calls the function once for every value in the Series\n      that `agg` is being called on), OR\n    2) Treat the function as a N->1 aggregation function (i.e. calls the\n      function once on the entire Series)\n    Pandas `agg` will use behavior #1 unless an error is raised when doing so.\n\n    We want to force Pandas `agg` to use behavior #2. To do this, we will wrap\n    the function with logic that checks that a Series is being passed in, and\n    raises a TypeError otherwise. When Pandas `agg` is attempting to use\n    behavior #1 but sees the TypeError, it will fall back to behavior #2.\n\n    Parameters\n    ----------\n    function : Callable\n        An aggregation function to be used with Pandas `agg`.\n    args : Tuple[Any, ...]\n        args to be passed to function when it is called by Pandas `agg`\n    kwargs : Dict[str, Any]\n        kwargs to be passed to function when it is called by Pandas `agg`\n    '
    assert callable(function), f'function {function} is not callable'

    @functools.wraps(function)
    def wrapped_func(data: Any, function: Callable=function, args: tuple[Any, ...]=args, kwargs: dict[str, Any]=kwargs) -> Callable:
        if False:
            return 10
        if not isinstance(data, pd.Series):
            raise TypeError(f'This function expects a Series, but saw an object of type {type(data)} instead.')
        return function(data, *args, **kwargs)
    return wrapped_func

class Summarize(AggregationContext):
    __slots__ = ()

    def agg(self, grouped_data, function, *args, **kwargs):
        if False:
            while True:
                i = 10
        if isinstance(function, str):
            return getattr(grouped_data, function)(*args, **kwargs)
        if not callable(function):
            raise TypeError(f'Object {function} is not callable or a string')
        if isinstance(grouped_data, pd.core.groupby.generic.SeriesGroupBy) and len(grouped_data):
            aggs = {}
            for (k, v) in grouped_data:
                func_args = [d.get_group(k) for d in args]
                aggs[k] = function(v, *func_args, **kwargs)
                grouped_col_name = v.name
            return pd.Series(aggs).rename(grouped_col_name).rename_axis(grouped_data.grouper.names)
        else:
            return grouped_data.agg(wrap_for_agg(function, args, kwargs))

class Transform(AggregationContext):
    __slots__ = ()

    def agg(self, grouped_data, function, *args, **kwargs):
        if False:
            return 10
        if self.output_type.is_struct():
            res = grouped_data.apply(function, *args, **kwargs)
        else:
            res = grouped_data.transform(function, *args, **kwargs)
        res.name = None
        return res

@functools.singledispatch
def compute_window_spec(dtype, obj):
    if False:
        i = 10
        return i + 15
    raise com.IbisTypeError(f'Unknown dtype type {dtype} and object {obj} for compute_window_spec')

@compute_window_spec.register(dt.Integer)
def compute_window_spec_none(_, obj):
    if False:
        while True:
            i = 10
    'Helper method only used for row-based windows.\n\n    Window spec in ibis is an inclusive window bound. A bound of 0\n    indicates the current row. Window spec in Pandas indicates window\n    size. Therefore, we must add 1 to the ibis window bound to get the\n    expected behavior.\n    '
    from ibis.backends.pandas.core import execute
    value = execute(obj)
    return value + 1

@compute_window_spec.register(dt.Interval)
def compute_window_spec_interval(_, obj):
    if False:
        return 10
    from ibis.backends.pandas.core import execute
    value = execute(obj)
    return pd.tseries.frequencies.to_offset(value)

def window_agg_built_in(frame: pd.DataFrame, windowed: pd.core.window.Window, function: str, max_lookback: ops.Literal, *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> pd.Series:
    if False:
        return 10
    'Apply window aggregation with built-in aggregators.'
    assert isinstance(function, str)
    method = operator.methodcaller(function, *args, **kwargs)
    if max_lookback is not None:
        agg_method = method

        def sliced_agg(s):
            if False:
                print('Hello World!')
            return agg_method(s.iloc[-max_lookback.value:])
        method = operator.methodcaller('apply', sliced_agg, raw=False)
    result = method(windowed)
    index = result.index
    result.index = pd.MultiIndex.from_arrays([frame.index] + list(map(index.get_level_values, range(index.nlevels))), names=[frame.index.name] + index.names)
    return result

def create_window_input_iter(grouped_data: SeriesGroupBy | pd.Series, masked_window_lower_indices: pd.Series, masked_window_upper_indices: pd.Series) -> Iterator[np.ndarray]:
    if False:
        for i in range(10):
            print('nop')
    data = getattr(grouped_data, 'obj', grouped_data).values
    lower_indices_array = masked_window_lower_indices.values
    upper_indices_array = masked_window_upper_indices.values
    for i in range(len(lower_indices_array)):
        lower_index = lower_indices_array[i]
        upper_index = upper_indices_array[i]
        yield data[lower_index:upper_index]

def window_agg_udf(grouped_data: SeriesGroupBy, function: Callable, window_lower_indices: pd.Series, window_upper_indices: pd.Series, mask: pd.Series, result_index: pd.Index, dtype: np.dtype, max_lookback: int, *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> pd.Series:
    if False:
        for i in range(10):
            print('nop')
    "Apply window aggregation with UDFs.\n\n    Notes\n    -----\n    Use custom logic to computing rolling window UDF instead of\n    using pandas's rolling function.\n    This is because pandas's rolling function doesn't support\n    multi param UDFs.\n    "
    assert len(window_lower_indices) == len(window_upper_indices)
    assert len(window_lower_indices) == len(mask)
    window_lower_indices = window_lower_indices.reset_index(drop=True)
    window_upper_indices = window_upper_indices.reset_index(drop=True)
    mask = mask.reset_index(drop=True)
    inputs = (grouped_data,) + args
    masked_window_lower_indices = window_lower_indices[mask].astype('i8')
    masked_window_upper_indices = window_upper_indices[mask].astype('i8')
    input_iters = [create_window_input_iter(arg, masked_window_lower_indices, masked_window_upper_indices) if isinstance(arg, (pd.Series, SeriesGroupBy)) else itertools.repeat(arg) for arg in inputs]
    valid_result = pd.Series((function(*(next(gen) for gen in input_iters)) for i in range(len(masked_window_lower_indices))))
    valid_result = pd.Series(valid_result)
    valid_result.index = masked_window_lower_indices.index
    result = pd.Series(index=mask.index, dtype=dtype)
    result[mask] = valid_result
    result.index = result_index
    return result

class Window(AggregationContext):
    __slots__ = ('construct_window',)

    def __init__(self, kind, *args, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(parent=kwargs.pop('parent', None), group_by=kwargs.pop('group_by', None), order_by=kwargs.pop('order_by', None), output_type=kwargs.pop('output_type'), max_lookback=kwargs.pop('max_lookback', None))
        self.construct_window = operator.methodcaller(kind, *args, **kwargs)

    def agg(self, grouped_data: pd.Series | SeriesGroupBy, function: str | Callable, *args: Any, **kwargs: Any) -> pd.Series:
        if False:
            i = 10
            return i + 15
        group_by = self.group_by
        order_by = self.order_by
        assert group_by or order_by
        parent = self.parent
        frame = getattr(parent, 'obj', parent)
        obj = getattr(grouped_data, 'obj', grouped_data)
        name = obj.name
        if frame[name] is not obj or name in group_by or name in order_by:
            name = f'{name}_{ibis.util.guid()}'
            frame = frame.assign(**{name: obj})
        columns = group_by + order_by + [name]
        indexed_by_ordering = frame[columns].copy()
        indexed_by_ordering['_placeholder'] = 0
        indexed_by_ordering = indexed_by_ordering.set_index(order_by)
        if group_by:
            grouped_frame = indexed_by_ordering.groupby(group_by, group_keys=False)
        else:
            grouped_frame = indexed_by_ordering
        grouped = grouped_frame[name]
        if callable(function):
            windowed_frame = self.construct_window(grouped_frame)
            window_sizes = windowed_frame['_placeholder'].count().reset_index(drop=True)
            mask = ~window_sizes.isna()
            window_upper_indices = pd.Series(range(len(window_sizes))) + 1
            window_lower_indices = window_upper_indices - window_sizes
            if get_time_col() in frame:
                result_index = construct_time_context_aware_series(obj, frame).index
            else:
                result_index = obj.index
            result = window_agg_udf(grouped_data, function, window_lower_indices, window_upper_indices, mask, result_index, self.dtype, self.max_lookback, *args, **kwargs)
        else:
            windowed = self.construct_window(grouped)
            result = window_agg_built_in(frame, windowed, function, self.max_lookback, *args, **kwargs)
        try:
            return result.astype(self.dtype, copy=False)
        except (TypeError, ValueError):
            return result

class Cumulative(Window):
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__('expanding', *args, **kwargs)

class Moving(Window):
    __slots__ = ()

    def __init__(self, start, max_lookback, *args, **kwargs):
        if False:
            while True:
                i = 10
        from ibis.backends.pandas.core import timedelta_types
        start = compute_window_spec(start.dtype, start.value)
        if isinstance(start, timedelta_types + (pd.offsets.DateOffset,)):
            closed = 'both'
        else:
            closed = None
        super().__init__('rolling', start, *args, max_lookback=max_lookback, closed=closed, min_periods=1, **kwargs)

    def short_circuit_method(self, grouped_data, function):
        if False:
            while True:
                i = 10
        raise AttributeError('No short circuit method for rolling operations')