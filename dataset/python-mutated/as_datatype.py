from __future__ import annotations
import contextlib
from typing import TYPE_CHECKING, Iterable, overload
from polars import functions as F
from polars.datatypes import Date, Struct, Time
from polars.utils._parse_expr_input import parse_as_expression, parse_as_list_of_expressions
from polars.utils._wrap import wrap_expr
from polars.utils.deprecation import rename_use_earliest_to_ambiguous
with contextlib.suppress(ImportError):
    import polars.polars as plr
if TYPE_CHECKING:
    from typing import Literal
    from polars import Expr, Series
    from polars.type_aliases import Ambiguous, IntoExpr, SchemaDict, TimeUnit

def datetime_(year: int | IntoExpr, month: int | IntoExpr, day: int | IntoExpr, hour: int | IntoExpr | None=None, minute: int | IntoExpr | None=None, second: int | IntoExpr | None=None, microsecond: int | IntoExpr | None=None, *, time_unit: TimeUnit='us', time_zone: str | None=None, use_earliest: bool | None=None, ambiguous: Ambiguous | Expr='raise') -> Expr:
    if False:
        while True:
            i = 10
    "\n    Create a Polars literal expression of type Datetime.\n\n    Parameters\n    ----------\n    year\n        Column or literal.\n    month\n        Column or literal, ranging from 1-12.\n    day\n        Column or literal, ranging from 1-31.\n    hour\n        Column or literal, ranging from 0-23.\n    minute\n        Column or literal, ranging from 0-59.\n    second\n        Column or literal, ranging from 0-59.\n    microsecond\n        Column or literal, ranging from 0-999999.\n    time_unit : {'us', 'ms', 'ns'}\n        Time unit of the resulting expression.\n    time_zone\n        Time zone of the resulting expression.\n    use_earliest\n        Determine how to deal with ambiguous datetimes:\n\n        - `None` (default): raise\n        - `True`: use the earliest datetime\n        - `False`: use the latest datetime\n\n        .. deprecated:: 0.19.0\n            Use `ambiguous` instead\n    ambiguous\n        Determine how to deal with ambiguous datetimes:\n\n        - `'raise'` (default): raise\n        - `'earliest'`: use the earliest datetime\n        - `'latest'`: use the latest datetime\n\n\n    Returns\n    -------\n    Expr\n        Expression of data type :class:`Datetime`.\n\n    "
    ambiguous = parse_as_expression(rename_use_earliest_to_ambiguous(use_earliest, ambiguous), str_as_lit=True)
    year_expr = parse_as_expression(year)
    month_expr = parse_as_expression(month)
    day_expr = parse_as_expression(day)
    if hour is not None:
        hour = parse_as_expression(hour)
    if minute is not None:
        minute = parse_as_expression(minute)
    if second is not None:
        second = parse_as_expression(second)
    if microsecond is not None:
        microsecond = parse_as_expression(microsecond)
    return wrap_expr(plr.datetime(year_expr, month_expr, day_expr, hour, minute, second, microsecond, time_unit, time_zone, ambiguous))

def date_(year: Expr | str | int, month: Expr | str | int, day: Expr | str | int) -> Expr:
    if False:
        print('Hello World!')
    '\n    Create a Polars literal expression of type Date.\n\n    Parameters\n    ----------\n    year\n        column or literal.\n    month\n        column or literal, ranging from 1-12.\n    day\n        column or literal, ranging from 1-31.\n\n    Returns\n    -------\n    Expr\n        Expression of data type :class:`Date`.\n\n    '
    return datetime_(year, month, day).cast(Date).alias('date')

def time_(hour: Expr | str | int | None=None, minute: Expr | str | int | None=None, second: Expr | str | int | None=None, microsecond: Expr | str | int | None=None) -> Expr:
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a Polars literal expression of type Time.\n\n    Parameters\n    ----------\n    hour\n        column or literal, ranging from 0-23.\n    minute\n        column or literal, ranging from 0-59.\n    second\n        column or literal, ranging from 0-59.\n    microsecond\n        column or literal, ranging from 0-999999.\n\n    Returns\n    -------\n    Expr\n        Expression of data type :class:`Date`.\n\n    '
    epoch_start = (1970, 1, 1)
    return datetime_(*epoch_start, hour, minute, second, microsecond).cast(Time).alias('time')

def duration(*, weeks: Expr | str | int | None=None, days: Expr | str | int | None=None, hours: Expr | str | int | None=None, minutes: Expr | str | int | None=None, seconds: Expr | str | int | None=None, milliseconds: Expr | str | int | None=None, microseconds: Expr | str | int | None=None, nanoseconds: Expr | str | int | None=None, time_unit: TimeUnit='us') -> Expr:
    if False:
        return 10
    '\n    Create polars `Duration` from distinct time components.\n\n    Parameters\n    ----------\n    weeks\n        Number of weeks.\n    days\n        Number of days.\n    hours\n        Number of hours.\n    minutes\n        Number of minutes.\n    seconds\n        Number of seconds.\n    milliseconds\n        Number of milliseconds.\n    microseconds\n        Number of microseconds.\n    nanoseconds\n        Number of nanoseconds.\n    time_unit : {\'us\', \'ms\', \'ns\'}\n        Time unit of the resulting expression.\n\n    Returns\n    -------\n    Expr\n        Expression of data type :class:`Duration`.\n\n    Notes\n    -----\n    A `duration` represents a fixed amount of time. For example,\n    `pl.duration(days=1)` means "exactly 24 hours". By contrast,\n    `Expr.dt.offset_by(\'1d\')` means "1 calendar day", which could sometimes be\n    23 hours or 25 hours depending on Daylight Savings Time.\n    For non-fixed durations such as "calendar month" or "calendar day",\n    please use :meth:`polars.Expr.dt.offset_by` instead.\n\n    Examples\n    --------\n    >>> from datetime import datetime\n    >>> df = pl.DataFrame(\n    ...     {\n    ...         "dt": [datetime(2022, 1, 1), datetime(2022, 1, 2)],\n    ...         "add": [1, 2],\n    ...     }\n    ... )\n    >>> df\n    shape: (2, 2)\n    ┌─────────────────────┬─────┐\n    │ dt                  ┆ add │\n    │ ---                 ┆ --- │\n    │ datetime[μs]        ┆ i64 │\n    ╞═════════════════════╪═════╡\n    │ 2022-01-01 00:00:00 ┆ 1   │\n    │ 2022-01-02 00:00:00 ┆ 2   │\n    └─────────────────────┴─────┘\n    >>> with pl.Config(tbl_width_chars=120):\n    ...     df.select(\n    ...         (pl.col("dt") + pl.duration(weeks="add")).alias("add_weeks"),\n    ...         (pl.col("dt") + pl.duration(days="add")).alias("add_days"),\n    ...         (pl.col("dt") + pl.duration(seconds="add")).alias("add_seconds"),\n    ...         (pl.col("dt") + pl.duration(milliseconds="add")).alias("add_millis"),\n    ...         (pl.col("dt") + pl.duration(hours="add")).alias("add_hours"),\n    ...     )\n    ...\n    shape: (2, 5)\n    ┌─────────────────────┬─────────────────────┬─────────────────────┬─────────────────────────┬─────────────────────┐\n    │ add_weeks           ┆ add_days            ┆ add_seconds         ┆ add_millis              ┆ add_hours           │\n    │ ---                 ┆ ---                 ┆ ---                 ┆ ---                     ┆ ---                 │\n    │ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]            ┆ datetime[μs]        │\n    ╞═════════════════════╪═════════════════════╪═════════════════════╪═════════════════════════╪═════════════════════╡\n    │ 2022-01-08 00:00:00 ┆ 2022-01-02 00:00:00 ┆ 2022-01-01 00:00:01 ┆ 2022-01-01 00:00:00.001 ┆ 2022-01-01 01:00:00 │\n    │ 2022-01-16 00:00:00 ┆ 2022-01-04 00:00:00 ┆ 2022-01-02 00:00:02 ┆ 2022-01-02 00:00:00.002 ┆ 2022-01-02 02:00:00 │\n    └─────────────────────┴─────────────────────┴─────────────────────┴─────────────────────────┴─────────────────────┘\n\n    If you need to add non-fixed durations, you should use :meth:`polars.Expr.dt.offset_by` instead:\n\n    >>> with pl.Config(tbl_width_chars=120):\n    ...     df.select(\n    ...         add_calendar_days=pl.col("dt").dt.offset_by(\n    ...             pl.format("{}d", pl.col("add"))\n    ...         ),\n    ...         add_calendar_months=pl.col("dt").dt.offset_by(\n    ...             pl.format("{}mo", pl.col("add"))\n    ...         ),\n    ...         add_calendar_years=pl.col("dt").dt.offset_by(\n    ...             pl.format("{}y", pl.col("add"))\n    ...         ),\n    ...     )\n    ...\n    shape: (2, 3)\n    ┌─────────────────────┬─────────────────────┬─────────────────────┐\n    │ add_calendar_days   ┆ add_calendar_months ┆ add_calendar_years  │\n    │ ---                 ┆ ---                 ┆ ---                 │\n    │ datetime[μs]        ┆ datetime[μs]        ┆ datetime[μs]        │\n    ╞═════════════════════╪═════════════════════╪═════════════════════╡\n    │ 2022-01-02 00:00:00 ┆ 2022-02-01 00:00:00 ┆ 2023-01-01 00:00:00 │\n    │ 2022-01-04 00:00:00 ┆ 2022-03-02 00:00:00 ┆ 2024-01-02 00:00:00 │\n    └─────────────────────┴─────────────────────┴─────────────────────┘\n\n    '
    if weeks is not None:
        weeks = parse_as_expression(weeks)
    if days is not None:
        days = parse_as_expression(days)
    if hours is not None:
        hours = parse_as_expression(hours)
    if minutes is not None:
        minutes = parse_as_expression(minutes)
    if seconds is not None:
        seconds = parse_as_expression(seconds)
    if milliseconds is not None:
        milliseconds = parse_as_expression(milliseconds)
    if microseconds is not None:
        microseconds = parse_as_expression(microseconds)
    if nanoseconds is not None:
        nanoseconds = parse_as_expression(nanoseconds)
    return wrap_expr(plr.duration(weeks, days, hours, minutes, seconds, milliseconds, microseconds, nanoseconds, time_unit))

def concat_list(exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr) -> Expr:
    if False:
        while True:
            i = 10
    '\n    Horizontally concatenate columns into a single list column.\n\n    Operates in linear time.\n\n    Parameters\n    ----------\n    exprs\n        Columns to concatenate into a single list column. Accepts expression input.\n        Strings are parsed as column names, other non-expression inputs are parsed as\n        literals.\n    *more_exprs\n        Additional columns to concatenate into a single list column, specified as\n        positional arguments.\n\n    Examples\n    --------\n    Create lagged columns and collect them into a list. This mimics a rolling window.\n\n    >>> df = pl.DataFrame({"A": [1.0, 2.0, 9.0, 2.0, 13.0]})\n    >>> df = df.select([pl.col("A").shift(i).alias(f"A_lag_{i}") for i in range(3)])\n    >>> df.select(\n    ...     pl.concat_list([f"A_lag_{i}" for i in range(3)][::-1]).alias("A_rolling")\n    ... )\n    shape: (5, 1)\n    ┌───────────────────┐\n    │ A_rolling         │\n    │ ---               │\n    │ list[f64]         │\n    ╞═══════════════════╡\n    │ [null, null, 1.0] │\n    │ [null, 1.0, 2.0]  │\n    │ [1.0, 2.0, 9.0]   │\n    │ [2.0, 9.0, 2.0]   │\n    │ [9.0, 2.0, 13.0]  │\n    └───────────────────┘\n\n    '
    exprs = parse_as_list_of_expressions(exprs, *more_exprs)
    return wrap_expr(plr.concat_list(exprs))

@overload
def struct(*exprs: IntoExpr | Iterable[IntoExpr], schema: SchemaDict | None=..., eager: Literal[False]=..., **named_exprs: IntoExpr) -> Expr:
    if False:
        i = 10
        return i + 15
    ...

@overload
def struct(*exprs: IntoExpr | Iterable[IntoExpr], schema: SchemaDict | None=..., eager: Literal[True], **named_exprs: IntoExpr) -> Series:
    if False:
        print('Hello World!')
    ...

@overload
def struct(*exprs: IntoExpr | Iterable[IntoExpr], schema: SchemaDict | None=..., eager: bool, **named_exprs: IntoExpr) -> Expr | Series:
    if False:
        return 10
    ...

def struct(*exprs: IntoExpr | Iterable[IntoExpr], schema: SchemaDict | None=None, eager: bool=False, **named_exprs: IntoExpr) -> Expr | Series:
    if False:
        for i in range(10):
            print('nop')
    '\n    Collect columns into a struct column.\n\n    Parameters\n    ----------\n    *exprs\n        Column(s) to collect into a struct column, specified as positional arguments.\n        Accepts expression input. Strings are parsed as column names,\n        other non-expression inputs are parsed as literals.\n    schema\n        Optional schema that explicitly defines the struct field dtypes. If no columns\n        or expressions are provided, schema keys are used to define columns.\n    eager\n        Evaluate immediately and return a `Series`. If set to `False` (default),\n        return an expression instead.\n    **named_exprs\n        Additional columns to collect into the struct column, specified as keyword\n        arguments. The columns will be renamed to the keyword used.\n\n    Examples\n    --------\n    Collect all columns of a dataframe into a struct by passing `pl.all()`.\n\n    >>> df = pl.DataFrame(\n    ...     {\n    ...         "int": [1, 2],\n    ...         "str": ["a", "b"],\n    ...         "bool": [True, None],\n    ...         "list": [[1, 2], [3]],\n    ...     }\n    ... )\n    >>> df.select(pl.struct(pl.all()).alias("my_struct"))\n    shape: (2, 1)\n    ┌─────────────────────┐\n    │ my_struct           │\n    │ ---                 │\n    │ struct[4]           │\n    ╞═════════════════════╡\n    │ {1,"a",true,[1, 2]} │\n    │ {2,"b",null,[3]}    │\n    └─────────────────────┘\n\n    Collect selected columns into a struct by either passing a list of columns, or by\n    specifying each column as a positional argument.\n\n    >>> df.select(pl.struct("int", False).alias("my_struct"))\n    shape: (2, 1)\n    ┌───────────┐\n    │ my_struct │\n    │ ---       │\n    │ struct[2] │\n    ╞═══════════╡\n    │ {1,false} │\n    │ {2,false} │\n    └───────────┘\n\n    Use keyword arguments to easily name each struct field.\n\n    >>> df.select(pl.struct(p="int", q="bool").alias("my_struct")).schema\n    OrderedDict([(\'my_struct\', Struct([Field(\'p\', Int64), Field(\'q\', Boolean)]))])\n\n    '
    pyexprs = parse_as_list_of_expressions(*exprs, **named_exprs)
    expr = wrap_expr(plr.as_struct(pyexprs))
    if schema:
        if not exprs:
            expr = wrap_expr(plr.as_struct(parse_as_list_of_expressions(list(schema.keys()))))
        expr = expr.cast(Struct(schema), strict=False)
    if eager:
        return F.select(expr).to_series()
    else:
        return expr

def concat_str(exprs: IntoExpr | Iterable[IntoExpr], *more_exprs: IntoExpr, separator: str='') -> Expr:
    if False:
        for i in range(10):
            print('nop')
    '\n    Horizontally concatenate columns into a single string column.\n\n    Operates in linear time.\n\n    Parameters\n    ----------\n    exprs\n        Columns to concatenate into a single string column. Accepts expression input.\n        Strings are parsed as column names, other non-expression inputs are parsed as\n        literals. Non-`Utf8` columns are cast to `Utf8`.\n    *more_exprs\n        Additional columns to concatenate into a single string column, specified as\n        positional arguments.\n    separator\n        String that will be used to separate the values of each column.\n\n    Examples\n    --------\n    >>> df = pl.DataFrame(\n    ...     {\n    ...         "a": [1, 2, 3],\n    ...         "b": ["dogs", "cats", None],\n    ...         "c": ["play", "swim", "walk"],\n    ...     }\n    ... )\n    >>> df.with_columns(\n    ...     pl.concat_str(\n    ...         [\n    ...             pl.col("a") * 2,\n    ...             pl.col("b"),\n    ...             pl.col("c"),\n    ...         ],\n    ...         separator=" ",\n    ...     ).alias("full_sentence"),\n    ... )\n    shape: (3, 4)\n    ┌─────┬──────┬──────┬───────────────┐\n    │ a   ┆ b    ┆ c    ┆ full_sentence │\n    │ --- ┆ ---  ┆ ---  ┆ ---           │\n    │ i64 ┆ str  ┆ str  ┆ str           │\n    ╞═════╪══════╪══════╪═══════════════╡\n    │ 1   ┆ dogs ┆ play ┆ 2 dogs play   │\n    │ 2   ┆ cats ┆ swim ┆ 4 cats swim   │\n    │ 3   ┆ null ┆ walk ┆ null          │\n    └─────┴──────┴──────┴───────────────┘\n\n    '
    exprs = parse_as_list_of_expressions(exprs, *more_exprs)
    return wrap_expr(plr.concat_str(exprs, separator))

def format(f_string: str, *args: Expr | str) -> Expr:
    if False:
        return 10
    '\n    Format expressions as a string.\n\n    Parameters\n    ----------\n    f_string\n        A string that with placeholders.\n        For example: "hello_{}" or "{}_world\n    args\n        Expression(s) that fill the placeholders\n\n    Examples\n    --------\n    >>> df = pl.DataFrame(\n    ...     {\n    ...         "a": ["a", "b", "c"],\n    ...         "b": [1, 2, 3],\n    ...     }\n    ... )\n    >>> df.select(\n    ...     [\n    ...         pl.format("foo_{}_bar_{}", pl.col("a"), "b").alias("fmt"),\n    ...     ]\n    ... )\n    shape: (3, 1)\n    ┌─────────────┐\n    │ fmt         │\n    │ ---         │\n    │ str         │\n    ╞═════════════╡\n    │ foo_a_bar_1 │\n    │ foo_b_bar_2 │\n    │ foo_c_bar_3 │\n    └─────────────┘\n\n    '
    if f_string.count('{}') != len(args):
        raise ValueError('number of placeholders should equal the number of arguments')
    exprs = []
    arguments = iter(args)
    for (i, s) in enumerate(f_string.split('{}')):
        if i > 0:
            e = wrap_expr(parse_as_expression(next(arguments)))
            exprs.append(e)
        if len(s) > 0:
            exprs.append(F.lit(s))
    return concat_str(exprs, separator='')