from __future__ import annotations
import contextlib
from typing import TYPE_CHECKING, overload
from polars import functions as F
from polars.datatypes import Int64
from polars.utils._parse_expr_input import parse_as_expression
from polars.utils._wrap import wrap_expr
with contextlib.suppress(ImportError):
    import polars.polars as plr
if TYPE_CHECKING:
    from typing import Literal
    from polars import Expr, Series
    from polars.type_aliases import IntoExprColumn, PolarsIntegerType

@overload
def arange(start: int | IntoExprColumn, end: int | IntoExprColumn, step: int=..., *, dtype: PolarsIntegerType=..., eager: Literal[False]=...) -> Expr:
    if False:
        return 10
    ...

@overload
def arange(start: int | IntoExprColumn, end: int | IntoExprColumn, step: int=..., *, dtype: PolarsIntegerType=..., eager: Literal[True]) -> Series:
    if False:
        while True:
            i = 10
    ...

@overload
def arange(start: int | IntoExprColumn, end: int | IntoExprColumn, step: int=..., *, dtype: PolarsIntegerType=..., eager: bool) -> Expr | Series:
    if False:
        return 10
    ...

def arange(start: int | IntoExprColumn, end: int | IntoExprColumn, step: int=1, *, dtype: PolarsIntegerType=Int64, eager: bool=False) -> Expr | Series:
    if False:
        while True:
            i = 10
    "\n    Generate a range of integers.\n\n    Alias for :func:`int_range`.\n\n    Parameters\n    ----------\n    start\n        Lower bound of the range (inclusive).\n    end\n        Upper bound of the range (exclusive).\n    step\n        Step size of the range.\n    dtype\n        Data type of the range. Defaults to `Int64`.\n    eager\n        Evaluate immediately and return a `Series`.\n        If set to `False` (default), return an expression instead.\n\n    Returns\n    -------\n    Column of data type `dtype`.\n\n    See Also\n    --------\n    int_range : Generate a range of integers.\n    int_ranges : Generate a range of integers for each row of the input columns.\n\n    Examples\n    --------\n    >>> pl.arange(0, 3, eager=True)\n    shape: (3,)\n    Series: 'int' [i64]\n    [\n            0\n            1\n            2\n    ]\n\n    "
    return int_range(start, end, step, dtype=dtype, eager=eager)

@overload
def int_range(start: int | IntoExprColumn, end: int | IntoExprColumn, step: int=..., *, dtype: PolarsIntegerType=..., eager: Literal[False]=...) -> Expr:
    if False:
        print('Hello World!')
    ...

@overload
def int_range(start: int | IntoExprColumn, end: int | IntoExprColumn, step: int=..., *, dtype: PolarsIntegerType=..., eager: Literal[True]) -> Series:
    if False:
        print('Hello World!')
    ...

@overload
def int_range(start: int | IntoExprColumn, end: int | IntoExprColumn, step: int=..., *, dtype: PolarsIntegerType=..., eager: bool) -> Expr | Series:
    if False:
        while True:
            i = 10
    ...

def int_range(start: int | IntoExprColumn, end: int | IntoExprColumn, step: int=1, *, dtype: PolarsIntegerType=Int64, eager: bool=False) -> Expr | Series:
    if False:
        i = 10
        return i + 15
    "\n    Generate a range of integers.\n\n    Parameters\n    ----------\n    start\n        Lower bound of the range (inclusive).\n    end\n        Upper bound of the range (exclusive).\n    step\n        Step size of the range.\n    dtype\n        Data type of the range. Defaults to `Int64`.\n    eager\n        Evaluate immediately and return a `Series`.\n        If set to `False` (default), return an expression instead.\n\n    Returns\n    -------\n    Expr or Series\n        Column of data type :class:`Int64`.\n\n    See Also\n    --------\n    int_ranges : Generate a range of integers for each row of the input columns.\n\n    Examples\n    --------\n    >>> pl.int_range(0, 3, eager=True)\n    shape: (3,)\n    Series: 'int' [i64]\n    [\n            0\n            1\n            2\n    ]\n\n    "
    start = parse_as_expression(start)
    end = parse_as_expression(end)
    result = wrap_expr(plr.int_range(start, end, step, dtype))
    if eager:
        return F.select(result).to_series()
    return result

@overload
def int_ranges(start: int | IntoExprColumn, end: int | IntoExprColumn, step: int=..., *, dtype: PolarsIntegerType=..., eager: Literal[False]=...) -> Expr:
    if False:
        i = 10
        return i + 15
    ...

@overload
def int_ranges(start: int | IntoExprColumn, end: int | IntoExprColumn, step: int=..., *, dtype: PolarsIntegerType=..., eager: Literal[True]) -> Series:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def int_ranges(start: int | IntoExprColumn, end: int | IntoExprColumn, step: int=..., *, dtype: PolarsIntegerType=..., eager: bool) -> Expr | Series:
    if False:
        return 10
    ...

def int_ranges(start: int | IntoExprColumn, end: int | IntoExprColumn, step: int=1, *, dtype: PolarsIntegerType=Int64, eager: bool=False) -> Expr | Series:
    if False:
        i = 10
        return i + 15
    '\n    Generate a range of integers for each row of the input columns.\n\n    Parameters\n    ----------\n    start\n        Lower bound of the range (inclusive).\n    end\n        Upper bound of the range (exclusive).\n    step\n        Step size of the range.\n    dtype\n        Integer data type of the ranges. Defaults to `Int64`.\n    eager\n        Evaluate immediately and return a `Series`.\n        If set to `False` (default), return an expression instead.\n\n    Returns\n    -------\n    Expr or Series\n        Column of data type `List(dtype)`.\n\n    See Also\n    --------\n    int_range : Generate a single range of integers.\n\n    Examples\n    --------\n    >>> df = pl.DataFrame({"start": [1, -1], "end": [3, 2]})\n    >>> df.with_columns(pl.int_ranges("start", "end"))\n    shape: (2, 3)\n    ┌───────┬─────┬────────────┐\n    │ start ┆ end ┆ int_range  │\n    │ ---   ┆ --- ┆ ---        │\n    │ i64   ┆ i64 ┆ list[i64]  │\n    ╞═══════╪═════╪════════════╡\n    │ 1     ┆ 3   ┆ [1, 2]     │\n    │ -1    ┆ 2   ┆ [-1, 0, 1] │\n    └───────┴─────┴────────────┘\n\n    '
    start = parse_as_expression(start)
    end = parse_as_expression(end)
    result = wrap_expr(plr.int_ranges(start, end, step, dtype))
    if eager:
        return F.select(result).to_series()
    return result