from __future__ import annotations
import contextlib
from typing import TYPE_CHECKING, Any, Iterable, Protocol, cast
from polars.datatypes import is_polars_dtype
from polars.utils._wrap import wrap_expr
plr: Any = None
with contextlib.suppress(ImportError):
    import polars.polars as plr
if TYPE_CHECKING:
    from polars.expr.expr import Expr
    from polars.type_aliases import PolarsDataType
__all__ = ['col']

def _create_col(name: str | PolarsDataType | Iterable[str] | Iterable[PolarsDataType], *more_names: str | PolarsDataType) -> Expr:
    if False:
        print('Hello World!')
    'Create one or more column expressions representing column(s) in a DataFrame.'
    if more_names:
        if isinstance(name, str):
            names_str = [name]
            names_str.extend(more_names)
            return wrap_expr(plr.cols(names_str))
        elif is_polars_dtype(name):
            dtypes = [name]
            dtypes.extend(more_names)
            return wrap_expr(plr.dtype_cols(dtypes))
        else:
            raise TypeError(f'invalid input for `col`\n\nExpected `str` or `DataType`, got {type(name).__name__!r}.')
    if isinstance(name, str):
        return wrap_expr(plr.col(name))
    elif is_polars_dtype(name):
        return wrap_expr(plr.dtype_cols([name]))
    elif isinstance(name, Iterable):
        names = list(name)
        if not names:
            return wrap_expr(plr.cols(names))
        item = names[0]
        if isinstance(item, str):
            return wrap_expr(plr.cols(names))
        elif is_polars_dtype(item):
            return wrap_expr(plr.dtype_cols(names))
        else:
            raise TypeError(f'invalid input for `col`\n\nExpected iterable of type `str` or `DataType`, got iterable of type {type(item).__name__!r}.')
    else:
        raise TypeError(f'invalid input for `col`\n\nExpected `str` or `DataType`, got {type(name).__name__!r}.')

class Column(Protocol):

    def __call__(self, name: str | PolarsDataType | Iterable[str] | Iterable[PolarsDataType], *more_names: str | PolarsDataType) -> Expr:
        if False:
            i = 10
            return i + 15
        ...

    def __getattr__(self, name: str) -> Expr:
        if False:
            return 10
        ...

class ColumnFactoryMeta(type):

    def __getattr__(self, name: str) -> Expr:
        if False:
            for i in range(10):
                print('nop')
        return _create_col(name)

class ColumnFactory(metaclass=ColumnFactoryMeta):
    """
    Create Polars column expressions.

    Notes
    -----
    An instance of this class is exported under the name `col`. It can be used as
    though it were a function by calling, for example, `pl.col("foo")`.
    See the :func:`__call__` method for further documentation.

    This helper class enables an alternative syntax for creating a column expression
    through attribute lookup. For example `col.foo` creates an expression equal to
    `col("foo")`.
    See the :func:`__getattr__` method for further documentation.

    The function call syntax is considered the idiomatic way of constructing a column
    expression. The alternative attribute syntax can be useful for quick prototyping as
    it can save some keystrokes, but has drawbacks in both expressiveness and
    readability.

    Examples
    --------
    >>> from polars import col
    >>> df = pl.DataFrame(
    ...     {
    ...         "foo": [1, 2],
    ...         "bar": [3, 4],
    ...     }
    ... )

    Create a new column expression using the standard syntax:

    >>> df.with_columns(baz=(col("foo") * col("bar")) / 2)
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ foo ┆ bar ┆ baz │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ f64 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ 1.5 │
    │ 2   ┆ 4   ┆ 4.0 │
    └─────┴─────┴─────┘

    Use attribute lookup to create a new column expression:

    >>> df.with_columns(baz=(col.foo + col.bar))
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ foo ┆ bar ┆ baz │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ i64 ┆ i64 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ 3   ┆ 4   │
    │ 2   ┆ 4   ┆ 6   │
    └─────┴─────┴─────┘

    """

    def __new__(cls, name: str | PolarsDataType | Iterable[str] | Iterable[PolarsDataType], *more_names: str | PolarsDataType) -> Expr:
        if False:
            i = 10
            return i + 15
        '\n        Create one or more column expressions representing column(s) in a DataFrame.\n\n        Parameters\n        ----------\n        name\n            The name or datatype of the column(s) to represent.\n            Accepts regular expression input.\n            Regular expressions should start with `^` and end with `$`.\n        *more_names\n            Additional names or datatypes of columns to represent,\n            specified as positional arguments.\n\n        Examples\n        --------\n        Pass a single column name to represent that column.\n\n        >>> df = pl.DataFrame(\n        ...     {\n        ...         "ham": [1, 2],\n        ...         "hamburger": [11, 22],\n        ...         "foo": [2, 1],\n        ...         "bar": ["a", "b"],\n        ...     }\n        ... )\n        >>> df.select(pl.col("foo"))\n        shape: (2, 1)\n        ┌─────┐\n        │ foo │\n        │ --- │\n        │ i64 │\n        ╞═════╡\n        │ 2   │\n        │ 1   │\n        └─────┘\n\n        Use dot syntax to save keystrokes for quick prototyping.\n\n        >>> from polars import col as c\n        >>> df.select(c.foo + c.ham)\n        shape: (2, 1)\n        ┌─────┐\n        │ foo │\n        │ --- │\n        │ i64 │\n        ╞═════╡\n        │ 3   │\n        │ 3   │\n        └─────┘\n\n        Use the wildcard `*` to represent all columns.\n\n        >>> df.select(pl.col("*"))\n        shape: (2, 4)\n        ┌─────┬───────────┬─────┬─────┐\n        │ ham ┆ hamburger ┆ foo ┆ bar │\n        │ --- ┆ ---       ┆ --- ┆ --- │\n        │ i64 ┆ i64       ┆ i64 ┆ str │\n        ╞═════╪═══════════╪═════╪═════╡\n        │ 1   ┆ 11        ┆ 2   ┆ a   │\n        │ 2   ┆ 22        ┆ 1   ┆ b   │\n        └─────┴───────────┴─────┴─────┘\n        >>> df.select(pl.col("*").exclude("ham"))\n        shape: (2, 3)\n        ┌───────────┬─────┬─────┐\n        │ hamburger ┆ foo ┆ bar │\n        │ ---       ┆ --- ┆ --- │\n        │ i64       ┆ i64 ┆ str │\n        ╞═══════════╪═════╪═════╡\n        │ 11        ┆ 2   ┆ a   │\n        │ 22        ┆ 1   ┆ b   │\n        └───────────┴─────┴─────┘\n\n        Regular expression input is supported.\n\n        >>> df.select(pl.col("^ham.*$"))\n        shape: (2, 2)\n        ┌─────┬───────────┐\n        │ ham ┆ hamburger │\n        │ --- ┆ ---       │\n        │ i64 ┆ i64       │\n        ╞═════╪═══════════╡\n        │ 1   ┆ 11        │\n        │ 2   ┆ 22        │\n        └─────┴───────────┘\n\n        Multiple columns can be represented by passing a list of names.\n\n        >>> df.select(pl.col(["hamburger", "foo"]))\n        shape: (2, 2)\n        ┌───────────┬─────┐\n        │ hamburger ┆ foo │\n        │ ---       ┆ --- │\n        │ i64       ┆ i64 │\n        ╞═══════════╪═════╡\n        │ 11        ┆ 2   │\n        │ 22        ┆ 1   │\n        └───────────┴─────┘\n\n        Or use positional arguments to represent multiple columns in the same way.\n\n        >>> df.select(pl.col("hamburger", "foo"))\n        shape: (2, 2)\n        ┌───────────┬─────┐\n        │ hamburger ┆ foo │\n        │ ---       ┆ --- │\n        │ i64       ┆ i64 │\n        ╞═══════════╪═════╡\n        │ 11        ┆ 2   │\n        │ 22        ┆ 1   │\n        └───────────┴─────┘\n\n        Easily select all columns that match a certain data type by passing that\n        datatype.\n\n        >>> df.select(pl.col(pl.Utf8))\n        shape: (2, 1)\n        ┌─────┐\n        │ bar │\n        │ --- │\n        │ str │\n        ╞═════╡\n        │ a   │\n        │ b   │\n        └─────┘\n        >>> df.select(pl.col(pl.Int64, pl.Float64))\n        shape: (2, 3)\n        ┌─────┬───────────┬─────┐\n        │ ham ┆ hamburger ┆ foo │\n        │ --- ┆ ---       ┆ --- │\n        │ i64 ┆ i64       ┆ i64 │\n        ╞═════╪═══════════╪═════╡\n        │ 1   ┆ 11        ┆ 2   │\n        │ 2   ┆ 22        ┆ 1   │\n        └─────┴───────────┴─────┘\n\n        '
        return _create_col(name, *more_names)

    def __call__(self, name: str | PolarsDataType | Iterable[str] | Iterable[PolarsDataType], *more_names: str | PolarsDataType) -> Expr:
        if False:
            return 10
        return _create_col(name, *more_names)

    def __getattr__(self, name: str) -> Expr:
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a column expression using attribute syntax.\n\n        Note that this syntax does not support passing data\n        types or multiple column names.\n\n        Parameters\n        ----------\n        name\n            The name of the column to represent.\n\n        Examples\n        --------\n        >>> from polars import col as c\n        >>> df = pl.DataFrame(\n        ...     {\n        ...         "foo": [1, 2],\n        ...         "bar": [3, 4],\n        ...     }\n        ... )\n        >>> df.select(c.foo + c.bar)\n        shape: (2, 1)\n        ┌─────┐\n        │ foo │\n        │ --- │\n        │ i64 │\n        ╞═════╡\n        │ 4   │\n        │ 6   │\n        └─────┘\n\n        '
        return getattr(type(self), name)
col = cast(Column, ColumnFactory)