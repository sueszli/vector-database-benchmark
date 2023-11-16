from __future__ import annotations
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from polars import Expr

class ExprNameNameSpace:
    """Namespace for expressions that operate on expression names."""
    _accessor = 'name'

    def __init__(self, expr: Expr):
        if False:
            i = 10
            return i + 15
        self._from_pyexpr = expr._from_pyexpr
        self._pyexpr = expr._pyexpr

    def keep(self) -> Expr:
        if False:
            for i in range(10):
                print('nop')
        '\n        Keep the original root name of the expression.\n\n        Notes\n        -----\n        Due to implementation constraints, this method can only be called as the last\n        expression in a chain.\n\n        See Also\n        --------\n        alias\n        map\n\n        Examples\n        --------\n        Prevent errors due to potential duplicate column names.\n\n        >>> df = pl.DataFrame(\n        ...     {\n        ...         "a": [1, 2],\n        ...         "b": [3, 4],\n        ...     }\n        ... )\n        >>> df.select((pl.lit(10) / pl.all()).name.keep())\n        shape: (2, 2)\n        ┌──────┬──────────┐\n        │ a    ┆ b        │\n        │ ---  ┆ ---      │\n        │ f64  ┆ f64      │\n        ╞══════╪══════════╡\n        │ 10.0 ┆ 3.333333 │\n        │ 5.0  ┆ 2.5      │\n        └──────┴──────────┘\n\n        Undo an alias operation.\n\n        >>> df.with_columns((pl.col("a") * 9).alias("c").name.keep())\n        shape: (2, 2)\n        ┌─────┬─────┐\n        │ a   ┆ b   │\n        │ --- ┆ --- │\n        │ i64 ┆ i64 │\n        ╞═════╪═════╡\n        │ 9   ┆ 3   │\n        │ 18  ┆ 4   │\n        └─────┴─────┘\n\n        '
        return self._from_pyexpr(self._pyexpr.name_keep())

    def map(self, function: Callable[[str], str]) -> Expr:
        if False:
            return 10
        '\n        Rename the output of an expression by mapping a function over the root name.\n\n        Parameters\n        ----------\n        function\n            Function that maps a root name to a new name.\n\n        See Also\n        --------\n        keep\n        prefix\n        suffix\n\n        Examples\n        --------\n        Remove a common suffix and convert to lower case.\n\n        >>> df = pl.DataFrame(\n        ...     {\n        ...         "A_reverse": [3, 2, 1],\n        ...         "B_reverse": ["z", "y", "x"],\n        ...     }\n        ... )\n        >>> df.with_columns(\n        ...     pl.all().reverse().name.map(lambda c: c.rstrip("_reverse").lower())\n        ... )\n        shape: (3, 4)\n        ┌───────────┬───────────┬─────┬─────┐\n        │ A_reverse ┆ B_reverse ┆ a   ┆ b   │\n        │ ---       ┆ ---       ┆ --- ┆ --- │\n        │ i64       ┆ str       ┆ i64 ┆ str │\n        ╞═══════════╪═══════════╪═════╪═════╡\n        │ 3         ┆ z         ┆ 1   ┆ x   │\n        │ 2         ┆ y         ┆ 2   ┆ y   │\n        │ 1         ┆ x         ┆ 3   ┆ z   │\n        └───────────┴───────────┴─────┴─────┘\n\n        '
        return self._from_pyexpr(self._pyexpr.name_map(function))

    def prefix(self, prefix: str) -> Expr:
        if False:
            while True:
                i = 10
        '\n        Add a prefix to the root column name of the expression.\n\n        Parameters\n        ----------\n        prefix\n            Prefix to add to the root column name.\n\n        Notes\n        -----\n        This will undo any previous renaming operations on the expression.\n\n        Due to implementation constraints, this method can only be called as the last\n        expression in a chain.\n\n        See Also\n        --------\n        suffix\n\n        Examples\n        --------\n        >>> df = pl.DataFrame(\n        ...     {\n        ...         "a": [1, 2, 3],\n        ...         "b": ["x", "y", "z"],\n        ...     }\n        ... )\n        >>> df.with_columns(pl.all().reverse().name.prefix("reverse_"))\n        shape: (3, 4)\n        ┌─────┬─────┬───────────┬───────────┐\n        │ a   ┆ b   ┆ reverse_a ┆ reverse_b │\n        │ --- ┆ --- ┆ ---       ┆ ---       │\n        │ i64 ┆ str ┆ i64       ┆ str       │\n        ╞═════╪═════╪═══════════╪═══════════╡\n        │ 1   ┆ x   ┆ 3         ┆ z         │\n        │ 2   ┆ y   ┆ 2         ┆ y         │\n        │ 3   ┆ z   ┆ 1         ┆ x         │\n        └─────┴─────┴───────────┴───────────┘\n\n        '
        return self._from_pyexpr(self._pyexpr.name_prefix(prefix))

    def suffix(self, suffix: str) -> Expr:
        if False:
            i = 10
            return i + 15
        '\n        Add a suffix to the root column name of the expression.\n\n        Parameters\n        ----------\n        suffix\n            Suffix to add to the root column name.\n\n        Notes\n        -----\n        This will undo any previous renaming operations on the expression.\n\n        Due to implementation constraints, this method can only be called as the last\n        expression in a chain.\n\n        See Also\n        --------\n        prefix\n\n        Examples\n        --------\n        >>> df = pl.DataFrame(\n        ...     {\n        ...         "a": [1, 2, 3],\n        ...         "b": ["x", "y", "z"],\n        ...     }\n        ... )\n        >>> df.with_columns(pl.all().reverse().name.suffix("_reverse"))\n        shape: (3, 4)\n        ┌─────┬─────┬───────────┬───────────┐\n        │ a   ┆ b   ┆ a_reverse ┆ b_reverse │\n        │ --- ┆ --- ┆ ---       ┆ ---       │\n        │ i64 ┆ str ┆ i64       ┆ str       │\n        ╞═════╪═════╪═══════════╪═══════════╡\n        │ 1   ┆ x   ┆ 3         ┆ z         │\n        │ 2   ┆ y   ┆ 2         ┆ y         │\n        │ 3   ┆ z   ┆ 1         ┆ x         │\n        └─────┴─────┴───────────┴───────────┘\n\n        '
        return self._from_pyexpr(self._pyexpr.name_suffix(suffix))

    def to_lowercase(self) -> Expr:
        if False:
            return 10
        '\n        Make the root column name lowercase.\n\n        Notes\n        -----\n        Due to implementation constraints, this method can only be called as the last\n        expression in a chain.\n\n        See Also\n        --------\n        prefix\n        suffix\n        to_uppercase\n\n        Examples\n        --------\n        >>> df = pl.DataFrame(\n        ...     {\n        ...         "ColX": [1, 2, 3],\n        ...         "ColY": ["x", "y", "z"],\n        ...     }\n        ... )\n        >>> df.with_columns(pl.all().name.to_lowercase())\n        shape: (3, 4)\n        ┌──────┬──────┬──────┬──────┐\n        │ ColX ┆ ColY ┆ colx ┆ coly │\n        │ ---  ┆ ---  ┆ ---  ┆ ---  │\n        │ i64  ┆ str  ┆ i64  ┆ str  │\n        ╞══════╪══════╪══════╪══════╡\n        │ 1    ┆ x    ┆ 1    ┆ x    │\n        │ 2    ┆ y    ┆ 2    ┆ y    │\n        │ 3    ┆ z    ┆ 3    ┆ z    │\n        └──────┴──────┴──────┴──────┘\n\n        '
        return self._from_pyexpr(self._pyexpr.name_to_lowercase())

    def to_uppercase(self) -> Expr:
        if False:
            while True:
                i = 10
        '\n        Make the root column name uppercase.\n\n        Notes\n        -----\n        Due to implementation constraints, this method can only be called as the last\n        expression in a chain.\n\n        See Also\n        --------\n        prefix\n        suffix\n        to_lowercase\n\n        Examples\n        --------\n        >>> df = pl.DataFrame(\n        ...     {\n        ...         "ColX": [1, 2, 3],\n        ...         "ColY": ["x", "y", "z"],\n        ...     }\n        ... )\n        >>> df.with_columns(pl.all().name.to_uppercase())\n        shape: (3, 4)\n        ┌──────┬──────┬──────┬──────┐\n        │ ColX ┆ ColY ┆ COLX ┆ COLY │\n        │ ---  ┆ ---  ┆ ---  ┆ ---  │\n        │ i64  ┆ str  ┆ i64  ┆ str  │\n        ╞══════╪══════╪══════╪══════╡\n        │ 1    ┆ x    ┆ 1    ┆ x    │\n        │ 2    ┆ y    ┆ 2    ┆ y    │\n        │ 3    ┆ z    ┆ 3    ┆ z    │\n        └──────┴──────┴──────┴──────┘\n\n        '
        return self._from_pyexpr(self._pyexpr.name_to_uppercase())