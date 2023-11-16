from __future__ import annotations
from functools import cached_property
from typing import TYPE_CHECKING, Sequence
if TYPE_CHECKING:
    import ibis.expr.types as ir
    import pyarrow as pa

class IbisDataFrame:
    """An implementation of the dataframe interchange protocol.

    This is a thin shim around the pyarrow implementation to allow for:

    - Accessing a few of the metadata queries without executing the expression.
    - Caching the execution on the dataframe object to avoid re-execution if
      multiple methods are accessed.

    The dataframe interchange protocol may be found here:
    https://data-apis.org/dataframe-protocol/latest/API.html
    """

    def __init__(self, table: ir.Table, nan_as_null: bool=False, allow_copy: bool=True, pyarrow_table: pa.Table | None=None):
        if False:
            i = 10
            return i + 15
        self._table = table
        self._nan_as_null = nan_as_null
        self._allow_copy = allow_copy
        self._pyarrow_table = pyarrow_table

    @cached_property
    def _pyarrow_df(self):
        if False:
            return 10
        "Returns the pyarrow implementation of the __dataframe__ protocol.\n\n        If the backing ibis Table hasn't been executed yet, this will result\n        in executing and caching the result."
        if self._pyarrow_table is None:
            self._pyarrow_table = self._table.to_pyarrow()
        return self._pyarrow_table.__dataframe__(nan_as_null=self._nan_as_null, allow_copy=self._allow_copy)

    @cached_property
    def _empty_pyarrow_df(self):
        if False:
            while True:
                i = 10
        'A pyarrow implementation of the __dataframe__ protocol for an\n        empty table with the same schema as this table.\n\n        Used for returning dtype information without executing the backing ibis\n        expression.\n        '
        return self._table.schema().to_pyarrow().empty_table().__dataframe__()

    def _get_dtype(self, name):
        if False:
            while True:
                i = 10
        'Get the dtype info for a column named `name`.'
        return self._empty_pyarrow_df.get_column_by_name(name).dtype

    def num_columns(self):
        if False:
            i = 10
            return i + 15
        return len(self._table.columns)

    def column_names(self):
        if False:
            while True:
                i = 10
        return self._table.columns

    def get_column(self, i: int) -> IbisColumn:
        if False:
            for i in range(10):
                print('nop')
        name = self._table.columns[i]
        return self.get_column_by_name(name)

    def get_column_by_name(self, name: str) -> IbisColumn:
        if False:
            print('Hello World!')
        return IbisColumn(self, name)

    def get_columns(self):
        if False:
            i = 10
            return i + 15
        return [IbisColumn(self, name) for name in self._table.columns]

    def select_columns(self, indices: Sequence[int]) -> IbisDataFrame:
        if False:
            return 10
        names = [self._table.columns[i] for i in indices]
        return self.select_columns_by_name(names)

    def select_columns_by_name(self, names: Sequence[str]) -> IbisDataFrame:
        if False:
            i = 10
            return i + 15
        names = list(names)
        table = self._table.select(names)
        if (pyarrow_table := self._pyarrow_table) is not None:
            pyarrow_table = pyarrow_table.select(names)
        return IbisDataFrame(table, nan_as_null=self._nan_as_null, allow_copy=self._allow_copy, pyarrow_table=pyarrow_table)

    def __dataframe__(self, nan_as_null: bool=False, allow_copy: bool=True) -> IbisDataFrame:
        if False:
            print('Hello World!')
        return IbisDataFrame(self._table, nan_as_null=nan_as_null, allow_copy=allow_copy, pyarrow_table=self._pyarrow_table)

    @property
    def metadata(self):
        if False:
            while True:
                i = 10
        return self._pyarrow_df.metadata

    def num_rows(self) -> int | None:
        if False:
            i = 10
            return i + 15
        return self._pyarrow_df.num_rows()

    def num_chunks(self) -> int:
        if False:
            return 10
        return self._pyarrow_df.num_chunks()

    def get_chunks(self, n_chunks: int | None=None):
        if False:
            return 10
        return self._pyarrow_df.get_chunks(n_chunks=n_chunks)

class IbisColumn:

    def __init__(self, df: IbisDataFrame, name: str):
        if False:
            print('Hello World!')
        self._df = df
        self._name = name

    @cached_property
    def _pyarrow_col(self):
        if False:
            print('Hello World!')
        "Returns the pyarrow implementation of the __dataframe__ protocol's\n        Column type.\n\n        If the backing ibis Table hasn't been executed yet, this will result\n        in executing and caching the result."
        return self._df._pyarrow_df.get_column_by_name(self._name)

    @property
    def dtype(self):
        if False:
            i = 10
            return i + 15
        return self._df._get_dtype(self._name)

    @property
    def describe_categorical(self):
        if False:
            i = 10
            return i + 15
        raise TypeError('describe_categorical only works on a column with categorical dtype')

    def size(self):
        if False:
            while True:
                i = 10
        return self._pyarrow_col.size()

    @property
    def offset(self):
        if False:
            print('Hello World!')
        return self._pyarrow_col.offset

    @property
    def describe_null(self):
        if False:
            return 10
        return self._pyarrow_col.describe_null

    @property
    def null_count(self):
        if False:
            print('Hello World!')
        return self._pyarrow_col.null_count

    @property
    def metadata(self):
        if False:
            print('Hello World!')
        return self._pyarrow_col.metadata

    def num_chunks(self) -> int:
        if False:
            i = 10
            return i + 15
        return self._pyarrow_col.num_chunks()

    def get_chunks(self, n_chunks: int | None=None):
        if False:
            return 10
        return self._pyarrow_col.get_chunks(n_chunks=n_chunks)

    def get_buffers(self):
        if False:
            i = 10
            return i + 15
        return self._pyarrow_col.get_buffers()