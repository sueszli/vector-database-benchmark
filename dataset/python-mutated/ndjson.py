from __future__ import annotations
from typing import TYPE_CHECKING
import polars._reexport as pl
from polars.datatypes import N_INFER_DEFAULT
if TYPE_CHECKING:
    from io import IOBase
    from pathlib import Path
    from polars import DataFrame, LazyFrame
    from polars.type_aliases import SchemaDefinition

def read_ndjson(source: str | Path | IOBase | bytes, *, schema: SchemaDefinition | None=None, schema_overrides: SchemaDefinition | None=None, ignore_errors: bool=False) -> DataFrame:
    if False:
        while True:
            i = 10
    '\n    Read into a DataFrame from a newline delimited JSON file.\n\n    Parameters\n    ----------\n    source\n        Path to a file or a file-like object (by file-like object, we refer to objects\n        that have a `read()` method, such as a file handler (e.g. via builtin `open`\n        function) or `BytesIO`).\n    schema : Sequence of str, (str,DataType) pairs, or a {str:DataType,} dict\n        The DataFrame schema may be declared in several ways:\n\n        * As a dict of {name:type} pairs; if type is None, it will be auto-inferred.\n        * As a list of column names; in this case types are automatically inferred.\n        * As a list of (name,type) pairs; this is equivalent to the dictionary form.\n\n        If you supply a list of column names that does not match the names in the\n        underlying data, the names given here will overwrite them. The number\n        of names given in the schema should match the underlying data dimensions.\n    schema_overrides : dict, default None\n        Support type specification or override of one or more columns; note that\n        any dtypes inferred from the schema param will be overridden.\n        underlying data, the names given here will overwrite them.\n    ignore_errors\n        Return `Null` if parsing fails because of schema mismatches.\n\n    '
    return pl.DataFrame._read_ndjson(source, schema=schema, schema_overrides=schema_overrides, ignore_errors=ignore_errors)

def scan_ndjson(source: str | Path | list[str] | list[Path], *, infer_schema_length: int | None=N_INFER_DEFAULT, batch_size: int | None=1024, n_rows: int | None=None, low_memory: bool=False, rechunk: bool=True, row_count_name: str | None=None, row_count_offset: int=0, schema: SchemaDefinition | None=None) -> LazyFrame:
    if False:
        for i in range(10):
            print('nop')
    '\n    Lazily read from a newline delimited JSON file or multiple files via glob patterns.\n\n    This allows the query optimizer to push down predicates and projections to the scan\n    level, thereby potentially reducing memory overhead.\n\n    Parameters\n    ----------\n    source\n        Path to a file.\n    infer_schema_length\n        Infer the schema from the first `infer_schema_length` rows.\n    batch_size\n        Number of rows to read in each batch.\n    n_rows\n        Stop reading from JSON file after reading `n_rows`.\n    low_memory\n        Reduce memory pressure at the expense of performance.\n    rechunk\n        Reallocate to contiguous memory when all chunks/ files are parsed.\n    row_count_name\n        If not None, this will insert a row count column with give name into the\n        DataFrame\n    row_count_offset\n        Offset to start the row_count column (only use if the name is set)\n    schema : Sequence of str, (str,DataType) pairs, or a {str:DataType,} dict\n        The DataFrame schema may be declared in several ways:\n\n        * As a dict of {name:type} pairs; if type is None, it will be auto-inferred.\n        * As a list of column names; in this case types are automatically inferred.\n        * As a list of (name,type) pairs; this is equivalent to the dictionary form.\n\n        If you supply a list of column names that does not match the names in the\n        underlying data, the names given here will overwrite them. The number\n        of names given in the schema should match the underlying data dimensions.\n\n    '
    return pl.LazyFrame._scan_ndjson(source, infer_schema_length=infer_schema_length, schema=schema, batch_size=batch_size, n_rows=n_rows, low_memory=low_memory, rechunk=rechunk, row_count_name=row_count_name, row_count_offset=row_count_offset)