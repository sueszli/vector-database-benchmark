from __future__ import annotations
from functools import partial
from typing import TYPE_CHECKING
import polars._reexport as pl
from polars.dependencies import pyarrow as pa
if TYPE_CHECKING:
    from polars import DataFrame, LazyFrame

def _scan_pyarrow_dataset(ds: pa.dataset.Dataset, *, allow_pyarrow_filter: bool=True, batch_size: int | None=None) -> LazyFrame:
    if False:
        i = 10
        return i + 15
    '\n    Pickle the partially applied function `_scan_pyarrow_dataset_impl`.\n\n    The bytes are then sent to the polars logical plan. It can be deserialized once\n    executed and ran.\n\n    Parameters\n    ----------\n    ds\n        pyarrow dataset\n    allow_pyarrow_filter\n        Allow predicates to be pushed down to pyarrow. This can lead to different\n        results if comparisons are done with null values as pyarrow handles this\n        different than polars does.\n    batch_size\n        The maximum row count for scanned pyarrow record batches.\n\n    '
    func = partial(_scan_pyarrow_dataset_impl, ds, batch_size=batch_size)
    return pl.LazyFrame._scan_python_function(ds.schema, func, pyarrow=allow_pyarrow_filter)

def _scan_pyarrow_dataset_impl(ds: pa.dataset.Dataset, with_columns: list[str] | None, predicate: str | None, n_rows: int | None, batch_size: int | None) -> DataFrame:
    if False:
        print('Hello World!')
    '\n    Take the projected columns and materialize an arrow table.\n\n    Parameters\n    ----------\n    ds\n        pyarrow dataset\n    with_columns\n        Columns that are projected\n    predicate\n        pyarrow expression that can be evaluated with eval\n    n_rows:\n        Materialize only n rows from the arrow dataset\n    batch_size\n        The maximum row count for scanned pyarrow record batches.\n\n    Returns\n    -------\n    DataFrame\n\n    '
    from polars import from_arrow
    _filter = None
    if predicate:
        from polars.datatypes import Date, Datetime, Duration
        from polars.utils.convert import _to_python_date, _to_python_datetime, _to_python_time, _to_python_timedelta
        _filter = eval(predicate, {'pa': pa, 'Date': Date, 'Datetime': Datetime, 'Duration': Duration, '_to_python_date': _to_python_date, '_to_python_datetime': _to_python_datetime, '_to_python_time': _to_python_time, '_to_python_timedelta': _to_python_timedelta})
    common_params = {'columns': with_columns, 'filter': _filter}
    if batch_size is not None:
        common_params['batch_size'] = batch_size
    if n_rows:
        return from_arrow(ds.head(n_rows, **common_params))
    return from_arrow(ds.to_table(**common_params))