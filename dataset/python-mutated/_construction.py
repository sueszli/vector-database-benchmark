from __future__ import annotations
import contextlib
import warnings
from datetime import date, datetime, time, timedelta
from decimal import Decimal as PyDecimal
from functools import lru_cache, singledispatch
from itertools import islice, zip_longest
from operator import itemgetter
from sys import version_info
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable, Iterator, Mapping, MutableMapping, Sequence, get_type_hints
import polars._reexport as pl
from polars import functions as F
from polars.datatypes import INTEGER_DTYPES, N_INFER_DEFAULT, TEMPORAL_DTYPES, Boolean, Categorical, Date, Datetime, Duration, Float32, List, Object, Struct, Time, UInt32, Unknown, Utf8, dtype_to_py_type, is_polars_dtype, numpy_char_code_to_dtype, py_type_to_dtype
from polars.datatypes.constructor import numpy_type_to_constructor, numpy_values_and_dtype, polars_type_to_constructor, py_type_to_constructor
from polars.dependencies import _NUMPY_AVAILABLE, _check_for_numpy, _check_for_pandas, _check_for_pydantic, dataclasses, pydantic
from polars.dependencies import numpy as np
from polars.dependencies import pandas as pd
from polars.dependencies import pyarrow as pa
from polars.exceptions import ComputeError, ShapeError, TimeZoneAwareConstructorWarning
from polars.utils._wrap import wrap_df, wrap_s
from polars.utils.meta import get_index_type, threadpool_size
from polars.utils.various import _is_generator, arrlen, find_stacklevel, parse_version, range_to_series
with contextlib.suppress(ImportError):
    from polars.polars import PyDataFrame, PySeries
if TYPE_CHECKING:
    from polars import DataFrame, Series
    from polars.type_aliases import Orientation, PolarsDataType, SchemaDefinition, SchemaDict

def _get_annotations(obj: type) -> dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    return getattr(obj, '__annotations__', {})
if version_info >= (3, 10):

    def type_hints(obj: type) -> dict[str, Any]:
        if False:
            print('Hello World!')
        try:
            return get_type_hints(obj)
        except TypeError:
            return _get_annotations(obj)
else:
    type_hints = _get_annotations

@lru_cache(64)
def is_namedtuple(cls: Any, *, annotated: bool=False) -> bool:
    if False:
        return 10
    'Check whether given class derives from NamedTuple.'
    if all((hasattr(cls, attr) for attr in ('_fields', '_field_defaults', '_replace'))):
        if not isinstance(cls._fields, property):
            if not annotated or len(cls.__annotations__) == len(cls._fields):
                return all((isinstance(fld, str) for fld in cls._fields))
    return False

def is_pydantic_model(value: Any) -> bool:
    if False:
        return 10
    'Check whether value derives from a pydantic.BaseModel.'
    return _check_for_pydantic(value) and isinstance(value, pydantic.BaseModel)

def contains_nested(value: Any, is_nested: Callable[[Any], bool]) -> bool:
    if False:
        while True:
            i = 10
    'Determine if value contains (or is) nested structured data.'
    if is_nested(value):
        return True
    elif isinstance(value, dict):
        return any((contains_nested(v, is_nested) for v in value.values()))
    elif isinstance(value, (list, tuple)):
        return any((contains_nested(v, is_nested) for v in value))
    return False

def include_unknowns(schema: SchemaDict, cols: Sequence[str]) -> MutableMapping[str, PolarsDataType]:
    if False:
        for i in range(10):
            print('nop')
    'Complete partial schema dict by including Unknown type.'
    return {col: schema.get(col, Unknown) or Unknown for col in cols}

def nt_unpack(obj: Any) -> Any:
    if False:
        for i in range(10):
            print('nop')
    'Recursively unpack a nested NamedTuple.'
    if isinstance(obj, dict):
        return {key: nt_unpack(value) for (key, value) in obj.items()}
    elif isinstance(obj, list):
        return [nt_unpack(value) for value in obj]
    elif is_namedtuple(obj.__class__):
        return {key: nt_unpack(value) for (key, value) in obj._asdict().items()}
    elif isinstance(obj, tuple):
        return tuple((nt_unpack(value) for value in obj))
    else:
        return obj

def series_to_pyseries(name: str, values: Series) -> PySeries:
    if False:
        return 10
    'Construct a new PySeries from a Polars Series.'
    py_s = values._s.clone()
    py_s.rename(name)
    return py_s

def arrow_to_pyseries(name: str, values: pa.Array, *, rechunk: bool=True) -> PySeries:
    if False:
        while True:
            i = 10
    'Construct a PySeries from an Arrow array.'
    array = coerce_arrow(values)
    if len(array) == 0 and isinstance(array.type, pa.DictionaryType) and (array.type.value_type in (pa.utf8(), pa.large_utf8())):
        pys = pl.Series(name, [], dtype=Categorical)._s
    elif not hasattr(array, 'num_chunks'):
        pys = PySeries.from_arrow(name, array)
    else:
        if array.num_chunks > 1:
            if isinstance(array.type, pa.StructType):
                pys = PySeries.from_arrow(name, array.combine_chunks())
            else:
                it = array.iterchunks()
                pys = PySeries.from_arrow(name, next(it))
                for a in it:
                    pys.append(PySeries.from_arrow(name, a))
        elif array.num_chunks == 0:
            pys = PySeries.from_arrow(name, pa.array([], array.type))
        else:
            pys = PySeries.from_arrow(name, array.chunks[0])
        if rechunk:
            pys.rechunk(in_place=True)
    return pys

def numpy_to_pyseries(name: str, values: np.ndarray[Any, Any], *, strict: bool=True, nan_to_null: bool=False) -> PySeries:
    if False:
        for i in range(10):
            print('nop')
    'Construct a PySeries from a numpy array.'
    if not values.flags['C_CONTIGUOUS']:
        values = np.array(values)
    if len(values.shape) == 1:
        (values, dtype) = numpy_values_and_dtype(values)
        constructor = numpy_type_to_constructor(dtype)
        return constructor(name, values, nan_to_null if dtype in (np.float32, np.float64) else strict)
    elif len(values.shape) == 2:
        pyseries_container = []
        for row in range(values.shape[0]):
            pyseries_container.append(numpy_to_pyseries('', values[row, :], strict=strict, nan_to_null=nan_to_null))
        return PySeries.new_series_list(name, pyseries_container, _strict=False)
    else:
        return PySeries.new_object(name, values, strict)

def _get_first_non_none(values: Sequence[Any | None]) -> Any:
    if False:
        for i in range(10):
            print('nop')
    "\n    Return the first value from a sequence that isn't None.\n\n    If sequence doesn't contain non-None values, return None.\n\n    "
    if values is not None:
        return next((v for v in values if v is not None), None)

def sequence_from_anyvalue_or_object(name: str, values: Sequence[Any]) -> PySeries:
    if False:
        while True:
            i = 10
    '\n    Last resort conversion.\n\n    AnyValues are most flexible and if they fail we go for object types\n\n    '
    try:
        return PySeries.new_from_anyvalues(name, values, strict=True)
    except RuntimeError:
        return PySeries.new_object(name, values, _strict=False)
    except ComputeError as exc:
        if 'mixed dtypes' in str(exc):
            return PySeries.new_object(name, values, _strict=False)
        raise

def sequence_from_anyvalue_and_dtype_or_object(name: str, values: Sequence[Any], dtype: PolarsDataType) -> PySeries:
    if False:
        for i in range(10):
            print('nop')
    '\n    Last resort conversion.\n\n    AnyValues are most flexible and if they fail we go for object types\n\n    '
    try:
        return PySeries.new_from_anyvalues_and_dtype(name, values, dtype, strict=True)
    except RuntimeError:
        return PySeries.new_object(name, values, _strict=False)
    except ComputeError as exc:
        if 'mixed dtypes' in str(exc):
            return PySeries.new_object(name, values, _strict=False)
        raise

def iterable_to_pyseries(name: str, values: Iterable[Any], dtype: PolarsDataType | None=None, *, dtype_if_empty: PolarsDataType | None=None, chunk_size: int=1000000, strict: bool=True) -> PySeries:
    if False:
        i = 10
        return i + 15
    'Construct a PySeries from an iterable/generator.'
    if not isinstance(values, (Generator, Iterator)):
        values = iter(values)

    def to_series_chunk(values: list[Any], dtype: PolarsDataType | None) -> Series:
        if False:
            while True:
                i = 10
        return pl.Series(name=name, values=values, dtype=dtype, strict=strict, dtype_if_empty=dtype_if_empty)
    n_chunks = 0
    series: Series = None
    while True:
        slice_values = list(islice(values, chunk_size))
        if not slice_values:
            break
        schunk = to_series_chunk(slice_values, dtype)
        if series is None:
            series = schunk
            dtype = series.dtype
        else:
            series.append(schunk)
            n_chunks += 1
    if series is None:
        series = to_series_chunk([], dtype)
    if n_chunks > 0:
        series.rechunk(in_place=True)
    return series._s

def _construct_series_with_fallbacks(constructor: Callable[[str, Sequence[Any], bool], PySeries], name: str, values: Sequence[Any], target_dtype: PolarsDataType | None, *, strict: bool) -> PySeries:
    if False:
        for i in range(10):
            print('nop')
    'Construct Series, with fallbacks for basic type mismatch (eg: bool/int).'
    while True:
        try:
            return constructor(name, values, strict)
        except TypeError as exc:
            str_exc = str(exc)
            if "'float'" in str_exc and target_dtype not in INTEGER_DTYPES | TEMPORAL_DTYPES:
                constructor = py_type_to_constructor(float)
            elif "'str'" in str_exc or str_exc == 'must be real number, not str':
                constructor = py_type_to_constructor(str)
            elif str_exc == "'int' object cannot be converted to 'PyBool'":
                constructor = py_type_to_constructor(int)
            elif 'decimal.Decimal' in str_exc:
                constructor = py_type_to_constructor(PyDecimal)
            else:
                raise

def sequence_to_pyseries(name: str, values: Sequence[Any], dtype: PolarsDataType | None=None, *, dtype_if_empty: PolarsDataType | None=None, strict: bool=True, nan_to_null: bool=False) -> PySeries:
    if False:
        return 10
    'Construct a PySeries from a sequence.'
    python_dtype: type | None = None
    if not values and dtype is None:
        dtype = dtype_if_empty or Float32
    elif dtype == List:
        getattr(dtype, 'inner', None)
        python_dtype = list
    py_temporal_types = {date, datetime, timedelta, time}
    pl_temporal_types = {Date, Datetime, Duration, Time}
    value = _get_first_non_none(values)
    if value is not None:
        if dataclasses.is_dataclass(value) or is_pydantic_model(value) or is_namedtuple(value.__class__):
            return pl.DataFrame(values).to_struct(name)._s
        elif isinstance(value, range):
            values = [range_to_series('', v) for v in values]
        elif dtype in py_temporal_types and isinstance(value, int):
            dtype = py_type_to_dtype(dtype)
        elif (dtype in pl_temporal_types or type(dtype) in pl_temporal_types) and (not isinstance(value, int)):
            python_dtype = dtype_to_py_type(dtype)
    if dtype is not None and dtype not in (List, Struct, Unknown) and is_polars_dtype(dtype) and (python_dtype is None):
        constructor = polars_type_to_constructor(dtype)
        pyseries = _construct_series_with_fallbacks(constructor, name, values, dtype, strict=strict)
        if dtype in (Date, Datetime, Duration, Time, Categorical, Boolean):
            if pyseries.dtype() != dtype:
                pyseries = pyseries.cast(dtype, strict=True)
        return pyseries
    elif dtype == Struct:
        struct_schema = dtype.to_schema() if isinstance(dtype, Struct) else None
        empty = {}
        return sequence_to_pydf(data=[empty if v is None else v for v in values], schema=struct_schema, orient='row').to_struct(name)
    else:
        if python_dtype is None:
            if value is None:
                constructor = polars_type_to_constructor(dtype_if_empty if dtype_if_empty else Float32)
                return _construct_series_with_fallbacks(constructor, name, values, dtype, strict=strict)
            python_dtype = type(value)
        if python_dtype in py_temporal_types:
            if dtype is None:
                dtype = py_type_to_dtype(python_dtype)
            elif dtype in py_temporal_types:
                dtype = py_type_to_dtype(dtype)
            values_dtype = None if value is None else py_type_to_dtype(type(value), raise_unmatched=False)
            if values_dtype is not None and values_dtype.is_float():
                raise TypeError(f"'float' object cannot be interpreted as a {python_dtype.__name__!r}")
            py_series = PySeries.new_from_anyvalues(name, values, strict)
            time_unit = getattr(dtype, 'time_unit', None)
            if time_unit is None or values_dtype == Date:
                s = wrap_s(py_series)
            else:
                s = wrap_s(py_series).dt.cast_time_unit(time_unit)
            time_zone = getattr(dtype, 'time_zone', None)
            if (values_dtype == Date) & (dtype == Datetime):
                return s.cast(Datetime(time_unit)).dt.replace_time_zone(time_zone)._s
            if dtype == Datetime and (value.tzinfo is not None or time_zone is not None):
                values_tz = str(value.tzinfo) if value.tzinfo is not None else None
                dtype_tz = dtype.time_zone
                if values_tz is not None and (dtype_tz is not None and dtype_tz != 'UTC'):
                    raise ValueError("time-zone-aware datetimes are converted to UTC\n\nPlease either drop the time zone from the dtype, or set it to 'UTC'. To convert to a different time zone, please use `.dt.convert_time_zone`.")
                if values_tz != 'UTC' and dtype_tz is None:
                    warnings.warn("Constructing a Series with time-zone-aware datetimes results in a Series with UTC time zone. To silence this warning, you can filter warnings of class TimeZoneAwareConstructorWarning, or set 'UTC' as the time zone of your datatype.", TimeZoneAwareConstructorWarning, stacklevel=find_stacklevel())
                return s.dt.replace_time_zone(dtype_tz or 'UTC')._s
            return s._s
        elif _check_for_numpy(value) and isinstance(value, np.ndarray) and (len(value.shape) == 1):
            return PySeries.new_series_list(name, [numpy_to_pyseries('', v, strict=strict, nan_to_null=nan_to_null) for v in values], strict)
        elif python_dtype in (list, tuple):
            if isinstance(dtype, Object):
                return PySeries.new_object(name, values, strict)
            if dtype:
                srs = sequence_from_anyvalue_and_dtype_or_object(name, values, dtype)
                if not dtype.is_(srs.dtype()):
                    srs = srs.cast(dtype, strict=False)
                return srs
            return sequence_from_anyvalue_or_object(name, values)
        elif python_dtype == pl.Series:
            return PySeries.new_series_list(name, [v._s for v in values], strict)
        elif python_dtype == PySeries:
            return PySeries.new_series_list(name, values, strict)
        else:
            constructor = py_type_to_constructor(python_dtype)
            if constructor == PySeries.new_object:
                try:
                    srs = PySeries.new_from_anyvalues(name, values, strict)
                    if _check_for_numpy(python_dtype, check_type=False) and isinstance(np.bool_(True), np.generic):
                        dtype = numpy_char_code_to_dtype(np.dtype(python_dtype).char)
                        return srs.cast(dtype, strict=strict)
                    else:
                        return srs
                except RuntimeError:
                    return sequence_from_anyvalue_or_object(name, values)
            return _construct_series_with_fallbacks(constructor, name, values, dtype, strict=strict)

def _pandas_series_to_arrow(values: pd.Series[Any] | pd.Index[Any], *, length: int | None=None, nan_to_null: bool=True) -> pa.Array:
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert a pandas Series to an Arrow Array.\n\n    Parameters\n    ----------\n    values : :class:`pandas.Series` or :class:`pandas.Index`.\n        Series to convert to arrow\n    nan_to_null : bool, default = True\n        Interpret `NaN` as missing values.\n    length : int, optional\n        in case all values are null, create a null array of this length.\n        if unset, length is inferred from values.\n\n    Returns\n    -------\n    :class:`pyarrow.Array`\n\n    '
    dtype = getattr(values, 'dtype', None)
    if dtype == 'object':
        first_non_none = _get_first_non_none(values.values)
        if isinstance(first_non_none, str):
            return pa.array(values, pa.large_utf8(), from_pandas=nan_to_null)
        elif first_non_none is None:
            return pa.nulls(length or len(values), pa.large_utf8())
        return pa.array(values, from_pandas=nan_to_null)
    elif dtype:
        return pa.array(values, from_pandas=nan_to_null)
    else:
        raise ValueError('duplicate column names found: ', f'{values.columns.tolist()!s}')

def pandas_to_pyseries(name: str, values: pd.Series[Any] | pd.DatetimeIndex, *, nan_to_null: bool=True) -> PySeries:
    if False:
        for i in range(10):
            print('nop')
    'Construct a PySeries from a pandas Series or DatetimeIndex.'
    if not name and values.name is not None:
        name = str(values.name)
    return arrow_to_pyseries(name, _pandas_series_to_arrow(values, nan_to_null=nan_to_null))

def _handle_columns_arg(data: list[PySeries], columns: Sequence[str] | None=None, *, from_dict: bool=False) -> list[PySeries]:
    if False:
        print('Hello World!')
    'Rename data according to columns argument.'
    if not columns:
        return data
    elif not data:
        return [pl.Series(c, None)._s for c in columns]
    elif len(data) == len(columns):
        if from_dict:
            series_map = {s.name(): s for s in data}
            if all((col in series_map for col in columns)):
                return [series_map[col] for col in columns]
        for (i, c) in enumerate(columns):
            if c != data[i].name():
                data[i] = data[i].clone()
                data[i].rename(c)
        return data
    else:
        raise ValueError(f'dimensions of columns arg ({len(columns)}) must match data dimensions ({len(data)})')

def _post_apply_columns(pydf: PyDataFrame, columns: SchemaDefinition | None, structs: dict[str, Struct] | None=None, schema_overrides: SchemaDict | None=None) -> PyDataFrame:
    if False:
        for i in range(10):
            print('nop')
    "Apply 'columns' param *after* PyDataFrame creation (if no alternative)."
    (pydf_columns, pydf_dtypes) = (pydf.columns(), pydf.dtypes())
    (columns, dtypes) = _unpack_schema(columns or pydf_columns, schema_overrides=schema_overrides)
    column_subset: list[str] = []
    if columns != pydf_columns:
        if len(columns) < len(pydf_columns) and columns == pydf_columns[:len(columns)]:
            column_subset = columns
        else:
            pydf.set_column_names(columns)
    column_casts = []
    for (i, col) in enumerate(columns):
        dtype = dtypes.get(col)
        pydf_dtype = pydf_dtypes[i]
        if dtype == Categorical != pydf_dtype:
            column_casts.append(F.col(col).cast(Categorical)._pyexpr)
        elif structs and (struct := structs.get(col)) and (struct != pydf_dtype):
            column_casts.append(F.col(col).cast(struct)._pyexpr)
        elif dtype is not None and dtype != Unknown and (dtype != pydf_dtype):
            column_casts.append(F.col(col).cast(dtype)._pyexpr)
    if column_casts or column_subset:
        pydf = pydf.lazy()
        if column_casts:
            pydf = pydf.with_columns(column_casts)
        if column_subset:
            pydf = pydf.select([F.col(col)._pyexpr for col in column_subset])
        pydf = pydf.collect()
    return pydf

def _unpack_schema(schema: SchemaDefinition | None, *, schema_overrides: SchemaDict | None=None, n_expected: int | None=None, lookup_names: Iterable[str] | None=None, include_overrides_in_columns: bool=False) -> tuple[list[str], SchemaDict]:
    if False:
        while True:
            i = 10
    '\n    Unpack column names and create dtype lookup.\n\n    Works for any (name, dtype) pairs or schema dict input,\n    overriding any inferred dtypes with explicit dtypes if supplied.\n    '
    if schema_overrides:
        schema_overrides = {name: dtype if is_polars_dtype(dtype, include_unknown=True) else py_type_to_dtype(dtype) for (name, dtype) in schema_overrides.items()}
    else:
        schema_overrides = {}
    if not schema:
        return ([f'column_{i}' for i in range(n_expected)] if n_expected else [], schema_overrides)
    if isinstance(schema, Mapping):
        column_names: list[str] = list(schema)
        schema = list(schema.items())
    else:
        column_names = [col or f'column_{i}' if isinstance(col, str) else col[0] for (i, col) in enumerate(schema)]
    lookup: dict[str, str] | None = {col: name for (col, name) in zip_longest(column_names, lookup_names) if name} if lookup_names else None
    column_dtypes: dict[str, PolarsDataType] = {lookup.get((name := col[0]), name) if lookup else col[0]: dtype if is_polars_dtype(dtype, include_unknown=True) else py_type_to_dtype(dtype) for col in schema if isinstance(col, tuple) and (dtype := col[1]) is not None}
    if schema_overrides:
        column_dtypes.update(schema_overrides)
        if include_overrides_in_columns:
            column_names.extend((col for col in column_dtypes if col not in column_names))
    return (column_names, column_dtypes)

def _expand_dict_data(data: Mapping[str, Sequence[object] | Mapping[str, Sequence[object]] | Series], dtypes: SchemaDict) -> Mapping[str, Sequence[object] | Mapping[str, Sequence[object]] | Series]:
    if False:
        print('Hello World!')
    '\n    Expand any unsized generators/iterators.\n\n    (Note that `range` is sized, and will take a fast-path on Series init).\n    '
    expanded_data = {}
    for (name, val) in data.items():
        expanded_data[name] = pl.Series(name, val, dtypes.get(name)) if _is_generator(val) else val
    return expanded_data

def _expand_dict_scalars(data: Mapping[str, Sequence[object] | Mapping[str, Sequence[object]] | Series], *, schema_overrides: SchemaDict | None=None, order: Sequence[str] | None=None, nan_to_null: bool=False) -> dict[str, Series]:
    if False:
        print('Hello World!')
    'Expand any scalar values in dict data (propagate literal as array).'
    updated_data = {}
    if data:
        dtypes = schema_overrides or {}
        data = _expand_dict_data(data, dtypes)
        array_len = max((arrlen(val) or 0 for val in data.values()))
        if array_len > 0:
            for (name, val) in data.items():
                dtype = dtypes.get(name)
                if isinstance(val, dict) and dtype != Struct:
                    updated_data[name] = pl.DataFrame(val).to_struct(name)
                elif isinstance(val, pl.Series):
                    s = val.rename(name) if name != val.name else val
                    if dtype and dtype != s.dtype:
                        s = s.cast(dtype)
                    updated_data[name] = s
                elif arrlen(val) is not None or _is_generator(val):
                    updated_data[name] = pl.Series(name=name, values=val, dtype=dtype, nan_to_null=nan_to_null)
                elif val is None or isinstance(val, (int, float, str, bool, date, datetime, time, timedelta)):
                    updated_data[name] = pl.Series(name=name, values=[val], dtype=dtype).extend_constant(val, array_len - 1)
                else:
                    updated_data[name] = pl.Series(name=name, values=[val] * array_len, dtype=dtype)
        elif all((arrlen(val) == 0 for val in data.values())):
            for (name, val) in data.items():
                updated_data[name] = pl.Series(name, values=val, dtype=dtypes.get(name))
        elif all((arrlen(val) is None for val in data.values())):
            for (name, val) in data.items():
                updated_data[name] = pl.Series(name, values=val if _is_generator(val) else [val], dtype=dtypes.get(name))
    if order and list(updated_data) != order:
        return {col: updated_data.pop(col) for col in order}
    return updated_data

def dict_to_pydf(data: Mapping[str, Sequence[object] | Mapping[str, Sequence[object]] | Series], schema: SchemaDefinition | None=None, *, schema_overrides: SchemaDict | None=None, nan_to_null: bool=False) -> PyDataFrame:
    if False:
        i = 10
        return i + 15
    'Construct a PyDataFrame from a dictionary of sequences.'
    if isinstance(schema, Mapping) and data:
        if not all((col in schema for col in data)):
            raise ValueError('the given column-schema names do not match the data dictionary')
        data = {col: data[col] for col in schema}
    (column_names, schema_overrides) = _unpack_schema(schema, lookup_names=data.keys(), schema_overrides=schema_overrides)
    if not column_names:
        column_names = list(data)
    if data and _NUMPY_AVAILABLE:
        count_numpy = sum((int(_check_for_numpy(val) and isinstance(val, np.ndarray) and (len(val) > 1000)) for val in data.values()))
        if count_numpy >= 3:
            import multiprocessing.dummy
            pool_size = threadpool_size()
            with multiprocessing.dummy.Pool(pool_size) as pool:
                data = dict(zip(column_names, pool.map(lambda t: pl.Series(t[0], t[1]) if isinstance(t[1], np.ndarray) else t[1], list(data.items()))))
    if not data and schema_overrides:
        data_series = [pl.Series(name, [], dtype=schema_overrides.get(name), nan_to_null=nan_to_null)._s for name in column_names]
    else:
        data_series = [s._s for s in _expand_dict_scalars(data, schema_overrides=schema_overrides, nan_to_null=nan_to_null).values()]
    data_series = _handle_columns_arg(data_series, columns=column_names, from_dict=True)
    pydf = PyDataFrame(data_series)
    if schema_overrides and pydf.dtypes() != list(schema_overrides.values()):
        pydf = _post_apply_columns(pydf, column_names, schema_overrides=schema_overrides)
    return pydf

def sequence_to_pydf(data: Sequence[Any], schema: SchemaDefinition | None=None, schema_overrides: SchemaDict | None=None, orient: Orientation | None=None, infer_schema_length: int | None=N_INFER_DEFAULT) -> PyDataFrame:
    if False:
        i = 10
        return i + 15
    'Construct a PyDataFrame from a sequence.'
    if len(data) == 0:
        return dict_to_pydf({}, schema=schema, schema_overrides=schema_overrides)
    return _sequence_to_pydf_dispatcher(data[0], data=data, schema=schema, schema_overrides=schema_overrides, orient=orient, infer_schema_length=infer_schema_length)

def _sequence_of_series_to_pydf(first_element: Series, data: Sequence[Any], schema: SchemaDefinition | None, schema_overrides: SchemaDict | None, **kwargs: Any) -> PyDataFrame:
    if False:
        for i in range(10):
            print('nop')
    series_names = [s.name for s in data]
    (column_names, schema_overrides) = _unpack_schema(schema or series_names, schema_overrides=schema_overrides, n_expected=len(data))
    data_series: list[PySeries] = []
    for (i, s) in enumerate(data):
        if not s.name:
            s = s.alias(column_names[i])
        new_dtype = schema_overrides.get(column_names[i])
        if new_dtype and new_dtype != s.dtype:
            s = s.cast(new_dtype)
        data_series.append(s._s)
    data_series = _handle_columns_arg(data_series, columns=column_names)
    return PyDataFrame(data_series)

@singledispatch
def _sequence_to_pydf_dispatcher(first_element: Any, data: Sequence[Any], schema: SchemaDefinition | None, schema_overrides: SchemaDict | None, orient: Orientation | None, infer_schema_length: int | None) -> PyDataFrame:
    if False:
        for i in range(10):
            print('nop')
    common_params = {'data': data, 'schema': schema, 'schema_overrides': schema_overrides, 'orient': orient, 'infer_schema_length': infer_schema_length}
    to_pydf: Callable[..., PyDataFrame]
    register_with_singledispatch = True
    if isinstance(first_element, Generator):
        to_pydf = _sequence_of_sequence_to_pydf
        data = [list(row) for row in data]
        first_element = data[0]
        register_with_singledispatch = False
    elif isinstance(first_element, pl.Series):
        to_pydf = _sequence_of_series_to_pydf
    elif _check_for_numpy(first_element) and isinstance(first_element, np.ndarray):
        to_pydf = _sequence_of_numpy_to_pydf
    elif _check_for_pandas(first_element) and isinstance(first_element, (pd.Series, pd.DatetimeIndex)):
        to_pydf = _sequence_of_pandas_to_pydf
    elif dataclasses.is_dataclass(first_element):
        to_pydf = _dataclasses_to_pydf
    elif is_pydantic_model(first_element):
        to_pydf = _pydantic_models_to_pydf
    else:
        to_pydf = _sequence_of_elements_to_pydf
    if register_with_singledispatch:
        _sequence_to_pydf_dispatcher.register(type(first_element), to_pydf)
    common_params['first_element'] = first_element
    return to_pydf(**common_params)

@_sequence_to_pydf_dispatcher.register(list)
def _sequence_of_sequence_to_pydf(first_element: Sequence[Any] | np.ndarray[Any, Any], data: Sequence[Any], schema: SchemaDefinition | None, schema_overrides: SchemaDict | None, orient: Orientation | None, infer_schema_length: int | None) -> PyDataFrame:
    if False:
        return 10
    if orient is None:
        if len(first_element) > 1000:
            orient = 'col' if schema and len(schema) == len(data) else 'row'
        elif schema is not None and len(schema) == len(data) or not schema:
            row_types = {type(value) for value in first_element if value is not None}
            if int in row_types and float in row_types:
                row_types.discard(int)
            orient = 'col' if len(row_types) == 1 else 'row'
        else:
            orient = 'row'
    if orient == 'row':
        (column_names, schema_overrides) = _unpack_schema(schema, schema_overrides=schema_overrides, n_expected=len(first_element))
        local_schema_override = include_unknowns(schema_overrides, column_names) if schema_overrides else {}
        if column_names and len(first_element) > 0 and (len(first_element) != len(column_names)):
            raise ShapeError('the row data does not match the number of columns')
        unpack_nested = False
        for (col, tp) in local_schema_override.items():
            if tp == Categorical:
                local_schema_override[col] = Utf8
            elif not unpack_nested and tp.base_type() in (Unknown, Struct):
                unpack_nested = contains_nested(getattr(first_element, col, None).__class__, is_namedtuple)
        if unpack_nested:
            dicts = [nt_unpack(d) for d in data]
            pydf = PyDataFrame.read_dicts(dicts, infer_schema_length)
        else:
            pydf = PyDataFrame.read_rows(data, infer_schema_length, local_schema_override or None)
        if column_names or schema_overrides:
            pydf = _post_apply_columns(pydf, column_names, schema_overrides=schema_overrides)
        return pydf
    if orient == 'col' or orient is None:
        (column_names, schema_overrides) = _unpack_schema(schema, schema_overrides=schema_overrides, n_expected=len(data))
        data_series: list[PySeries] = [pl.Series(column_names[i], element, schema_overrides.get(column_names[i]))._s for (i, element) in enumerate(data)]
        return PyDataFrame(data_series)
    raise ValueError(f"`orient` must be one of {{'col', 'row', None}}, got {orient!r}")

@_sequence_to_pydf_dispatcher.register(tuple)
def _sequence_of_tuple_to_pydf(first_element: tuple[Any, ...], data: Sequence[Any], schema: SchemaDefinition | None, schema_overrides: SchemaDict | None, orient: Orientation | None, infer_schema_length: int | None) -> PyDataFrame:
    if False:
        while True:
            i = 10
    if is_namedtuple(first_element.__class__):
        if schema is None:
            schema = first_element._fields
            annotations = getattr(first_element, '__annotations__', None)
            if annotations and len(annotations) == len(schema):
                schema = [(name, py_type_to_dtype(tp, raise_unmatched=False)) for (name, tp) in first_element.__annotations__.items()]
        if orient is None:
            orient = 'row'
    return _sequence_of_sequence_to_pydf(first_element, data=data, schema=schema, schema_overrides=schema_overrides, orient=orient, infer_schema_length=infer_schema_length)

@_sequence_to_pydf_dispatcher.register(dict)
def _sequence_of_dict_to_pydf(first_element: Any, data: Sequence[Any], schema: SchemaDefinition | None, schema_overrides: SchemaDict | None, infer_schema_length: int | None, **kwargs: Any) -> PyDataFrame:
    if False:
        return 10
    (column_names, schema_overrides) = _unpack_schema(schema, schema_overrides=schema_overrides)
    dicts_schema = include_unknowns(schema_overrides, column_names or list(schema_overrides)) if column_names else None
    pydf = PyDataFrame.read_dicts(data, infer_schema_length, dicts_schema, schema_overrides)
    if schema_overrides:
        pydf = _post_apply_columns(pydf, columns=column_names, schema_overrides=schema_overrides)
    return pydf

@_sequence_to_pydf_dispatcher.register(str)
def _sequence_of_elements_to_pydf(first_element: Any, data: Sequence[Any], schema: SchemaDefinition | None, schema_overrides: SchemaDict | None, **kwargs: Any) -> PyDataFrame:
    if False:
        print('Hello World!')
    (column_names, schema_overrides) = _unpack_schema(schema, schema_overrides=schema_overrides, n_expected=1)
    data_series: list[PySeries] = [pl.Series(column_names[0], data, schema_overrides.get(column_names[0]))._s]
    data_series = _handle_columns_arg(data_series, columns=column_names)
    return PyDataFrame(data_series)

def _sequence_of_numpy_to_pydf(first_element: np.ndarray[Any, Any], **kwargs: Any) -> PyDataFrame:
    if False:
        return 10
    to_pydf = _sequence_of_sequence_to_pydf if first_element.ndim == 1 else _sequence_of_elements_to_pydf
    return to_pydf(first_element, **kwargs)

def _sequence_of_pandas_to_pydf(first_element: pd.Series[Any] | pd.DatetimeIndex, data: Sequence[Any], schema: SchemaDefinition | None, schema_overrides: SchemaDict | None, **kwargs: Any) -> PyDataFrame:
    if False:
        return 10
    if schema is None:
        column_names: list[str] = []
    else:
        (column_names, schema_overrides) = _unpack_schema(schema, schema_overrides=schema_overrides, n_expected=1)
    schema_overrides = schema_overrides or {}
    data_series: list[PySeries] = []
    for (i, s) in enumerate(data):
        name = column_names[i] if column_names else s.name
        dtype = schema_overrides.get(name, None)
        pyseries = pandas_to_pyseries(name=name, values=s)
        if dtype is not None and dtype != pyseries.dtype():
            pyseries = pyseries.cast(dtype, strict=True)
        data_series.append(pyseries)
    return PyDataFrame(data_series)

def _establish_dataclass_or_model_schema(first_element: Any, schema: SchemaDefinition | None, schema_overrides: SchemaDict | None, model_fields: list[str] | None) -> tuple[bool, list[str], SchemaDict, SchemaDict]:
    if False:
        i = 10
        return i + 15
    'Shared utility code for establishing dataclasses/pydantic model cols/schema.'
    from dataclasses import asdict
    unpack_nested = False
    if schema:
        (column_names, schema_overrides) = _unpack_schema(schema, schema_overrides=schema_overrides)
        overrides = {col: schema_overrides.get(col, Unknown) for col in column_names}
    else:
        column_names = []
        overrides = {col: py_type_to_dtype(tp, raise_unmatched=False) or Unknown for (col, tp) in type_hints(first_element.__class__).items() if (col in model_fields if model_fields else col != '__slots__')}
        if schema_overrides:
            overrides.update(schema_overrides)
        elif not model_fields:
            dc_fields = set(asdict(first_element))
            schema_overrides = overrides = {nm: tp for (nm, tp) in overrides.items() if nm in dc_fields}
        else:
            schema_overrides = overrides
    for (col, tp) in overrides.items():
        if tp == Categorical:
            overrides[col] = Utf8
        elif not unpack_nested and tp.base_type() in (Unknown, Struct):
            unpack_nested = contains_nested(getattr(first_element, col, None), is_pydantic_model if model_fields else dataclasses.is_dataclass)
    if model_fields and len(model_fields) == len(overrides):
        overrides = dict(zip(model_fields, overrides.values()))
    return (unpack_nested, column_names, schema_overrides, overrides)

def _dataclasses_to_pydf(first_element: Any, data: Sequence[Any], schema: SchemaDefinition | None, schema_overrides: SchemaDict | None, infer_schema_length: int | None, **kwargs: Any) -> PyDataFrame:
    if False:
        for i in range(10):
            print('nop')
    'Initialise DataFrame from python dataclasses.'
    from dataclasses import asdict, astuple
    (unpack_nested, column_names, schema_overrides, overrides) = _establish_dataclass_or_model_schema(first_element, schema, schema_overrides, model_fields=None)
    if unpack_nested:
        dicts = [asdict(md) for md in data]
        pydf = PyDataFrame.read_dicts(dicts, infer_schema_length)
    else:
        rows = [astuple(dc) for dc in data]
        pydf = PyDataFrame.read_rows(rows, infer_schema_length, overrides or None)
    if overrides:
        structs = {c: tp for (c, tp) in overrides.items() if isinstance(tp, Struct)}
        pydf = _post_apply_columns(pydf, column_names, structs, schema_overrides)
    return pydf

def _pydantic_models_to_pydf(first_element: Any, data: Sequence[Any], schema: SchemaDefinition | None, schema_overrides: SchemaDict | None, infer_schema_length: int | None, **kwargs: Any) -> PyDataFrame:
    if False:
        i = 10
        return i + 15
    'Initialise DataFrame from pydantic model objects.'
    import pydantic
    old_pydantic = parse_version(pydantic.__version__) < parse_version('2.0')
    model_fields = list(first_element.__fields__ if old_pydantic else first_element.model_fields)
    (unpack_nested, column_names, schema_overrides, overrides) = _establish_dataclass_or_model_schema(first_element, schema, schema_overrides, model_fields)
    if unpack_nested:
        dicts = [md.dict() for md in data] if old_pydantic else [md.model_dump(mode='python') for md in data]
        pydf = PyDataFrame.read_dicts(dicts, infer_schema_length)
    elif len(model_fields) > 50:
        get_values = itemgetter(*model_fields)
        rows = [get_values(md.__dict__) for md in data]
        pydf = PyDataFrame.read_rows(rows, infer_schema_length, overrides)
    else:
        dicts = [md.__dict__ for md in data]
        pydf = PyDataFrame.read_dicts(dicts, infer_schema_length, overrides)
    if overrides:
        structs = {c: tp for (c, tp) in overrides.items() if isinstance(tp, Struct)}
        pydf = _post_apply_columns(pydf, column_names, structs, schema_overrides)
    return pydf

def numpy_to_pydf(data: np.ndarray[Any, Any], schema: SchemaDefinition | None=None, *, schema_overrides: SchemaDict | None=None, orient: Orientation | None=None, nan_to_null: bool=False) -> PyDataFrame:
    if False:
        for i in range(10):
            print('nop')
    'Construct a PyDataFrame from a numpy ndarray (including structured ndarrays).'
    shape = data.shape
    if data.dtype.names is not None:
        (structured_array, orient) = (True, 'col')
        record_names = list(data.dtype.names)
        n_columns = len(record_names)
        for nm in record_names:
            shape = data[nm].shape
            if len(data[nm].shape) > 2:
                raise ValueError(f'cannot create DataFrame from structured array with elements > 2D; shape[{nm!r}] = {shape}')
        if not schema:
            schema = record_names
    else:
        (structured_array, record_names) = (False, [])
        if shape == (0,):
            n_columns = 0
        elif len(shape) == 1:
            n_columns = 1
        elif len(shape) == 2:
            if orient is None and schema is None:
                n_columns = shape[1]
                orient = 'row'
            elif orient is None and schema is not None:
                n_schema_cols = len(schema)
                if n_schema_cols == shape[0] and n_schema_cols != shape[1]:
                    orient = 'col'
                    n_columns = shape[0]
                else:
                    orient = 'row'
                    n_columns = shape[1]
            elif orient == 'row':
                n_columns = shape[1]
            elif orient == 'col':
                n_columns = shape[0]
            else:
                raise ValueError(f"`orient` must be one of {{'col', 'row', None}}, got {orient!r}")
        else:
            raise ValueError(f'cannot create DataFrame from array with more than two dimensions; shape = {shape}')
    if schema is not None and len(schema) != n_columns:
        raise ValueError('dimensions of `schema` arg must match data dimensions')
    (column_names, schema_overrides) = _unpack_schema(schema, schema_overrides=schema_overrides, n_expected=n_columns)
    if structured_array:
        data_series = [pl.Series(name=series_name, values=data[record_name], dtype=schema_overrides.get(record_name), nan_to_null=nan_to_null)._s for (series_name, record_name) in zip(column_names, record_names)]
    elif shape == (0,):
        data_series = []
    elif len(shape) == 1:
        data_series = [pl.Series(name=column_names[0], values=data, dtype=schema_overrides.get(column_names[0]), nan_to_null=nan_to_null)._s]
    elif orient == 'row':
        data_series = [pl.Series(name=column_names[i], values=data[:, i], dtype=schema_overrides.get(column_names[i]), nan_to_null=nan_to_null)._s for i in range(n_columns)]
    else:
        data_series = [pl.Series(name=column_names[i], values=data[i], dtype=schema_overrides.get(column_names[i]), nan_to_null=nan_to_null)._s for i in range(n_columns)]
    data_series = _handle_columns_arg(data_series, columns=column_names)
    return PyDataFrame(data_series)

def arrow_to_pydf(data: pa.Table, schema: SchemaDefinition | None=None, *, schema_overrides: SchemaDict | None=None, rechunk: bool=True) -> PyDataFrame:
    if False:
        while True:
            i = 10
    'Construct a PyDataFrame from an Arrow Table.'
    original_schema = schema
    (column_names, schema_overrides) = _unpack_schema(schema or data.column_names, schema_overrides=schema_overrides)
    try:
        if column_names != data.column_names:
            data = data.rename_columns(column_names)
    except pa.lib.ArrowInvalid as e:
        raise ValueError('dimensions of columns arg must match data dimensions') from e
    data_dict = {}
    dictionary_cols = {}
    struct_cols = {}
    names = []
    for (i, column) in enumerate(data):
        name = f'column_{i}' if column._name is None else column._name
        names.append(name)
        column = coerce_arrow(column)
        if pa.types.is_dictionary(column.type):
            ps = arrow_to_pyseries(name, column, rechunk=rechunk)
            dictionary_cols[i] = wrap_s(ps)
        elif isinstance(column.type, pa.StructType) and column.num_chunks > 1:
            ps = arrow_to_pyseries(name, column, rechunk=rechunk)
            struct_cols[i] = wrap_s(ps)
        else:
            data_dict[name] = column
    if len(data_dict) > 0:
        tbl = pa.table(data_dict)
        if tbl.shape[0] == 0:
            pydf = pl.DataFrame([pl.Series(name, c) for (name, c) in zip(tbl.column_names, tbl.columns)])._df
        else:
            pydf = PyDataFrame.from_arrow_record_batches(tbl.to_batches())
    else:
        pydf = pl.DataFrame([])._df
    if rechunk:
        pydf = pydf.rechunk()
    reset_order = False
    if len(dictionary_cols) > 0:
        df = wrap_df(pydf)
        df = df.with_columns([F.lit(s).alias(s.name) for s in dictionary_cols.values()])
        reset_order = True
    if len(struct_cols) > 0:
        df = wrap_df(pydf)
        df = df.with_columns([F.lit(s).alias(s.name) for s in struct_cols.values()])
        reset_order = True
    if reset_order:
        df = df[names]
        pydf = df._df
    if column_names != original_schema and (schema_overrides or original_schema):
        pydf = _post_apply_columns(pydf, original_schema, schema_overrides=schema_overrides)
    elif schema_overrides:
        for (col, dtype) in zip(pydf.columns(), pydf.dtypes()):
            override_dtype = schema_overrides.get(col)
            if override_dtype is not None and dtype != override_dtype:
                pydf = _post_apply_columns(pydf, original_schema, schema_overrides=schema_overrides)
                break
    return pydf

def series_to_pydf(data: Series, schema: SchemaDefinition | None=None, schema_overrides: SchemaDict | None=None) -> PyDataFrame:
    if False:
        print('Hello World!')
    'Construct a PyDataFrame from a Polars Series.'
    data_series = [data._s]
    series_name = [s.name() for s in data_series]
    (column_names, schema_overrides) = _unpack_schema(schema or series_name, schema_overrides=schema_overrides, n_expected=1)
    if schema_overrides:
        new_dtype = next(iter(schema_overrides.values()))
        if new_dtype != data.dtype:
            data_series[0] = data_series[0].cast(new_dtype, strict=True)
    data_series = _handle_columns_arg(data_series, columns=column_names)
    return PyDataFrame(data_series)

def iterable_to_pydf(data: Iterable[Any], schema: SchemaDefinition | None=None, schema_overrides: SchemaDict | None=None, orient: Orientation | None=None, chunk_size: int | None=None, infer_schema_length: int | None=N_INFER_DEFAULT) -> PyDataFrame:
    if False:
        return 10
    'Construct a PyDataFrame from an iterable/generator.'
    original_schema = schema
    column_names: list[str] = []
    dtypes_by_idx: dict[int, PolarsDataType] = {}
    if schema is not None:
        (column_names, schema_overrides) = _unpack_schema(schema, schema_overrides=schema_overrides)
    elif schema_overrides:
        (_, schema_overrides) = _unpack_schema(schema, schema_overrides=schema_overrides)
    if not isinstance(data, Generator):
        data = iter(data)
    if orient == 'col':
        if column_names and schema_overrides:
            dtypes_by_idx = {idx: schema_overrides.get(col, Unknown) for (idx, col) in enumerate(column_names)}
        return pl.DataFrame({column_names[idx] if column_names else f'column_{idx}': pl.Series(coldata, dtype=dtypes_by_idx.get(idx)) for (idx, coldata) in enumerate(data)})._df

    def to_frame_chunk(values: list[Any], schema: SchemaDefinition | None) -> DataFrame:
        if False:
            return 10
        return pl.DataFrame(data=values, schema=schema, orient='row', infer_schema_length=infer_schema_length)
    n_chunks = 0
    n_chunk_elems = 1000000
    if chunk_size:
        adaptive_chunk_size = chunk_size
    elif column_names:
        adaptive_chunk_size = n_chunk_elems // len(column_names)
    else:
        adaptive_chunk_size = None
    df: DataFrame = None
    chunk_size = max(infer_schema_length or 0, adaptive_chunk_size or 1000)
    while True:
        values = list(islice(data, chunk_size))
        if not values:
            break
        frame_chunk = to_frame_chunk(values, original_schema)
        if df is None:
            df = frame_chunk
            if not original_schema:
                original_schema = list(df.schema.items())
            if chunk_size != adaptive_chunk_size:
                if (n_columns := len(df.columns)) > 0:
                    chunk_size = adaptive_chunk_size = n_chunk_elems // n_columns
        else:
            df.vstack(frame_chunk, in_place=True)
            n_chunks += 1
    if df is None:
        df = to_frame_chunk([], original_schema)
    return (df.rechunk() if n_chunks > 0 else df)._df

def pandas_has_default_index(df: pd.DataFrame) -> bool:
    if False:
        while True:
            i = 10
    'Identify if the pandas frame only has a default (or equivalent) index.'
    from pandas.core.indexes.range import RangeIndex
    index_cols = df.index.names
    if len(index_cols) > 1 or index_cols not in ([None], ['']):
        return False
    elif df.index.equals(RangeIndex(start=0, stop=len(df), step=1)):
        return True
    else:
        return str(df.index.dtype).startswith('int') and (df.index.sort_values() == np.arange(len(df))).all()

def pandas_to_pydf(data: pd.DataFrame, schema: SchemaDefinition | None=None, *, schema_overrides: SchemaDict | None=None, rechunk: bool=True, nan_to_null: bool=True, include_index: bool=False) -> PyDataFrame:
    if False:
        print('Hello World!')
    'Construct a PyDataFrame from a pandas DataFrame.'
    arrow_dict = {}
    length = data.shape[0]
    if include_index and (not pandas_has_default_index(data)):
        for idxcol in data.index.names:
            arrow_dict[str(idxcol)] = _pandas_series_to_arrow(data.index.get_level_values(idxcol), nan_to_null=nan_to_null, length=length)
    for col in data.columns:
        arrow_dict[str(col)] = _pandas_series_to_arrow(data[col], nan_to_null=nan_to_null, length=length)
    arrow_table = pa.table(arrow_dict)
    return arrow_to_pydf(arrow_table, schema=schema, schema_overrides=schema_overrides, rechunk=rechunk)

def coerce_arrow(array: pa.Array, *, rechunk: bool=True) -> pa.Array:
    if False:
        return 10
    import pyarrow.compute as pc
    if hasattr(array, 'num_chunks') and array.num_chunks > 1 and rechunk:
        if pa.types.is_dictionary(array.type) and (pa.types.is_int8(array.type.index_type) or pa.types.is_uint8(array.type.index_type) or pa.types.is_int16(array.type.index_type) or pa.types.is_uint16(array.type.index_type) or pa.types.is_int32(array.type.index_type)):
            array = pc.cast(array, pa.dictionary(pa.uint32(), pa.large_string())).combine_chunks()
    return array

def numpy_to_idxs(idxs: np.ndarray[Any, Any], size: int) -> pl.Series:
    if False:
        while True:
            i = 10
    if idxs.ndim != 1:
        raise ValueError('only 1D numpy array is supported as index')
    idx_type = get_index_type()
    if len(idxs) == 0:
        return pl.Series('', [], dtype=idx_type)
    if idxs.dtype.kind not in ('i', 'u'):
        raise NotImplementedError('unsupported idxs datatype')
    if idx_type == UInt32:
        if idxs.dtype in {np.int64, np.uint64} and idxs.max() >= 2 ** 32:
            raise ValueError('index positions should be smaller than 2^32')
        if idxs.dtype == np.int64 and idxs.min() < -2 ** 32:
            raise ValueError('index positions should be bigger than -2^32 + 1')
    if idxs.dtype.kind == 'i' and idxs.min() < 0:
        if idx_type == UInt32:
            if idxs.dtype in (np.int8, np.int16):
                idxs = idxs.astype(np.int32)
        elif idxs.dtype in (np.int8, np.int16, np.int32):
            idxs = idxs.astype(np.int64)
        idxs = np.where(idxs < 0, size + idxs, idxs)
    idxs = idxs.astype(np.uint32) if idx_type == UInt32 else idxs.astype(np.uint64)
    return pl.Series('', idxs, dtype=idx_type)