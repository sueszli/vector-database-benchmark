"""A bunch of useful utilities for dealing with types."""
from __future__ import annotations
import contextlib
import copy
import re
import types
from enum import Enum, EnumMeta, auto
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, cast, overload
import numpy as np
import pyarrow as pa
from pandas import DataFrame, Index, MultiIndex, Series
from pandas.api.types import infer_dtype, is_dict_like, is_list_like
from typing_extensions import Final, Literal, Protocol, TypeAlias, TypeGuard, get_args
import streamlit as st
from streamlit import config, errors
from streamlit import logger as _logger
from streamlit import string_util
if TYPE_CHECKING:
    import graphviz
    import sympy
    from pandas.core.indexing import _iLocIndexer
    from pandas.io.formats.style import Styler
    from pandas.io.formats.style_renderer import StyleRenderer
    from plotly.graph_objs import Figure
    from pydeck import Deck
MAX_UNEVALUATED_DF_ROWS = 10000
_LOGGER = _logger.get_logger('root')
ArrayValueFieldName: TypeAlias = Literal['double_array_value', 'int_array_value', 'string_array_value']
ARRAY_VALUE_FIELD_NAMES: Final = frozenset(cast('tuple[ArrayValueFieldName, ...]', get_args(ArrayValueFieldName)))
ValueFieldName: TypeAlias = Literal[ArrayValueFieldName, 'arrow_value', 'bool_value', 'bytes_value', 'double_value', 'file_uploader_state_value', 'int_value', 'json_value', 'string_value', 'trigger_value', 'string_trigger_value']
V_co = TypeVar('V_co', covariant=True)
T = TypeVar('T')

class DataFrameGenericAlias(Protocol[V_co]):
    """Technically not a GenericAlias, but serves the same purpose in
    OptionSequence below, in that it is a type which admits DataFrame,
    but is generic. This allows OptionSequence to be a fully generic type,
    significantly increasing its usefulness.

    We can't use types.GenericAlias, as it is only available from python>=3.9,
    and isn't easily back-ported.
    """

    @property
    def iloc(self) -> _iLocIndexer:
        if False:
            print('Hello World!')
        ...
OptionSequence: TypeAlias = Union[Iterable[V_co], DataFrameGenericAlias[V_co]]
Key: TypeAlias = Union[str, int]
LabelVisibility = Literal['visible', 'hidden', 'collapsed']

class SupportsStr(Protocol):

    def __str__(self) -> str:
        if False:
            print('Hello World!')
        ...

def is_array_value_field_name(obj: object) -> TypeGuard[ArrayValueFieldName]:
    if False:
        return 10
    return obj in ARRAY_VALUE_FIELD_NAMES

@overload
def is_type(obj: object, fqn_type_pattern: Literal['pydeck.bindings.deck.Deck']) -> TypeGuard[Deck]:
    if False:
        return 10
    ...

@overload
def is_type(obj: object, fqn_type_pattern: Literal['plotly.graph_objs._figure.Figure']) -> TypeGuard[Figure]:
    if False:
        i = 10
        return i + 15
    ...

@overload
def is_type(obj: object, fqn_type_pattern: Union[str, re.Pattern[str]]) -> bool:
    if False:
        return 10
    ...

def is_type(obj: object, fqn_type_pattern: Union[str, re.Pattern[str]]) -> bool:
    if False:
        while True:
            i = 10
    "Check type without importing expensive modules.\n\n    Parameters\n    ----------\n    obj : object\n        The object to type-check.\n    fqn_type_pattern : str or regex\n        The fully-qualified type string or a regular expression.\n        Regexes should start with `^` and end with `$`.\n\n    Example\n    -------\n\n    To check whether something is a Matplotlib Figure without importing\n    matplotlib, use:\n\n    >>> is_type(foo, 'matplotlib.figure.Figure')\n\n    "
    fqn_type = get_fqn_type(obj)
    if isinstance(fqn_type_pattern, str):
        return fqn_type_pattern == fqn_type
    else:
        return fqn_type_pattern.match(fqn_type) is not None

def get_fqn(the_type: type) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Get module.type_name for a given type.'
    return f'{the_type.__module__}.{the_type.__qualname__}'

def get_fqn_type(obj: object) -> str:
    if False:
        return 10
    'Get module.type_name for a given object.'
    return get_fqn(type(obj))
_PANDAS_DF_TYPE_STR: Final = 'pandas.core.frame.DataFrame'
_PANDAS_INDEX_TYPE_STR: Final = 'pandas.core.indexes.base.Index'
_PANDAS_SERIES_TYPE_STR: Final = 'pandas.core.series.Series'
_PANDAS_STYLER_TYPE_STR: Final = 'pandas.io.formats.style.Styler'
_NUMPY_ARRAY_TYPE_STR: Final = 'numpy.ndarray'
_SNOWPARK_DF_TYPE_STR: Final = 'snowflake.snowpark.dataframe.DataFrame'
_SNOWPARK_DF_ROW_TYPE_STR: Final = 'snowflake.snowpark.row.Row'
_SNOWPARK_TABLE_TYPE_STR: Final = 'snowflake.snowpark.table.Table'
_PYSPARK_DF_TYPE_STR: Final = 'pyspark.sql.dataframe.DataFrame'
_DATAFRAME_LIKE_TYPES: Final[tuple[str, ...]] = (_PANDAS_DF_TYPE_STR, _PANDAS_INDEX_TYPE_STR, _PANDAS_SERIES_TYPE_STR, _PANDAS_STYLER_TYPE_STR, _NUMPY_ARRAY_TYPE_STR)
DataFrameLike: TypeAlias = 'Union[DataFrame, Index, Series, Styler]'
_DATAFRAME_COMPATIBLE_TYPES: Final[tuple[type, ...]] = (dict, list, set, tuple, type(None))
_DataFrameCompatible: TypeAlias = Union[dict, list, set, Tuple[Any], None]
DataFrameCompatible: TypeAlias = Union[_DataFrameCompatible, DataFrameLike]
_BYTES_LIKE_TYPES: Final[tuple[type, ...]] = (bytes, bytearray)
BytesLike: TypeAlias = Union[bytes, bytearray]

class DataFormat(Enum):
    """DataFormat is used to determine the format of the data."""
    UNKNOWN = auto()
    EMPTY = auto()
    PANDAS_DATAFRAME = auto()
    PANDAS_SERIES = auto()
    PANDAS_INDEX = auto()
    NUMPY_LIST = auto()
    NUMPY_MATRIX = auto()
    PYARROW_TABLE = auto()
    SNOWPARK_OBJECT = auto()
    PYSPARK_OBJECT = auto()
    PANDAS_STYLER = auto()
    LIST_OF_RECORDS = auto()
    LIST_OF_ROWS = auto()
    LIST_OF_VALUES = auto()
    TUPLE_OF_VALUES = auto()
    SET_OF_VALUES = auto()
    COLUMN_INDEX_MAPPING = auto()
    COLUMN_VALUE_MAPPING = auto()
    COLUMN_SERIES_MAPPING = auto()
    KEY_VALUE_DICT = auto()

def is_dataframe(obj: object) -> TypeGuard[DataFrame]:
    if False:
        i = 10
        return i + 15
    return is_type(obj, _PANDAS_DF_TYPE_STR)

def is_dataframe_like(obj: object) -> TypeGuard[DataFrameLike]:
    if False:
        return 10
    return any((is_type(obj, t) for t in _DATAFRAME_LIKE_TYPES))

def is_snowpark_or_pyspark_data_object(obj: object) -> bool:
    if False:
        return 10
    'True if if obj is of type snowflake.snowpark.dataframe.DataFrame, snowflake.snowpark.table.Table or\n    True when obj is a list which contains snowflake.snowpark.row.Row or True when obj is of type pyspark.sql.dataframe.DataFrame\n    False otherwise.\n    '
    return is_snowpark_data_object(obj) or is_pyspark_data_object(obj)

def is_snowpark_data_object(obj: object) -> bool:
    if False:
        while True:
            i = 10
    'True if obj is of type snowflake.snowpark.dataframe.DataFrame, snowflake.snowpark.table.Table or\n    True when obj is a list which contains snowflake.snowpark.row.Row,\n    False otherwise.\n    '
    if is_type(obj, _SNOWPARK_TABLE_TYPE_STR):
        return True
    if is_type(obj, _SNOWPARK_DF_TYPE_STR):
        return True
    if not isinstance(obj, list):
        return False
    if len(obj) < 1:
        return False
    if not hasattr(obj[0], '__class__'):
        return False
    return is_type(obj[0], _SNOWPARK_DF_ROW_TYPE_STR)

def is_pyspark_data_object(obj: object) -> bool:
    if False:
        print('Hello World!')
    'True if obj is of type pyspark.sql.dataframe.DataFrame'
    return is_type(obj, _PYSPARK_DF_TYPE_STR) and hasattr(obj, 'toPandas') and callable(getattr(obj, 'toPandas'))

def is_dataframe_compatible(obj: object) -> TypeGuard[DataFrameCompatible]:
    if False:
        i = 10
        return i + 15
    'True if type that can be passed to convert_anything_to_df.'
    return is_dataframe_like(obj) or type(obj) in _DATAFRAME_COMPATIBLE_TYPES

def is_bytes_like(obj: object) -> TypeGuard[BytesLike]:
    if False:
        for i in range(10):
            print('nop')
    'True if the type is considered bytes-like for the purposes of\n    protobuf data marshalling.\n    '
    return isinstance(obj, _BYTES_LIKE_TYPES)

def to_bytes(obj: BytesLike) -> bytes:
    if False:
        return 10
    'Converts the given object to bytes.\n\n    Only types for which `is_bytes_like` is true can be converted; anything\n    else will result in an exception.\n    '
    if isinstance(obj, bytearray):
        return bytes(obj)
    elif isinstance(obj, bytes):
        return obj
    raise RuntimeError(f'{obj} is not convertible to bytes')
_SYMPY_RE: Final = re.compile('^sympy.*$')

def is_sympy_expession(obj: object) -> TypeGuard[sympy.Expr]:
    if False:
        i = 10
        return i + 15
    'True if input is a SymPy expression.'
    if not is_type(obj, _SYMPY_RE):
        return False
    try:
        import sympy
        return isinstance(obj, sympy.Expr)
    except ImportError:
        return False
_ALTAIR_RE: Final = re.compile('^altair\\.vegalite\\.v\\d+\\.api\\.\\w*Chart$')

def is_altair_chart(obj: object) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'True if input looks like an Altair chart.'
    return is_type(obj, _ALTAIR_RE)

def is_keras_model(obj: object) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'True if input looks like a Keras model.'
    return is_type(obj, 'keras.engine.sequential.Sequential') or is_type(obj, 'keras.engine.training.Model') or is_type(obj, 'tensorflow.python.keras.engine.sequential.Sequential') or is_type(obj, 'tensorflow.python.keras.engine.training.Model')

def is_list_of_scalars(data: Iterable[Any]) -> bool:
    if False:
        while True:
            i = 10
    'Check if the list only contains scalar values.'
    return infer_dtype(data, skipna=True) not in ['mixed', 'unknown-array']

def is_plotly_chart(obj: object) -> TypeGuard[Union[Figure, list[Any], dict[str, Any]]]:
    if False:
        for i in range(10):
            print('nop')
    'True if input looks like a Plotly chart.'
    return is_type(obj, 'plotly.graph_objs._figure.Figure') or _is_list_of_plotly_objs(obj) or _is_probably_plotly_dict(obj)

def is_graphviz_chart(obj: object) -> TypeGuard[Union[graphviz.Graph, graphviz.Digraph]]:
    if False:
        return 10
    'True if input looks like a GraphViz chart.'
    return is_type(obj, 'graphviz.dot.Graph') or is_type(obj, 'graphviz.dot.Digraph') or is_type(obj, 'graphviz.graphs.Graph') or is_type(obj, 'graphviz.graphs.Digraph')

def _is_plotly_obj(obj: object) -> bool:
    if False:
        i = 10
        return i + 15
    'True if input if from a type that lives in plotly.plotly_objs.'
    the_type = type(obj)
    return the_type.__module__.startswith('plotly.graph_objs')

def _is_list_of_plotly_objs(obj: object) -> TypeGuard[list[Any]]:
    if False:
        return 10
    if not isinstance(obj, list):
        return False
    if len(obj) == 0:
        return False
    return all((_is_plotly_obj(item) for item in obj))

def _is_probably_plotly_dict(obj: object) -> TypeGuard[dict[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(obj, dict):
        return False
    if len(obj.keys()) == 0:
        return False
    if any((k not in ['config', 'data', 'frames', 'layout'] for k in obj.keys())):
        return False
    if any((_is_plotly_obj(v) for v in obj.values())):
        return True
    if any((_is_list_of_plotly_objs(v) for v in obj.values())):
        return True
    return False

def is_function(x: object) -> TypeGuard[types.FunctionType]:
    if False:
        print('Hello World!')
    'Return True if x is a function.'
    return isinstance(x, types.FunctionType)

def is_namedtuple(x: object) -> TypeGuard[NamedTuple]:
    if False:
        for i in range(10):
            print('nop')
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple):
        return False
    return all((type(n).__name__ == 'str' for n in f))

def is_pandas_styler(obj: object) -> TypeGuard['Styler']:
    if False:
        while True:
            i = 10
    return is_type(obj, _PANDAS_STYLER_TYPE_STR)

def is_pydeck(obj: object) -> TypeGuard[Deck]:
    if False:
        while True:
            i = 10
    'True if input looks like a pydeck chart.'
    return is_type(obj, 'pydeck.bindings.deck.Deck')

def is_iterable(obj: object) -> TypeGuard[Iterable[Any]]:
    if False:
        i = 10
        return i + 15
    try:
        iter(obj)
    except TypeError:
        return False
    return True

def is_sequence(seq: Any) -> bool:
    if False:
        while True:
            i = 10
    'True if input looks like a sequence.'
    if isinstance(seq, str):
        return False
    try:
        len(seq)
    except Exception:
        return False
    return True

@overload
def convert_anything_to_df(data: Any, max_unevaluated_rows: int=MAX_UNEVALUATED_DF_ROWS, ensure_copy: bool=False) -> DataFrame:
    if False:
        return 10
    ...

@overload
def convert_anything_to_df(data: Any, max_unevaluated_rows: int=MAX_UNEVALUATED_DF_ROWS, ensure_copy: bool=False, allow_styler: bool=False) -> Union[DataFrame, 'Styler']:
    if False:
        print('Hello World!')
    ...

def convert_anything_to_df(data: Any, max_unevaluated_rows: int=MAX_UNEVALUATED_DF_ROWS, ensure_copy: bool=False, allow_styler: bool=False) -> Union[DataFrame, 'Styler']:
    if False:
        i = 10
        return i + 15
    "Try to convert different formats to a Pandas Dataframe.\n\n    Parameters\n    ----------\n    data : ndarray, Iterable, dict, DataFrame, Styler, pa.Table, None, dict, list, or any\n\n    max_unevaluated_rows: int\n        If unevaluated data is detected this func will evaluate it,\n        taking max_unevaluated_rows, defaults to 10k and 100 for st.table\n\n    ensure_copy: bool\n        If True, make sure to always return a copy of the data. If False, it depends on the\n        type of the data. For example, a Pandas DataFrame will be returned as-is.\n\n    allow_styler: bool\n        If True, allows this to return a Pandas Styler object as well. If False, returns\n        a plain Pandas DataFrame (which, of course, won't contain the Styler's styles).\n\n    Returns\n    -------\n    pandas.DataFrame or pandas.Styler\n\n    "
    if is_type(data, _PANDAS_DF_TYPE_STR):
        return data.copy() if ensure_copy else cast(DataFrame, data)
    if is_pandas_styler(data):
        sr = cast('StyleRenderer', data)
        if allow_styler:
            if ensure_copy:
                out = copy.deepcopy(sr)
                out.data = sr.data.copy()
                return cast('Styler', out)
            else:
                return data
        else:
            return cast('Styler', sr.data.copy() if ensure_copy else sr.data)
    if is_type(data, 'numpy.ndarray'):
        if len(data.shape) == 0:
            return DataFrame([])
        return DataFrame(data)
    if is_type(data, _SNOWPARK_DF_TYPE_STR) or is_type(data, _SNOWPARK_TABLE_TYPE_STR) or is_type(data, _PYSPARK_DF_TYPE_STR):
        if is_type(data, _PYSPARK_DF_TYPE_STR):
            data = data.limit(max_unevaluated_rows).toPandas()
        else:
            data = DataFrame(data.take(max_unevaluated_rows))
        if data.shape[0] == max_unevaluated_rows:
            st.caption(f'⚠️ Showing only {string_util.simplify_number(max_unevaluated_rows)} rows. Call `collect()` on the dataframe to show more.')
        return cast(DataFrame, data)
    if hasattr(data, 'to_pandas'):
        return cast(DataFrame, data.to_pandas())
    try:
        return DataFrame(data)
    except ValueError as ex:
        if isinstance(data, dict):
            with contextlib.suppress(ValueError):
                return DataFrame.from_dict(data, orient='index')
        raise errors.StreamlitAPIException(f'\nUnable to convert object of type `{type(data)}` to `pandas.DataFrame`.\nOffending object:\n```py\n{data}\n```') from ex

@overload
def ensure_iterable(obj: Iterable[V_co]) -> Iterable[V_co]:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def ensure_iterable(obj: OptionSequence[V_co]) -> Iterable[Any]:
    if False:
        while True:
            i = 10
    ...

def ensure_iterable(obj: Union[OptionSequence[V_co], Iterable[V_co]]) -> Iterable[Any]:
    if False:
        while True:
            i = 10
    'Try to convert different formats to something iterable. Most inputs\n    are assumed to be iterable, but if we have a DataFrame, we can just\n    select the first column to iterate over. If the input is not iterable,\n    a TypeError is raised.\n\n    Parameters\n    ----------\n    obj : list, tuple, numpy.ndarray, pandas.Series, pandas.DataFrame, pyspark.sql.DataFrame, snowflake.snowpark.dataframe.DataFrame or snowflake.snowpark.table.Table\n\n    Returns\n    -------\n    iterable\n\n    '
    if is_snowpark_or_pyspark_data_object(obj):
        obj = convert_anything_to_df(obj)
    if is_dataframe(obj):
        return cast(Iterable[Any], obj.iloc[:, 0])
    if is_iterable(obj):
        return obj
    raise TypeError(f'Object is not an iterable and could not be converted to one. Object: {obj}')

def ensure_indexable(obj: OptionSequence[V_co]) -> Sequence[V_co]:
    if False:
        while True:
            i = 10
    'Try to ensure a value is an indexable Sequence. If the collection already\n    is one, it has the index method that we need. Otherwise, convert it to a list.\n    '
    it = ensure_iterable(obj)
    index_fn = getattr(it, 'index', None)
    if callable(index_fn):
        return it
    else:
        return list(it)

def is_pandas_version_less_than(v: str) -> bool:
    if False:
        return 10
    'Return True if the current Pandas version is less than the input version.\n\n    Parameters\n    ----------\n    v : str\n        Version string, e.g. "0.25.0"\n\n    Returns\n    -------\n    bool\n\n    '
    import pandas as pd
    from packaging import version
    return version.parse(pd.__version__) < version.parse(v)

def is_pyarrow_version_less_than(v: str) -> bool:
    if False:
        return 10
    'Return True if the current Pyarrow version is less than the input version.\n\n    Parameters\n    ----------\n    v : str\n        Version string, e.g. "0.25.0"\n\n    Returns\n    -------\n    bool\n\n    '
    from packaging import version
    return version.parse(pa.__version__) < version.parse(v)

def pyarrow_table_to_bytes(table: pa.Table) -> bytes:
    if False:
        return 10
    'Serialize pyarrow.Table to bytes using Apache Arrow.\n\n    Parameters\n    ----------\n    table : pyarrow.Table\n        A table to convert.\n\n    '
    sink = pa.BufferOutputStream()
    writer = pa.RecordBatchStreamWriter(sink, table.schema)
    writer.write_table(table)
    writer.close()
    return cast(bytes, sink.getvalue().to_pybytes())

def is_colum_type_arrow_incompatible(column: Union[Series[Any], Index]) -> bool:
    if False:
        while True:
            i = 10
    'Return True if the column type is known to cause issues during Arrow conversion.'
    if column.dtype.kind in ['c']:
        return True
    if column.dtype == 'object':
        inferred_type = infer_dtype(column, skipna=True)
        if inferred_type in ['mixed-integer', 'complex']:
            return True
        elif inferred_type == 'mixed':
            if len(column) == 0 or not hasattr(column, 'iloc'):
                return True
            first_value = column.iloc[0]
            if not is_list_like(first_value) or is_dict_like(first_value) or isinstance(first_value, frozenset):
                return True
            return False
    return False

def fix_arrow_incompatible_column_types(df: DataFrame, selected_columns: Optional[List[str]]=None) -> DataFrame:
    if False:
        print('Hello World!')
    'Fix column types that are not supported by Arrow table.\n\n    This includes mixed types (e.g. mix of integers and strings)\n    as well as complex numbers (complex128 type). These types will cause\n    errors during conversion of the dataframe to an Arrow table.\n    It is fixed by converting all values of the column to strings\n    This is sufficient for displaying the data on the frontend.\n\n    Parameters\n    ----------\n    df : pandas.DataFrame\n        A dataframe to fix.\n\n    selected_columns: Optional[List[str]]\n        A list of columns to fix. If None, all columns are evaluated.\n\n    Returns\n    -------\n    The fixed dataframe.\n    '
    df_copy: DataFrame | None = None
    for col in selected_columns or df.columns:
        if is_colum_type_arrow_incompatible(df[col]):
            if df_copy is None:
                df_copy = df.copy()
            df_copy[col] = df[col].astype('string')
    if not selected_columns and (not isinstance(df.index, MultiIndex) and is_colum_type_arrow_incompatible(df.index)):
        if df_copy is None:
            df_copy = df.copy()
        df_copy.index = df.index.astype('string')
    return df_copy if df_copy is not None else df

def data_frame_to_bytes(df: DataFrame) -> bytes:
    if False:
        return 10
    'Serialize pandas.DataFrame to bytes using Apache Arrow.\n\n    Parameters\n    ----------\n    df : pandas.DataFrame\n        A dataframe to convert.\n\n    '
    try:
        table = pa.Table.from_pandas(df)
    except (pa.ArrowTypeError, pa.ArrowInvalid, pa.ArrowNotImplementedError) as ex:
        _LOGGER.info('Serialization of dataframe to Arrow table was unsuccessful due to: %s. Applying automatic fixes for column types to make the dataframe Arrow-compatible.', ex)
        df = fix_arrow_incompatible_column_types(df)
        table = pa.Table.from_pandas(df)
    return pyarrow_table_to_bytes(table)

def bytes_to_data_frame(source: bytes) -> DataFrame:
    if False:
        while True:
            i = 10
    'Convert bytes to pandas.DataFrame.\n\n    Using this function in production needs to make sure that\n    the pyarrow version >= 14.0.1.\n\n    Parameters\n    ----------\n    source : bytes\n        A bytes object to convert.\n\n    '
    reader = pa.RecordBatchStreamReader(source)
    return cast(DataFrame, reader.read_pandas())

def determine_data_format(input_data: Any) -> DataFormat:
    if False:
        i = 10
        return i + 15
    'Determine the data format of the input data.\n\n    Parameters\n    ----------\n    input_data : Any\n        The input data to determine the data format of.\n\n    Returns\n    -------\n    DataFormat\n        The data format of the input data.\n    '
    if input_data is None:
        return DataFormat.EMPTY
    elif isinstance(input_data, DataFrame):
        return DataFormat.PANDAS_DATAFRAME
    elif isinstance(input_data, np.ndarray):
        if len(input_data.shape) == 1:
            return DataFormat.NUMPY_LIST
        return DataFormat.NUMPY_MATRIX
    elif isinstance(input_data, pa.Table):
        return DataFormat.PYARROW_TABLE
    elif isinstance(input_data, Series):
        return DataFormat.PANDAS_SERIES
    elif isinstance(input_data, Index):
        return DataFormat.PANDAS_INDEX
    elif is_pandas_styler(input_data):
        return DataFormat.PANDAS_STYLER
    elif is_snowpark_data_object(input_data):
        return DataFormat.SNOWPARK_OBJECT
    elif is_pyspark_data_object(input_data):
        return DataFormat.PYSPARK_OBJECT
    elif isinstance(input_data, (list, tuple, set)):
        if is_list_of_scalars(input_data):
            if isinstance(input_data, tuple):
                return DataFormat.TUPLE_OF_VALUES
            if isinstance(input_data, set):
                return DataFormat.SET_OF_VALUES
            return DataFormat.LIST_OF_VALUES
        else:
            first_element = next(iter(input_data))
            if isinstance(first_element, dict):
                return DataFormat.LIST_OF_RECORDS
            if isinstance(first_element, (list, tuple, set)):
                return DataFormat.LIST_OF_ROWS
    elif isinstance(input_data, dict):
        if not input_data:
            return DataFormat.KEY_VALUE_DICT
        if len(input_data) > 0:
            first_value = next(iter(input_data.values()))
            if isinstance(first_value, dict):
                return DataFormat.COLUMN_INDEX_MAPPING
            if isinstance(first_value, (list, tuple)):
                return DataFormat.COLUMN_VALUE_MAPPING
            if isinstance(first_value, Series):
                return DataFormat.COLUMN_SERIES_MAPPING
            if is_list_of_scalars(input_data.values()):
                return DataFormat.KEY_VALUE_DICT
    return DataFormat.UNKNOWN

def _unify_missing_values(df: DataFrame) -> DataFrame:
    if False:
        print('Hello World!')
    'Unify all missing values in a DataFrame to None.\n\n    Pandas uses a variety of values to represent missing values, including np.nan,\n    NaT, None, and pd.NA. This function replaces all of these values with None,\n    which is the only missing value type that is supported by all data\n    '
    return df.fillna(np.nan).replace([np.nan], [None])

def convert_df_to_data_format(df: DataFrame, data_format: DataFormat) -> Union[DataFrame, Series[Any], pa.Table, np.ndarray[Any, np.dtype[Any]], Tuple[Any], List[Any], Set[Any], Dict[str, Any]]:
    if False:
        i = 10
        return i + 15
    'Convert a dataframe to the specified data format.\n\n    Parameters\n    ----------\n    df : pd.DataFrame\n        The dataframe to convert.\n\n    data_format : DataFormat\n        The data format to convert to.\n\n    Returns\n    -------\n    pd.DataFrame, pd.Series, pyarrow.Table, np.ndarray, list, set, tuple, or dict.\n        The converted dataframe.\n    '
    if data_format in [DataFormat.EMPTY, DataFormat.PANDAS_DATAFRAME, DataFormat.SNOWPARK_OBJECT, DataFormat.PYSPARK_OBJECT, DataFormat.PANDAS_INDEX, DataFormat.PANDAS_STYLER]:
        return df
    elif data_format == DataFormat.NUMPY_LIST:
        return np.ndarray(0) if df.empty else df.iloc[:, 0].to_numpy()
    elif data_format == DataFormat.NUMPY_MATRIX:
        return np.ndarray(0) if df.empty else df.to_numpy()
    elif data_format == DataFormat.PYARROW_TABLE:
        return pa.Table.from_pandas(df)
    elif data_format == DataFormat.PANDAS_SERIES:
        if len(df.columns) != 1:
            raise ValueError(f'DataFrame is expected to have a single column but has {len(df.columns)}.')
        return df[df.columns[0]]
    elif data_format == DataFormat.LIST_OF_RECORDS:
        return _unify_missing_values(df).to_dict(orient='records')
    elif data_format == DataFormat.LIST_OF_ROWS:
        return _unify_missing_values(df).to_numpy().tolist()
    elif data_format == DataFormat.COLUMN_INDEX_MAPPING:
        return _unify_missing_values(df).to_dict(orient='dict')
    elif data_format == DataFormat.COLUMN_VALUE_MAPPING:
        return _unify_missing_values(df).to_dict(orient='list')
    elif data_format == DataFormat.COLUMN_SERIES_MAPPING:
        return df.to_dict(orient='series')
    elif data_format in [DataFormat.LIST_OF_VALUES, DataFormat.TUPLE_OF_VALUES, DataFormat.SET_OF_VALUES]:
        df = _unify_missing_values(df)
        return_list = []
        if len(df.columns) == 1:
            return_list = df[df.columns[0]].tolist()
        elif len(df.columns) >= 1:
            raise ValueError(f'DataFrame is expected to have a single column but has {len(df.columns)}.')
        if data_format == DataFormat.TUPLE_OF_VALUES:
            return tuple(return_list)
        if data_format == DataFormat.SET_OF_VALUES:
            return set(return_list)
        return return_list
    elif data_format == DataFormat.KEY_VALUE_DICT:
        df = _unify_missing_values(df)
        return dict() if df.empty else df.iloc[:, 0].to_dict()
    raise ValueError(f'Unsupported input data format: {data_format}')

@overload
def to_key(key: None) -> None:
    if False:
        while True:
            i = 10
    ...

@overload
def to_key(key: Key) -> str:
    if False:
        return 10
    ...

def to_key(key: Optional[Key]) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    if key is None:
        return None
    else:
        return str(key)

def maybe_tuple_to_list(item: Any) -> Any:
    if False:
        for i in range(10):
            print('nop')
    "Convert a tuple to a list. Leave as is if it's not a tuple."
    return list(item) if isinstance(item, tuple) else item

def maybe_raise_label_warnings(label: Optional[str], label_visibility: Optional[str]):
    if False:
        for i in range(10):
            print('nop')
    if not label:
        _LOGGER.warning('`label` got an empty value. This is discouraged for accessibility reasons and may be disallowed in the future by raising an exception. Please provide a non-empty label and hide it with label_visibility if needed.')
    if label_visibility not in ('visible', 'hidden', 'collapsed'):
        raise errors.StreamlitAPIException(f"Unsupported label_visibility option '{label_visibility}'. Valid values are 'visible', 'hidden' or 'collapsed'.")

def infer_vegalite_type(data: Series[Any]) -> Union[str, Tuple[str, List[Any]]]:
    if False:
        while True:
            i = 10
    "\n    From an array-like input, infer the correct vega typecode\n    ('ordinal', 'nominal', 'quantitative', or 'temporal')\n\n    Parameters\n    ----------\n    data: Numpy array or Pandas Series\n    "
    typ = infer_dtype(data)
    if typ in ['floating', 'mixed-integer-float', 'integer', 'mixed-integer', 'complex']:
        return 'quantitative'
    elif typ == 'categorical' and data.cat.ordered:
        return ('ordinal', data.cat.categories.tolist())
    elif typ in ['string', 'bytes', 'categorical', 'boolean', 'mixed', 'unicode']:
        return 'nominal'
    elif typ in ['datetime', 'datetime64', 'timedelta', 'timedelta64', 'date', 'time', 'period']:
        return 'temporal'
    else:
        return 'nominal'
E1 = TypeVar('E1', bound=Enum)
E2 = TypeVar('E2', bound=Enum)
ALLOWED_ENUM_COERCION_CONFIG_SETTINGS = ('off', 'nameOnly', 'nameAndValue')

def coerce_enum(from_enum_value: E1, to_enum_class: Type[E2]) -> E1 | E2:
    if False:
        while True:
            i = 10
    'Attempt to coerce an Enum value to another EnumMeta.\n\n    An Enum value of EnumMeta E1 is considered coercable to EnumType E2\n    if the EnumMeta __qualname__ match and the names of their members\n    match as well. (This is configurable in streamlist configs)\n    '
    if not isinstance(from_enum_value, Enum):
        raise ValueError(f'Expected an Enum in the first argument. Got {type(from_enum_value)}')
    if not isinstance(to_enum_class, EnumMeta):
        raise ValueError(f'Expected an EnumMeta/Type in the second argument. Got {type(to_enum_class)}')
    if isinstance(from_enum_value, to_enum_class):
        return from_enum_value
    coercion_type = config.get_option('runner.enumCoercion')
    if coercion_type not in ALLOWED_ENUM_COERCION_CONFIG_SETTINGS:
        raise errors.StreamlitAPIException(f"Invalid value for config option runner.enumCoercion. Expected one of {ALLOWED_ENUM_COERCION_CONFIG_SETTINGS}, but got '{coercion_type}'.")
    if coercion_type == 'off':
        return from_enum_value
    from_enum_class = from_enum_value.__class__
    if from_enum_class.__qualname__ != to_enum_class.__qualname__ or (coercion_type == 'nameOnly' and set(to_enum_class._member_names_) != set(from_enum_class._member_names_)) or (coercion_type == 'nameAndValue' and set(to_enum_class._value2member_map_) != set(from_enum_class._value2member_map_)):
        _LOGGER.debug('Failed to coerce %s to class %s', from_enum_value, to_enum_class)
        return from_enum_value
    _LOGGER.debug('Coerced %s to class %s', from_enum_value, to_enum_class)
    return to_enum_class[from_enum_value._name_]