from __future__ import annotations
import json
from enum import Enum
from typing import Dict, List, Mapping, Optional, Union
import pandas as pd
import pyarrow as pa
from typing_extensions import Final, Literal, TypeAlias
from streamlit.elements.lib.column_types import ColumnConfig, ColumnType
from streamlit.elements.lib.dicttools import remove_none_values
from streamlit.errors import StreamlitAPIException
from streamlit.proto.Arrow_pb2 import Arrow as ArrowProto
from streamlit.type_util import DataFormat, is_colum_type_arrow_incompatible
IndexIdentifierType = Literal['_index']
INDEX_IDENTIFIER: IndexIdentifierType = '_index'
_NUMERICAL_POSITION_PREFIX = '_pos:'

class ColumnDataKind(str, Enum):
    INTEGER = 'integer'
    FLOAT = 'float'
    DATE = 'date'
    TIME = 'time'
    DATETIME = 'datetime'
    BOOLEAN = 'boolean'
    STRING = 'string'
    TIMEDELTA = 'timedelta'
    PERIOD = 'period'
    INTERVAL = 'interval'
    BYTES = 'bytes'
    DECIMAL = 'decimal'
    COMPLEX = 'complex'
    LIST = 'list'
    DICT = 'dict'
    EMPTY = 'empty'
    UNKNOWN = 'unknown'
DataframeSchema: TypeAlias = Dict[str, ColumnDataKind]
_EDITING_COMPATIBILITY_MAPPING: Final[Dict[ColumnType, List[ColumnDataKind]]] = {'text': [ColumnDataKind.STRING, ColumnDataKind.EMPTY], 'number': [ColumnDataKind.INTEGER, ColumnDataKind.FLOAT, ColumnDataKind.DECIMAL, ColumnDataKind.STRING, ColumnDataKind.TIMEDELTA, ColumnDataKind.EMPTY], 'checkbox': [ColumnDataKind.BOOLEAN, ColumnDataKind.STRING, ColumnDataKind.INTEGER, ColumnDataKind.EMPTY], 'selectbox': [ColumnDataKind.STRING, ColumnDataKind.BOOLEAN, ColumnDataKind.INTEGER, ColumnDataKind.FLOAT, ColumnDataKind.EMPTY], 'date': [ColumnDataKind.DATE, ColumnDataKind.DATETIME, ColumnDataKind.EMPTY], 'time': [ColumnDataKind.TIME, ColumnDataKind.DATETIME, ColumnDataKind.EMPTY], 'datetime': [ColumnDataKind.DATETIME, ColumnDataKind.DATE, ColumnDataKind.TIME, ColumnDataKind.EMPTY], 'link': [ColumnDataKind.STRING, ColumnDataKind.EMPTY]}

def is_type_compatible(column_type: ColumnType, data_kind: ColumnDataKind) -> bool:
    if False:
        i = 10
        return i + 15
    'Check if the column type is compatible with the underlying data kind.\n\n    This check only applies to editable column types (e.g. number or text).\n    Non-editable column types (e.g. bar_chart or image) can be configured for\n    all data kinds (this might change in the future).\n\n    Parameters\n    ----------\n    column_type : ColumnType\n        The column type to check.\n\n    data_kind : ColumnDataKind\n        The data kind to check.\n\n    Returns\n    -------\n    bool\n        True if the column type is compatible with the data kind, False otherwise.\n    '
    if column_type not in _EDITING_COMPATIBILITY_MAPPING:
        return True
    return data_kind in _EDITING_COMPATIBILITY_MAPPING[column_type]

def _determine_data_kind_via_arrow(field: pa.Field) -> ColumnDataKind:
    if False:
        return 10
    'Determine the data kind via the arrow type information.\n\n    The column data kind refers to the shared data type of the values\n    in the column (e.g. int, float, str, bool).\n\n    Parameters\n    ----------\n\n    field : pa.Field\n        The arrow field from the arrow table schema.\n\n    Returns\n    -------\n    ColumnDataKind\n        The data kind of the field.\n    '
    field_type = field.type
    if pa.types.is_integer(field_type):
        return ColumnDataKind.INTEGER
    if pa.types.is_floating(field_type):
        return ColumnDataKind.FLOAT
    if pa.types.is_boolean(field_type):
        return ColumnDataKind.BOOLEAN
    if pa.types.is_string(field_type):
        return ColumnDataKind.STRING
    if pa.types.is_date(field_type):
        return ColumnDataKind.DATE
    if pa.types.is_time(field_type):
        return ColumnDataKind.TIME
    if pa.types.is_timestamp(field_type):
        return ColumnDataKind.DATETIME
    if pa.types.is_duration(field_type):
        return ColumnDataKind.TIMEDELTA
    if pa.types.is_list(field_type):
        return ColumnDataKind.LIST
    if pa.types.is_decimal(field_type):
        return ColumnDataKind.DECIMAL
    if pa.types.is_null(field_type):
        return ColumnDataKind.EMPTY
    if pa.types.is_binary(field_type):
        return ColumnDataKind.BYTES
    if pa.types.is_struct(field_type):
        return ColumnDataKind.DICT
    return ColumnDataKind.UNKNOWN

def _determine_data_kind_via_pandas_dtype(column: pd.Series | pd.Index) -> ColumnDataKind:
    if False:
        i = 10
        return i + 15
    'Determine the data kind by using the pandas dtype.\n\n    The column data kind refers to the shared data type of the values\n    in the column (e.g. int, float, str, bool).\n\n    Parameters\n    ----------\n    column : pd.Series, pd.Index\n        The column for which the data kind should be determined.\n\n    Returns\n    -------\n    ColumnDataKind\n        The data kind of the column.\n    '
    column_dtype = column.dtype
    if pd.api.types.is_bool_dtype(column_dtype):
        return ColumnDataKind.BOOLEAN
    if pd.api.types.is_integer_dtype(column_dtype):
        return ColumnDataKind.INTEGER
    if pd.api.types.is_float_dtype(column_dtype):
        return ColumnDataKind.FLOAT
    if pd.api.types.is_datetime64_any_dtype(column_dtype):
        return ColumnDataKind.DATETIME
    if pd.api.types.is_timedelta64_dtype(column_dtype):
        return ColumnDataKind.TIMEDELTA
    if isinstance(column_dtype, pd.PeriodDtype):
        return ColumnDataKind.PERIOD
    if isinstance(column_dtype, pd.IntervalDtype):
        return ColumnDataKind.INTERVAL
    if pd.api.types.is_complex_dtype(column_dtype):
        return ColumnDataKind.COMPLEX
    if pd.api.types.is_object_dtype(column_dtype) is False and pd.api.types.is_string_dtype(column_dtype):
        return ColumnDataKind.STRING
    return ColumnDataKind.UNKNOWN

def _determine_data_kind_via_inferred_type(column: pd.Series | pd.Index) -> ColumnDataKind:
    if False:
        for i in range(10):
            print('nop')
    'Determine the data kind by inferring it from the underlying data.\n\n    The column data kind refers to the shared data type of the values\n    in the column (e.g. int, float, str, bool).\n\n    Parameters\n    ----------\n    column : pd.Series, pd.Index\n        The column to determine the data kind for.\n\n    Returns\n    -------\n    ColumnDataKind\n        The data kind of the column.\n    '
    inferred_type = pd.api.types.infer_dtype(column)
    if inferred_type == 'string':
        return ColumnDataKind.STRING
    if inferred_type == 'bytes':
        return ColumnDataKind.BYTES
    if inferred_type in ['floating', 'mixed-integer-float']:
        return ColumnDataKind.FLOAT
    if inferred_type == 'integer':
        return ColumnDataKind.INTEGER
    if inferred_type == 'decimal':
        return ColumnDataKind.DECIMAL
    if inferred_type == 'complex':
        return ColumnDataKind.COMPLEX
    if inferred_type == 'boolean':
        return ColumnDataKind.BOOLEAN
    if inferred_type in ['datetime64', 'datetime']:
        return ColumnDataKind.DATETIME
    if inferred_type == 'date':
        return ColumnDataKind.DATE
    if inferred_type in ['timedelta64', 'timedelta']:
        return ColumnDataKind.TIMEDELTA
    if inferred_type == 'time':
        return ColumnDataKind.TIME
    if inferred_type == 'period':
        return ColumnDataKind.PERIOD
    if inferred_type == 'interval':
        return ColumnDataKind.INTERVAL
    if inferred_type == 'empty':
        return ColumnDataKind.EMPTY
    return ColumnDataKind.UNKNOWN

def _determine_data_kind(column: pd.Series | pd.Index, field: Optional[pa.Field]=None) -> ColumnDataKind:
    if False:
        while True:
            i = 10
    'Determine the data kind of a column.\n\n    The column data kind refers to the shared data type of the values\n    in the column (e.g. int, float, str, bool).\n\n    Parameters\n    ----------\n    column : pd.Series, pd.Index\n        The column to determine the data kind for.\n    field : pa.Field, optional\n        The arrow field from the arrow table schema.\n\n    Returns\n    -------\n    ColumnDataKind\n        The data kind of the column.\n    '
    if isinstance(column.dtype, pd.CategoricalDtype):
        return _determine_data_kind_via_inferred_type(column.dtype.categories)
    if field is not None:
        data_kind = _determine_data_kind_via_arrow(field)
        if data_kind != ColumnDataKind.UNKNOWN:
            return data_kind
    if column.dtype.name == 'object':
        return _determine_data_kind_via_inferred_type(column)
    return _determine_data_kind_via_pandas_dtype(column)

def determine_dataframe_schema(data_df: pd.DataFrame, arrow_schema: pa.Schema) -> DataframeSchema:
    if False:
        return 10
    'Determine the schema of a dataframe.\n\n    Parameters\n    ----------\n    data_df : pd.DataFrame\n        The dataframe to determine the schema of.\n    arrow_schema : pa.Schema\n        The Arrow schema of the dataframe.\n\n    Returns\n    -------\n\n    DataframeSchema\n        A mapping that contains the detected data type for the index and columns.\n        The key is the column name in the underlying dataframe or ``_index`` for index columns.\n    '
    dataframe_schema: DataframeSchema = {}
    dataframe_schema[INDEX_IDENTIFIER] = _determine_data_kind(data_df.index)
    for (i, column) in enumerate(data_df.items()):
        (column_name, column_data) = column
        dataframe_schema[column_name] = _determine_data_kind(column_data, arrow_schema.field(i))
    return dataframe_schema
ColumnConfigMapping: TypeAlias = Dict[Union[IndexIdentifierType, str], ColumnConfig]
ColumnConfigMappingInput: TypeAlias = Mapping[Union[IndexIdentifierType, str], Union[ColumnConfig, None, str]]

def process_config_mapping(column_config: ColumnConfigMappingInput | None=None) -> ColumnConfigMapping:
    if False:
        return 10
    'Transforms a user-provided column config mapping into a valid column config mapping\n    that can be used by the frontend.\n\n    Parameters\n    ----------\n    column_config: dict or None\n        The user-provided column config mapping.\n\n    Returns\n    -------\n    dict\n        The transformed column config mapping.\n    '
    if column_config is None:
        return {}
    transformed_column_config: ColumnConfigMapping = {}
    for (column, config) in column_config.items():
        if config is None:
            transformed_column_config[column] = ColumnConfig(hidden=True)
        elif isinstance(config, str):
            transformed_column_config[column] = ColumnConfig(label=config)
        elif isinstance(config, dict):
            transformed_column_config[column] = config
        else:
            raise StreamlitAPIException(f'Invalid column config for column `{column}`. Expected `None`, `str` or `dict`, but got `{type(config)}`.')
    return transformed_column_config

def update_column_config(column_config_mapping: ColumnConfigMapping, column: str, column_config: ColumnConfig) -> None:
    if False:
        while True:
            i = 10
    'Updates the column config value for a single column within the mapping.\n\n    Parameters\n    ----------\n\n    column_config_mapping : ColumnConfigMapping\n        The column config mapping to update.\n\n    column : str\n        The column to update the config value for.\n\n    column_config : ColumnConfig\n        The column config to update.\n    '
    if column not in column_config_mapping:
        column_config_mapping[column] = {}
    column_config_mapping[column].update(column_config)

def apply_data_specific_configs(columns_config: ColumnConfigMapping, data_df: pd.DataFrame, data_format: DataFormat, check_arrow_compatibility: bool=False) -> None:
    if False:
        print('Hello World!')
    'Apply data specific configurations to the provided dataframe.\n\n    This will apply inplace changes to the dataframe and the column configurations\n    depending on the data format.\n\n    Parameters\n    ----------\n    columns_config : ColumnConfigMapping\n        A mapping of column names/ids to column configurations.\n\n    data_df : pd.DataFrame\n        The dataframe to apply the configurations to.\n\n    data_format : DataFormat\n        The format of the data.\n\n    check_arrow_compatibility : bool\n        Whether to check if the data is compatible with arrow.\n    '
    if check_arrow_compatibility:
        for (column_name, column_data) in data_df.items():
            if is_colum_type_arrow_incompatible(column_data):
                update_column_config(columns_config, column_name, {'disabled': True})
                data_df[column_name] = column_data.astype('string')
    if data_format in [DataFormat.SET_OF_VALUES, DataFormat.TUPLE_OF_VALUES, DataFormat.LIST_OF_VALUES, DataFormat.NUMPY_LIST, DataFormat.NUMPY_MATRIX, DataFormat.LIST_OF_RECORDS, DataFormat.LIST_OF_ROWS, DataFormat.COLUMN_VALUE_MAPPING]:
        update_column_config(columns_config, INDEX_IDENTIFIER, {'hidden': True})
    if data_format in [DataFormat.SET_OF_VALUES, DataFormat.TUPLE_OF_VALUES, DataFormat.LIST_OF_VALUES, DataFormat.NUMPY_LIST, DataFormat.KEY_VALUE_DICT]:
        data_df.rename(columns={0: 'value'}, inplace=True)
    if not isinstance(data_df.index, pd.RangeIndex):
        update_column_config(columns_config, INDEX_IDENTIFIER, {'required': True})

def marshall_column_config(proto: ArrowProto, column_config_mapping: ColumnConfigMapping) -> None:
    if False:
        while True:
            i = 10
    'Marshall the column config into the Arrow proto.\n\n    Parameters\n    ----------\n    proto : ArrowProto\n        The proto to marshall into.\n\n    column_config_mapping : ColumnConfigMapping\n        The column config to marshall.\n    '
    proto.columns = json.dumps({f'{_NUMERICAL_POSITION_PREFIX}{str(k)}' if isinstance(k, int) else k: v for (k, v) in remove_none_values(column_config_mapping).items()})