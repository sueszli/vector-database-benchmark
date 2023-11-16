""" Superset wrapper around pyarrow.Table.
"""
import datetime
import json
import logging
from typing import Any, Optional
import numpy as np
import pandas as pd
import pyarrow as pa
from numpy.typing import NDArray
from superset.db_engine_specs import BaseEngineSpec
from superset.superset_typing import DbapiDescription, DbapiResult, ResultSetColumnType
from superset.utils import core as utils
logger = logging.getLogger(__name__)

def dedup(l: list[str], suffix: str='__', case_sensitive: bool=True) -> list[str]:
    if False:
        i = 10
        return i + 15
    "De-duplicates a list of string by suffixing a counter\n\n    Always returns the same number of entries as provided, and always returns\n    unique values. Case sensitive comparison by default.\n\n    >>> print(','.join(dedup(['foo', 'bar', 'bar', 'bar', 'Bar'])))\n    foo,bar,bar__1,bar__2,Bar\n    >>> print(\n        ','.join(dedup(['foo', 'bar', 'bar', 'bar', 'Bar'], case_sensitive=False))\n    )\n    foo,bar,bar__1,bar__2,Bar__3\n    "
    new_l: list[str] = []
    seen: dict[str, int] = {}
    for item in l:
        s_fixed_case = item if case_sensitive else item.lower()
        if s_fixed_case in seen:
            seen[s_fixed_case] += 1
            item += suffix + str(seen[s_fixed_case])
        else:
            seen[s_fixed_case] = 0
        new_l.append(item)
    return new_l

def stringify(obj: Any) -> str:
    if False:
        return 10
    return json.dumps(obj, default=utils.json_iso_dttm_ser)

def stringify_values(array: NDArray[Any]) -> NDArray[Any]:
    if False:
        for i in range(10):
            print('nop')
    result = np.copy(array)
    with np.nditer(result, flags=['refs_ok'], op_flags=[['readwrite']]) as it:
        for obj in it:
            if (na_obj := pd.isna(obj)):
                obj[na_obj] = None
            else:
                try:
                    obj[...] = obj.astype(str)
                except ValueError:
                    obj[...] = stringify(obj)
    return result

def destringify(obj: str) -> Any:
    if False:
        for i in range(10):
            print('nop')
    return json.loads(obj)

def convert_to_string(value: Any) -> str:
    if False:
        print('Hello World!')
    '\n    Used to ensure column names from the cursor description are strings.\n    '
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)

class SupersetResultSet:

    def __init__(self, data: DbapiResult, cursor_description: DbapiDescription, db_engine_spec: type[BaseEngineSpec]):
        if False:
            for i in range(10):
                print('nop')
        self.db_engine_spec = db_engine_spec
        data = data or []
        column_names: list[str] = []
        pa_data: list[pa.Array] = []
        deduped_cursor_desc: list[tuple[Any, ...]] = []
        numpy_dtype: list[tuple[str, ...]] = []
        stringified_arr: NDArray[Any]
        if cursor_description:
            column_names = dedup([convert_to_string(col[0]) for col in cursor_description])
            deduped_cursor_desc = [tuple([column_name, *list(description)[1:]]) for (column_name, description) in zip(column_names, cursor_description)]
            numpy_dtype = [(column_name, 'object') for column_name in column_names]
        if data and (not isinstance(data, list) or not isinstance(data[0], tuple)):
            data = [tuple(row) for row in data]
        array = np.array(data, dtype=numpy_dtype)
        if array.size > 0:
            for column in column_names:
                try:
                    pa_data.append(pa.array(array[column].tolist()))
                except (pa.lib.ArrowInvalid, pa.lib.ArrowTypeError, pa.lib.ArrowNotImplementedError, ValueError, TypeError):
                    stringified_arr = stringify_values(array[column])
                    pa_data.append(pa.array(stringified_arr.tolist()))
        if pa_data:
            for (i, column) in enumerate(column_names):
                if pa.types.is_nested(pa_data[i].type):
                    stringified_arr = stringify_values(array[column])
                    pa_data[i] = pa.array(stringified_arr.tolist())
                elif pa.types.is_temporal(pa_data[i].type):
                    sample = self.first_nonempty(array[column])
                    if sample and isinstance(sample, datetime.datetime):
                        try:
                            if sample.tzinfo:
                                tz = sample.tzinfo
                                series = pd.Series(array[column])
                                series = pd.to_datetime(series)
                                pa_data[i] = pa.Array.from_pandas(series, type=pa.timestamp('ns', tz=tz))
                        except Exception as ex:
                            logger.exception(ex)
        if not pa_data:
            column_names = []
        self.table = pa.Table.from_arrays(pa_data, names=column_names)
        self._type_dict: dict[str, Any] = {}
        try:
            self._type_dict = {col: db_engine_spec.get_datatype(deduped_cursor_desc[i][1]) for (i, col) in enumerate(column_names) if deduped_cursor_desc}
        except Exception as ex:
            logger.exception(ex)

    @staticmethod
    def convert_pa_dtype(pa_dtype: pa.DataType) -> Optional[str]:
        if False:
            while True:
                i = 10
        if pa.types.is_boolean(pa_dtype):
            return 'BOOL'
        if pa.types.is_integer(pa_dtype):
            return 'INT'
        if pa.types.is_floating(pa_dtype):
            return 'FLOAT'
        if pa.types.is_string(pa_dtype):
            return 'STRING'
        if pa.types.is_temporal(pa_dtype):
            return 'DATETIME'
        return None

    @staticmethod
    def convert_table_to_df(table: pa.Table) -> pd.DataFrame:
        if False:
            i = 10
            return i + 15
        try:
            return table.to_pandas(integer_object_nulls=True)
        except pa.lib.ArrowInvalid:
            return table.to_pandas(integer_object_nulls=True, timestamp_as_object=True)

    @staticmethod
    def first_nonempty(items: NDArray[Any]) -> Any:
        if False:
            while True:
                i = 10
        return next((i for i in items if i), None)

    def is_temporal(self, db_type_str: Optional[str]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        column_spec = self.db_engine_spec.get_column_spec(db_type_str)
        if column_spec is None:
            return False
        return column_spec.is_dttm

    def data_type(self, col_name: str, pa_dtype: pa.DataType) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        'Given a pyarrow data type, Returns a generic database type'
        if (set_type := self._type_dict.get(col_name)):
            return set_type
        if (mapped_type := self.convert_pa_dtype(pa_dtype)):
            return mapped_type
        return None

    def to_pandas_df(self) -> pd.DataFrame:
        if False:
            while True:
                i = 10
        return self.convert_table_to_df(self.table)

    @property
    def pa_table(self) -> pa.Table:
        if False:
            while True:
                i = 10
        return self.table

    @property
    def size(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.table.num_rows

    @property
    def columns(self) -> list[ResultSetColumnType]:
        if False:
            while True:
                i = 10
        if not self.table.column_names:
            return []
        columns = []
        for col in self.table.schema:
            db_type_str = self.data_type(col.name, col.type)
            column: ResultSetColumnType = {'column_name': col.name, 'name': col.name, 'type': db_type_str, 'is_dttm': self.is_temporal(db_type_str)}
            columns.append(column)
        return columns