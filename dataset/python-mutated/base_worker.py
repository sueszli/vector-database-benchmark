"""Module provides ``BaseDbWorker`` class."""
import abc
import uuid
from typing import List, Tuple
import numpy as np
import pyarrow as pa
from modin.error_message import ErrorMessage
_UINT_TO_INT_MAP = {pa.uint8(): pa.int16(), pa.uint16(): pa.int32(), pa.uint32(): pa.int64(), pa.uint64(): pa.int64()}

class DbTable(abc.ABC):
    """
    Base class, representing a table in the HDK database.

    Attributes
    ----------
    name : str
        Table name.
    """

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, int]:
        if False:
            i = 10
            return i + 15
        '\n        Return a tuple with the number of rows and columns.\n\n        Returns\n        -------\n        tuple of int\n        '
        pass

    @property
    @abc.abstractmethod
    def column_names(self) -> List[str]:
        if False:
            i = 10
            return i + 15
        '\n        Return a list of the table column names.\n\n        Returns\n        -------\n        tuple of str\n        '
        pass

    @abc.abstractmethod
    def to_arrow(self) -> pa.Table:
        if False:
            for i in range(10):
                print('nop')
        '\n        Convert this table to arrow.\n\n        Returns\n        -------\n        pyarrow.Table\n        '
        pass

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        Return the number of rows in the table.\n\n        Returns\n        -------\n        int\n        '
        return self.shape[0]

class BaseDbWorker(abc.ABC):
    """Base class for HDK storage format based execution engine ."""

    @classmethod
    @abc.abstractmethod
    def dropTable(cls, name):
        if False:
            print('Hello World!')
        '\n        Drops table with the specified name.\n\n        Parameters\n        ----------\n        name : str\n            A table to drop.\n        '
        pass

    @classmethod
    @abc.abstractmethod
    def executeDML(cls, query):
        if False:
            while True:
                i = 10
        '\n        Execute DML SQL query.\n\n        Parameters\n        ----------\n        query : str\n            SQL query.\n\n        Returns\n        -------\n        DbTable\n            Execution result.\n        '
        pass

    @classmethod
    @abc.abstractmethod
    def executeRA(cls, query):
        if False:
            print('Hello World!')
        '\n        Execute calcite query.\n\n        Parameters\n        ----------\n        query : str\n            Serialized calcite query.\n\n        Returns\n        -------\n        DbTable\n            Execution result.\n        '
        pass

    @classmethod
    def _genName(cls, name):
        if False:
            while True:
                i = 10
        '\n        Generate or mangle a table name.\n\n        Parameters\n        ----------\n        name : str or None\n            Table name to mangle or None to generate a unique\n            table name.\n\n        Returns\n        -------\n        str\n            Table name.\n        '
        if not name:
            name = 'frame_' + str(uuid.uuid4()).replace('-', '')
        return name

    @classmethod
    def cast_to_compatible_types(cls, table, cast_dict):
        if False:
            print('Hello World!')
        '\n        Cast PyArrow table to be fully compatible with HDK.\n\n        Parameters\n        ----------\n        table : pyarrow.Table\n            Source table.\n        cast_dict : bool\n            Cast dictionary columns to string.\n\n        Returns\n        -------\n        pyarrow.Table\n            Table with fully compatible types with HDK.\n        '
        schema = table.schema
        new_schema = schema
        need_cast = False
        uint_to_int_cast = False
        for (i, field) in enumerate(schema):
            if pa.types.is_dictionary(field.type):
                value_type = field.type.value_type
                if pa.types.is_null(value_type):
                    mask = np.full(table.num_rows, True, dtype=bool)
                    new_col_data = np.empty(table.num_rows, dtype=str)
                    new_col = pa.array(new_col_data, pa.string(), mask)
                    new_field = pa.field(field.name, pa.string(), field.nullable, field.metadata)
                    table = table.set_column(i, new_field, new_col)
                elif pa.types.is_string(value_type):
                    if cast_dict:
                        need_cast = True
                        new_field = pa.field(field.name, pa.string(), field.nullable, field.metadata)
                    else:
                        new_field = field
                else:
                    (new_field, int_cast) = cls._convert_field(field, value_type)
                    need_cast = True
                    uint_to_int_cast = uint_to_int_cast or int_cast
                    if new_field == field:
                        new_field = pa.field(field.name, value_type, field.nullable, field.metadata)
                new_schema = new_schema.set(i, new_field)
            else:
                (new_field, int_cast) = cls._convert_field(field, field.type)
                need_cast = need_cast or new_field is not field
                uint_to_int_cast = uint_to_int_cast or int_cast
                new_schema = new_schema.set(i, new_field)
        if uint_to_int_cast:
            ErrorMessage.single_warning('HDK does not support unsigned integer types, such types will be rounded up to the signed equivalent.')
        if need_cast:
            try:
                table = table.cast(new_schema)
            except pa.lib.ArrowInvalid as err:
                raise (OverflowError if uint_to_int_cast else RuntimeError)("An error occurred when trying to convert unsupported by HDK 'dtypes' " + f'to the supported ones, the schema to cast was: \n{new_schema}.') from err
        return table

    @staticmethod
    def _convert_field(field, field_type):
        if False:
            print('Hello World!')
        '\n        Convert the specified arrow field, if required.\n\n        Parameters\n        ----------\n        field : pyarrow.Field\n        field_type : pyarrow.DataType\n\n        Returns\n        -------\n        Tuple[pyarrow.Field, boolean]\n            A tuple, containing (new_field, uint_to_int_cast)\n        '
        if pa.types.is_date(field_type):
            return (pa.field(field.name, pa.timestamp('s'), field.nullable, field.metadata), False)
        elif pa.types.is_unsigned_integer(field_type):
            return (pa.field(field.name, _UINT_TO_INT_MAP[field_type], field.nullable, field.metadata), True)
        return (field, False)

    @classmethod
    @abc.abstractmethod
    def import_arrow_table(cls, table, name=None):
        if False:
            i = 10
            return i + 15
        '\n        Import Arrow table to the worker.\n\n        Parameters\n        ----------\n        table : pyarrow.Table\n            A table to import.\n        name : str, optional\n            A table name to use. None to generate a unique name.\n\n        Returns\n        -------\n        DbTable\n            Imported table.\n        '
        pass

    @classmethod
    def import_pandas_dataframe(cls, df, name=None):
        if False:
            print('Hello World!')
        '\n        Import ``pandas.DataFrame`` to the worker.\n\n        Parameters\n        ----------\n        df : pandas.DataFrame\n            A frame to import.\n        name : str, optional\n            A table name to use. None to generate a unique name.\n\n        Returns\n        -------\n        DbTable\n            Imported table.\n        '
        return cls.import_arrow_table(pa.Table.from_pandas(df), name=name)