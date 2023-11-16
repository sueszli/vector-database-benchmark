from typing import List, Optional, Union
from pyflink.java_gateway import get_gateway
from pyflink.table.types import DataType, RowType, _to_java_data_type, _from_java_data_type
from pyflink.util.java_utils import to_jarray
__all__ = ['TableSchema']

class TableSchema(object):
    """
    A table schema that represents a table's structure with field names and data types.
    """

    def __init__(self, field_names: List[str]=None, data_types: List[DataType]=None, j_table_schema=None):
        if False:
            i = 10
            return i + 15
        if j_table_schema is None:
            gateway = get_gateway()
            j_field_names = to_jarray(gateway.jvm.String, field_names)
            j_data_types = to_jarray(gateway.jvm.DataType, [_to_java_data_type(item) for item in data_types])
            self._j_table_schema = gateway.jvm.TableSchema.builder().fields(j_field_names, j_data_types).build()
        else:
            self._j_table_schema = j_table_schema

    def copy(self) -> 'TableSchema':
        if False:
            return 10
        '\n        Returns a deep copy of the table schema.\n\n        :return: A deep copy of the table schema.\n        '
        return TableSchema(j_table_schema=self._j_table_schema.copy())

    def get_field_data_types(self) -> List[DataType]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns all field data types as a list.\n\n        :return: A list of all field data types.\n        '
        return [_from_java_data_type(item) for item in self._j_table_schema.getFieldDataTypes()]

    def get_field_data_type(self, field: Union[int, str]) -> Optional[DataType]:
        if False:
            print('Hello World!')
        '\n        Returns the specified data type for the given field index or field name.\n\n        :param field: The index of the field or the name of the field.\n        :return: The data type of the specified field.\n        '
        if not isinstance(field, (int, str)):
            raise TypeError('Expected field index or field name, got %s' % type(field))
        optional_result = self._j_table_schema.getFieldDataType(field)
        if optional_result.isPresent():
            return _from_java_data_type(optional_result.get())
        else:
            return None

    def get_field_count(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Returns the number of fields.\n\n        :return: The number of fields.\n        '
        return self._j_table_schema.getFieldCount()

    def get_field_names(self) -> List[str]:
        if False:
            while True:
                i = 10
        '\n        Returns all field names as a list.\n\n        :return: The list of all field names.\n        '
        return list(self._j_table_schema.getFieldNames())

    def get_field_name(self, field_index: int) -> Optional[str]:
        if False:
            while True:
                i = 10
        '\n        Returns the specified name for the given field index.\n\n        :param field_index: The index of the field.\n        :return: The field name.\n        '
        optional_result = self._j_table_schema.getFieldName(field_index)
        if optional_result.isPresent():
            return optional_result.get()
        else:
            return None

    def to_row_data_type(self) -> RowType:
        if False:
            i = 10
            return i + 15
        '\n        Converts a table schema into a (nested) data type describing a\n        :func:`pyflink.table.types.DataTypes.ROW`.\n\n        :return: The row data type.\n        '
        return _from_java_data_type(self._j_table_schema.toRowDataType())

    def __repr__(self):
        if False:
            print('Hello World!')
        return self._j_table_schema.toString()

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return isinstance(other, self.__class__) and self._j_table_schema == other._j_table_schema

    def __hash__(self):
        if False:
            return 10
        return self._j_table_schema.hashCode()

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return not self.__eq__(other)

    @classmethod
    def builder(cls):
        if False:
            return 10
        return TableSchema.Builder()

    class Builder(object):
        """
        Builder for creating a :class:`TableSchema`.
        """

        def __init__(self):
            if False:
                print('Hello World!')
            self._field_names = []
            self._field_data_types = []

        def field(self, name: str, data_type: DataType) -> 'TableSchema.Builder':
            if False:
                i = 10
                return i + 15
            '\n            Add a field with name and data type.\n\n            The call order of this method determines the order of fields in the schema.\n\n            :param name: The field name.\n            :param data_type: The field data type.\n            :return: This object.\n            '
            assert name is not None
            assert data_type is not None
            self._field_names.append(name)
            self._field_data_types.append(data_type)
            return self

        def build(self) -> 'TableSchema':
            if False:
                i = 10
                return i + 15
            '\n            Returns a :class:`TableSchema` instance.\n\n            :return: The :class:`TableSchema` instance.\n            '
            return TableSchema(self._field_names, self._field_data_types)