from __future__ import annotations
from unittest.mock import Mock, PropertyMock, patch
import pytest
try:
    import pymssql
    from airflow.providers.apache.hive.transfers.mssql_to_hive import MsSqlToHiveOperator
except ImportError:
    pytest.skip('MSSQL not available', allow_module_level=True)

@pytest.mark.backend('mssql')
class TestMsSqlToHiveTransfer:

    def setup_method(self):
        if False:
            return 10
        self.kwargs = dict(sql='sql', hive_table='table', task_id='test_mssql_to_hive', dag=None)

    def test_type_map_binary(self):
        if False:
            while True:
                i = 10
        mapped_type = MsSqlToHiveOperator(**self.kwargs).type_map(pymssql.BINARY.value)
        assert mapped_type == 'INT'

    def test_type_map_decimal(self):
        if False:
            for i in range(10):
                print('nop')
        mapped_type = MsSqlToHiveOperator(**self.kwargs).type_map(pymssql.DECIMAL.value)
        assert mapped_type == 'FLOAT'

    def test_type_map_number(self):
        if False:
            i = 10
            return i + 15
        mapped_type = MsSqlToHiveOperator(**self.kwargs).type_map(pymssql.NUMBER.value)
        assert mapped_type == 'INT'

    def test_type_map_string(self):
        if False:
            return 10
        mapped_type = MsSqlToHiveOperator(**self.kwargs).type_map(None)
        assert mapped_type == 'STRING'

    @patch('airflow.providers.apache.hive.transfers.mssql_to_hive.csv')
    @patch('airflow.providers.apache.hive.transfers.mssql_to_hive.NamedTemporaryFile')
    @patch('airflow.providers.apache.hive.transfers.mssql_to_hive.MsSqlHook')
    @patch('airflow.providers.apache.hive.transfers.mssql_to_hive.HiveCliHook')
    def test_execute(self, mock_hive_hook, mock_mssql_hook, mock_tmp_file, mock_csv):
        if False:
            i = 10
            return i + 15
        type(mock_tmp_file).name = PropertyMock(return_value='tmp_file')
        mock_tmp_file.return_value.__enter__ = Mock(return_value=mock_tmp_file)
        mock_mssql_hook_get_conn = mock_mssql_hook.return_value.get_conn.return_value.__enter__
        mock_mssql_hook_cursor = mock_mssql_hook_get_conn.return_value.cursor.return_value.__enter__
        mock_mssql_hook_cursor.return_value.description = [('anything', 'some-other-thing')]
        mssql_to_hive_transfer = MsSqlToHiveOperator(**self.kwargs)
        mssql_to_hive_transfer.execute(context={})
        mock_mssql_hook_cursor.return_value.execute.assert_called_once_with(mssql_to_hive_transfer.sql)
        mock_tmp_file.assert_called_with(mode='w', encoding='utf-8')
        mock_csv.writer.assert_called_once_with(mock_tmp_file, delimiter=mssql_to_hive_transfer.delimiter)
        field_dict = {}
        for field in mock_mssql_hook_cursor.return_value.description:
            field_dict[field[0]] = mssql_to_hive_transfer.type_map(field[1])
        mock_csv.writer.return_value.writerows.assert_called_once_with(mock_mssql_hook_cursor.return_value)
        mock_hive_hook.return_value.load_file.assert_called_once_with(mock_tmp_file.name, mssql_to_hive_transfer.hive_table, field_dict=field_dict, create=mssql_to_hive_transfer.create, partition=mssql_to_hive_transfer.partition, delimiter=mssql_to_hive_transfer.delimiter, recreate=mssql_to_hive_transfer.recreate, tblproperties=mssql_to_hive_transfer.tblproperties)

    @patch('airflow.providers.apache.hive.transfers.mssql_to_hive.csv')
    @patch('airflow.providers.apache.hive.transfers.mssql_to_hive.NamedTemporaryFile')
    @patch('airflow.providers.apache.hive.transfers.mssql_to_hive.MsSqlHook')
    @patch('airflow.providers.apache.hive.transfers.mssql_to_hive.HiveCliHook')
    def test_execute_empty_description_field(self, mock_hive_hook, mock_mssql_hook, mock_tmp_file, mock_csv):
        if False:
            while True:
                i = 10
        type(mock_tmp_file).name = PropertyMock(return_value='tmp_file')
        mock_tmp_file.return_value.__enter__ = Mock(return_value=mock_tmp_file)
        mock_mssql_hook_get_conn = mock_mssql_hook.return_value.get_conn.return_value.__enter__
        mock_mssql_hook_cursor = mock_mssql_hook_get_conn.return_value.cursor.return_value.__enter__
        mock_mssql_hook_cursor.return_value.description = [('', '')]
        mssql_to_hive_transfer = MsSqlToHiveOperator(**self.kwargs)
        mssql_to_hive_transfer.execute(context={})
        field_dict = {}
        for (col_count, field) in enumerate(mock_mssql_hook_cursor.return_value.description, start=1):
            col_position = f'Column{col_count}'
            field_dict[col_position] = mssql_to_hive_transfer.type_map(field[1])
        mock_hive_hook.return_value.load_file.assert_called_once_with(mock_tmp_file.name, mssql_to_hive_transfer.hive_table, field_dict=field_dict, create=mssql_to_hive_transfer.create, partition=mssql_to_hive_transfer.partition, delimiter=mssql_to_hive_transfer.delimiter, recreate=mssql_to_hive_transfer.recreate, tblproperties=mssql_to_hive_transfer.tblproperties)