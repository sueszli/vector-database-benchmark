from __future__ import annotations
from unittest.mock import Mock, patch
from airflow.providers.google.suite.transfers.sql_to_sheets import SQLToGoogleSheetsOperator

class TestSQLToGoogleSheets:
    """
    Test class for SQLToGoogleSheetsOperator
    """

    def setup_method(self):
        if False:
            return 10
        '\n        setup\n        '
        self.gcp_conn_id = 'test'
        self.sql_conn_id = 'test'
        self.sql = 'select 1 as my_col'
        self.spreadsheet_id = '1234567890'
        self.values = [[1, 2, 3]]

    @patch('airflow.providers.google.suite.transfers.sql_to_sheets.GSheetsHook')
    def test_execute(self, mock_sheet_hook):
        if False:
            for i in range(10):
                print('nop')
        op = SQLToGoogleSheetsOperator(task_id='test_task', spreadsheet_id=self.spreadsheet_id, gcp_conn_id=self.gcp_conn_id, sql_conn_id=self.sql_conn_id, sql=self.sql)
        op._get_data = Mock(return_value=self.values)
        op.execute(None)
        mock_sheet_hook.assert_called_once_with(gcp_conn_id=self.gcp_conn_id, delegate_to=None, impersonation_chain=None)
        mock_sheet_hook.return_value.update_values.assert_called_once_with(spreadsheet_id=self.spreadsheet_id, range_='Sheet1', values=self.values)