import unittest.mock as mock
from textwrap import dedent
import numpy as np
import pandas as pd
from sqlalchemy.types import NVARCHAR
from superset.db_engine_specs.redshift import RedshiftEngineSpec
from superset.errors import ErrorLevel, SupersetError, SupersetErrorType
from superset.sql_parse import Table
from tests.integration_tests.db_engine_specs.base_tests import TestDbEngineSpec
from tests.integration_tests.test_app import app

class TestRedshiftDbEngineSpec(TestDbEngineSpec):

    def test_extract_errors(self):
        if False:
            print('Hello World!')
        '\n        Test that custom error messages are extracted correctly.\n        '
        msg = 'FATAL:  password authentication failed for user "wronguser"'
        result = RedshiftEngineSpec.extract_errors(Exception(msg))
        assert result == [SupersetError(error_type=SupersetErrorType.CONNECTION_ACCESS_DENIED_ERROR, message='Either the username "wronguser" or the password is incorrect.', level=ErrorLevel.ERROR, extra={'invalid': ['username', 'password'], 'engine_name': 'Amazon Redshift', 'issue_codes': [{'code': 1014, 'message': 'Issue 1014 - Either the username or the password is wrong.'}, {'code': 1015, 'message': 'Issue 1015 - Either the database is spelled incorrectly or does not exist.'}]})]
        msg = 'redshift: error: could not translate host name "badhost" to address: nodename nor servname provided, or not known'
        result = RedshiftEngineSpec.extract_errors(Exception(msg))
        assert result == [SupersetError(error_type=SupersetErrorType.CONNECTION_INVALID_HOSTNAME_ERROR, message='The hostname "badhost" cannot be resolved.', level=ErrorLevel.ERROR, extra={'invalid': ['host'], 'engine_name': 'Amazon Redshift', 'issue_codes': [{'code': 1007, 'message': "Issue 1007 - The hostname provided can't be resolved."}]})]
        msg = dedent('\npsql: error: could not connect to server: Connection refused\n        Is the server running on host "localhost" (::1) and accepting\n        TCP/IP connections on port 12345?\ncould not connect to server: Connection refused\n        Is the server running on host "localhost" (127.0.0.1) and accepting\n        TCP/IP connections on port 12345?\n            ')
        result = RedshiftEngineSpec.extract_errors(Exception(msg))
        assert result == [SupersetError(error_type=SupersetErrorType.CONNECTION_PORT_CLOSED_ERROR, message='Port 12345 on hostname "localhost" refused the connection.', level=ErrorLevel.ERROR, extra={'invalid': ['host', 'port'], 'engine_name': 'Amazon Redshift', 'issue_codes': [{'code': 1008, 'message': 'Issue 1008 - The port is closed.'}]})]
        msg = dedent('\npsql: error: could not connect to server: Operation timed out\n        Is the server running on host "example.com" (93.184.216.34) and accepting\n        TCP/IP connections on port 12345?\n            ')
        result = RedshiftEngineSpec.extract_errors(Exception(msg))
        assert result == [SupersetError(error_type=SupersetErrorType.CONNECTION_HOST_DOWN_ERROR, message='The host "example.com" might be down, and can\'t be reached on port 12345.', level=ErrorLevel.ERROR, extra={'engine_name': 'Amazon Redshift', 'issue_codes': [{'code': 1009, 'message': "Issue 1009 - The host might be down, and can't be reached on the provided port."}], 'invalid': ['host', 'port']})]
        msg = dedent('\npsql: error: could not connect to server: Operation timed out\n        Is the server running on host "93.184.216.34" and accepting\n        TCP/IP connections on port 12345?\n            ')
        result = RedshiftEngineSpec.extract_errors(Exception(msg))
        assert result == [SupersetError(error_type=SupersetErrorType.CONNECTION_HOST_DOWN_ERROR, message='The host "93.184.216.34" might be down, and can\'t be reached on port 12345.', level=ErrorLevel.ERROR, extra={'engine_name': 'Amazon Redshift', 'issue_codes': [{'code': 1009, 'message': "Issue 1009 - The host might be down, and can't be reached on the provided port."}], 'invalid': ['host', 'port']})]
        msg = 'database "badDB" does not exist'
        result = RedshiftEngineSpec.extract_errors(Exception(msg))
        assert result == [SupersetError(error_type=SupersetErrorType.CONNECTION_UNKNOWN_DATABASE_ERROR, message='We were unable to connect to your database named "badDB". Please verify your database name and try again.', level=ErrorLevel.ERROR, extra={'engine_name': 'Amazon Redshift', 'issue_codes': [{'code': 10015, 'message': 'Issue 1015 - Either the database is spelled incorrectly or does not exist.'}], 'invalid': ['database']})]

    def test_df_to_sql_no_dtype(self):
        if False:
            while True:
                i = 10
        mock_database = mock.MagicMock()
        mock_database.get_df.return_value.empty = False
        table_name = 'foobar'
        data = [('foo', 'bar', pd.NA, None), ('foo', 'bar', pd.NA, True), ('foo', 'bar', pd.NA, None)]
        numpy_dtype = [('id', 'object'), ('value', 'object'), ('num', 'object'), ('bool', 'object')]
        column_names = ['id', 'value', 'num', 'bool']
        test_array = np.array(data, dtype=numpy_dtype)
        df = pd.DataFrame(test_array, columns=column_names)
        df.to_sql = mock.MagicMock()
        with app.app_context():
            RedshiftEngineSpec.df_to_sql(mock_database, Table(table=table_name), df, to_sql_kwargs={})
        assert df.to_sql.call_args[1]['dtype'] == {}

    def test_df_to_sql_with_string_dtype(self):
        if False:
            while True:
                i = 10
        mock_database = mock.MagicMock()
        mock_database.get_df.return_value.empty = False
        table_name = 'foobar'
        data = [('foo', 'bar', pd.NA, None), ('foo', 'bar', pd.NA, True), ('foo', 'bar', pd.NA, None)]
        column_names = ['id', 'value', 'num', 'bool']
        df = pd.DataFrame(data, columns=column_names)
        df = df.astype(dtype={'value': 'string'})
        df.to_sql = mock.MagicMock()
        with app.app_context():
            RedshiftEngineSpec.df_to_sql(mock_database, Table(table=table_name), df, to_sql_kwargs={})
        dtype = df.to_sql.call_args[1]['dtype']
        assert isinstance(dtype['value'], NVARCHAR)
        assert dtype['value'].length == 65535