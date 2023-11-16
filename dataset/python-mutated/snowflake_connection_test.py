import threading
import unittest
from unittest.mock import MagicMock, PropertyMock, patch
import pytest
import streamlit as st
from streamlit.connections import SnowflakeConnection
from streamlit.errors import StreamlitAPIException
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.secrets import AttrDict
from tests.testutil import create_mock_script_run_ctx

@pytest.mark.require_snowflake
class SnowflakeConnectionTest(unittest.TestCase):

    def tearDown(self) -> None:
        if False:
            while True:
                i = 10
        st.cache_data.clear()

    @patch('snowflake.snowpark.context.get_active_session')
    @patch('streamlit.connections.snowflake_connection.running_in_sis', MagicMock(return_value=True))
    def test_uses_active_session_if_in_sis(self, patched_get_active_session):
        if False:
            for i in range(10):
                print('nop')
        active_session_mock = MagicMock()
        active_session_mock.connection = 'some active session'
        patched_get_active_session.return_value = active_session_mock
        conn = SnowflakeConnection('my_snowflake_connection')
        assert conn._instance == 'some active session'

    @patch('streamlit.connections.snowflake_connection.SnowflakeConnection._secrets', PropertyMock(return_value=AttrDict({'account': 'some_val_1', 'some_key': 'some_val_2'})))
    @patch('snowflake.connector.connect')
    def test_uses_streamlit_secrets_if_available(self, patched_connect):
        if False:
            for i in range(10):
                print('nop')
        SnowflakeConnection('my_snowflake_connection')
        patched_connect.assert_called_once_with(account='some_val_1', some_key='some_val_2')

    @patch('snowflake.connector.connect')
    def test_uses_config_manager_if_available(self, patched_connect):
        if False:
            i = 10
            return i + 15
        SnowflakeConnection('snowflake', some_kwarg='some_value')
        patched_connect.assert_called_once_with(connection_name='default', some_kwarg='some_value')

    @patch('snowflake.connector.connection')
    @patch('snowflake.connector.connect')
    def test_falls_back_to_using_kwargs_last(self, patched_connect, patched_connection):
        if False:
            return 10
        delattr(patched_connection, 'CONFIG_MANAGER')
        SnowflakeConnection('snowflake', account='account', some_kwarg='some_value')
        patched_connect.assert_called_once_with(account='account', some_kwarg='some_value')

    def test_throws_friendly_error_if_no_config_set(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(StreamlitAPIException) as e:
            SnowflakeConnection('snowflake')
        assert 'Missing Snowflake connection configuration.' in str(e.value)

    @patch('streamlit.connections.snowflake_connection.SnowflakeConnection._connect', MagicMock())
    def test_query_caches_value(self):
        if False:
            for i in range(10):
                print('nop')
        add_script_run_ctx(threading.current_thread(), create_mock_script_run_ctx())
        mock_cursor = MagicMock()
        mock_cursor.fetch_pandas_all = MagicMock(return_value='i am a dataframe')
        conn = SnowflakeConnection('my_snowflake_connection')
        conn._instance.cursor.return_value = mock_cursor
        assert conn.query('SELECT 1;') == 'i am a dataframe'
        assert conn.query('SELECT 1;') == 'i am a dataframe'
        conn._instance.cursor.assert_called_once()
        mock_cursor.execute.assert_called_once_with('SELECT 1;', params=None)

    @patch('streamlit.connections.snowflake_connection.SnowflakeConnection._connect', MagicMock())
    def test_retry_behavior(self):
        if False:
            print('Hello World!')
        from snowflake.connector.errors import ProgrammingError
        from snowflake.connector.network import MASTER_TOKEN_EXPIRED_GS_CODE
        mock_cursor = MagicMock()
        mock_cursor.fetch_pandas_all = MagicMock(side_effect=ProgrammingError('oh noes :(', errno=int(MASTER_TOKEN_EXPIRED_GS_CODE)))
        conn = SnowflakeConnection('my_snowflake_connection')
        conn._instance.cursor.return_value = mock_cursor
        with patch.object(conn, 'reset', wraps=conn.reset) as wrapped_reset:
            with pytest.raises(ProgrammingError):
                conn.query('SELECT 1;')
            assert wrapped_reset.call_count == 3
        assert conn._connect.call_count == 3

    @patch('streamlit.connections.snowflake_connection.SnowflakeConnection._connect', MagicMock())
    def test_retry_fails_fast_for_programming_errors_with_wrong_code(self):
        if False:
            return 10
        from snowflake.connector.errors import ProgrammingError
        mock_cursor = MagicMock()
        mock_cursor.fetch_pandas_all = MagicMock(side_effect=ProgrammingError('oh noes :(', errno=42))
        conn = SnowflakeConnection('my_snowflake_connection')
        conn._instance.cursor.return_value = mock_cursor
        with pytest.raises(ProgrammingError):
            conn.query('SELECT 1;')
        assert conn._connect.call_count == 1

    @patch('streamlit.connections.snowflake_connection.SnowflakeConnection._connect', MagicMock())
    def test_retry_fails_fast_for_general_snowflake_errors(self):
        if False:
            while True:
                i = 10
        from snowflake.connector.errors import Error as SnowflakeError
        mock_cursor = MagicMock()
        mock_cursor.fetch_pandas_all = MagicMock(side_effect=SnowflakeError('oh noes :('))
        conn = SnowflakeConnection('my_snowflake_connection')
        conn._instance.cursor.return_value = mock_cursor
        with pytest.raises(SnowflakeError):
            conn.query('SELECT 1;')
        assert conn._connect.call_count == 1

    @patch('streamlit.connections.snowflake_connection.SnowflakeConnection._connect', MagicMock())
    def test_retry_fails_fast_for_other_errors(self):
        if False:
            i = 10
            return i + 15
        mock_cursor = MagicMock()
        mock_cursor.fetch_pandas_all = MagicMock(side_effect=Exception('oh noes :('))
        conn = SnowflakeConnection('my_snowflake_connection')
        conn._instance.cursor.return_value = mock_cursor
        with pytest.raises(Exception):
            conn.query('SELECT 1;')
        assert conn._connect.call_count == 1