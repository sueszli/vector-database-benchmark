import threading
import unittest
from unittest.mock import MagicMock, PropertyMock, patch
import pytest
import streamlit as st
from streamlit.connections import SnowparkConnection
from streamlit.errors import StreamlitAPIException
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit.runtime.secrets import AttrDict
from tests.testutil import create_mock_script_run_ctx

@pytest.mark.require_snowflake
class SnowparkConnectionTest(unittest.TestCase):

    def tearDown(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        st.cache_data.clear()

    @patch('snowflake.snowpark.context.get_active_session', MagicMock(return_value='some active session'))
    @patch('streamlit.connections.snowpark_connection.running_in_sis', MagicMock(return_value=True))
    def test_uses_active_session_if_in_sis(self):
        if False:
            for i in range(10):
                print('nop')
        conn = SnowparkConnection('my_snowpark_connection')
        assert conn._instance == 'some active session'

    @patch('streamlit.connections.snowpark_connection.load_from_snowsql_config_file', MagicMock(return_value={'account': 'some_val_1', 'password': 'i get overwritten'}))
    @patch('streamlit.connections.snowpark_connection.SnowparkConnection._secrets', PropertyMock(return_value=AttrDict({'user': 'some_val_2', 'some_key': 'i get overwritten'})))
    @patch('snowflake.snowpark.session.Session')
    def test_merges_params_from_all_config_sources(self, patched_session):
        if False:
            for i in range(10):
                print('nop')
        SnowparkConnection('my_snowpark_connection', some_key='some_val_3', password='hunter2')
        patched_session.builder.configs.assert_called_with({'account': 'some_val_1', 'user': 'some_val_2', 'some_key': 'some_val_3', 'password': 'hunter2'})

    def test_error_if_no_conn_params(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(StreamlitAPIException) as e:
            SnowparkConnection('my_snowpark_connection')
        assert 'Missing Snowpark connection configuration.' in str(e.value)

    def test_error_if_missing_required_conn_params(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(StreamlitAPIException) as e:
            SnowparkConnection('my_snowpark_connection', user='my_user')
        assert 'Missing Snowpark connection param: account' == str(e.value)

    @patch('streamlit.connections.snowpark_connection.SnowparkConnection._connect', MagicMock())
    def test_query_caches_value(self):
        if False:
            print('Hello World!')
        add_script_run_ctx(threading.current_thread(), create_mock_script_run_ctx())
        mock_sql_return = MagicMock()
        mock_sql_return.to_pandas = MagicMock(return_value='i am a dataframe')
        conn = SnowparkConnection('my_snowpark_connection')
        conn._instance.sql.return_value = mock_sql_return
        assert conn.query('SELECT 1;') == 'i am a dataframe'
        assert conn.query('SELECT 1;') == 'i am a dataframe'
        conn._instance.sql.assert_called_once()

    @patch('streamlit.connections.snowpark_connection.SnowparkConnection._connect', MagicMock())
    def test_retry_behavior(self):
        if False:
            return 10
        from snowflake.snowpark.exceptions import SnowparkServerException
        mock_sql_return = MagicMock()
        mock_sql_return.to_pandas = MagicMock(side_effect=SnowparkServerException('oh noes :('))
        conn = SnowparkConnection('my_snowpark_connection')
        conn._instance.sql.return_value = mock_sql_return
        with patch.object(conn, 'reset', wraps=conn.reset) as wrapped_reset:
            with pytest.raises(SnowparkServerException):
                conn.query('SELECT 1;')
            assert wrapped_reset.call_count == 3
        assert conn._connect.call_count == 3

    @patch('streamlit.connections.snowpark_connection.SnowparkConnection._connect', MagicMock())
    def test_retry_fails_fast_for_most_errors(self):
        if False:
            return 10
        mock_sql_return = MagicMock()
        mock_sql_return.to_pandas = MagicMock(side_effect=Exception('oh noes :('))
        conn = SnowparkConnection('my_snowpark_connection')
        conn._instance.sql.return_value = mock_sql_return
        with pytest.raises(Exception):
            conn.query('SELECT 1;')
        assert conn._connect.call_count == 1