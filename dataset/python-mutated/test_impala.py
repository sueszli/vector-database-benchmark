from __future__ import annotations
from unittest.mock import MagicMock, patch
import pytest
from airflow.models import Connection
from airflow.providers.apache.impala.hooks.impala import ImpalaHook

@pytest.fixture()
def impala_hook_fixture() -> ImpalaHook:
    if False:
        while True:
            i = 10
    hook = ImpalaHook()
    mock_get_conn = MagicMock()
    mock_get_conn.return_value.cursor = MagicMock()
    mock_get_conn.return_value.cursor.return_value.rowcount = 2
    hook.get_conn = mock_get_conn
    return hook

@patch('airflow.providers.apache.impala.hooks.impala.connect', autospec=True)
def test_get_conn(mock_connect):
    if False:
        i = 10
        return i + 15
    hook = ImpalaHook()
    hook.get_connection = MagicMock(return_value=Connection(login='login', password='password', host='host', port=21050, schema='test', extra={'use_ssl': True}))
    hook.get_conn()
    mock_connect.assert_called_once_with(host='host', port=21050, user='login', password='password', database='test', use_ssl=True)

@patch('airflow.providers.apache.impala.hooks.impala.connect', autospec=True)
def test_get_conn_kerberos(mock_connect):
    if False:
        return 10
    hook = ImpalaHook()
    hook.get_connection = MagicMock(return_value=Connection(login='login', password='password', host='host', port=21050, schema='test', extra={'auth_mechanism': 'GSSAPI', 'use_ssl': True}))
    hook.get_conn()
    mock_connect.assert_called_once_with(host='host', port=21050, user='login', password='password', database='test', use_ssl=True, auth_mechanism='GSSAPI')

@patch('airflow.providers.common.sql.hooks.sql.DbApiHook.insert_rows')
def test_insert_rows(mock_insert_rows, impala_hook_fixture):
    if False:
        return 10
    table = 'table'
    rows = [('hello',), ('world',)]
    target_fields = None
    commit_every = 10
    impala_hook_fixture.insert_rows(table, rows, target_fields, commit_every)
    mock_insert_rows.assert_called_once_with(table, rows, None, 10)

def test_get_first_record(impala_hook_fixture):
    if False:
        return 10
    statement = 'SQL'
    result_sets = [('row1',), ('row2',)]
    impala_hook_fixture.get_conn.return_value.cursor.return_value.fetchone.return_value = result_sets[0]
    assert result_sets[0] == impala_hook_fixture.get_first(statement)
    impala_hook_fixture.get_conn.return_value.cursor.return_value.execute.assert_called_once_with(statement)

def test_get_records(impala_hook_fixture):
    if False:
        print('Hello World!')
    statement = 'SQL'
    result_sets = [('row1',), ('row2',)]
    impala_hook_fixture.get_conn.return_value.cursor.return_value.fetchall.return_value = result_sets
    assert result_sets == impala_hook_fixture.get_records(statement)
    impala_hook_fixture.get_conn.return_value.cursor.return_value.execute.assert_called_once_with(statement)

def test_get_pandas_df(impala_hook_fixture):
    if False:
        print('Hello World!')
    statement = 'SQL'
    column = 'col'
    result_sets = [('row1',), ('row2',)]
    impala_hook_fixture.get_conn.return_value.cursor.return_value.description = [(column,)]
    impala_hook_fixture.get_conn.return_value.cursor.return_value.fetchall.return_value = result_sets
    df = impala_hook_fixture.get_pandas_df(statement)
    assert column == df.columns[0]
    assert result_sets[0][0] == df.values.tolist()[0][0]
    assert result_sets[1][0] == df.values.tolist()[1][0]
    impala_hook_fixture.get_conn.return_value.cursor.return_value.execute.assert_called_once_with(statement)