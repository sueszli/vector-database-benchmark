from __future__ import annotations
from unittest.mock import Mock
import pytest
from airflow.providers.pgvector.hooks.pgvector import PgVectorHook

@pytest.fixture
def pg_vector_hook():
    if False:
        return 10
    return PgVectorHook(postgres_conn_id='your_postgres_conn_id')

def test_create_table(pg_vector_hook):
    if False:
        return 10
    pg_vector_hook.run = Mock()
    table_name = 'my_table'
    columns = ['id SERIAL PRIMARY KEY', 'name VARCHAR(255)', 'value INTEGER']
    pg_vector_hook.create_table(table_name, columns, if_not_exists=True)
    pg_vector_hook.run.assert_called_with('CREATE TABLE IF NOT EXISTS my_table (id SERIAL PRIMARY KEY, name VARCHAR(255), value INTEGER)')

def test_create_extension(pg_vector_hook):
    if False:
        i = 10
        return i + 15
    pg_vector_hook.run = Mock()
    extension_name = 'timescaledb'
    pg_vector_hook.create_extension(extension_name, if_not_exists=True)
    pg_vector_hook.run.assert_called_with('CREATE EXTENSION IF NOT EXISTS timescaledb')

def test_drop_table(pg_vector_hook):
    if False:
        print('Hello World!')
    pg_vector_hook.run = Mock()
    table_name = 'my_table'
    pg_vector_hook.drop_table(table_name, if_exists=True)
    pg_vector_hook.run.assert_called_with('DROP TABLE IF EXISTS my_table')

def test_truncate_table(pg_vector_hook):
    if False:
        while True:
            i = 10
    pg_vector_hook.run = Mock()
    table_name = 'my_table'
    pg_vector_hook.truncate_table(table_name, restart_identity=True)
    pg_vector_hook.run.assert_called_with('TRUNCATE TABLE my_table RESTART IDENTITY')