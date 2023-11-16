from __future__ import annotations
import os
import uuid
import pytest
import sqlalchemy
from snippets.cloud_sql_connection_pool import init_db, init_tcp_connection_engine, init_unix_connection_engine

@pytest.fixture(name='conn_vars')
def setup() -> dict[str, str]:
    if False:
        print('Hello World!')
    try:
        conn_vars = {}
        conn_vars['db_user'] = os.environ['POSTGRES_USER']
        conn_vars['db_pass'] = os.environ['POSTGRES_PASSWORD']
        conn_vars['db_name'] = os.environ['POSTGRES_DATABASE']
        conn_vars['db_host'] = os.environ['POSTGRES_HOST']
        conn_vars['instance_conn_name'] = os.environ['POSTGRES_INSTANCE']
        conn_vars['db_socket_dir'] = os.getenv('DB_SOCKET_DIR', '/cloudsql')
    except KeyError:
        raise Exception('The following env variables must be set to run these tests:POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DATABASE, POSTGRES_HOST, POSTGRES_INSTANCE')
    else:
        yield conn_vars

def test_init_tcp_connection_engine(conn_vars: dict[str, str]) -> None:
    if False:
        while True:
            i = 10
    engine = init_tcp_connection_engine(db_user=conn_vars['db_user'], db_name=conn_vars['db_name'], db_pass=conn_vars['db_pass'], db_host=conn_vars['db_host'])
    assert isinstance(engine, sqlalchemy.engine.base.Engine)
    assert conn_vars['db_name'] in engine.url

def test_init_unix_connection_engine(conn_vars: dict[str, str]) -> None:
    if False:
        i = 10
        return i + 15
    engine = init_unix_connection_engine(db_user=conn_vars['db_user'], db_name=conn_vars['db_name'], db_pass=conn_vars['db_pass'], instance_connection_name=conn_vars['instance_conn_name'], db_socket_dir=conn_vars['db_socket_dir'])
    assert isinstance(engine, sqlalchemy.engine.base.Engine)
    assert conn_vars['db_name'] in engine.url

def test_init_db(conn_vars: dict[str, str]) -> None:
    if False:
        print('Hello World!')
    table_name = f'votes_{uuid.uuid4().hex}'
    engine = init_db(db_user=conn_vars['db_user'], db_name=conn_vars['db_name'], db_pass=conn_vars['db_pass'], table_name=table_name, db_host=conn_vars['db_host'])
    assert isinstance(engine, sqlalchemy.engine.base.Engine)
    try:
        with engine.connect() as conn:
            conn.execute(f'SELECT count(*) FROM {table_name}').all()
    except Exception as error:
        pytest.fail(f"Database wasn't initialized properly: {error}")