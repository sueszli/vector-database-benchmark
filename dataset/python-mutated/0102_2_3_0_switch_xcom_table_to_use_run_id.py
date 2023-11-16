"""Switch XCom table to use ``run_id`` and add ``map_index``.

Revision ID: c306b5b5ae4a
Revises: a3bcd0914482
Create Date: 2022-01-19 03:20:35.329037
"""
from __future__ import annotations
from typing import Sequence
from alembic import op
from sqlalchemy import Column, Integer, LargeBinary, MetaData, Table, and_, literal_column, select
from airflow.migrations.db_types import TIMESTAMP, StringID
from airflow.migrations.utils import get_mssql_table_constraints
revision = 'c306b5b5ae4a'
down_revision = 'a3bcd0914482'
branch_labels = None
depends_on = None
airflow_version = '2.3.0'
metadata = MetaData()

def _get_new_xcom_columns() -> Sequence[Column]:
    if False:
        print('Hello World!')
    return [Column('dag_run_id', Integer(), nullable=False), Column('task_id', StringID(), nullable=False), Column('key', StringID(length=512), nullable=False), Column('value', LargeBinary), Column('timestamp', TIMESTAMP, nullable=False), Column('dag_id', StringID(), nullable=False), Column('run_id', StringID(), nullable=False), Column('map_index', Integer, nullable=False, server_default='-1')]

def _get_old_xcom_columns() -> Sequence[Column]:
    if False:
        print('Hello World!')
    return [Column('key', StringID(length=512), nullable=False, primary_key=True), Column('value', LargeBinary), Column('timestamp', TIMESTAMP, nullable=False), Column('task_id', StringID(length=250), nullable=False, primary_key=True), Column('dag_id', StringID(length=250), nullable=False, primary_key=True), Column('execution_date', TIMESTAMP, nullable=False, primary_key=True)]

def _get_dagrun_table() -> Table:
    if False:
        i = 10
        return i + 15
    return Table('dag_run', metadata, Column('id', Integer, primary_key=True), Column('dag_id', StringID(), nullable=False), Column('run_id', StringID(), nullable=False), Column('execution_date', TIMESTAMP, nullable=False))

def upgrade():
    if False:
        while True:
            i = 10
    'Switch XCom table to use run_id.\n\n    For performance reasons, this is done by creating a new table with needed\n    data pre-populated, adding back constraints we need, and renaming it to\n    replace the existing XCom table.\n    '
    conn = op.get_bind()
    is_sqlite = conn.dialect.name == 'sqlite'
    op.create_table('__airflow_tmp_xcom', *_get_new_xcom_columns())
    xcom = Table('xcom', metadata, *_get_old_xcom_columns())
    dagrun = _get_dagrun_table()
    query = select([dagrun.c.id, xcom.c.task_id, xcom.c.key, xcom.c.value, xcom.c.timestamp, xcom.c.dag_id, dagrun.c.run_id, literal_column('-1')]).select_from(xcom.join(right=dagrun, onclause=and_(xcom.c.dag_id == dagrun.c.dag_id, xcom.c.execution_date == dagrun.c.execution_date)))
    op.execute(f'INSERT INTO __airflow_tmp_xcom {query.selectable.compile(op.get_bind())}')
    if is_sqlite:
        op.execute('PRAGMA foreign_keys=off')
    op.drop_table('xcom')
    if is_sqlite:
        op.execute('PRAGMA foreign_keys=on')
    op.rename_table('__airflow_tmp_xcom', 'xcom')
    with op.batch_alter_table('xcom') as batch_op:
        batch_op.create_primary_key('xcom_pkey', ['dag_run_id', 'task_id', 'map_index', 'key'])
        batch_op.create_index('idx_xcom_key', ['key'])
        batch_op.create_foreign_key('xcom_task_instance_fkey', 'task_instance', ['dag_id', 'task_id', 'run_id', 'map_index'], ['dag_id', 'task_id', 'run_id', 'map_index'], ondelete='CASCADE')

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    'Switch XCom table back to use execution_date.\n\n    Basically an inverse operation.\n    '
    conn = op.get_bind()
    op.create_table('__airflow_tmp_xcom', *_get_old_xcom_columns())
    xcom = Table('xcom', metadata, *_get_new_xcom_columns())
    op.execute(xcom.delete().where(xcom.c.map_index != -1))
    dagrun = _get_dagrun_table()
    query = select([xcom.c.key, xcom.c.value, xcom.c.timestamp, xcom.c.task_id, xcom.c.dag_id, dagrun.c.execution_date]).select_from(xcom.join(right=dagrun, onclause=and_(xcom.c.dag_id == dagrun.c.dag_id, xcom.c.run_id == dagrun.c.run_id)))
    op.execute(f'INSERT INTO __airflow_tmp_xcom {query.selectable.compile(op.get_bind())}')
    op.drop_table('xcom')
    op.rename_table('__airflow_tmp_xcom', 'xcom')
    if conn.dialect.name == 'mssql':
        constraints = get_mssql_table_constraints(conn, 'xcom')
        (pk, _) = constraints['PRIMARY KEY'].popitem()
        op.drop_constraint(pk, 'xcom', type_='primary')
        op.create_primary_key(constraint_name='pk_xcom', table_name='xcom', columns=['dag_id', 'task_id', 'execution_date', 'key'])