"""Add time zone awareness

Revision ID: 0e2a74e0fc9f
Revises: d2ae31099d61
Create Date: 2017-11-10 22:22:31.326152

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from sqlalchemy import text
from sqlalchemy.dialects import mysql
revision = '0e2a74e0fc9f'
down_revision = 'd2ae31099d61'
branch_labels = None
depends_on = None
airflow_version = '1.10.0'

def upgrade():
    if False:
        i = 10
        return i + 15
    conn = op.get_bind()
    if conn.dialect.name == 'mysql':
        conn.execute(text("SET time_zone = '+00:00'"))
        cur = conn.execute(text('SELECT @@explicit_defaults_for_timestamp'))
        res = cur.fetchall()
        if res[0][0] == 0:
            raise Exception('Global variable explicit_defaults_for_timestamp needs to be on (1) for mysql')
        op.alter_column(table_name='chart', column_name='last_modified', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='dag', column_name='last_scheduler_run', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='dag', column_name='last_pickled', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='dag', column_name='last_expired', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='dag_pickle', column_name='created_dttm', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='dag_run', column_name='execution_date', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='dag_run', column_name='start_date', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='dag_run', column_name='end_date', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='import_error', column_name='timestamp', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='job', column_name='start_date', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='job', column_name='end_date', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='job', column_name='latest_heartbeat', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='log', column_name='dttm', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='log', column_name='execution_date', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='sla_miss', column_name='execution_date', type_=mysql.TIMESTAMP(fsp=6), nullable=False)
        op.alter_column(table_name='sla_miss', column_name='timestamp', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='task_fail', column_name='execution_date', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='task_fail', column_name='start_date', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='task_fail', column_name='end_date', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='task_instance', column_name='execution_date', type_=mysql.TIMESTAMP(fsp=6), nullable=False)
        op.alter_column(table_name='task_instance', column_name='start_date', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='task_instance', column_name='end_date', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='task_instance', column_name='queued_dttm', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='xcom', column_name='timestamp', type_=mysql.TIMESTAMP(fsp=6))
        op.alter_column(table_name='xcom', column_name='execution_date', type_=mysql.TIMESTAMP(fsp=6))
    else:
        if conn.dialect.name in ('sqlite', 'mssql'):
            return
        if conn.dialect.name == 'postgresql':
            conn.execute(text('set timezone=UTC'))
        op.alter_column(table_name='chart', column_name='last_modified', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='dag', column_name='last_scheduler_run', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='dag', column_name='last_pickled', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='dag', column_name='last_expired', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='dag_pickle', column_name='created_dttm', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='dag_run', column_name='execution_date', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='dag_run', column_name='start_date', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='dag_run', column_name='end_date', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='import_error', column_name='timestamp', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='job', column_name='start_date', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='job', column_name='end_date', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='job', column_name='latest_heartbeat', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='log', column_name='dttm', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='log', column_name='execution_date', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='sla_miss', column_name='execution_date', type_=sa.TIMESTAMP(timezone=True), nullable=False)
        op.alter_column(table_name='sla_miss', column_name='timestamp', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='task_fail', column_name='execution_date', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='task_fail', column_name='start_date', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='task_fail', column_name='end_date', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='task_instance', column_name='execution_date', type_=sa.TIMESTAMP(timezone=True), nullable=False)
        op.alter_column(table_name='task_instance', column_name='start_date', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='task_instance', column_name='end_date', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='task_instance', column_name='queued_dttm', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='xcom', column_name='timestamp', type_=sa.TIMESTAMP(timezone=True))
        op.alter_column(table_name='xcom', column_name='execution_date', type_=sa.TIMESTAMP(timezone=True))

def downgrade():
    if False:
        return 10
    conn = op.get_bind()
    if conn.dialect.name == 'mysql':
        conn.execute(text("SET time_zone = '+00:00'"))
        op.alter_column(table_name='chart', column_name='last_modified', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='dag', column_name='last_scheduler_run', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='dag', column_name='last_pickled', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='dag', column_name='last_expired', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='dag_pickle', column_name='created_dttm', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='dag_run', column_name='execution_date', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='dag_run', column_name='start_date', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='dag_run', column_name='end_date', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='import_error', column_name='timestamp', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='job', column_name='start_date', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='job', column_name='end_date', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='job', column_name='latest_heartbeat', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='log', column_name='dttm', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='log', column_name='execution_date', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='sla_miss', column_name='execution_date', type_=mysql.DATETIME(fsp=6), nullable=False)
        op.alter_column(table_name='sla_miss', column_name='timestamp', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='task_fail', column_name='execution_date', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='task_fail', column_name='start_date', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='task_fail', column_name='end_date', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='task_instance', column_name='execution_date', type_=mysql.DATETIME(fsp=6), nullable=False)
        op.alter_column(table_name='task_instance', column_name='start_date', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='task_instance', column_name='end_date', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='task_instance', column_name='queued_dttm', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='xcom', column_name='timestamp', type_=mysql.DATETIME(fsp=6))
        op.alter_column(table_name='xcom', column_name='execution_date', type_=mysql.DATETIME(fsp=6))
    else:
        if conn.dialect.name in ('sqlite', 'mssql'):
            return
        if conn.dialect.name == 'postgresql':
            conn.execute(text('set timezone=UTC'))
        op.alter_column(table_name='chart', column_name='last_modified', type_=sa.DateTime())
        op.alter_column(table_name='dag', column_name='last_scheduler_run', type_=sa.DateTime())
        op.alter_column(table_name='dag', column_name='last_pickled', type_=sa.DateTime())
        op.alter_column(table_name='dag', column_name='last_expired', type_=sa.DateTime())
        op.alter_column(table_name='dag_pickle', column_name='created_dttm', type_=sa.DateTime())
        op.alter_column(table_name='dag_run', column_name='execution_date', type_=sa.DateTime())
        op.alter_column(table_name='dag_run', column_name='start_date', type_=sa.DateTime())
        op.alter_column(table_name='dag_run', column_name='end_date', type_=sa.DateTime())
        op.alter_column(table_name='import_error', column_name='timestamp', type_=sa.DateTime())
        op.alter_column(table_name='job', column_name='start_date', type_=sa.DateTime())
        op.alter_column(table_name='job', column_name='end_date', type_=sa.DateTime())
        op.alter_column(table_name='job', column_name='latest_heartbeat', type_=sa.DateTime())
        op.alter_column(table_name='log', column_name='dttm', type_=sa.DateTime())
        op.alter_column(table_name='log', column_name='execution_date', type_=sa.DateTime())
        op.alter_column(table_name='sla_miss', column_name='execution_date', type_=sa.DateTime(), nullable=False)
        op.alter_column(table_name='sla_miss', column_name='timestamp', type_=sa.DateTime())
        op.alter_column(table_name='task_fail', column_name='execution_date', type_=sa.DateTime())
        op.alter_column(table_name='task_fail', column_name='start_date', type_=sa.DateTime())
        op.alter_column(table_name='task_fail', column_name='end_date', type_=sa.DateTime())
        op.alter_column(table_name='task_instance', column_name='execution_date', type_=sa.DateTime(), nullable=False)
        op.alter_column(table_name='task_instance', column_name='start_date', type_=sa.DateTime())
        op.alter_column(table_name='task_instance', column_name='end_date', type_=sa.DateTime())
        op.alter_column(table_name='task_instance', column_name='queued_dttm', type_=sa.DateTime())
        op.alter_column(table_name='xcom', column_name='timestamp', type_=sa.DateTime())
        op.alter_column(table_name='xcom', column_name='execution_date', type_=sa.DateTime())