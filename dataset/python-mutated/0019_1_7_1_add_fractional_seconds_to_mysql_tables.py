"""Add fractional seconds to MySQL tables

Revision ID: 4addfa1236f1
Revises: f2ca10b85618
Create Date: 2016-09-11 13:39:18.592072

"""
from __future__ import annotations
from alembic import op
from sqlalchemy.dialects import mysql
revision = '4addfa1236f1'
down_revision = 'f2ca10b85618'
branch_labels = None
depends_on = None
airflow_version = '1.7.1.3'

def upgrade():
    if False:
        i = 10
        return i + 15
    conn = op.get_bind()
    if conn.dialect.name == 'mysql':
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

def downgrade():
    if False:
        print('Hello World!')
    conn = op.get_bind()
    if conn.dialect.name == 'mysql':
        op.alter_column(table_name='dag', column_name='last_scheduler_run', type_=mysql.DATETIME())
        op.alter_column(table_name='dag', column_name='last_pickled', type_=mysql.DATETIME())
        op.alter_column(table_name='dag', column_name='last_expired', type_=mysql.DATETIME())
        op.alter_column(table_name='dag_pickle', column_name='created_dttm', type_=mysql.DATETIME())
        op.alter_column(table_name='dag_run', column_name='execution_date', type_=mysql.DATETIME())
        op.alter_column(table_name='dag_run', column_name='start_date', type_=mysql.DATETIME())
        op.alter_column(table_name='dag_run', column_name='end_date', type_=mysql.DATETIME())
        op.alter_column(table_name='import_error', column_name='timestamp', type_=mysql.DATETIME())
        op.alter_column(table_name='job', column_name='start_date', type_=mysql.DATETIME())
        op.alter_column(table_name='job', column_name='end_date', type_=mysql.DATETIME())
        op.alter_column(table_name='job', column_name='latest_heartbeat', type_=mysql.DATETIME())
        op.alter_column(table_name='log', column_name='dttm', type_=mysql.DATETIME())
        op.alter_column(table_name='log', column_name='execution_date', type_=mysql.DATETIME())
        op.alter_column(table_name='sla_miss', column_name='execution_date', type_=mysql.DATETIME(), nullable=False)
        op.alter_column(table_name='sla_miss', column_name='timestamp', type_=mysql.DATETIME())
        op.alter_column(table_name='task_fail', column_name='execution_date', type_=mysql.DATETIME())
        op.alter_column(table_name='task_fail', column_name='start_date', type_=mysql.DATETIME())
        op.alter_column(table_name='task_fail', column_name='end_date', type_=mysql.DATETIME())
        op.alter_column(table_name='task_instance', column_name='execution_date', type_=mysql.DATETIME(), nullable=False)
        op.alter_column(table_name='task_instance', column_name='start_date', type_=mysql.DATETIME())
        op.alter_column(table_name='task_instance', column_name='end_date', type_=mysql.DATETIME())
        op.alter_column(table_name='task_instance', column_name='queued_dttm', type_=mysql.DATETIME())
        op.alter_column(table_name='xcom', column_name='timestamp', type_=mysql.DATETIME())
        op.alter_column(table_name='xcom', column_name='execution_date', type_=mysql.DATETIME())