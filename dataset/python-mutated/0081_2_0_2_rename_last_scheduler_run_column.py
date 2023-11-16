"""Rename ``last_scheduler_run`` column in ``DAG`` table to ``last_parsed_time``

Revision ID: 2e42bb497a22
Revises: 8646922c8a04
Create Date: 2021-03-04 19:50:38.880942

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql
revision = '2e42bb497a22'
down_revision = '8646922c8a04'
branch_labels = None
depends_on = None
airflow_version = '2.0.2'

def upgrade():
    if False:
        while True:
            i = 10
    'Apply Rename ``last_scheduler_run`` column in ``DAG`` table to ``last_parsed_time``'
    conn = op.get_bind()
    if conn.dialect.name == 'mssql':
        with op.batch_alter_table('dag') as batch_op:
            batch_op.alter_column('last_scheduler_run', new_column_name='last_parsed_time', type_=mssql.DATETIME2(precision=6))
    else:
        with op.batch_alter_table('dag') as batch_op:
            batch_op.alter_column('last_scheduler_run', new_column_name='last_parsed_time', type_=sa.TIMESTAMP(timezone=True))

def downgrade():
    if False:
        print('Hello World!')
    'Unapply Rename ``last_scheduler_run`` column in ``DAG`` table to ``last_parsed_time``'
    conn = op.get_bind()
    if conn.dialect.name == 'mssql':
        with op.batch_alter_table('dag') as batch_op:
            batch_op.alter_column('last_parsed_time', new_column_name='last_scheduler_run', type_=mssql.DATETIME2(precision=6))
    else:
        with op.batch_alter_table('dag') as batch_op:
            batch_op.alter_column('last_parsed_time', new_column_name='last_scheduler_run', type_=sa.TIMESTAMP(timezone=True))