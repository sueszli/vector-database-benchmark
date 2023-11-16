"""fix_mssql_exec_date_rendered_task_instance_fields_for_MSSQL

Revision ID: 52d53670a240
Revises: 98271e7606e2
Create Date: 2020-10-13 15:13:24.911486

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mssql
revision = '52d53670a240'
down_revision = '98271e7606e2'
branch_labels = None
depends_on = None
airflow_version = '2.0.0'
TABLE_NAME = 'rendered_task_instance_fields'

def upgrade():
    if False:
        while True:
            i = 10
    '\n    Recreate RenderedTaskInstanceFields table changing timestamp to datetime2(6) when using MSSQL as\n    backend\n    '
    conn = op.get_bind()
    if conn.dialect.name == 'mssql':
        json_type = sa.Text
        op.drop_table(TABLE_NAME)
        op.create_table(TABLE_NAME, sa.Column('dag_id', sa.String(length=250), nullable=False), sa.Column('task_id', sa.String(length=250), nullable=False), sa.Column('execution_date', mssql.DATETIME2, nullable=False), sa.Column('rendered_fields', json_type(), nullable=False), sa.PrimaryKeyConstraint('dag_id', 'task_id', 'execution_date'))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    '\n    Recreate RenderedTaskInstanceFields table changing datetime2(6) to timestamp when using MSSQL as\n    backend\n    '
    conn = op.get_bind()
    if conn.dialect.name == 'mssql':
        json_type = sa.Text
        op.drop_table(TABLE_NAME)
        op.create_table(TABLE_NAME, sa.Column('dag_id', sa.String(length=250), nullable=False), sa.Column('task_id', sa.String(length=250), nullable=False), sa.Column('execution_date', sa.TIMESTAMP, nullable=False), sa.Column('rendered_fields', json_type(), nullable=False), sa.PrimaryKeyConstraint('dag_id', 'task_id', 'execution_date'))