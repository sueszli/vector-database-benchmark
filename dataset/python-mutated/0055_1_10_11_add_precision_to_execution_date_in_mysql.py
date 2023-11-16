"""Add Precision to ``execution_date`` in ``RenderedTaskInstanceFields`` table

Revision ID: a66efa278eea
Revises: 952da73b5eff
Create Date: 2020-06-16 21:44:02.883132

"""
from __future__ import annotations
from alembic import op
from sqlalchemy.dialects import mysql
revision = 'a66efa278eea'
down_revision = '952da73b5eff'
branch_labels = None
depends_on = None
airflow_version = '1.10.11'
TABLE_NAME = 'rendered_task_instance_fields'
COLUMN_NAME = 'execution_date'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    'Add Precision to ``execution_date`` in ``RenderedTaskInstanceFields`` table for MySQL'
    conn = op.get_bind()
    if conn.dialect.name == 'mysql':
        op.alter_column(table_name=TABLE_NAME, column_name=COLUMN_NAME, type_=mysql.TIMESTAMP(fsp=6), nullable=False)

def downgrade():
    if False:
        while True:
            i = 10
    'Unapply Add Precision to ``execution_date`` in ``RenderedTaskInstanceFields`` table'
    conn = op.get_bind()
    if conn.dialect.name == 'mysql':
        op.alter_column(table_name=TABLE_NAME, column_name=COLUMN_NAME, type_=mysql.TIMESTAMP(), nullable=False)