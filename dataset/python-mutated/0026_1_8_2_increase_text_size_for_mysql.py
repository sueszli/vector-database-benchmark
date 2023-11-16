"""Increase text size for MySQL (not relevant for other DBs' text types)

Revision ID: d2ae31099d61
Revises: 947454bf1dff
Create Date: 2017-08-18 17:07:16.686130

"""
from __future__ import annotations
from alembic import op
from sqlalchemy.dialects import mysql
revision = 'd2ae31099d61'
down_revision = '947454bf1dff'
branch_labels = None
depends_on = None
airflow_version = '1.8.2'

def upgrade():
    if False:
        print('Hello World!')
    conn = op.get_bind()
    if conn.dialect.name == 'mysql':
        op.alter_column(table_name='variable', column_name='val', type_=mysql.MEDIUMTEXT)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    conn = op.get_bind()
    if conn.dialect.name == 'mysql':
        op.alter_column(table_name='variable', column_name='val', type_=mysql.TEXT)