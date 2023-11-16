"""Change field in ``DagCode`` to ``MEDIUMTEXT`` for MySql

Revision ID: e959f08ac86c
Revises: 64a7d6477aae
Create Date: 2020-12-07 16:31:43.982353

"""
from __future__ import annotations
from alembic import op
from sqlalchemy.dialects import mysql
revision = 'e959f08ac86c'
down_revision = '64a7d6477aae'
branch_labels = None
depends_on = None
airflow_version = '2.0.0'

def upgrade():
    if False:
        print('Hello World!')
    conn = op.get_bind()
    if conn.dialect.name == 'mysql':
        op.alter_column(table_name='dag_code', column_name='source_code', type_=mysql.MEDIUMTEXT, nullable=False)

def downgrade():
    if False:
        print('Hello World!')
    pass