"""Make XCom primary key columns non-nullable

Revision ID: e9304a3141f0
Revises: 83f031fd9f1c
Create Date: 2021-04-06 13:22:02.197726

"""
from __future__ import annotations
from alembic import op
from airflow.migrations.db_types import TIMESTAMP, StringID
revision = 'e9304a3141f0'
down_revision = '83f031fd9f1c'
branch_labels = None
depends_on = None
airflow_version = '2.2.0'

def upgrade():
    if False:
        while True:
            i = 10
    'Apply Make XCom primary key columns non-nullable'
    conn = op.get_bind()
    with op.batch_alter_table('xcom') as bop:
        bop.alter_column('key', type_=StringID(length=512), nullable=False)
        bop.alter_column('execution_date', type_=TIMESTAMP, nullable=False)
        if conn.dialect.name == 'mssql':
            bop.create_primary_key(constraint_name='pk_xcom', columns=['dag_id', 'task_id', 'key', 'execution_date'])

def downgrade():
    if False:
        print('Hello World!')
    'Unapply Make XCom primary key columns non-nullable'
    conn = op.get_bind()
    with op.batch_alter_table('xcom') as bop:
        if conn.dialect.name == 'mssql':
            bop.drop_constraint('pk_xcom', type_='primary')
            bop.alter_column('key', type_=StringID(length=512), nullable=True)
            bop.alter_column('execution_date', type_=TIMESTAMP, nullable=True)