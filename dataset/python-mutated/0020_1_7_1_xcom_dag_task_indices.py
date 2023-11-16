"""Add indices on ``xcom`` table

Revision ID: 8504051e801b
Revises: 4addfa1236f1
Create Date: 2016-11-29 08:13:03.253312

"""
from __future__ import annotations
from alembic import op
revision = '8504051e801b'
down_revision = '4addfa1236f1'
branch_labels = None
depends_on = None
airflow_version = '1.7.1.3'

def upgrade():
    if False:
        while True:
            i = 10
    'Create Index.'
    op.create_index('idx_xcom_dag_task_date', 'xcom', ['dag_id', 'task_id', 'execution_date'], unique=False)

def downgrade():
    if False:
        while True:
            i = 10
    'Drop Index.'
    op.drop_index('idx_xcom_dag_task_date', table_name='xcom')