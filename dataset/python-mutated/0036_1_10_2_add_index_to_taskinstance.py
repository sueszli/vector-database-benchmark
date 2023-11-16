"""Add index to ``task_instance`` table

Revision ID: bf00311e1990
Revises: dd25f486b8ea
Create Date: 2018-09-12 09:53:52.007433

"""
from __future__ import annotations
from alembic import op
revision = 'bf00311e1990'
down_revision = 'dd25f486b8ea'
branch_labels = None
depends_on = None
airflow_version = '1.10.2'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.create_index('ti_dag_date', 'task_instance', ['dag_id', 'execution_date'], unique=False)

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_index('ti_dag_date', table_name='task_instance')