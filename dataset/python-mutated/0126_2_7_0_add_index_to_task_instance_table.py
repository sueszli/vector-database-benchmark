"""Add index to task_instance table

Revision ID: 937cbd173ca1
Revises: c804e5c76e3e
Create Date: 2023-05-03 11:31:32.527362

"""
from __future__ import annotations
from alembic import op
revision = '937cbd173ca1'
down_revision = 'c804e5c76e3e'
branch_labels = None
depends_on = None
airflow_version = '2.7.0'

def upgrade():
    if False:
        return 10
    'Apply Add index to task_instance table'
    op.create_index('ti_state_incl_start_date', 'task_instance', ['dag_id', 'task_id', 'state'], postgresql_include=['start_date'])

def downgrade():
    if False:
        return 10
    'Unapply Add index to task_instance table'
    op.drop_index('ti_state_incl_start_date', table_name='task_instance')