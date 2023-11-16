"""Add TI state index

Revision ID: 211e584da130
Revises: 2e82aab8ef20
Create Date: 2016-06-30 10:54:24.323588

"""
from __future__ import annotations
from alembic import op
revision = '211e584da130'
down_revision = '2e82aab8ef20'
branch_labels = None
depends_on = None
airflow_version = '1.7.1.3'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.create_index('ti_state', 'task_instance', ['state'], unique=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_index('ti_state', table_name='task_instance')