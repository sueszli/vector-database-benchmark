"""add dttm index on log table

Revision ID: 6abdffdd4815
Revises: 290244fb8b83
Create Date: 2023-01-13 13:57:14.412028

"""
from __future__ import annotations
from alembic import op
revision = '6abdffdd4815'
down_revision = '290244fb8b83'
branch_labels = None
depends_on = None
airflow_version = '2.6.0'

def upgrade():
    if False:
        i = 10
        return i + 15
    'Apply add dttm index on log table'
    op.create_index('idx_log_dttm', 'log', ['dttm'], unique=False)

def downgrade():
    if False:
        print('Hello World!')
    'Unapply add dttm index on log table'
    op.drop_index('idx_log_dttm', table_name='log')