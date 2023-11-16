"""Add index on ``log`` table

Revision ID: dd25f486b8ea
Revises: 9635ae0956e7
Create Date: 2018-08-07 06:41:41.028249

"""
from __future__ import annotations
from alembic import op
revision = 'dd25f486b8ea'
down_revision = '9635ae0956e7'
branch_labels = None
depends_on = None
airflow_version = '1.10.2'

def upgrade():
    if False:
        print('Hello World!')
    op.create_index('idx_log_dag', 'log', ['dag_id'], unique=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_index('idx_log_dag', table_name='log')