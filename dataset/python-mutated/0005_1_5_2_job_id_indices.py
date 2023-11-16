"""Add indices in ``job`` table

Revision ID: 52d714495f0
Revises: 338e90f54d61
Create Date: 2015-10-20 03:17:01.962542

"""
from __future__ import annotations
from alembic import op
revision = '52d714495f0'
down_revision = '338e90f54d61'
branch_labels = None
depends_on = None
airflow_version = '1.5.2'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.create_index('idx_job_state_heartbeat', 'job', ['state', 'latest_heartbeat'], unique=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_index('idx_job_state_heartbeat', table_name='job')