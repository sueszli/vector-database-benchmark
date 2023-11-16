"""Create index on ``job_id`` column in ``task_instance`` table

Revision ID: 947454bf1dff
Revises: bdaa763e6c56
Create Date: 2017-08-15 15:12:13.845074

"""
from __future__ import annotations
from alembic import op
revision = '947454bf1dff'
down_revision = 'bdaa763e6c56'
branch_labels = None
depends_on = None
airflow_version = '1.8.2'

def upgrade():
    if False:
        return 10
    op.create_index('ti_job_id', 'task_instance', ['job_id'], unique=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_index('ti_job_id', table_name='task_instance')