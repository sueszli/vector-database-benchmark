"""Add index for ``dag_id`` column in ``job`` table.

Revision ID: 587bdf053233
Revises: c381b21cb7e4
Create Date: 2021-12-14 10:20:12.482940

"""
from __future__ import annotations
from alembic import op
revision = '587bdf053233'
down_revision = 'c381b21cb7e4'
branch_labels = None
depends_on = None
airflow_version = '2.2.4'

def upgrade():
    if False:
        return 10
    'Apply Add index for ``dag_id`` column in ``job`` table.'
    op.create_index('idx_job_dag_id', 'job', ['dag_id'], unique=False)

def downgrade():
    if False:
        print('Hello World!')
    'Unapply Add index for ``dag_id`` column in ``job`` table.'
    op.drop_index('idx_job_dag_id', table_name='job')