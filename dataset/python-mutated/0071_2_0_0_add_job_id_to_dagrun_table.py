"""Add ``creating_job_id`` to ``DagRun`` table

Revision ID: 364159666cbd
Revises: 52d53670a240
Create Date: 2020-10-10 09:08:07.332456

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = '364159666cbd'
down_revision = '52d53670a240'
branch_labels = None
depends_on = None
airflow_version = '2.0.0'

def upgrade():
    if False:
        print('Hello World!')
    'Apply Add ``creating_job_id`` to ``DagRun`` table'
    op.add_column('dag_run', sa.Column('creating_job_id', sa.Integer))

def downgrade():
    if False:
        print('Hello World!')
    'Unapply Add job_id to DagRun table'
    op.drop_column('dag_run', 'creating_job_id')