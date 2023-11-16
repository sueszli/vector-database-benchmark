"""Add updated_at column to DagRun and TaskInstance

Revision ID: ee8d93fcc81e
Revises: e07f49787c9d
Create Date: 2022-09-08 19:08:37.623121

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from airflow.migrations.db_types import TIMESTAMP
revision = 'ee8d93fcc81e'
down_revision = 'e07f49787c9d'
branch_labels = None
depends_on = None
airflow_version = '2.5.0'

def upgrade():
    if False:
        return 10
    'Apply add updated_at column to DagRun and TaskInstance'
    with op.batch_alter_table('task_instance') as batch_op:
        batch_op.add_column(sa.Column('updated_at', TIMESTAMP, default=sa.func.now))
    with op.batch_alter_table('dag_run') as batch_op:
        batch_op.add_column(sa.Column('updated_at', TIMESTAMP, default=sa.func.now))

def downgrade():
    if False:
        return 10
    'Unapply add updated_at column to DagRun and TaskInstance'
    with op.batch_alter_table('task_instance') as batch_op:
        batch_op.drop_column('updated_at')
    with op.batch_alter_table('dag_run') as batch_op:
        batch_op.drop_column('updated_at')