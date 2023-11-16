"""Add ``queued_at`` column in ``dag_run`` table

Revision ID: 97cdd93827b8
Revises: a13f7613ad25
Create Date: 2021-06-29 21:53:48.059438

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from airflow.migrations.db_types import TIMESTAMP
revision = '97cdd93827b8'
down_revision = 'a13f7613ad25'
branch_labels = None
depends_on = None
airflow_version = '2.1.3'

def upgrade():
    if False:
        i = 10
        return i + 15
    'Apply Add ``queued_at`` column in ``dag_run`` table'
    op.add_column('dag_run', sa.Column('queued_at', TIMESTAMP, nullable=True))

def downgrade():
    if False:
        return 10
    'Unapply Add ``queued_at`` column in ``dag_run`` table'
    with op.batch_alter_table('dag_run') as batch_op:
        batch_op.drop_column('queued_at')