"""Add ``dag_stats`` table

Revision ID: f2ca10b85618
Revises: 64de9cddf6c9
Create Date: 2016-07-20 15:08:28.247537

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from airflow.migrations.db_types import StringID
revision = 'f2ca10b85618'
down_revision = '64de9cddf6c9'
branch_labels = None
depends_on = None
airflow_version = '1.7.1.3'

def upgrade():
    if False:
        return 10
    op.create_table('dag_stats', sa.Column('dag_id', StringID(), nullable=False), sa.Column('state', sa.String(length=50), nullable=False), sa.Column('count', sa.Integer(), nullable=False, default=0), sa.Column('dirty', sa.Boolean(), nullable=False, default=False), sa.PrimaryKeyConstraint('dag_id', 'state'))

def downgrade():
    if False:
        print('Hello World!')
    op.drop_table('dag_stats')