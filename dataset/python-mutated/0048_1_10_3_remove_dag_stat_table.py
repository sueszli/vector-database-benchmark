"""Remove ``dag_stat`` table

Revision ID: a56c9515abdc
Revises: c8ffec048a3b
Create Date: 2018-12-27 10:27:59.715872

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = 'a56c9515abdc'
down_revision = 'c8ffec048a3b'
branch_labels = None
depends_on = None
airflow_version = '1.10.3'

def upgrade():
    if False:
        return 10
    'Drop dag_stats table'
    op.drop_table('dag_stats')

def downgrade():
    if False:
        while True:
            i = 10
    'Create dag_stats table'
    op.create_table('dag_stats', sa.Column('dag_id', sa.String(length=250), nullable=False), sa.Column('state', sa.String(length=50), nullable=False), sa.Column('count', sa.Integer(), nullable=False, default=0), sa.Column('dirty', sa.Boolean(), nullable=False, default=False), sa.PrimaryKeyConstraint('dag_id', 'state'))