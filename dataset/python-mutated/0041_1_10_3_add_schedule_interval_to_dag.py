"""Add schedule interval to dag

Revision ID: dd4ecb8fbee3
Revises: c8ffec048a3b
Create Date: 2018-12-27 18:39:25.748032

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = 'dd4ecb8fbee3'
down_revision = 'c8ffec048a3b'
branch_labels = None
depends_on = None
airflow_version = '1.10.3'

def upgrade():
    if False:
        return 10
    op.add_column('dag', sa.Column('schedule_interval', sa.Text(), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('dag', 'schedule_interval')