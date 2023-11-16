"""Add ``pool_slots`` field to ``task_instance``

Revision ID: a4c2fd67d16b
Revises: 7939bcff74ba
Create Date: 2020-01-14 03:35:01.161519

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = 'a4c2fd67d16b'
down_revision = '7939bcff74ba'
branch_labels = None
depends_on = None
airflow_version = '1.10.10'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('task_instance', sa.Column('pool_slots', sa.Integer, default=1))

def downgrade():
    if False:
        return 10
    op.drop_column('task_instance', 'pool_slots')