"""Adding ``extra`` column to ``Log`` table

Revision ID: 502898887f84
Revises: 52d714495f0
Create Date: 2015-11-03 22:50:49.794097

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = '502898887f84'
down_revision = '52d714495f0'
branch_labels = None
depends_on = None
airflow_version = '1.6.0'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('log', sa.Column('extra', sa.Text(), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('log', 'extra')