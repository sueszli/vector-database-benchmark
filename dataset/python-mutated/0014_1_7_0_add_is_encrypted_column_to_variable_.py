"""Add ``is_encrypted`` column to variable table

Revision ID: 1968acfc09e3
Revises: bba5a7cfc896
Create Date: 2016-02-02 17:20:55.692295

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = '1968acfc09e3'
down_revision = 'bba5a7cfc896'
branch_labels = None
depends_on = None
airflow_version = '1.7.0'

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column('variable', sa.Column('is_encrypted', sa.Boolean, default=False))

def downgrade():
    if False:
        return 10
    op.drop_column('variable', 'is_encrypted')