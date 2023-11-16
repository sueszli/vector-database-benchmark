"""Add ``password`` column to ``user`` table

Revision ID: 561833c1c74b
Revises: 40e67319e3a9
Create Date: 2015-11-30 06:51:25.872557

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = '561833c1c74b'
down_revision = '40e67319e3a9'
branch_labels = None
depends_on = None
airflow_version = '1.6.2'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('user', sa.Column('password', sa.String(255)))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('user', 'password')