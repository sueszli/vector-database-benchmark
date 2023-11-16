"""Add map_index to Log.

Revision ID: 75d5ed6c2b43
Revises: 909884dea523
Create Date: 2022-03-15 16:35:54.816863
"""
from __future__ import annotations
from alembic import op
from sqlalchemy import Column, Integer
revision = '75d5ed6c2b43'
down_revision = '909884dea523'
branch_labels = None
depends_on = None
airflow_version = '2.3.0'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    'Add map_index to Log.'
    op.add_column('log', Column('map_index', Integer))

def downgrade():
    if False:
        i = 10
        return i + 15
    'Remove map_index from Log.'
    with op.batch_alter_table('log') as batch_op:
        batch_op.drop_column('map_index')