"""Add ``start_date`` and ``end_date`` in ``dag_run`` table

Revision ID: 4446e08588
Revises: 561833c1c74b
Create Date: 2015-12-10 11:26:18.439223

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = '4446e08588'
down_revision = '561833c1c74b'
branch_labels = None
depends_on = None
airflow_version = '1.6.2'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('dag_run', sa.Column('end_date', sa.DateTime(), nullable=True))
    op.add_column('dag_run', sa.Column('start_date', sa.DateTime(), nullable=True))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('dag_run', 'start_date')
    op.drop_column('dag_run', 'end_date')