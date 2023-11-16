"""Add ``description`` and ``default_view`` column to ``dag`` table

Revision ID: c8ffec048a3b
Revises: 41f5f12752f8
Create Date: 2018-12-23 21:55:46.463634

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = 'c8ffec048a3b'
down_revision = '41f5f12752f8'
branch_labels = None
depends_on = None
airflow_version = '1.10.3'

def upgrade():
    if False:
        print('Hello World!')
    op.add_column('dag', sa.Column('description', sa.Text(), nullable=True))
    op.add_column('dag', sa.Column('default_view', sa.String(25), nullable=True))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('dag', 'description')
    op.drop_column('dag', 'default_view')