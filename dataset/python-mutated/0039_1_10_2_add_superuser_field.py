"""Add superuser field

Revision ID: 41f5f12752f8
Revises: 03bc53e68815
Create Date: 2018-12-04 15:50:04.456875

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = '41f5f12752f8'
down_revision = '03bc53e68815'
branch_labels = None
depends_on = None
airflow_version = '1.10.2'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('users', sa.Column('superuser', sa.Boolean(), default=False))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_column('users', 'superuser')