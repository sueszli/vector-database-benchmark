"""Create a ``session`` table to store web session data

Revision ID: c381b21cb7e4
Revises: be2bfac3da23
Create Date: 2022-01-25 13:56:35.069429

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = 'c381b21cb7e4'
down_revision = 'be2bfac3da23'
branch_labels = None
depends_on = None
airflow_version = '2.2.4'
TABLE_NAME = 'session'

def upgrade():
    if False:
        return 10
    'Apply Create a ``session`` table to store web session data'
    op.create_table(TABLE_NAME, sa.Column('id', sa.Integer()), sa.Column('session_id', sa.String(255)), sa.Column('data', sa.LargeBinary()), sa.Column('expiry', sa.DateTime()), sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('session_id'))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    'Unapply Create a ``session`` table to store web session data'
    op.drop_table(TABLE_NAME)