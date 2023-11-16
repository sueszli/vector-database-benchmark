"""Add description field to ``Variable`` model

Revision ID: e165e7455d70
Revises: 90d1635d7b86
Create Date: 2021-04-11 22:28:02.107290

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = 'e165e7455d70'
down_revision = '90d1635d7b86'
branch_labels = None
depends_on = None
airflow_version = '2.1.0'

def upgrade():
    if False:
        print('Hello World!')
    'Apply Add description field to ``Variable`` model'
    with op.batch_alter_table('variable', schema=None) as batch_op:
        batch_op.add_column(sa.Column('description', sa.Text(), nullable=True))

def downgrade():
    if False:
        i = 10
        return i + 15
    'Unapply Add description field to ``Variable`` model'
    with op.batch_alter_table('variable', schema=None) as batch_op:
        batch_op.drop_column('description')