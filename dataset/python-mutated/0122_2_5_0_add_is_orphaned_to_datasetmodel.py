"""Add is_orphaned to DatasetModel

Revision ID: 290244fb8b83
Revises: 1986afd32c1b
Create Date: 2022-11-22 00:12:53.432961

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = '290244fb8b83'
down_revision = '1986afd32c1b'
branch_labels = None
depends_on = None
airflow_version = '2.5.0'

def upgrade():
    if False:
        i = 10
        return i + 15
    'Add is_orphaned to DatasetModel'
    with op.batch_alter_table('dataset') as batch_op:
        batch_op.add_column(sa.Column('is_orphaned', sa.Boolean, default=False, nullable=False, server_default='0'))

def downgrade():
    if False:
        while True:
            i = 10
    'Remove is_orphaned from DatasetModel'
    with op.batch_alter_table('dataset') as batch_op:
        batch_op.drop_column('is_orphaned', mssql_drop_default=True)