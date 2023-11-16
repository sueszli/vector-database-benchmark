"""Add ``operator`` and ``queued_dttm`` to ``task_instance`` table

Revision ID: 338e90f54d61
Revises: 13eb55f81627
Create Date: 2015-08-25 06:09:20.460147

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = '338e90f54d61'
down_revision = '13eb55f81627'
branch_labels = None
depends_on = None
airflow_version = '1.5.0'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('task_instance', sa.Column('operator', sa.String(length=1000), nullable=True))
    op.add_column('task_instance', sa.Column('queued_dttm', sa.DateTime(), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('task_instance', 'queued_dttm')
    op.drop_column('task_instance', 'operator')