"""Add ``executor_config`` column to ``task_instance`` table

Revision ID: 33ae817a1ff4
Revises: 947454bf1dff
Create Date: 2017-09-11 15:26:47.598494

"""
from __future__ import annotations
import dill
import sqlalchemy as sa
from alembic import op
revision = '27c6a30d7c24'
down_revision = '33ae817a1ff4'
branch_labels = None
depends_on = None
airflow_version = '1.10.0'
TASK_INSTANCE_TABLE = 'task_instance'
NEW_COLUMN = 'executor_config'

def upgrade():
    if False:
        print('Hello World!')
    op.add_column(TASK_INSTANCE_TABLE, sa.Column(NEW_COLUMN, sa.PickleType(pickler=dill)))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_column(TASK_INSTANCE_TABLE, NEW_COLUMN)