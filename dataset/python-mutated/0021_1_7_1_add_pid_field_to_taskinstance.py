"""Add ``pid`` field to ``TaskInstance``

Revision ID: 5e7d17757c7a
Revises: 8504051e801b
Create Date: 2016-12-07 15:51:37.119478

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = '5e7d17757c7a'
down_revision = '8504051e801b'
branch_labels = None
depends_on = None
airflow_version = '1.7.1.3'

def upgrade():
    if False:
        print('Hello World!')
    'Add pid column to task_instance table.'
    op.add_column('task_instance', sa.Column('pid', sa.Integer))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    'Drop pid column from task_instance table.'
    op.drop_column('task_instance', 'pid')