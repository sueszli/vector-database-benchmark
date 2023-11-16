"""Rename BackgroundTask.task_uuid column to uuid

Revision ID: c276b66607aa
Revises: 92b3dc62f774
Create Date: 2021-01-27 09:50:29.164512

"""
from alembic import op
revision = 'c276b66607aa'
down_revision = '92b3dc62f774'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    op.alter_column('background_tasks', 'task_uuid', new_column_name='uuid')

def downgrade():
    if False:
        print('Hello World!')
    op.alter_column('background_tasks', 'uuid', new_column_name='task_uuid')