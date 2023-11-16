"""Add user_id and dttm composite index to Log model

Revision ID: cdcf3d64daf4
Revises: 7fb8bca906d2
Create Date: 2022-04-05 13:27:06.028908

"""
revision = 'cdcf3d64daf4'
down_revision = '7fb8bca906d2'
from alembic import op

def upgrade():
    if False:
        return 10
    op.create_index(op.f('ix_logs_user_id_dttm'), 'logs', ['user_id', 'dttm'], unique=False)

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_index(op.f('ix_logs_user_id_dttm'), table_name='logs')