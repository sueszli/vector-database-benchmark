"""Add retry and restart metadata

Revision ID: af52717cf201
Revises: ad4b1b4d1e9d
Create Date: 2022-10-19 15:58:10.016251

"""
import sqlalchemy as sa
from alembic import op
revision = 'af52717cf201'
down_revision = '3ced59d8806b'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('task_run', schema=None) as batch_op:
        batch_op.add_column(sa.Column('flow_run_run_count', sa.Integer(), server_default='0', nullable=False))

def downgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('task_run', schema=None) as batch_op:
        batch_op.drop_column('flow_run_run_count')