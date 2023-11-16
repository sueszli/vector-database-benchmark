"""Add retry and restart metadata

Revision ID: 8ea825da948d
Revises: ad4b1b4d1e9d
Create Date: 2022-10-19 16:51:10.239643

"""
import sqlalchemy as sa
from alembic import op
revision = '8ea825da948d'
down_revision = '3ced59d8806b'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('task_run', sa.Column('flow_run_run_count', sa.Integer(), server_default='0', nullable=False))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('task_run', 'flow_run_run_count')