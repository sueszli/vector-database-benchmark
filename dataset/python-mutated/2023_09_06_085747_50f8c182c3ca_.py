"""Adds heartbeat_interval_seconds column to worker table

Revision ID: 50f8c182c3ca
Revises: 5f623ddbf7fe
Create Date: 2023-09-06 08:57:47.166983

"""
import sqlalchemy as sa
from alembic import op
revision = '50f8c182c3ca'
down_revision = '5f623ddbf7fe'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('worker', sa.Column('heartbeat_interval_seconds', sa.Integer(), nullable=True))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('worker', 'heartbeat_interval_seconds')