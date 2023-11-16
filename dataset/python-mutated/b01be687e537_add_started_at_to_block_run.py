"""Add started_at to block run

Revision ID: b01be687e537
Revises: 067326f43bc3
Create Date: 2022-09-07 10:37:29.955476

"""
from alembic import op
import sqlalchemy as sa
revision = 'b01be687e537'
down_revision = '067326f43bc3'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.add_column('block_run', sa.Column('started_at', sa.DateTime(timezone=True), nullable=True))

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('block_run', 'started_at')