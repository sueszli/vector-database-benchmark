"""Add deleted field to user

Revision ID: f0e18084aae4
Revises: 78f16015b904
Create Date: 2023-04-12 20:33:28.239793

"""
import sqlalchemy as sa
from alembic import op
revision = 'f0e18084aae4'
down_revision = '78f16015b904'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.add_column('user', sa.Column('deleted', sa.Boolean(), server_default=sa.text('false'), nullable=False))

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('user', 'deleted')