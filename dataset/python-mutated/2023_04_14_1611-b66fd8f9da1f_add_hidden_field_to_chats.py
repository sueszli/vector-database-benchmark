"""Add hidden field to chats

Revision ID: b66fd8f9da1f
Revises: f0e18084aae4
Create Date: 2023-04-14 16:11:35.361507

"""
import sqlalchemy as sa
from alembic import op
revision = 'b66fd8f9da1f'
down_revision = 'f0e18084aae4'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.add_column('chat', sa.Column('hidden', sa.Boolean(), server_default=sa.text('false'), nullable=False))

def downgrade() -> None:
    if False:
        return 10
    op.drop_column('chat', 'hidden')