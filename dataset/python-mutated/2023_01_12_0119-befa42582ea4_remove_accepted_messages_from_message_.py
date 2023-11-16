"""remove accepted_messages from message_tree_state

Revision ID: befa42582ea4
Revises: 05975b274a81
Create Date: 2023-01-12 01:19:59.654864

"""
import sqlalchemy as sa
from alembic import op
revision = 'befa42582ea4'
down_revision = '05975b274a81'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        while True:
            i = 10
    op.drop_column('message_tree_state', 'accepted_messages')

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.add_column('message_tree_state', sa.Column('accepted_messages', sa.INTEGER(), autoincrement=False, nullable=False))