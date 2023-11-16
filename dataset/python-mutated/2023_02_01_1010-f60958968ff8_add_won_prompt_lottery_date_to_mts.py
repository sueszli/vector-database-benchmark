"""add won_prompt_lottery_date to mts

Revision ID: f60958968ff8
Revises: 7b8f0011e0b0
Create Date: 2023-02-01 10:10:38.301707

"""
import sqlalchemy as sa
from alembic import op
revision = 'f60958968ff8'
down_revision = '7b8f0011e0b0'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.add_column('message_tree_state', sa.Column('won_prompt_lottery_date', sa.DateTime(timezone=True), nullable=True))

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.drop_column('message_tree_state', 'won_prompt_lottery_date')