"""add rank and indices to user_stats

Revision ID: 0964ac95170d
Revises: 423557e869e4
Create Date: 2023-01-15 16:54:09.510018

"""
import sqlalchemy as sa
from alembic import op
revision = '0964ac95170d'
down_revision = '423557e869e4'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.add_column('user_stats', sa.Column('rank', sa.Integer(), nullable=True))
    op.create_index('ix_user_stats__timeframe__rank__user_id', 'user_stats', ['time_frame', 'rank', 'user_id'], unique=True)
    op.create_index('ix_user_stats__timeframe__user_id', 'user_stats', ['time_frame', 'user_id'], unique=True)

def downgrade() -> None:
    if False:
        while True:
            i = 10
    op.drop_index('ix_user_stats__timeframe__user_id', table_name='user_stats')
    op.drop_index('ix_user_stats__timeframe__rank__user_id', table_name='user_stats')
    op.drop_column('user_stats', 'rank')