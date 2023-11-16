"""renamed_tokens_calls

Revision ID: 35e47f20475b
Revises: 598cfb37292a
Create Date: 2023-06-06 04:34:15.101672

"""
from alembic import op
import sqlalchemy as sa
revision = '35e47f20475b'
down_revision = '598cfb37292a'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        while True:
            i = 10
    op.add_column('agent_executions', sa.Column('num_of_calls', sa.Integer(), nullable=True))
    op.add_column('agent_executions', sa.Column('num_of_tokens', sa.Integer(), nullable=True))
    op.drop_column('agent_executions', 'calls')
    op.drop_column('agent_executions', 'tokens')

def downgrade() -> None:
    if False:
        return 10
    op.add_column('agent_executions', sa.Column('tokens', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('agent_executions', sa.Column('calls', sa.INTEGER(), autoincrement=False, nullable=True))
    op.drop_column('agent_executions', 'num_of_tokens')
    op.drop_column('agent_executions', 'num_of_calls')