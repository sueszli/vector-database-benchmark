"""agent_executions_modified

Revision ID: 2cc1179834b0
Revises: 2f97c068fab9
Create Date: 2023-06-02 21:01:43.303961

"""
from alembic import op
import sqlalchemy as sa
revision = '2cc1179834b0'
down_revision = '2f97c068fab9'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    op.add_column('agent_executions', sa.Column('calls', sa.Integer(), nullable=True))
    op.add_column('agent_executions', sa.Column('tokens', sa.Integer(), nullable=True))

def downgrade() -> None:
    if False:
        print('Hello World!')
    op.drop_column('agent_executions', 'tokens')
    op.drop_column('agent_executions', 'calls')