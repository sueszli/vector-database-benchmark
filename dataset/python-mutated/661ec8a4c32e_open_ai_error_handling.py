"""open_ai_error_handling

Revision ID: 661ec8a4c32e
Revises: 40affbf3022b
Create Date: 2023-09-07 10:41:07.462436

"""
from alembic import op
import sqlalchemy as sa
revision = '661ec8a4c32e'
down_revision = 'c4f2f6ba602a'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.add_column('agent_execution_feeds', sa.Column('error_message', sa.String(), nullable=True))
    op.add_column('agent_executions', sa.Column('last_shown_error_id', sa.Integer(), nullable=True))

def downgrade() -> None:
    if False:
        print('Hello World!')
    op.drop_column('agent_executions', 'last_shown_error_id')
    op.drop_column('agent_execution_feeds', 'error_message')