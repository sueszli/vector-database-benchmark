"""add lang to message_tree_state

Revision ID: 9db92d504f64
Revises: 8cd0c34d0c3c
Create Date: 2023-02-26 00:52:40.624843

"""
import sqlalchemy as sa
from alembic import op
revision = '9db92d504f64'
down_revision = '8cd0c34d0c3c'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        print('Hello World!')
    op.add_column('message_tree_state', sa.Column('lang', sa.String(length=32), nullable=True))
    op.execute('WITH msg AS (SELECT id, lang FROM message WHERE parent_id is NULL) UPDATE message_tree_state mts SET lang = msg.lang FROM msg WHERE mts.message_tree_id = msg.id')
    op.alter_column('message_tree_state', 'lang', nullable=False)
    op.drop_index('ix_message_tree_state_state', table_name='message_tree_state')
    op.create_index('ix_message_tree_state__lang__state', 'message_tree_state', ['state', 'lang'], unique=False)

def downgrade() -> None:
    if False:
        return 10
    op.drop_index('ix_message_tree_state__lang__state', table_name='message_tree_state')
    op.create_index('ix_message_tree_state_state', 'message_tree_state', ['state'], unique=False)
    op.drop_column('message_tree_state', 'lang')