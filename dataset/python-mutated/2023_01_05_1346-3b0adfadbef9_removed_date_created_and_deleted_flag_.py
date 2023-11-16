"""removed date_created and deleted flag from message_tree_state

Revision ID: 3b0adfadbef9
Revises: d4161e384f83
Create Date: 2023-01-05 13:46:11.338655

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = '3b0adfadbef9'
down_revision = 'd4161e384f83'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('message_tree_state', 'deleted')
    op.drop_column('message_tree_state', 'created_date')

def downgrade() -> None:
    if False:
        return 10
    op.add_column('message_tree_state', sa.Column('created_date', postgresql.TIMESTAMP(), server_default=sa.text('CURRENT_TIMESTAMP'), autoincrement=False, nullable=False))
    op.add_column('message_tree_state', sa.Column('deleted', sa.BOOLEAN(), server_default=sa.text('false'), autoincrement=False, nullable=False))