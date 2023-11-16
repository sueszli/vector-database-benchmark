"""add ix_user_display_name_id

Revision ID: 4f26fec4d204
Revises: 0964ac95170d
Create Date: 2023-01-19 22:00:00

"""
from alembic import op
revision = '4f26fec4d204'
down_revision = '7f0a28a156f4'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        while True:
            i = 10
    op.create_index('ix_user_display_name_id', 'user', ['display_name', 'id'], unique=True)

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.drop_index('ix_user_display_name_id', table_name='user')