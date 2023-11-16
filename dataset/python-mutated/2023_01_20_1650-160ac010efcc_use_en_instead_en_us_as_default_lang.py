"""use 'en' instead 'en-US' as default lang

Revision ID: 160ac010efcc
Revises: 4f26fec4d204
Create Date: 2023-01-20 16:50:00

"""
import sqlalchemy as sa
from alembic import op
revision = '160ac010efcc'
down_revision = '4f26fec4d204'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.drop_column('message', 'lang')
    op.add_column('message', sa.Column('lang', sa.String(length=32), server_default='en', nullable=False))

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.drop_column('message', 'lang')
    op.add_column('message', sa.Column('lang', sa.VARCHAR(length=200), autoincrement=False, nullable=False))