"""
Add field on User model for is_moderator

Revision ID: 3db69c05dd11
Revises: 67f52a64a389
Create Date: 2019-01-04 21:29:45.455607
"""
import sqlalchemy as sa
from alembic import op
revision = '3db69c05dd11'
down_revision = '67f52a64a389'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('users', sa.Column('is_moderator', sa.Boolean(), nullable=False, server_default=sa.sql.false()))

def downgrade():
    if False:
        return 10
    op.drop_column('users', 'is_moderator')