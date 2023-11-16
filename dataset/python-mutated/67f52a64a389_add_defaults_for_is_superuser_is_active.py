"""
Add defaults for is_superuser, is_active

Revision ID: 67f52a64a389
Revises: 2d6390eebe90
Create Date: 2019-01-10 17:53:20.246417
"""
import sqlalchemy as sa
from alembic import op
revision = '67f52a64a389'
down_revision = '2d6390eebe90'

def upgrade():
    if False:
        print('Hello World!')
    op.alter_column('users', 'is_active', existing_type=sa.BOOLEAN(), server_default=sa.text('false'), existing_nullable=False)
    op.alter_column('users', 'is_superuser', existing_type=sa.BOOLEAN(), server_default=sa.text('false'), existing_nullable=False)

def downgrade():
    if False:
        return 10
    op.alter_column('users', 'is_superuser', existing_type=sa.BOOLEAN(), server_default=None, existing_nullable=False)
    op.alter_column('users', 'is_active', existing_type=sa.BOOLEAN(), server_default=None, existing_nullable=False)