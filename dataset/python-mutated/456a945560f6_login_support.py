"""login support

Revision ID: 456a945560f6
Revises: 38c4e85512a9
Create Date: 2013-12-29 00:18:35.795259

"""
revision = '456a945560f6'
down_revision = '38c4e85512a9'
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('users', sa.Column('email', sa.String(length=64), nullable=True))
    op.add_column('users', sa.Column('password_hash', sa.String(length=128), nullable=True))
    op.create_index('ix_users_email', 'users', ['email'], unique=True)

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_index('ix_users_email', 'users')
    op.drop_column('users', 'password_hash')
    op.drop_column('users', 'email')