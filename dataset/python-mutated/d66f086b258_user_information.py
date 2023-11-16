"""user information

Revision ID: d66f086b258
Revises: 56ed7d33de8d
Create Date: 2013-12-29 23:50:49.566954

"""
revision = 'd66f086b258'
down_revision = '56ed7d33de8d'
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        return 10
    op.add_column('users', sa.Column('about_me', sa.Text(), nullable=True))
    op.add_column('users', sa.Column('last_seen', sa.DateTime(), nullable=True))
    op.add_column('users', sa.Column('location', sa.String(length=64), nullable=True))
    op.add_column('users', sa.Column('member_since', sa.DateTime(), nullable=True))
    op.add_column('users', sa.Column('name', sa.String(length=64), nullable=True))

def downgrade():
    if False:
        print('Hello World!')
    op.drop_column('users', 'name')
    op.drop_column('users', 'member_since')
    op.drop_column('users', 'location')
    op.drop_column('users', 'last_seen')
    op.drop_column('users', 'about_me')