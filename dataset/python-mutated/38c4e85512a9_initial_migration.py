"""initial migration

Revision ID: 38c4e85512a9
Revises: None
Create Date: 2013-12-27 01:23:59.392801

"""
revision = '38c4e85512a9'
down_revision = None
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        i = 10
        return i + 15
    op.create_table('roles', sa.Column('id', sa.Integer(), nullable=False), sa.Column('name', sa.String(length=64), nullable=True), sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('name'))
    op.create_table('users', sa.Column('id', sa.Integer(), nullable=False), sa.Column('username', sa.String(length=64), nullable=True), sa.Column('role_id', sa.Integer(), nullable=True), sa.ForeignKeyConstraint(['role_id'], ['roles.id']), sa.PrimaryKeyConstraint('id'))
    op.create_index('ix_users_username', 'users', ['username'], unique=True)

def downgrade():
    if False:
        print('Hello World!')
    op.drop_index('ix_users_username', 'users')
    op.drop_table('users')
    op.drop_table('roles')