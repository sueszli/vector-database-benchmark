"""tables creation

Revision ID: acb725fe8e8e
Revises: 
Create Date: 2020-12-18 13:58:03.727009

"""
import sqlalchemy as sa
from alembic import op
revision = 'acb725fe8e8e'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.create_table('users', sa.Column('uuid', sa.String(length=36), nullable=False), sa.Column('username', sa.String(length=255), nullable=False), sa.Column('password_hash', sa.String(length=255), nullable=False), sa.Column('created', sa.DateTime(), server_default=sa.text("timezone('utc', now())"), nullable=False), sa.PrimaryKeyConstraint('uuid', 'username', 'password_hash', name=op.f('pk_users')), sa.UniqueConstraint('uuid', name=op.f('uq_users_uuid')))
    op.create_table('tokens', sa.Column('token', sa.String(length=255), nullable=True), sa.Column('user', sa.String(length=36), nullable=False), sa.Column('created', sa.DateTime(), server_default=sa.text("timezone('utc', now())"), nullable=False), sa.ForeignKeyConstraint(['user'], ['users.uuid'], name=op.f('fk_tokens_user_users')), sa.PrimaryKeyConstraint('user', name=op.f('pk_tokens')))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_table('tokens')
    op.drop_table('users')