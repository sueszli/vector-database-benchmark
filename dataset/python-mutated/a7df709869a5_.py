"""Add GitConfig model

Revision ID: a7df709869a5
Revises: b2cf0b907670
Create Date: 2022-12-21 10:32:20.932167

"""
import sqlalchemy as sa
from alembic import op
revision = 'a7df709869a5'
down_revision = 'b2cf0b907670'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.create_table('git_configs', sa.Column('uuid', sa.String(length=36), nullable=False), sa.Column('auth_user_uuid', sa.String(length=36), nullable=False), sa.Column('name', sa.String(), nullable=False), sa.Column('email', sa.String(), nullable=False), sa.ForeignKeyConstraint(['auth_user_uuid'], ['auth_users.uuid'], name=op.f('fk_git_configs_auth_user_uuid_auth_users'), ondelete='CASCADE'), sa.PrimaryKeyConstraint('uuid', name=op.f('pk_git_configs')))
    op.create_index(op.f('ix_git_configs_auth_user_uuid'), 'git_configs', ['auth_user_uuid'], unique=True)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_index(op.f('ix_git_configs_auth_user_uuid'), table_name='git_configs')
    op.drop_table('git_configs')