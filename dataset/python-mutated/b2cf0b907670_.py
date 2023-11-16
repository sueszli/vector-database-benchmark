"""Add AuthUser model

Revision ID: b2cf0b907670
Revises: 68591a046415
Create Date: 2022-12-21 09:03:34.038538

"""
import sqlalchemy as sa
from alembic import op
revision = 'b2cf0b907670'
down_revision = '68591a046415'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    op.create_table('auth_users', sa.Column('uuid', sa.String(length=36), nullable=False), sa.PrimaryKeyConstraint('uuid', name=op.f('pk_auth_users')))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_table('auth_users')