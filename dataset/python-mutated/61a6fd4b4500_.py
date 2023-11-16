"""Adding new fields to the User model for SECURITY_TRACKABLE

Revision ID: 61a6fd4b4500
Revises: 538eeb160af6
Create Date: 2016-04-23 18:17:47.216434

"""
revision = '61a6fd4b4500'
down_revision = '538eeb160af6'
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('user', sa.Column('current_login_at', sa.DateTime(), nullable=True))
    op.add_column('user', sa.Column('current_login_ip', sa.String(length=45), nullable=True))
    op.add_column('user', sa.Column('last_login_at', sa.DateTime(), nullable=True))
    op.add_column('user', sa.Column('last_login_ip', sa.String(length=45), nullable=True))
    op.add_column('user', sa.Column('login_count', sa.Integer(), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('user', 'login_count')
    op.drop_column('user', 'last_login_ip')
    op.drop_column('user', 'last_login_at')
    op.drop_column('user', 'current_login_ip')
    op.drop_column('user', 'current_login_at')