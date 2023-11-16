"""
Set User.last_login automatically in the DB

Revision ID: a65114e48d6f
Revises: 104b4c56862b
Create Date: 2016-06-11 00:28:39.176496
"""
import sqlalchemy as sa
from alembic import op
revision = 'a65114e48d6f'
down_revision = '104b4c56862b'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.alter_column('accounts_user', 'last_login', server_default=sa.func.now())

def downgrade():
    if False:
        while True:
            i = 10
    op.alter_column('accounts_user', 'last_login', server_default=None)