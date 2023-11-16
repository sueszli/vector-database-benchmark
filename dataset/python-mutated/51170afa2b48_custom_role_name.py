"""custom role name

Revision ID: 51170afa2b48
Revises: 595e27f36454
Create Date: 2015-03-11 16:29:51.037379

"""
revision = '51170afa2b48'
down_revision = '595e27f36454'
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('account', sa.Column('role_name', sa.String(length=256), nullable=True))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('account', 'role_name')