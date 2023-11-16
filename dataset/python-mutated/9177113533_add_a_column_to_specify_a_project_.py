"""
Add a column to specify a project specific upload limit

Revision ID: 9177113533
Revises: 10cb17aea73
Create Date: 2015-09-04 21:06:59.950947
"""
import sqlalchemy as sa
from alembic import op
revision = '9177113533'
down_revision = '10cb17aea73'

def upgrade():
    if False:
        return 10
    op.add_column('packages', sa.Column('upload_limit', sa.Integer(), nullable=True))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('packages', 'upload_limit')