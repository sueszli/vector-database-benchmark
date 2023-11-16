"""Adding networkwhitelist table

Revision ID: fb592c81e71
Revises: 331ca47ce8ad
Create Date: 2014-10-16 20:30:14.435034

"""
revision = 'fb592c81e71'
down_revision = '331ca47ce8ad'
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.create_table('networkwhitelist', sa.Column('id', sa.Integer(), nullable=False), sa.Column('name', sa.String(length=512), nullable=True), sa.Column('notes', sa.String(length=512), nullable=True), sa.Column('cidr', postgresql.CIDR(), nullable=True), sa.PrimaryKeyConstraint('id'))

def downgrade():
    if False:
        return 10
    op.drop_table('networkwhitelist')