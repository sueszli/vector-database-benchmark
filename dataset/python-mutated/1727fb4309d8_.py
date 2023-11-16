"""Updating max length of s3_name in account table

Revision ID: 1727fb4309d8
Revises: 51170afa2b48
Create Date: 2015-07-06 12:29:48.859104

"""
revision = '1727fb4309d8'
down_revision = '51170afa2b48'
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        while True:
            i = 10
    op.alter_column('account', 's3_name', type_=sa.VARCHAR(64), existing_type=sa.VARCHAR(length=32), nullable=True)

def downgrade():
    if False:
        print('Hello World!')
    op.alter_column('account', 's3_name', type_=sa.VARCHAR(32), existing_type=sa.VARCHAR(length=64), nullable=True)