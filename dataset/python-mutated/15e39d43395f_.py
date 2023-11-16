"""Expand the Hash fields due to the adoption of DeepHash

Revision ID: 15e39d43395f
Revises: 7c54b06e227b
Create Date: 2020-09-07 12:29:37.391979

"""
revision = '15e39d43395f'
down_revision = '7c54b06e227b'
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        print('Hello World!')
    'This revision is going to expand the hash database fields from `varchar(32)` to `varchar(64)`. Due to the usage of DeepHash.\n    Note: This is going to result in all change items getting re-updated as the hashes are going to be different.\n    '
    op.alter_column('item', 'latest_revision_complete_hash', type_=sa.VARCHAR(64), existing_type=sa.VARCHAR(length=32), existing_nullable=True)
    op.alter_column('item', 'latest_revision_durable_hash', type_=sa.VARCHAR(64), existing_type=sa.VARCHAR(length=32), existing_nullable=True)

def downgrade():
    if False:
        while True:
            i = 10
    raise ValueError('You cannot downgrade from this DB revision!! Sorry!')