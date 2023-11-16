"""
mandate sha256 hashes for all files

Revision ID: f392e419ea1b
Revises: d8301a1bf519
Create Date: 2016-01-04 16:20:50.428491
"""
from alembic import op
revision = 'f392e419ea1b'
down_revision = 'd8301a1bf519'

def upgrade():
    if False:
        print('Hello World!')
    op.alter_column('release_files', 'sha256_digest', nullable=False)

def downgrade():
    if False:
        i = 10
        return i + 15
    op.alter_column('release_files', 'sha256_digest', nullable=True)