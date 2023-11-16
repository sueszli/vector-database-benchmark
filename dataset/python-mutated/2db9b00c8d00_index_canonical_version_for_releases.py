"""
index canonical_version for releases

Revision ID: 2db9b00c8d00
Revises: 8a335305fd39
Create Date: 2022-07-22 16:54:20.701648
"""
from alembic import op
revision = '2db9b00c8d00'
down_revision = '8a335305fd39'

def upgrade():
    if False:
        print('Hello World!')
    op.get_bind().commit()
    with op.get_context().autocommit_block():
        op.create_index('release_canonical_version_idx', 'releases', ['canonical_version'], unique=False, postgresql_concurrently=True)

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_index('release_canonical_version_idx', table_name='releases')