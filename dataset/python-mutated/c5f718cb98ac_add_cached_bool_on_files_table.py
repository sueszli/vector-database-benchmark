"""
add cached bool on files table

Revision ID: c5f718cb98ac
Revises: 6073f65a2767
Create Date: 2023-05-12 08:00:47.726442
"""
import sqlalchemy as sa
from alembic import op
revision = 'c5f718cb98ac'
down_revision = '6073f65a2767'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('release_files', sa.Column('cached', sa.Boolean(), server_default=sa.text('false'), nullable=False, comment='If True, the object has been populated to our cache bucket.'))
    op.get_bind().commit()
    with op.get_context().autocommit_block():
        op.create_index('release_files_cached_idx', 'release_files', ['cached'], unique=False, postgresql_concurrently=True)

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_index('release_files_cached_idx', table_name='release_files')
    op.drop_column('release_files', 'cached')