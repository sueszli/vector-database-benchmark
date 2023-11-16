"""make_filesystem_id_non_nullable

Revision ID: b7f98cfd6a70
Revises: d9d36b6f4d1e
Create Date: 2022-03-18 18:10:27.842201

"""
import sqlalchemy as sa
from alembic import op
revision = 'b7f98cfd6a70'
down_revision = 'd9d36b6f4d1e'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.execute('DELETE FROM seen_replies WHERE reply_id IN (SELECT replies.id FROM replies JOIN sources ON sources.id=replies.source_id WHERE filesystem_id IS NULL)')
    op.execute('DELETE FROM seen_files WHERE file_id IN (SELECT submissions.id FROM submissions JOIN sources ON sources.id=submissions.source_id WHERE filesystem_id IS NULL)')
    op.execute('DELETE FROM seen_messages WHERE message_id IN (SELECT submissions.id FROM submissions JOIN sources ON sources.id=submissions.source_id WHERE filesystem_id IS NULL)')
    for table in ('source_stars', 'submissions', 'replies'):
        op.execute(f'DELETE FROM {table} WHERE source_id IN (SELECT id FROM sources WHERE filesystem_id IS NULL)')
    op.execute('DELETE FROM sources WHERE filesystem_id IS NULL')
    with op.batch_alter_table('sources', schema=None) as batch_op:
        batch_op.alter_column('filesystem_id', existing_type=sa.VARCHAR(length=96), nullable=False)

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('sources', schema=None) as batch_op:
        batch_op.alter_column('filesystem_id', existing_type=sa.VARCHAR(length=96), nullable=True)