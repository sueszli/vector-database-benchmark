"""
Refactor description to its own table

Revision ID: 9ca7d5668af4
Revises: 42f0409bb702
Create Date: 2019-05-10 16:19:04.008388
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = '9ca7d5668af4'
down_revision = '42f0409bb702'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute('SET statement_timeout = 0')
    op.create_table('release_descriptions', sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False), sa.Column('content_type', sa.Text(), nullable=True), sa.Column('raw', sa.Text(), nullable=False), sa.Column('html', sa.Text(), nullable=False), sa.Column('rendered_by', sa.Text(), nullable=False), sa.Column('release_id', postgresql.UUID(as_uuid=True), nullable=False), sa.PrimaryKeyConstraint('id'))
    op.add_column('releases', sa.Column('description_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(None, 'releases', 'release_descriptions', ['description_id'], ['id'], onupdate='CASCADE', ondelete='CASCADE')
    op.execute(" WITH inserted_descriptions AS (\n                INSERT INTO release_descriptions\n                        (content_type, raw, html, rendered_by, release_id)\n                    SELECT\n                        description_content_type, COALESCE(description, ''), '', '', id\n                    FROM releases\n                    RETURNING release_id, id AS description_id\n            )\n            UPDATE releases\n            SET description_id = ids.description_id\n            FROM inserted_descriptions AS ids\n            WHERE id = release_id\n        ")
    op.alter_column('releases', 'description_id', nullable=False)
    op.drop_column('releases', 'description_content_type')
    op.drop_column('releases', 'description')
    op.drop_column('release_descriptions', 'release_id')

def downgrade():
    if False:
        i = 10
        return i + 15
    raise RuntimeError(f'Cannot downgrade past revision: {revision!r}')