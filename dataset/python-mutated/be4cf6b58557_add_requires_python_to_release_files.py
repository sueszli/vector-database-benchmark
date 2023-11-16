"""
Add a requires_python column to release_files; pursuant to enabling PEP 503.

Revision ID: be4cf6b58557
Revises: 3d2b8a42219a
Create Date: 2016-09-15 04:12:53.430363
"""
import sqlalchemy as sa
from alembic import op
revision = 'be4cf6b58557'
down_revision = '3d2b8a42219a'

def upgrade():
    if False:
        i = 10
        return i + 15
    '\n    Add column `requires_python` in the `release_files` table.\n    '
    op.add_column('release_files', sa.Column('requires_python', sa.Text(), nullable=True))
    op.execute(' UPDATE release_files\n            SET requires_python = releases.requires_python\n            FROM releases\n            WHERE\n                release_files.name=releases.name\n                AND release_files.version=releases.version;\n        ')
    op.execute('CREATE OR REPLACE FUNCTION update_release_files_requires_python()\n            RETURNS TRIGGER AS $$\n            BEGIN\n                UPDATE\n                    release_files\n                SET\n                    requires_python = releases.requires_python\n                FROM releases\n                WHERE\n                    release_files.name=releases.name\n                    AND release_files.version=releases.version\n                    AND release_files.name = NEW.name\n                    AND releases.version = NEW.version;\n                RETURN NULL;\n            END;\n            $$ LANGUAGE plpgsql;\n        ')
    op.execute(' CREATE TRIGGER releases_requires_python\n            AFTER INSERT OR UPDATE OF requires_python ON releases\n            FOR EACH ROW\n                EXECUTE PROCEDURE update_release_files_requires_python();\n        ')

def downgrade():
    if False:
        i = 10
        return i + 15
    '\n    Drop trigger and function that synchronize `releases`.\n    '
    op.execute('DROP TRIGGER releases_requires_python ON releases')
    op.execute('DROP FUNCTION update_release_files_requires_python()')
    op.drop_column('release_files', 'requires_python')